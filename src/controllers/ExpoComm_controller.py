from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th

from .basic_controller import BasicMAC


def get_exp_neighbors(bs, n_agents, topk):
    """
    positions: (batch_size, n_agents* 2)"""

    topk_indices = th.arange(topk - 1)
    topk_indices = th.pow(2, topk_indices)
    topk_indices = th.cat([th.zeros(1), topk_indices])

    agent_ind = th.arange(n_agents)
    topk_indices = agent_ind[:, None] + topk_indices[None, :]
    topk_indices = topk_indices % n_agents
    topk_indices = topk_indices[None, :, :].expand(bs, -1, -1)

    # (bs, n_agents, topk)
    return topk_indices.long()


class ExpoCommMAC(BasicMAC):

    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)
        if hasattr(args, "one_peer"):
            self.one_peer = args.one_peer
        else:
            self.one_peer = False
        self.packet_loss_recovery = getattr(args, "packet_loss_recovery", "hold")
        if self.packet_loss_recovery not in {"hold", "mse_imputer"}:
            raise ValueError(
                "ExpoCommMAC supports packet_loss_recovery 'hold' or "
                f"'mse_imputer', got {self.packet_loss_recovery!r}"
            )
        if self.packet_loss_recovery == "mse_imputer" and not self.one_peer:
            raise ValueError("mse_imputer currently supports one_peer ExpoComm only")
        self.communication_recovery_state = None
        self.last_imputer_training_data = None

    def init_hidden(self, batch_size):

        h, msg = self.agent.init_hidden()
        h = h.unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav
        msg = msg.unsqueeze(0).expand(batch_size, self.n_agents, -1)
        self.hidden_states = (h, msg)
        if self.packet_loss_recovery == "mse_imputer":
            device = msg.device
            hidden_dim = msg.size(-1)
            self.communication_recovery_state = {
                "cache": msg.new_zeros(batch_size, self.n_agents, self.n_agents, hidden_dim),
                "age": th.zeros(batch_size, self.n_agents, self.n_agents, device=device, dtype=th.long),
                "valid": th.zeros(batch_size, self.n_agents, self.n_agents, device=device, dtype=th.bool),
            }
        else:
            self.communication_recovery_state = None
        self.last_imputer_training_data = None

    def load_models(self, path):
        super().load_models(path)
        imputer_path = getattr(self.args, "packet_imputer_checkpoint", "")
        if not imputer_path:
            return
        if self.packet_loss_recovery != "mse_imputer":
            raise ValueError(
                "packet_imputer_checkpoint requires packet_loss_recovery=mse_imputer"
            )
        payload = th.load(imputer_path, map_location=lambda storage, loc: storage)
        state_dict = payload.get("message_imputer", payload)
        self.agent.message_imputer.load_state_dict(state_dict)

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions foser the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs, _ = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(
            agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode
        )
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs, topk_indices = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        delivery_mask = None
        recovery_context = None
        # ExpoCommAgent returns recurrent states flattened as (B * N, H),
        # whereas the receiver--sender cache is indexed as (B, N, N, H).
        # Restore the agent axis before selecting each sender's old message.
        previous_msg = self.hidden_states[1].reshape(
            ep_batch.batch_size, self.n_agents, -1
        )
        if self.packet_loss_recovery == "mse_imputer":
            delivery_mask = self.agent._packet_keep_mask(topk_indices, test_mode)
            recovery_context = self._build_imputer_context(topk_indices, t)
            if getattr(self.args, "imputer_collect_targets", False):
                source_index = topk_indices.unsqueeze(-1).expand(
                    -1, -1, previous_msg.size(-1)
                )
                # Centralized training target only. It is never requested in
                # decentralized evaluation because imputer_collect_targets is
                # false there.
                imputer_target = previous_msg.gather(1, source_index).detach()
        agent_outs, self.hidden_states, msgs = self.agent(
            agent_inputs,
            self.hidden_states,
            topk_indices,
            test_mode=test_mode,
            delivery_mask=delivery_mask,
            recovery_context=recovery_context,
        )
        if self.packet_loss_recovery == "mse_imputer":
            self._update_message_cache(topk_indices, delivery_mask, previous_msg)
            if getattr(self.args, "imputer_collect_targets", False):
                receiver_indices = th.arange(
                    self.n_agents, device=topk_indices.device
                ).unsqueeze(0).expand_as(topk_indices)
                remote_loss = topk_indices.ne(receiver_indices) & ~delivery_mask
                self.last_imputer_training_data = {
                    "prediction": self.agent.last_imputed_remote_message,
                    "target": imputer_target,
                    "eligible": remote_loss & recovery_context["cache_usable"],
                    "cached_msg": recovery_context["cached_msg"].detach(),
                }
        states_predicted = self.agent.aux_forward(msgs)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(
                    ep_batch.batch_size * self.n_agents, -1
                )
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        # CAUTION: API changed
        return agent_outs.view(
            ep_batch.batch_size, self.n_agents, -1
        ), states_predicted.view(ep_batch.batch_size, self.n_agents, -1)

    def _build_imputer_context(self, sender_indices, t):
        state = self.communication_recovery_state
        batch_size, n_agents = sender_indices.shape
        hidden_dim = state["cache"].size(-1)
        gather_index = sender_indices[:, :, None, None].expand(
            -1, -1, 1, hidden_dim
        )
        cached_msg = state["cache"].gather(2, gather_index)[:, :, 0, :]
        scalar_index = sender_indices[:, :, None]
        cache_age = state["age"].gather(2, scalar_index)[:, :, 0]
        cache_valid = state["valid"].gather(2, scalar_index)[:, :, 0]
        ttl = int(getattr(self.args, "packet_cache_ttl", 8))
        if ttl < 0:
            raise ValueError("packet_cache_ttl must be non-negative")
        cache_usable = cache_valid & (cache_age <= ttl)
        denominator = max(ttl, 1)
        normalized_age = cache_age.clamp(max=ttl + 1).to(cached_msg.dtype)
        normalized_age = (normalized_age / denominator).unsqueeze(-1)
        offset_index = th.full_like(sender_indices, t % self.args.topk_neighbors)
        return {
            "cached_msg": cached_msg,
            "cache_usable": cache_usable,
            "normalized_age": normalized_age,
            "offset_index": offset_index,
        }

    def _update_message_cache(self, sender_indices, delivery_mask, previous_msg):
        """Update only links genuinely delivered before this step's msg update."""
        state = self.communication_recovery_state
        batch_size, n_agents = sender_indices.shape
        hidden_dim = previous_msg.size(-1)
        receiver_indices = th.arange(n_agents, device=sender_indices.device)
        receiver_indices = receiver_indices.unsqueeze(0).expand(batch_size, -1)
        remote_link = sender_indices.ne(receiver_indices)
        delivered_remote = delivery_mask & remote_link

        source_index = sender_indices.unsqueeze(-1).expand(-1, -1, hidden_dim)
        source_msg = previous_msg.gather(1, source_index)
        cache_index = sender_indices[:, :, None, None].expand(-1, -1, 1, hidden_dim)
        old_selected = state["cache"].gather(2, cache_index)[:, :, 0, :]
        new_selected = th.where(
            delivered_remote.unsqueeze(-1), source_msg, old_selected
        )
        state["cache"] = state["cache"].scatter(
            2, cache_index, new_selected.unsqueeze(2)
        )

        valid_index = sender_indices.unsqueeze(-1)
        old_valid = state["valid"].gather(2, valid_index)[:, :, 0]
        new_valid = old_valid | delivered_remote
        state["valid"] = state["valid"].scatter(
            2, valid_index, new_valid.unsqueeze(-1)
        )

        incremented_age = th.where(state["valid"], state["age"] + 1, state["age"])
        old_age = incremented_age.gather(2, valid_index)[:, :, 0]
        new_age = th.where(delivered_remote, th.zeros_like(old_age), old_age)
        state["age"] = incremented_age.scatter(
            2, valid_index, new_age.unsqueeze(-1)
        )

    def detach_imputer_history(self):
        """Truncate BPTT between steps for frozen-policy imputer training."""
        if self.packet_loss_recovery != "mse_imputer":
            return
        h, msg = self.hidden_states
        self.hidden_states = (h.detach(), msg.detach())
        self.communication_recovery_state["cache"] = (
            self.communication_recovery_state["cache"].detach()
        )

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []

        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t - 1])
        if self.args.obs_agent_id:
            inputs.append(
                th.eye(self.n_agents, device=batch.device)
                .unsqueeze(0)
                .expand(bs, -1, -1)
            )

        inputs = th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)
        topk_indices = get_exp_neighbors(
            bs=bs, n_agents=self.n_agents, topk=self.args.topk_neighbors
        )
        topk_indices = topk_indices.to(device=batch.device)
        if self.one_peer:
            topk_index = t % self.args.topk_neighbors
            topk_indices = topk_indices[:, :, topk_index]
        return inputs, topk_indices
