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

    def init_hidden(self, batch_size):

        h, msg = self.agent.init_hidden()
        h = h.unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav
        msg = msg.unsqueeze(0).expand(batch_size, self.n_agents, -1)
        self.hidden_states = (h, msg)

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
        agent_outs, self.hidden_states, msgs = self.agent(
            agent_inputs,
            self.hidden_states,
            topk_indices,
            test_mode=test_mode,
        )
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

    def communication_stats(self, batch, t):
        """Scheduled non-self payload before drops; no headers/control traffic.

        Counts fixed slots like the original execution, including zero-message
        inactive slots. These are model link counts, not measured network IO.
        """
        neighbors = get_exp_neighbors(batch.batch_size, self.n_agents, self.args.topk_neighbors)
        if self.one_peer:
            neighbors = neighbors[:, :, t % self.args.topk_neighbors:t % self.args.topk_neighbors + 1]
        receivers = th.arange(self.n_agents).view(1, -1, 1)
        packets = (neighbors != receivers).sum().item()
        byte_width = self.agent.fc1.weight.element_size()
        return {"comm_packets": packets,
                "comm_payload_bytes": packets * self.args.hidden_dim * byte_width}
