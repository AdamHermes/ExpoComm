import torch as th
from .ExpoComm_controller import ExpoCommMAC


class ExpoCommMultiViewMAC(ExpoCommMAC):
    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)
        if not self.one_peer or args.agent_output_type != "q":
            raise ValueError("Multiview MAC requires one_peer=True and Q outputs")

    def forward(self, ep_batch, t, test_mode=False, return_features=False):
        active = None
        if "agent_active" in ep_batch.scheme:
            active = ep_batch["agent_active"][:, t].to(th.bool)
            retain = active
            if "agent_reset" in ep_batch.scheme:
                retain = active & ~ep_batch["agent_reset"][:, t].bool()
            # Clear inactive slots, including state carried across absence.
            self.hidden_states = tuple(
                value.reshape(ep_batch.batch_size, self.n_agents, -1) * retain
                for value in self.hidden_states)
        inputs, neighbors = self._build_inputs(ep_batch, t)
        if self.args.obs_last_action and "agent_reset" in ep_batch.scheme:
            # The previous action belonged to the old occupant of this slot.
            start = ep_batch["obs"].size(-1)
            end = start + self.args.n_actions
            inputs = inputs.clone()
            inputs[:, start:end] *= ~ep_batch["agent_reset"][:, t].reshape(-1, 1).bool()
        q, self.hidden_states, message = self.agent(inputs, self.hidden_states, neighbors, test_mode)
        h, m = (v.reshape(ep_batch.batch_size, self.n_agents, -1) for v in self.hidden_states)
        if active is not None:
            h, m = h * active, m * active
        self.hidden_states = (h, m)
        q = q.reshape(ep_batch.batch_size, self.n_agents, -1)
        # Team-state concatenations can be very large. Do not materialize
        # [B,N,state_dim] predictions when reconstruction is disabled.
        prediction = self.agent.aux_forward(m) if self.args.aux_coef > 0 else m.new_zeros(m.shape[:2] + (0,))
        if return_features:
            return q, prediction, h, m
        return q, prediction
