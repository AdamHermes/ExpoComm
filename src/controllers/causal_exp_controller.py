from .basic_controller import BasicMAC
from .ExpoComm_controller import ExpoCommMAC


class CausalExpMAC(ExpoCommMAC):
    """Reuse ExpoComm episode lifecycle and auxiliary-Q learner interface."""

    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)
        if self.agent_output_type != "q":
            raise ValueError("CausalExpMAC currently supports Q-learning only")

    def forward(self, ep_batch, t, test_mode=False):
        inputs = BasicMAC._build_inputs(self, ep_batch, t)
        q, self.hidden_states, message = self.agent(
            inputs, self.hidden_states, t=t, test_mode=test_mode)
        prediction = self.agent.aux_forward(message)
        return (q.reshape(ep_batch.batch_size, self.n_agents, -1),
                prediction.reshape(ep_batch.batch_size, self.n_agents, -1))
