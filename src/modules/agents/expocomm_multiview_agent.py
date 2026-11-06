from .ExpoComm_agent import ExpoCommOAgent
from modules.communication.multiview_encoder import MultiViewEncoder


class ExpoCommMultiViewAgent(ExpoCommOAgent):
    """Original one-peer execution; multiview is invoked only by the learner."""

    def __init__(self, input_shape, args):
        super().__init__(input_shape, args)
        self.multiview = MultiViewEncoder(args)
