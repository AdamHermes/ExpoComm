REGISTRY = {}

from .rnn_agent import RNNAgent
from .rnn_ns_agent import RNNNSAgent
from .rnn_feature_agent import RNNFeatureAgent
from .ExpoComm_agent import (
    ExpoCommSAgent,
    ExpoCommOAgent,
    ExpoCommSContAgent,
    ExpoCommOContAgent,
)


REGISTRY["ExpoComm_static"] = ExpoCommSAgent
REGISTRY["ExpoComm_one_peer"] = ExpoCommOAgent
REGISTRY["ExpoComm_static_cont"] = ExpoCommSContAgent
REGISTRY["ExpoComm_one_peer_cont"] = ExpoCommOContAgent

from .causal_exp_agent import CausalExpAgent

REGISTRY["causal_exp"] = CausalExpAgent

from .expocomm_multiview_agent import ExpoCommMultiViewAgent
REGISTRY["expocomm_multiview"] = ExpoCommMultiViewAgent
