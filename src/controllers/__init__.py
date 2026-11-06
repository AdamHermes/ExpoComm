REGISTRY = {}

from .basic_controller import BasicMAC
from .non_shared_controller import NonSharedMAC
from .maddpg_controller import MADDPGMAC
from .ExpoComm_controller import ExpoCommMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["non_shared_mac"] = NonSharedMAC
REGISTRY["maddpg_mac"] = MADDPGMAC
REGISTRY["ExpoComm_mac"] = ExpoCommMAC

from .causal_exp_controller import CausalExpMAC

REGISTRY["causal_exp_mac"] = CausalExpMAC

from .expocomm_multiview_controller import ExpoCommMultiViewMAC
REGISTRY["expocomm_multiview_mac"] = ExpoCommMultiViewMAC
