"""Lazy factories: a Battle run need not import SC2 or IMP dependencies."""
import os
import sys
from importlib import import_module
from functools import partial
from .multiagentenv import MultiAgentEnv

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH", os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))


def env_fn(env, **kwargs):
    return env(**kwargs)


def _make(module, name, **kwargs):
    return getattr(import_module(module), name)(**kwargs)


REGISTRY = {
    "sc2": partial(_make, "smac.env", "StarCraft2Env"),
    "struct_marl": partial(_make, "imp_marl.imp_wrappers.epymarl_wrapper.epymarl_wrap_ma_struct", "EPymarlMAStruct"),
    "gymma": partial(_make, "envs.gymma_wrapper", "_GymmaWrapper"),
    "MAgent_Battle": partial(_make, "envs.battle_wrappers", "Battle_w_PretrainedOpp"),
    "MAgent_AdvPursuit": partial(_make, "envs.adv_pursuit_wrappers", "AdvPursuit_w_PretrainedOpp"),
    "ic3_traffic": partial(_make, "envs.traffic_junction_wrapper", "TrafficJunctionWrapper"),
    "ic3_predator": partial(_make, "envs.predator_prey_wrapper", "PredatorPreyWrapper"),
    "smac_limited_view": partial(_make, "envs.smac_limited_view", "LimitedViewSMAC"),
}
