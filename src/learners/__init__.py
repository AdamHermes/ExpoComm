"""Load only the selected learner's optional dependencies."""
from importlib import import_module
from functools import partial


def _construct(module, name, *args, **kwargs):
    return getattr(import_module("." + module, __name__), name)(*args, **kwargs)


REGISTRY = {
    key: partial(_construct, module, name)
    for key, module, name in (
        ("q_learner", "q_learner", "QLearner"),
        ("aux_q_learner", "q_learner", "AuxQLearner"),
        ("cont_q_learner", "q_learner", "ContQLearner"),
        ("multiview_q_learner", "multiview_q_learner", "MultiViewQLearner"),
        ("coma_learner", "coma_learner", "COMALearner"),
        ("qtran_learner", "qtran_learner", "QLearner"),
        ("actor_critic_learner", "actor_critic_learner", "ActorCriticLearner"),
        ("maddpg_learner", "maddpg_learner", "MADDPGLearner"),
        ("ppo_learner", "ppo_learner", "PPOLearner"),
        ("pac_learner", "actor_critic_pac_learner", "PACActorCriticLearner"),
        ("pac_dcg_learner", "actor_critic_pac_dcg_learner", "PACDCGLearner"),
    )
}
