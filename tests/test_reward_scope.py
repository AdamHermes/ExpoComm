import unittest
from types import SimpleNamespace
from multiview_fixtures import args_for, make_batch, make_learner
from components.reward_scope import is_global_reward


class RewardTests(unittest.TestCase):
    def test_legacy_and_explicit(self):
        self.assertTrue(is_global_reward(SimpleNamespace(env="struct_marl")))
        self.assertFalse(is_global_reward(SimpleNamespace(env="sc2")))
        self.assertTrue(is_global_reward(SimpleNamespace(env="sc2", reward_scope="team")))
        with self.assertRaises(ValueError):
            is_global_reward(SimpleNamespace(env="sc2", reward_scope="wrong"))

    def test_incompatible_mixer_and_schema_rejected(self):
        args = args_for(reward_scope="team", mixer=None)
        with self.assertRaises(ValueError):
            make_learner(args, make_batch(args))
        args = args_for()
        batch = make_batch(args)
        batch.scheme["reward"].pop("group")
        with self.assertRaises(ValueError):
            make_learner(args, batch)
