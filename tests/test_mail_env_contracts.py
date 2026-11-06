"""Real upstream contracts when IC3Net is installed; explicit skips otherwise."""
import importlib.util
import unittest
import numpy as np
from multiview_fixtures import args_for, make_batch, make_learner
import torch as th
from envs.traffic_junction_wrapper import TrafficJunctionWrapper
from envs.predator_prey_wrapper import PredatorPreyWrapper


@unittest.skipUnless(importlib.util.find_spec("ic3net_envs") and importlib.util.find_spec("gym"), "IC3Net/Gym not installed")
class EnvironmentContracts(unittest.TestCase):
    def test_real_reset_steps_reward_scope_and_rng(self):
        for cls, options in ((TrafficJunctionWrapper, dict(dim=14, difficulty="medium", vision=1,
                                                         add_rate_min=.2, add_rate_max=.2)),
                             (PredatorPreyWrapper, dict(dim=10, vision=1, mode="mixed", no_stay=True))):
            first = cls(n_agents=5, episode_limit=3, seed=3, **options)
            second = cls(n_agents=5, episode_limit=3, seed=3, **options)
            np.testing.assert_array_equal(first.get_obs(), second.get_obs())
            info = first.get_env_info()
            self.assertEqual(first.get_obs().shape, (5, info["obs_shape"]))
            self.assertEqual(len(first.get_state()), info["state_shape"])
            for t in range(3):
                actions = np.zeros(5, dtype=np.int64)
                reward, done, result = first.step(actions)
                np.random.randn(20)  # must not change the other env's stream
                other_reward, other_done, _ = second.step(actions)
                self.assertEqual(reward, other_reward)
                self.assertEqual(done, other_done)
                np.testing.assert_array_equal(first.get_obs(), second.get_obs())
                self.assertTrue(np.isfinite(reward))
            self.assertTrue(done)
            self.assertIn("success", result)
            with self.assertRaises(RuntimeError):
                first.step(actions)
            first.reset()
            self.assertTrue(first.get_agent_reset().all())


class SlotResetTests(unittest.TestCase):
    def test_reused_slot_discards_old_hidden_and_last_action(self):
        args = args_for(obs_last_action=True)
        batch = make_batch(args)
        batch["agent_reset"][:, 1] = 1
        learner, _ = make_learner(args, batch)
        learner.mac.init_hidden(2)
        learner.mac.forward(batch, 0)
        q1 = learner.mac.forward(batch, 1)[0]
        learner.mac.init_hidden(2)
        batch["actions_onehot"][:, 0] = 1e4
        q2 = learner.mac.forward(batch, 1)[0]
        th.testing.assert_close(q1, q2, rtol=0, atol=0)
