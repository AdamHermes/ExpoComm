import copy
import tempfile
import unittest
from unittest.mock import patch
import torch as th
from multiview_fixtures import args_for, make_batch, make_learner
from controllers.ExpoComm_controller import ExpoCommMAC
from learners.q_learner import AuxQLearner
from multiview_fixtures import Logger


class TrainingTests(unittest.TestCase):
    def setUp(self):
        th.manual_seed(7)
        th.set_num_threads(1)

    def test_execution_matches_original_and_never_calls_views(self):
        args = args_for(obs_last_action=True, obs_agent_id=True)
        batch = make_batch(args)
        learner, _ = make_learner(args, batch)
        base_args = copy.copy(args)
        base_args.agent = "ExpoComm_one_peer"
        baseline = ExpoCommMAC(batch.scheme, {"agents": args.n_agents}, base_args)
        state = {k: v for k, v in learner.mac.agent.state_dict().items() if not k.startswith("multiview.")}
        baseline.agent.load_state_dict(state)
        baseline.init_hidden(2)
        learner.mac.init_hidden(2)
        with patch.object(learner.mac.agent.multiview, "forward", side_effect=AssertionError("Execution used training graph")):
            for t in range(batch.max_seq_length):
                q, _ = learner.mac.forward(batch, t, test_mode=True)
                original, _ = baseline.forward(batch, t, test_mode=True)
                th.testing.assert_close(q, original, rtol=0, atol=0)

    def test_train_all_variants_and_router_gradients(self):
        for variant in ("fixed", "adaptive", "no_feature", "no_topology", "no_cross", "no_state_aux", "no_multiview"):
            args = args_for(variant)
            batch = make_batch(args)
            batch["terminated"][0, 2] = 1
            batch["filled"][0, 3:] = 0
            batch["agent_active"][1, :, -1] = 0
            learner, logger = make_learner(args, batch)
            before = learner.mac.agent.multiview.gate[-1].weight.detach().clone()
            learner.train(batch, 0, 0)
            self.assertEqual(learner.training_steps, 1)
            self.assertTrue(all(th.isfinite(th.tensor(v)) for v in logger.stats.values()))
            if variant == "adaptive":
                self.assertFalse(th.equal(before, learner.mac.agent.multiview.gate[-1].weight))
            self.assertTrue(all(p.grad is None for p in learner.target_mac.parameters()))

    def test_team_qmix_and_vdn(self):
        for mixer in ("qmix", "vdn"):
            args = args_for(reward_scope="team", mixer=mixer, mixing_embed_dim=8,
                            hypernet_layers=2, hypernet_embed=16)
            batch = make_batch(args)
            learner, logger = make_learner(args, batch)
            learner.train(batch, 0, 0)
            self.assertTrue(th.isfinite(th.tensor(logger.stats["loss"])))
            if mixer == "qmix":
                self.assertTrue(any(p.grad is not None for p in learner.mixer.parameters()))

    def test_padding_invariance_including_normalization(self):
        args = args_for()
        batch = make_batch(args)
        batch["terminated"][:, 1] = 1
        batch["filled"][:, 2:] = 0
        learner, log1 = make_learner(args, batch)
        other, log2 = make_learner(args, batch)
        other.mac.load_state(learner.mac)
        other.target_mac.load_state(learner.target_mac)
        th.manual_seed(100)
        learner.train(batch, 0, 0)
        for key in ("obs", "state", "reward"):
            batch[key][:, 2:] += 10000
        th.manual_seed(100)
        other.train(batch, 0, 0)
        for key in log1.stats:
            self.assertAlmostEqual(log1.stats[key], log2.stats[key], places=5, msg=key)
        for a, b in zip(learner.mac.parameters(), other.mac.parameters()):
            th.testing.assert_close(a, b)

    def test_hard_soft_targets_and_checkpoint_resume(self):
        for interval in (2, .1):
            args = args_for(target_update_interval_or_tau=interval)
            batch = make_batch(args)
            learner, _ = make_learner(args, batch)
            old_target = [p.detach().clone() for p in learner.target_mac.parameters()]
            learner.train(batch, 0, 0)
            for old, target, online in zip(old_target, learner.target_mac.parameters(), learner.mac.parameters()):
                expected = old if interval == 2 else .9 * old + .1 * online
                th.testing.assert_close(target, expected)
            with tempfile.TemporaryDirectory() as directory:
                learner.save_models(directory)
                restored, _ = make_learner(args, batch)
                restored.load_models(directory)
                self.assertEqual(restored.training_steps, 1)
                th.testing.assert_close(restored.rew_ms.mean, learner.rew_ms.mean)
                for a, b in zip(learner.target_mac.parameters(), restored.target_mac.parameters()):
                    th.testing.assert_close(a, b, rtol=0, atol=0)
                th.manual_seed(50)
                learner.train(batch, 1, 1)
                th.manual_seed(50)
                restored.train(batch, 1, 1)
                for a, b in zip(learner.mac.parameters(), restored.mac.parameters()):
                    th.testing.assert_close(a, b, rtol=0, atol=0)

    def test_single_agent_and_all_inactive(self):
        args = args_for(n_agents=1)
        batch = make_batch(args, batch_size=1, length=2)
        learner, log = make_learner(args, batch)
        learner.train(batch, 0, 0)
        self.assertTrue(th.isfinite(th.tensor(log.stats["loss"])))
        batch["agent_active"].zero_()
        learner.train(batch, 1, 1)
        self.assertEqual(learner.training_steps, 1)

    def test_communication_excludes_self_slot(self):
        args = args_for()
        batch = make_batch(args)
        learner, _ = make_learner(args, batch)
        self.assertEqual(learner.mac.communication_stats(batch, 0)["comm_packets"], 0)
        self.assertEqual(learner.mac.communication_stats(batch, 1)["comm_packets"], 10)

    def test_no_multiview_matches_original_td_and_aux_update(self):
        args = args_for("no_multiview", standardise_rewards=False)
        batch = make_batch(args)
        learner, logger = make_learner(args, batch)
        base_args = copy.copy(args)
        base_args.agent = "ExpoComm_one_peer"
        baseline = ExpoCommMAC(batch.scheme, {"agents": args.n_agents}, base_args)
        baseline.agent.load_state_dict({k: v for k, v in learner.mac.agent.state_dict().items()
                                        if not k.startswith("multiview.")})
        base_logger = Logger()
        original = AuxQLearner(baseline, batch.scheme, base_logger, base_args)
        original.train(batch, 0, 0)
        learner.train(batch, 0, 0)
        for key in ("loss", "q_loss", "aux_loss"):
            th.testing.assert_close(th.tensor(logger.stats[key]), th.tensor(base_logger.stats[key]), rtol=1e-6, atol=1e-6)
        for key, value in baseline.agent.state_dict().items():
            th.testing.assert_close(value, learner.mac.agent.state_dict()[key])

    def test_magent_observation_shapes_train(self):
        for obs, n in ((98, 20), (128, 25)):
            args = args_for(n_agents=n, state_shape=obs)
            batch = make_batch(args, batch_size=1, length=4, obs_dim=obs)
            learner, logger = make_learner(args, batch)
            learner.train(batch, 0, 0)
            self.assertTrue(th.isfinite(th.tensor(logger.stats["loss"])))

    def test_disabled_state_aux_does_not_decode(self):
        args = args_for("no_state_aux")
        batch = make_batch(args)
        learner, _ = make_learner(args, batch)
        with patch.object(learner.mac.agent, "aux_forward", side_effect=AssertionError("Unneeded state decoding")):
            learner.train(batch, 0, 0)

    def test_individual_reset_cuts_bootstrap(self):
        args = args_for("no_multiview", standardise_rewards=False, aux_coef=0)
        batch = make_batch(args, batch_size=1, length=2)
        batch["reward"].zero_()
        batch["agent_reset"][:, 1] = 1
        learner, logger = make_learner(args, batch)
        # Q_online(t,a)=0, Q_target(t,a)=1. Reused slots must have TD target 0.
        with th.no_grad():
            for agent in (learner.mac.agent, learner.target_mac.agent):
                for parameter in agent.parameters():
                    parameter.zero_()
            learner.target_mac.agent.fc2[-1].bias.fill_(1)
        learner.train(batch, 0, 0)
        self.assertEqual(logger.stats["q_loss"], 0)

    def test_double_q_available_actions_and_terminal_target(self):
        for terminal in (False, True):
            args = args_for("no_multiview", standardise_rewards=False, aux_coef=0)
            batch = make_batch(args, batch_size=1, length=2)
            batch["actions"].zero_()
            batch["reward"].fill_(2)
            batch["terminated"][:, 0] = terminal
            batch["avail_actions"][:, 1, :, 2] = 0
            learner, logger = make_learner(args, batch)
            with th.no_grad():
                for agent in (learner.mac.agent, learner.target_mac.agent):
                    for parameter in agent.parameters():
                        parameter.zero_()
                learner.mac.agent.fc2[-1].bias.copy_(th.tensor([0., 2., 8., 1.]))
                learner.target_mac.agent.fc2[-1].bias.copy_(th.tensor([3., 7., 100., 5.]))
            learner.train(batch, 0, 0)
            expected = (2 + (0 if terminal else args.gamma * 7)) ** 2
            self.assertAlmostEqual(logger.stats["q_loss"], expected, places=4)


if __name__ == "__main__":
    unittest.main()
