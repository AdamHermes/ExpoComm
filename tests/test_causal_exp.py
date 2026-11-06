"""CPU contract/integration tests; no MAgent installation required."""
import copy
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch as th
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from modules.agents.causal_exp_agent import CausalExpAgent
from controllers import REGISTRY
from components.episode_buffer import EpisodeBatch
# Load the real learner without importing unrelated COMA/PAC dependencies from
# learners/__init__.py. This is not an environment/entrypoint smoke test.
spec = importlib.util.spec_from_file_location("causal_test_q_learner", ROOT / "src/learners/q_learner.py")
q_learner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q_learner)
AuxQLearner = q_learner.AuxQLearner


def arguments(**overrides):
    config = {}
    for file in ("default.yaml", "algs/iql_causal_exp.yaml"):
        config.update(yaml.safe_load((ROOT / "src/config" / file).read_text()))
    config.update(n_agents=5, n_actions=4, hidden_dim=16, state_shape=12,
                  env="MAgent_Battle", use_cuda=False)
    config.update(overrides)
    return SimpleNamespace(**config)


def initial(agent, b=2):
    return tuple(x.unsqueeze(0).expand(b, agent.n_agents, -1).clone()
                 for x in agent.init_hidden())


class Logger:
    def __init__(self):
        self.stats = {}
        self.console_logger = self

    def info(self, *args):
        pass

    def log_stat(self, key, value, step):
        self.stats[key] = value


class CausalExpTests(unittest.TestCase):
    def setUp(self):
        th.manual_seed(31)
        th.set_num_threads(1)

    def test_budget_shapes_and_nonself_edges(self):
        for n in (1, 2, 5, 20, 37):
            agent = CausalExpAgent(8, arguments(n_agents=n))
            hidden = initial(agent)
            for t in range(4):
                q, hidden, m = agent(th.randn(2 * n, 8), hidden, t)
                self.assertEqual(q.shape, (2 * n, 4))
                self.assertEqual(m.shape, (2 * n, 16))
                info = agent.last_routing
                self.assertTrue(th.equal(info["sent"], th.full((2, n), float(n > 1))))
                self.assertEqual(info["received"].sum().item(), 2 * n if n > 1 else 0)
                if n > 1:
                    self.assertTrue((info["destinations"] != th.arange(n)).all())

    def test_local_router_does_not_read_other_agents(self):
        agent = CausalExpAgent(8, arguments())
        h, m = th.randn(2, 5, 16), th.randn(2, 5, 16)
        first, _ = agent.route(h, m, 1)
        h[:, 1:] += 100
        m[:, 1:] -= 100
        second, _ = agent.route(h, m, 1)
        th.testing.assert_close(first[:, 0], second[:, 0], rtol=0, atol=0)
        # Check differentiable dependencies too, not just argmax equality.
        h.requires_grad_()
        gates, _ = agent.route(h, m, 1)
        grad, = th.autograd.grad(gates[:, 0, 0].sum(), h)
        self.assertEqual(grad[:, 1:].abs().sum().item(), 0)
        self.assertGreater(grad[:, 0].abs().sum().item(), 0)

    def test_backbone_temporal_reachability(self):
        for n in (2, 3, 5, 20, 37, 100):
            agent = CausalExpAgent(8, arguments(n_agents=n))
            knowledge = th.eye(n, dtype=th.bool)
            for t in range(0, 2 * agent.offsets.numel(), 2):
                _, selected = agent.route(th.zeros(1, n, 16), th.zeros(1, n, 16), t)
                destinations = agent.destinations.gather(1, selected[0, :, None]).squeeze(1)
                previous = knowledge.clone()
                knowledge[destinations] |= previous
            self.assertTrue(knowledge.all())  # potential paths, not lossless neural recall

    def test_packet_loss_keeps_local_update(self):
        agent = CausalExpAgent(8, arguments(packet_loss_percent=100,
                                          packet_loss_train=True, packet_loss_test=True))
        for test_mode in (False, True):
            _, _, message = agent(th.randn(10, 8), initial(agent), 1, test_mode)
            self.assertEqual(agent.last_routing["received"].sum().item(), 0)
            self.assertGreater(message.abs().sum().item(), 0)
            _, _, other = agent(th.zeros(10, 8), initial(agent), 1, test_mode)
            self.assertFalse(th.equal(message, other))

    def test_hard_sparse_inbox_matches_explicit_delivery(self):
        agent = CausalExpAgent(8, arguments())
        captured = []
        hook = agent.msg_processor.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
        inputs, hidden = th.randn(10, 8), initial(agent)
        agent(inputs, hidden, 1)
        h = agent.rnn(th.relu(agent.fc1(inputs)), hidden[0].reshape(-1, 16)).reshape(2, 5, 16)
        packets = agent.packet_encoder(th.cat((h, hidden[1]), -1))
        expected, count = th.zeros_like(h), th.zeros(2, 5, 1)
        for b in range(2):
            for sender in range(5):
                destination = agent.last_routing["destinations"][b, sender]
                expected[b, destination] += packets[b, sender]
                count[b, destination] += 1
        th.testing.assert_close(captured[0][..., 16:], expected / count.clamp_min(1))
        hook.remove()

    def test_replay_determinism_and_no_communication_ablation(self):
        for mode in ("hybrid", "fixed", "adaptive", "none"):
            agent = CausalExpAgent(8, arguments(routing_mode=mode))
            inputs = th.randn(10, 8)
            first = agent(inputs, initial(agent), 1, test_mode=False)[0]
            second = agent(inputs, initial(agent), 1, test_mode=True)[0]
            th.testing.assert_close(first, second, rtol=0, atol=0)
            if mode == "none":
                self.assertEqual(agent.last_routing["received"].sum().item(), 0)

    def test_real_learner_optimizer_targets_and_checkpoint(self):
        args = arguments(target_update_interval_or_tau=2)
        scheme = {
            "obs": {"vshape": 8, "group": "agents"},
            "state": {"vshape": 12},
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
            "avail_actions": {"vshape": (4,), "group": "agents", "dtype": th.int},
            "reward": {"vshape": (1,), "group": "agents"},
            "terminated": {"vshape": (1,), "dtype": th.uint8},
        }
        batch = EpisodeBatch(scheme, {"agents": 5}, 2, 5)
        batch.update({"obs": th.randn(2, 5, 5, 8), "state": th.randn(2, 5, 12),
                      "actions": th.randint(4, (2, 5, 5, 1)),
                      "avail_actions": th.ones(2, 5, 5, 4),
                      "reward": th.randn(2, 5, 5, 1), "terminated": th.zeros(2, 5, 1)})
        batch["terminated"][0, 2] = 1
        batch["filled"][0, 3:] = 0
        mac = REGISTRY[args.mac](scheme, {"agents": 5}, args)
        logger = Logger()
        learner = AuxQLearner(mac, scheme, logger, args)
        before = copy.deepcopy(mac.agent.router.state_dict())
        learner.train(batch, 0, 0)
        self.assertTrue(all(th.isfinite(th.tensor(v)) for v in logger.stats.values()))
        self.assertTrue(any(not th.equal(v, before[k]) for k, v in mac.agent.router.state_dict().items()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in mac.agent.router.parameters()))
        learner.train(batch, 1, 1)
        for online, target in zip(mac.parameters(), learner.target_mac.parameters()):
            th.testing.assert_close(online, target, rtol=0, atol=0)
        mac.init_hidden(2)
        actions = mac.select_actions(batch, 0, 0, test_mode=True)
        self.assertEqual(actions.shape, (2, 5))
        with tempfile.TemporaryDirectory() as path:
            learner.save_models(path)
            restored = REGISTRY[args.mac](scheme, {"agents": 5}, args)
            restored.load_models(path)
            for original, loaded in zip(mac.parameters(), restored.parameters()):
                th.testing.assert_close(original, loaded, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
