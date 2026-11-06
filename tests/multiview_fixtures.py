import sys
from pathlib import Path
from types import SimpleNamespace
import torch as th
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from components.episode_buffer import EpisodeBatch
from controllers import REGISTRY
from learners.multiview_q_learner import MultiViewQLearner


def args_for(variant="adaptive", **changes):
    data = {}
    for path in ("default.yaml", "algs/iql_expocomm_mv_" + variant + ".yaml"):
        data.update(yaml.safe_load((ROOT / "src/config" / path).read_text()))
    data.update(n_agents=5, n_actions=4, hidden_dim=16, mv_dim=8, state_shape=12,
                env="MAgent_Battle", use_cuda=False, device="cpu", learner_log_interval=1)
    data.update(changes)
    return SimpleNamespace(**data)


class Logger:
    def __init__(self):
        self.stats = {}
        self.console_logger = self

    def log_stat(self, key, value, t):
        self.stats[key] = value

    def info(self, *args):
        pass


def make_batch(args, batch_size=2, length=5, obs_dim=8):
    n, a = args.n_agents, args.n_actions
    scheme = {
        "state": {"vshape": args.state_shape},
        "obs": {"vshape": obs_dim, "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "actions_onehot": {"vshape": (a,), "group": "agents"},
        "avail_actions": {"vshape": (a,), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
        "agent_active": {"vshape": (1,), "group": "agents", "dtype": th.uint8},
        "agent_reset": {"vshape": (1,), "group": "agents", "dtype": th.uint8},
    }
    if args.reward_scope != "team":
        scheme["reward"]["group"] = "agents"
    batch = EpisodeBatch(scheme, {"agents": n}, batch_size, length)
    batch.update({"obs": th.randn(batch_size, length, n, obs_dim),
                  "state": th.randn(batch_size, length, args.state_shape),
                  "actions": th.randint(a, (batch_size, length, n, 1)),
                  "avail_actions": th.ones(batch_size, length, n, a),
                  "reward": th.randn_like(batch["reward"]),
                  "terminated": th.zeros(batch_size, length, 1),
                  "agent_active": th.ones(batch_size, length, n, 1)})
    return batch


def make_learner(args, batch):
    mac = REGISTRY[args.mac](batch.scheme, {"agents": args.n_agents}, args)
    logger = Logger()
    return MultiViewQLearner(mac, batch.scheme, logger, args), logger
