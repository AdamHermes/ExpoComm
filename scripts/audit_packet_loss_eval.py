"""Headless audit of the packet-loss path and Battle wrapper logic.

Runs the actual agent methods with synthetic tensors and extracts actual local
scenario/wrapper methods for logic probes. No environment simulator, checkpoint
or Slurm job is run. These probes cannot explain a trained policy's tactics.
"""
import argparse
import ast
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch as th

ROOT = Path(__file__).resolve().parents[1]


def load_file(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def method(relative, class_name, function_name, scope):
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    body = next(node.body for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    function = next(node for node in body if isinstance(node, ast.FunctionDef) and node.name == function_name)
    scope = dict(scope)
    exec(compile(ast.Module(body=[function], type_ignores=[]), relative, "exec"), scope)
    return scope[function_name]


def neighbors(agents, topk, t, batch=1):
    offsets = th.tensor([0] + [2**i for i in range(topk-1)])
    indices = (th.arange(agents)[:, None] + offsets) % agents
    return indices[:, t % topk][None].expand(batch, -1)


def channel_tests():
    agents = load_file("audited_agents", "src/modules/agents/ExpoComm_agent.py")
    results = []
    for count, topk in [(20, 6), (42, 7), (64, 7), (25, 6), (45, 7), (61, 7)]:
        args = SimpleNamespace(n_agents=count, hidden_dim=16, attention_dim=8,
                               state_shape=(3,), use_rnn=True, n_actions=5,
                               packet_loss_percent=100, packet_loss_train=False, packet_loss_test=True)
        th.manual_seed(2026)
        agent = agents.ExpoCommOAgent(7, args)
        for t in range(topk):
            senders = neighbors(count, topk, t)
            keep = agent._packet_keep_mask(senders, test_mode=True)
            is_self = senders.eq(th.arange(count)[None])
            assert th.equal(keep, is_self), "100% mask allowed a remote sender"
        # Change every other agent's observations over a full recurrent rollout.
        # Agent 0 receives exactly the same own observations in both trajectories.
        observations = th.randn(topk * 3, count, 7)
        changed = observations.clone()
        changed[:, 1:] = 10 * th.randn_like(changed[:, 1:])

        def rollout(inputs):
            h, msg = agent.init_hidden()
            state = (h.expand(count, -1).clone(), msg.expand(count, -1).clone())
            values = []
            with th.no_grad():
                for t, obs in enumerate(inputs):
                    q, state, _ = agent(obs, state, neighbors(count, topk, t), test_mode=True)
                    values.append(q[0].clone())
            return th.stack(values)

        no_remote_a, no_remote_b = rollout(observations), rollout(changed)
        assert th.equal(no_remote_a, no_remote_b), "Other observations influenced agent 0 at loss100"
        args.packet_loss_percent = 0
        with_remote_a, with_remote_b = rollout(observations), rollout(changed)
        difference = float((with_remote_a - with_remote_b).abs().max())
        assert difference > 1e-7, "Positive control did not exercise communication"
        h, msg = agent.init_hidden()
        assert th.count_nonzero(h) == 0 and th.count_nonzero(msg) == 0
        results.append(dict(agents=count, topk=topk,
                            loss100_remote_keeps=0, altered_remote_observations_change_q_at100=False,
                            positive_control_max_q_difference_at0=difference,
                            initial_hidden_and_message_zero=True))
    return results


def scenario_probes():
    generate = method("env/battle_v3_view7.py", "_parallel_env", "generate_map", {"math": math})
    reset = method("src/envs/battle_wrappers.py", "_BattleWrapper", "reset", {"np": np})
    results = []
    for size in [25, 35, 40]:
        class Grid:
            def __init__(self):
                self.positions = {}

            def add_agents(self, handle, method, pos):
                self.positions[handle] = pos

        grid = Grid()
        scenario = SimpleNamespace(env=grid, map_size=size, handles=[0, 1], leftID=0, rightID=1)

        class ParallelProxy:
            calls = 0

            def reset(self):
                self.calls += 1
                generate(scenario)
                return {"red_0": np.zeros(1), "blue_0": np.zeros(1)}

            def __getattr__(self, name):
                # Satisfy the wrapper's unused nested env-access expression.
                return self

        proxy = ParallelProxy()
        wrapper = SimpleNamespace(env=proxy, agents=["red_0", "blue_0"],
                                  get_obs=lambda: None, get_state=lambda: None)
        double_reset_sides = []
        for _ in range(4):
            reset(wrapper)
            double_reset_sides.append(scenario.leftID)
        assert proxy.calls == 8 and len(set(double_reset_sides)) == 1
        single_reset_sides = []
        for _ in range(4):
            proxy.reset()
            single_reset_sides.append(scenario.leftID)
        assert single_reset_sides[0] != single_reset_sides[1]
        positions = grid.positions[scenario.leftID]
        results.append(dict(map_size=size, team_size=len(positions),
                            formation_columns=len({p[0] for p in positions}),
                            formation_rows=len({p[1] for p in positions}),
                            left_team_ids_under_double_reset=double_reset_sides,
                            left_team_ids_under_single_reset=single_reset_sides))
    return results


def survivor_count_probe():
    step = method("src/envs/battle_wrappers.py", "_BattleWrapper", "step",
                  {"np": np, "torch": th, "math": math})
    names = ["red_0", "red_1", "red_2", "blue_0", "blue_1"]

    class Proxy:
        def step(self, actions):
            # Upstream PettingZoo 1.12 returns observations for the agent list
            # BEFORE removing newly dead agents. Their final observations may
            # be zero, but dictionary membership alone does not imply survival.
            obs = {name: np.ones(9) for name in names}
            obs["red_1"] = np.zeros(9)
            obs["red_2"] = np.zeros(9)
            return obs, {name: 0.0 for name in names}, {"__all__": True}, {name: {} for name in names}

    wrapper = SimpleNamespace(env=Proxy(), agents=names, _episode_length=199, episode_limit=200,
                              observation_space={"obs": SimpleNamespace(shape=(1, 1, 9))})
    _, _, info = step(wrapper, [0] * len(names))
    assert info["red_team_alives"] == 3 and info["blue_team_alives"] == 2 and info["red_team_win"]
    return dict(synthetic_true_survivors=dict(red=1, blue=2),
                wrapper_reported=dict(red=info["red_team_alives"], blue=info["blue_team_alives"]),
                wrapper_reported_red_win=bool(info["red_team_win"]),
                true_red_win=False,
                scope="Synthetic reproduction of counting dictionary membership, not a measurement of the W&B episodes")


def cli_probe():
    module = load_file("audited_overrides", "src/config_overrides.py")
    config = dict(seed=0, run_name="probe", runner="episode")
    params = ["--eval-seed=4", "--eval-result-path=probe.json", "--packet-loss-percent=100", "--packet-loss-mode=test"]
    module.apply_packet_loss_cli_overrides(config, params)
    assert config["seed"] == 0 and params == ["--eval-seed=4", "--eval-result-path=probe.json"]
    return dict(requested_eval_seed=4, effective_config_seed=config["seed"], ignored_options=params,
                percent_and_test_mode_applied=(config["packet_loss_percent"] == 100 and config["packet_loss_test"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("output/packet_loss_eval_audit/probe_results.json"))
    args = parser.parse_args()
    result = dict(channel=channel_tests(), battle_reset_and_formation=scenario_probes(),
                  survivor_count_logic=survivor_count_probe(), cli=cli_probe())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved audit probes to {args.out.resolve()}")


if __name__ == "__main__":
    main()
