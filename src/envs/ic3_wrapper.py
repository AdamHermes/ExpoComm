"""Adapter for upstream IC3Net environments; NOT a verified MAIL protocol.

The package must be installed separately. Defaults not specified in MAIL
remain explicit experiment choices; see docs/multiview_experiments.md.
"""
import argparse
from contextlib import contextmanager
from importlib import import_module
import random
import numpy as np
from .multiagentenv import MultiAgentEnv


class IC3Wrapper(MultiAgentEnv):
    task = None

    def __init__(self, n_agents, episode_limit, seed=0, reward_aggregation="mean", **options):
        if reward_aggregation not in ("mean", "sum", "individual"):
            raise ValueError("Unknown reward_aggregation")
        self.reward_aggregation = reward_aggregation
        self.n_agents, self.episode_limit = int(n_agents), int(episode_limit)
        if self.n_agents < 1 or self.episode_limit < 1:
            raise ValueError("Agent count and episode limit must be positive")
        module, name = {
            "traffic": ("traffic_junction_env", "TrafficJunctionEnv"),
            "predator": ("predator_prey_env", "PredatorPreyEnv"),
        }[self.task]
        cls = getattr(import_module("ic3net_envs." + module), name)
        self.env = cls()
        parser = argparse.ArgumentParser()
        self.env.init_args(parser)
        config = parser.parse_args([])
        unknown = set(options) - set(vars(config))
        if unknown:
            raise ValueError("Unknown IC3Net options: " + str(sorted(unknown)))
        vars(config).update(options)
        config.nagents = config.nfriendly = self.n_agents
        if self.task == "predator" and (config.nenemies != 1 or config.enemy_comm or config.moving_prey):
            raise ValueError("This adapter supports one fixed prey, without prey communication")
        self._np_rng = np.random.RandomState(seed).get_state()
        self._py_rng = random.Random(seed).getstate()
        with self._rng():
            self.env.multi_agent_init(config)
        self.n_actions = self.env.naction
        self._done = True
        self.reset()

    @contextmanager
    def _rng(self):
        # Upstream uses global numpy/random RNGs. Isolate environment streams
        # from replay sampling and other environments in the same process.
        old_np, old_py = np.random.get_state(), random.getstate()
        np.random.set_state(self._np_rng)
        random.setstate(self._py_rng)
        try:
            yield
        finally:
            self._np_rng, self._py_rng = np.random.get_state(), random.getstate()
            np.random.set_state(old_np)
            random.setstate(old_py)

    def _set_obs(self, obs):
        # Traffic emits tuples (last action, route, local grid); Predator emits
        # arrays. Flatten each component in the upstream order, preserving the
        # already-normalized scalar features (do not one-hot them a second time).
        def flatten(value):
            if isinstance(value, tuple):
                return np.concatenate([flatten(part) for part in value])
            return np.asarray(value, dtype=np.float32).reshape(-1)
        self.obs = np.stack([flatten(agent_obs) for agent_obs in obs])

    def reset(self):
        with self._rng():
            self._set_obs(self.env.reset())
        self.t, self._done = 0, False
        self._reset_slots = np.ones(self.n_agents, dtype=np.uint8)
        return self.get_obs(), self.get_state()

    def step(self, actions):
        if self._done:
            raise RuntimeError("Reset is required after episode termination")
        if hasattr(actions, "detach"):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions).reshape(self.n_agents)
        if np.any(actions < 0) or np.any(actions >= self.n_actions):
            raise ValueError("Action index out of range")
        previous_active = self.get_agent_active().astype(bool)
        with self._rng():
            obs, reward, terminated, _ = self.env.step(actions)
        self._set_obs(obs)
        self.t += 1
        if self.task == "predator":
            success = bool(np.all(self.env.reached_prey))
            terminated = bool(terminated) or success
        else:
            success = not bool(self.env.has_failed)
        self._reset_slots = (self.get_agent_active().astype(bool) & ~previous_active)
        if self.task == "traffic":
            # A completed car's slot can be reused within the same env step.
            self._reset_slots |= np.asarray(self.env.is_completed, dtype=bool)
        truncated = self.t >= self.episode_limit and not terminated
        self._done = bool(terminated or truncated)
        info = {"episode_limit": bool(truncated)}
        if self._done:
            info.update(success=float(success), episode_length=self.t)
        reward = np.asarray(reward, dtype=np.float32).reshape(self.n_agents)
        if self.reward_aggregation == "mean":
            reward = float(reward.mean())
        elif self.reward_aggregation == "sum":
            reward = float(reward.sum())
        else:
            reward = reward.tolist()
        return reward, self._done, info

    def get_agent_active(self):
        if self.task == "traffic":
            return np.asarray(self.env.alive_mask, dtype=np.uint8).copy()
        # Reached predators remain information sources until episode ends.
        return np.ones(self.n_agents, dtype=np.uint8)

    def get_agent_reset(self):
        return self._reset_slots.astype(np.uint8).copy()

    def get_obs(self):
        return self.obs.copy()

    def get_obs_agent(self, agent_id):
        return self.obs[agent_id].copy()

    def get_obs_size(self):
        return self.obs.shape[-1]

    def get_state(self):
        # Centralized observation concatenation; not physical simulator state.
        return self.obs.reshape(-1).copy()

    def get_state_size(self):
        return self.obs.size

    def get_avail_actions(self):
        return np.ones((self.n_agents, self.n_actions), dtype=np.int32)

    def get_avail_agent_actions(self, agent_id):
        return self.get_avail_actions()[agent_id]

    def get_total_actions(self):
        return self.n_actions

    def close(self):
        pass  # no renderer/process is initialized by this headless adapter

    def render(self, **kwargs):
        raise NotImplementedError("IC3Net adapter is headless")

    def save_replay(self):
        raise NotImplementedError("IC3Net does not provide replay export")
