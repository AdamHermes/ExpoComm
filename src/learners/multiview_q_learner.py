"""Recurrent Double Q with MAIL-inspired training-only supervision."""
import math
import os
import torch as th
from .q_learner import QLearner
from modules.communication.multiview_losses import masked_mean


class MultiViewQLearner(QLearner):
    def __init__(self, mac, scheme, logger, args):
        if args.standardise_returns:
            raise ValueError("Multiview learner does not support return normalization")
        if args.runner != "episode":
            raise ValueError("Multiview experiments currently require the episode runner")
        super().__init__(mac, scheme, logger, args)
        if self.global_reward != (self.mixer is not None):
            raise ValueError("Use no mixer for individual rewards; QMIX/VDN for team rewards")
        if (scheme["reward"].get("group") == "agents") == self.global_reward:
            raise ValueError("Replay reward schema disagrees with reward_scope")
        if args.mv_samples_per_episode < 1 or args.target_update_interval_or_tau <= 0:
            raise ValueError("Sample count and target update interval must be positive")
        for key in ("aux_coef", "mv_gcl_coef", "mv_align_coef", "mv_reg_coef"):
            if not math.isfinite(getattr(args, key)) or getattr(args, key) < 0:
                raise ValueError(key + " must be finite and nonnegative")

    def train(self, batch, t_env, episode_num):
        steps = batch.max_seq_length - 1
        if steps < 1:
            raise ValueError("Need a transition and a bootstrap observation")
        filled = batch["filled"][:, :-1, 0].bool()
        done = batch["terminated"][:, :-1, 0].bool()
        preceding = th.cat((th.ones_like(done[:, :1]), (~done[:, :-1]).cumprod(1).bool()), 1)
        transition = filled & preceding
        if "agent_active" in batch.scheme:
            active = batch["agent_active"][..., 0].bool()
        else:
            active = th.ones(batch["obs"].shape[:3], dtype=th.bool, device=batch["obs"].device)
        node_mask = transition.unsqueeze(-1) & active[:, :-1]
        td_mask = transition.unsqueeze(-1) if self.global_reward else node_mask
        if not td_mask.any():
            return
        actions = batch["actions"][:, :-1].long()
        available = batch["avail_actions"].bool()
        chosen_valid = available[:, :-1].gather(-1, actions).squeeze(-1)
        if (node_mask & ~chosen_valid).any():
            raise ValueError("Replay contains an unavailable executed action")
        reward = batch["reward"][:, :-1].detach()
        if not self.global_reward:
            reward = reward.squeeze(-1)
        if self.args.standardise_rewards:
            # Population variance is defined even for a single valid sample.
            values = reward[td_mask.expand_as(reward)].reshape(-1, 1)
            self.rew_ms.update_from_moments(values.mean(0), values.var(0, unbiased=False), len(values))
            reward = (reward - self.rew_ms.mean) / self.rew_ms.var.clamp_min(1e-8).sqrt()

        self.mac.init_hidden(batch.batch_size)
        online, predictions, features, messages = [], [], [], []
        for t in range(steps + 1):
            q, pred, h, m = self.mac.forward(batch, t, return_features=True)
            online.append(q)
            if t < steps:
                predictions.append(pred)
                features.append(h)
                messages.append(m)
        online = th.stack(online, 1)
        selected = online[:, :-1].gather(-1, actions).squeeze(-1)
        with th.no_grad():
            self.target_mac.init_hidden(batch.batch_size)
            target = th.stack([self.target_mac.forward(batch, t)[0] for t in range(steps + 1)], 1)[:, 1:]
            next_available = available[:, 1:]
            next_has_action = next_available.any(-1)
            next_scores = online.detach()[:, 1:] if self.args.double_q else target
            best = next_scores.masked_fill(~next_available, -th.inf).argmax(-1, keepdim=True)
            next_q = target.gather(-1, best).squeeze(-1)
            next_q = next_q.masked_fill(~(active[:, 1:] & next_has_action), 0)
            if self.mixer is not None:
                next_q = self.target_mixer(next_q, batch["state"][:, 1:])
                bootstrap = (~done).unsqueeze(-1)
            else:
                bootstrap = (~done).unsqueeze(-1) & active[:, 1:] & next_has_action
                if "agent_reset" in batch.scheme:
                    # A new occupant's value does not continue the previous
                    # occupant's individual trajectory (team value is different).
                    bootstrap &= ~batch["agent_reset"][:, 1:, :, 0].bool()
            targets = reward + self.args.gamma * bootstrap * next_q
        if self.mixer is not None:
            selected = self.mixer(selected * active[:, :-1], batch["state"][:, :-1])
        td = selected - targets
        q_loss = masked_mean(td.square(), td_mask)
        zero = q_loss * 0
        state_loss = zero
        if self.args.aux_coef > 0:
            predictions = th.stack(predictions, 1)
            state = batch["state"][:, :-1].reshape(batch.batch_size, steps, 1, -1)
            # Preserve original auxiliary scaling: sum state dims, average nodes.
            state_loss = masked_mean((predictions - state).square().sum(-1), node_mask)
        gcl = alignment = regularizer = zero
        mv_metrics = {}
        if any(getattr(self.args, key) > 0 for key in ("mv_gcl_coef", "mv_align_coef", "mv_reg_coef")):
            h, m = th.stack(features, 1), th.stack(messages, 1)
            batch_indices, time_indices = [], []
            for b in range(batch.batch_size):
                valid_t = th.where(node_mask[b].any(-1))[0]
                chosen = valid_t[th.randperm(len(valid_t), device=h.device)[:self.args.mv_samples_per_episode]]
                batch_indices.extend([b] * len(chosen))
                time_indices.extend(chosen.unbind())
            if time_indices:
                bi = th.tensor(batch_indices, device=h.device)
                ti = th.stack(time_indices)
                gcl, alignment, regularizer, mv_metrics = self.mac.agent.multiview(
                    h[bi, ti], m[bi, ti], node_mask[bi, ti])
        loss = (q_loss + self.args.aux_coef * state_loss + self.args.mv_gcl_coef * gcl
                + self.args.mv_align_coef * alignment + self.args.mv_reg_coef * regularizer)
        if not th.isfinite(loss):
            raise FloatingPointError("Non-finite multiview loss")
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        if not th.isfinite(grad_norm):
            raise FloatingPointError("Non-finite multiview gradients")
        self.optimiser.step()
        self.training_steps += 1
        interval = self.args.target_update_interval_or_tau
        if interval > 1 and self.training_steps - self.last_target_update_step >= interval:
            self._update_targets_hard()
            self.last_target_update_step = self.training_steps
        elif 0 < interval <= 1:
            self._update_targets_soft(interval)
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            stats = dict(loss=loss, q_loss=q_loss, aux_loss=state_loss, mv_gcl=gcl,
                         mv_alignment=alignment, mv_regularizer=regularizer, grad_norm=grad_norm,
                         td_error_abs=masked_mean(td.abs(), td_mask))
            stats.update({"mv_" + k: v for k, v in mv_metrics.items()})
            for key, value in stats.items():
                self.logger.log_stat(key, value.detach().item(), t_env)
            self.log_stats_t = t_env

    def save_models(self, path):
        super().save_models(path)
        state = {"target_agent": self.target_mac.agent.state_dict(),
                 "training_steps": self.training_steps,
                 "last_target_update_step": self.last_target_update_step}
        if self.mixer is not None:
            state["target_mixer"] = self.target_mixer.state_dict()
        if self.args.standardise_rewards:
            state["reward_stats"] = (self.rew_ms.mean, self.rew_ms.var, self.rew_ms.count)
        th.save(state, os.path.join(path, "multiview_training.th"))

    def load_models(self, path):
        super().load_models(path)
        device = next(self.mac.parameters()).device
        state = th.load(os.path.join(path, "multiview_training.th"), map_location=device)
        self.target_mac.agent.load_state_dict(state["target_agent"])
        self.training_steps = state["training_steps"]
        self.last_target_update_step = state["last_target_update_step"]
        if self.mixer is not None:
            self.target_mixer.load_state_dict(state["target_mixer"])
        if self.args.standardise_rewards:
            self.rew_ms.mean, self.rew_ms.var, self.rew_ms.count = state["reward_stats"]
