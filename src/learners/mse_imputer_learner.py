"""Frozen-policy supervised trainer for the ExpoComm residual imputer."""

import os

import torch as th
import torch.nn.functional as F
from torch.optim import Adam


class MSEImputerLearner:
    """Train only ``message_imputer`` on receiver-side packet-loss contexts."""

    def __init__(self, mac, scheme, logger, args):
        self.mac = mac
        self.logger = logger
        self.args = args
        if args.packet_loss_recovery != "mse_imputer":
            raise ValueError("MSEImputerLearner requires packet_loss_recovery=mse_imputer")
        if not args.packet_loss_train:
            raise ValueError("MSEImputerLearner requires packet_loss_train=True")
        if not hasattr(mac.agent, "message_imputer"):
            raise ValueError("MSEImputerLearner requires an ExpoComm message_imputer")

        for parameter in self.mac.agent.parameters():
            parameter.requires_grad_(False)
        for parameter in self.mac.agent.message_imputer.parameters():
            parameter.requires_grad_(True)

        self.params = list(self.mac.agent.message_imputer.parameters())
        self.optimiser = Adam(self.params, lr=args.imputer_lr)
        self.log_stats_t = -args.learner_log_interval - 1
        self.training_steps = 0

    def train(self, batch, t_env, episode_num):
        self.mac.init_hidden(batch.batch_size)
        self.optimiser.zero_grad()

        total_loss = 0.0
        total_cache_mse = 0.0
        total_eligible = 0
        optimisation_steps = 0
        filled = batch["filled"][:, :, 0].float()

        # The final sequence element exists only to bootstrap normal RL. It has
        # no environment transition and is excluded from imputer training.
        for t in range(batch.max_seq_length - 1):
            self.mac.forward(batch, t=t, test_mode=False)
            record = self.mac.last_imputer_training_data
            if record is None:
                raise RuntimeError("ExpoCommMAC did not expose imputer training data")

            eligible = record["eligible"] & filled[:, t].bool().unsqueeze(-1)
            eligible_count = int(eligible.sum().item())
            if eligible_count:
                per_agent_mse = F.mse_loss(
                    record["prediction"], record["target"], reduction="none"
                ).mean(dim=-1)
                cache_mse = F.mse_loss(
                    record["cached_msg"], record["target"], reduction="none"
                ).mean(dim=-1)
                loss_t = per_agent_mse[eligible].mean()
                loss_t.backward()
                total_loss += loss_t.item() * eligible_count
                total_cache_mse += cache_mse[eligible].sum().item()
                total_eligible += eligible_count
                optimisation_steps += 1

            # This is one-step supervised learning, not a new BPTT/RL
            # objective. Detach histories to bound GPU memory.
            self.mac.detach_imputer_history()

        if optimisation_steps == 0:
            raise RuntimeError(
                "No eligible imputation samples. Increase packet loss, cache TTL, "
                "or batch size so a dropped remote packet follows a valid cache."
            )

        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()
        self.training_steps += 1

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("imputer_mse", total_loss / total_eligible, t_env)
            self.logger.log_stat(
                "imputer_cache_mse", total_cache_mse / total_eligible, t_env
            )
            self.logger.log_stat("imputer_eligible_count", total_eligible, t_env)
            self.logger.log_stat("imputer_grad_norm", grad_norm.item(), t_env)
            self.log_stats_t = t_env

    def cuda(self):
        self.mac.cuda()

    def load_models(self, path):
        # Load agent.th only. Never load opt.th from the source RL run because
        # this optimizer owns only newly added imputer weights.
        self.mac.load_models(path)

    def save_models(self, path):
        os.makedirs(path, exist_ok=True)
        th.save(
            {"message_imputer": self.mac.agent.message_imputer.state_dict()},
            os.path.join(path, "imputer.th"),
        )
        th.save(self.optimiser.state_dict(), os.path.join(path, "imputer_opt.th"))
