import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch as th
import torch.nn.functional as F
from torch.optim import Adam
from components.standarize_stream import RunningMeanStd


class QLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.global_reward = args.env == "struct_marl"  # only true for IMP env

        self.params = list(mac.parameters())
        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = Adam(params=self.params, lr=args.lr)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.training_steps = 0
        self.last_target_update_step = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

        device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=device)
        if self.args.standardise_rewards:
            self.rew_ms = RunningMeanStd(shape=(1,), device=device)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        # DIFF: terminated expandes as rewards
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        if self.global_reward:
            pass
        else:
            terminated = terminated[:, :, None, :].expand_as(rewards)
            mask = mask[:, :, None, :].expand_as(rewards)
        avail_actions = batch["avail_actions"]

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(
            3
        )  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(
                chosen_action_qvals, batch["state"][:, :-1]
            )
            target_max_qvals = self.target_mixer(
                target_max_qvals, batch["state"][:, 1:]
            )

        if self.args.standardise_returns:
            target_max_qvals = (
                target_max_qvals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean
            )

        # Calculate 1-step Q-Learning targets
        if self.global_reward:
            targets = (
                rewards + self.args.gamma * (1 - terminated) * target_max_qvals.detach()
            )
        else:
            targets = (
                rewards
                + self.args.gamma
                * (1 - terminated)
                * target_max_qvals.detach()[:, :, :, None]
            )

        if self.args.standardise_returns:
            self.ret_ms.update(targets)
            targets = (targets - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

        # Td-error
        if self.global_reward:
            td_error = chosen_action_qvals - targets.detach()
        else:
            td_error = chosen_action_qvals[:, :, :, None] - targets.detach()

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error**2).sum() / mask.sum()

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        self.training_steps += 1
        if (
            self.args.target_update_interval_or_tau > 1
            and (self.training_steps - self.last_target_update_step)
            / self.args.target_update_interval_or_tau
            >= 1.0
        ):
            self._update_targets_hard()
            self.last_target_update_step = self.training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat(
                "td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env
            )
            if self.global_reward:
                self.logger.log_stat(
                    "q_taken_mean",
                    (chosen_action_qvals * mask).sum().item()
                    / (mask_elems * self.args.n_agents),
                    t_env,
                )
            else:
                self.logger.log_stat(
                    "q_taken_mean",
                    (chosen_action_qvals[:, :, :, None] * mask).sum().item()
                    / (mask_elems * self.args.n_agents),
                    t_env,
                )
            self.logger.log_stat(
                "target_mean",
                (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                t_env,
            )
            self.log_stats_t = t_env

    def _update_targets_hard(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def _update_targets_soft(self, tau):
        for target_param, param in zip(
            self.target_mac.parameters(), self.mac.parameters()
        ):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        if self.mixer is not None:
            for target_param, param in zip(
                self.target_mixer.parameters(), self.mixer.parameters()
            ):
                target_param.data.copy_(
                    target_param.data * (1.0 - tau) + param.data * tau
                )

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(
                th.load(
                    "{}/mixer.th".format(path),
                    map_location=lambda storage, loc: storage,
                )
            )
        # Evaluation and replay export only need the policy/Q-network weights.
        # In particular, an old ExpoComm checkpoint can be evaluated with a
        # newly attached message-imputer whose parameters did not exist in the
        # checkpoint's optimiser state.  Loading opt.th in that case fails due
        # to a parameter-group size mismatch and provides no evaluation value.
        # Keep strict optimiser loading for an actual training resume.
        if not (self.args.evaluate or self.args.save_replay):
            self.optimiser.load_state_dict(
                th.load(
                    "{}/opt.th".format(path),
                    map_location=lambda storage, loc: storage,
                )
            )


class AuxQLearner(QLearner):
    def __init__(self, mac, scheme, logger, args):
        super().__init__(mac, scheme, logger, args)

        self.aux_coef = args.aux_coef

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        states = batch["state"][:, :-1]
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        # DIFF: terminated expandes as rewards
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        if self.global_reward:
            pass
        else:
            terminated = terminated[:, :, None, :].expand_as(rewards)
            mask = mask[:, :, None, :].expand_as(rewards)
        avail_actions = batch["avail_actions"]

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        # Calculate estimated Q-Values
        mac_out = []
        state_predicts = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            # different
            agent_outs, state_predicted = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
            state_predicts.append(state_predicted)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        state_predicts = th.stack(state_predicts, dim=1)  # Concat over time
        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(
            3
        )  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            # diffeent
            target_agent_outs, _ = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(
                chosen_action_qvals, batch["state"][:, :-1]
            )
            target_max_qvals = self.target_mixer(
                target_max_qvals, batch["state"][:, 1:]
            )

        if self.args.standardise_returns:
            target_max_qvals = (
                target_max_qvals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean
            )

        # Calculate 1-step Q-Learning targets
        if self.global_reward:
            targets = (
                rewards + self.args.gamma * (1 - terminated) * target_max_qvals.detach()
            )
        else:
            targets = (
                rewards
                + self.args.gamma
                * (1 - terminated)
                * target_max_qvals.detach()[:, :, :, None]
            )

        if self.args.standardise_returns:
            self.ret_ms.update(targets)
            targets = (targets - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

        # Td-error
        if self.global_reward:
            td_error = chosen_action_qvals - targets.detach()
        else:
            td_error = chosen_action_qvals[:, :, :, None] - targets.detach()
        predict_states_error = state_predicts[:, :-1] - states[:, :, None, :]

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        if self.global_reward:
            predict_mask = mask[:, :, None, :].expand_as(predict_states_error)
        else:
            predict_mask = mask
        masked_predict_states_error = predict_states_error * predict_mask

        # Normal L2 loss, take mean over actual data
        q_loss = (masked_td_error**2).sum() / mask.sum()
        aux_loss = (masked_predict_states_error**2).sum() / predict_mask.sum()
        loss = q_loss + self.aux_coef * aux_loss

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        self.training_steps += 1
        if (
            self.args.target_update_interval_or_tau > 1
            and (self.training_steps - self.last_target_update_step)
            / self.args.target_update_interval_or_tau
            >= 1.0
        ):
            self._update_targets_hard()
            self.last_target_update_step = self.training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("q_loss", q_loss.item(), t_env)
            self.logger.log_stat("aux_loss", aux_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat(
                "td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env
            )
            if self.global_reward:
                self.logger.log_stat(
                    "q_taken_mean",
                    (chosen_action_qvals * mask).sum().item()
                    / (mask_elems * self.args.n_agents),
                    t_env,
                )
            else:
                self.logger.log_stat(
                    "q_taken_mean",
                    (chosen_action_qvals[:, :, :, None] * mask).sum().item()
                    / (mask_elems * self.args.n_agents),
                    t_env,
                )
            self.logger.log_stat(
                "target_mean",
                (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                t_env,
            )
            self.log_stats_t = t_env


class ContQLearner(QLearner):

    def __init__(self, mac, scheme, logger, args):
        super().__init__(mac, scheme, logger, args)

        self.aux_coef = args.aux_coef
        self.cont_t_interval = args.topk_neighbors - 1
        self.temperature = args.temperature
        self.neg_num = args.neg_num
        assert self.global_reward, "Contrastive Q-Learner only supports global reward"

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        states = batch["state"][:, :-1]
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        # DIFF: terminated expandes as rewards
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        # Calculate estimated Q-Values
        mac_out = []
        state_predicts = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            # different
            agent_outs, state_predicted = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
            state_predicts.append(state_predicted)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        state_predicts = th.stack(state_predicts, dim=1)  # Concat over time
        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(
            3
        )  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            # diffeent
            target_agent_outs, _ = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(
                chosen_action_qvals, batch["state"][:, :-1]
            )
            target_max_qvals = self.target_mixer(
                target_max_qvals, batch["state"][:, 1:]
            )

        if self.args.standardise_returns:
            target_max_qvals = (
                target_max_qvals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean
            )

        # Calculate 1-step Q-Learning targets
        targets = (
            rewards + self.args.gamma * (1 - terminated) * target_max_qvals.detach()
        )
        if self.args.standardise_returns:
            self.ret_ms.update(targets)
            targets = (targets - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

        # Td-error
        td_error = chosen_action_qvals - targets.detach()
        info_nce_loss = self.info_nce_loss(state_predicts[:, :-1], mask)

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        q_loss = (masked_td_error**2).sum() / mask.sum()
        eps = 1e-10
        loss = (
            q_loss
            + self.aux_coef
            * (q_loss / (info_nce_loss + eps)).abs().detach()
            * info_nce_loss
        )

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        self.training_steps += 1
        if (
            self.args.target_update_interval_or_tau > 1
            and (self.training_steps - self.last_target_update_step)
            / self.args.target_update_interval_or_tau
            >= 1.0
        ):
            self._update_targets_hard()
            self.last_target_update_step = self.training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("q_loss", q_loss.item(), t_env)
            self.logger.log_stat("info_nce_loss", info_nce_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat(
                "td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env
            )
            self.logger.log_stat(
                "q_taken_mean",
                (chosen_action_qvals * mask).sum().item()
                / (mask_elems * self.args.n_agents),
                t_env,
            )
            self.logger.log_stat(
                "target_mean",
                (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                t_env,
            )
            self.log_stats_t = t_env

    def info_nce_loss(self, predict_feat, mask):
        """Implementation following https://github.com/sthalles/SimCLR/blob/master/simclr.py
        Args:
            predict_feat (_type_): (bs, n_steps, n_agents, feat_dim)
            mask (_type_): (bs, n_steps, 1)
        """

        # same timestep, different agent as postive pairs, different timestep as negative pairs

        bs, n_steps, n_agents, feat_dim = predict_feat.size()
        # ref_timestep is a random int value from 0 to n_steps-1
        # implement a smaller version first to avoid oom
        ref_timestep = th.randint(0, n_steps, (1,)).to(predict_feat.device)
        # ref_agent = th.randint(0, n_agents, (1,)).to(predict_feat.device)
        # (bs, featdim)
        # ref_feat = predict_feat[:, ref_timestep, ref_agent]
        # (bs, n_agents, featdim)
        ref_feat = predict_feat[:, ref_timestep[0]]
        ref_mask = mask[:, ref_timestep[0]][:, None, :].expand(-1, n_agents, -1)

        # generate random indices pos_ind with shape (bs, n_agents), where pos_ind[i,j] != j
        pos_ind_temp = (
            th.arange(n_agents).to(predict_feat.device)[None, :].expand(bs, -1)
        )
        pos_ind = th.randint(0, n_agents - 1, (bs, n_agents)).to(predict_feat.device)
        pos_ind[pos_ind >= pos_ind_temp] += 1
        # (bs, n_agents, featdim)
        pos_ind = pos_ind[:, :, None].expand(-1, -1, feat_dim)

        pos_feat = predict_feat[:, ref_timestep[0]]
        # (bs, n_agents, featdim)
        pos_feat = th.gather(pos_feat, 1, pos_ind)

        neg_time_ind_candidates = []
        for x in range(n_steps):
            if (
                x < ref_timestep - self.cont_t_interval
                or x > ref_timestep + self.cont_t_interval
            ):
                neg_time_ind_candidates.append(x)
        neg_time_ind_candidates = th.tensor(neg_time_ind_candidates).to(
            predict_feat.device
        )
        ind_ind = th.randint(0, len(neg_time_ind_candidates), (bs,)).to(
            predict_feat.device
        )
        # (bs,)
        neg_time_ind = neg_time_ind_candidates[ind_ind]
        neg_mask = mask[th.arange(bs), neg_time_ind, 0]
        neg_mask = neg_mask[:, None].expand(-1, n_agents)
        neg_time_ind = neg_time_ind[:, None, None, None].expand(
            -1, -1, n_agents, feat_dim
        )
        # (bs, n_agents, featdim)
        neg_feat = th.gather(predict_feat, 1, neg_time_ind)[:, 0, :, :]

        # neg_agent_ind = th.randperm(n_agents).to(predict_feat.device)[: self.neg_num]
        # (bs, neg_num), correlation along the n_agent dim but otherwise will slow down the code
        neg_agent_ind = th.stack(
            [th.randperm(n_agents)[: self.neg_num] for _ in range(bs)]
        ).to(predict_feat.device)
        neg_agent_ind = neg_agent_ind[:, None, :, None].expand(
            -1, n_agents, -1, feat_dim
        )
        neg_feat = neg_feat[:, None, :, :].expand(-1, n_agents, -1, -1)
        # (bs, n_agents, neg_num, featdim)
        neg_feat = th.gather(neg_feat, 2, neg_agent_ind)

        ref_feat = ref_feat.reshape(bs * n_agents, feat_dim)
        pos_feat = pos_feat.reshape(bs * n_agents, feat_dim)
        neg_feat = neg_feat.reshape(bs * n_agents, self.neg_num, feat_dim)
        ref_mask = ref_mask.reshape(bs * n_agents)
        neg_mask = neg_mask.reshape(bs * n_agents)

        # (bs*n_agents, 1, 1)
        pos_pairs = th.matmul(pos_feat[:, None, :], ref_feat[:, :, None])
        # (bs*n_agents, 1)
        pos_pairs = pos_pairs[:, 0]
        # (bs*n_agents, 1, neg_num)
        neg_pairs = th.matmul(pos_feat[:, None, :], neg_feat.permute(0, 2, 1))
        neg_pairs = neg_pairs[:, 0]

        # (bs*n_agents, neg_num + 1)
        logits = th.cat([pos_pairs, neg_pairs], dim=1)
        final_mask = th.ones(logits.shape[0], dtype=th.long).to(predict_feat.device)
        final_mask = final_mask * ref_mask
        final_mask = final_mask * neg_mask

        # select only valid values
        logits = logits[final_mask.bool()]

        labels = th.zeros(logits.shape[0], dtype=th.long).to(predict_feat.device)

        logits = logits / self.temperature

        # this shall automatically average the loss
        info_nce_loss = F.cross_entropy(logits, labels)  # need to rescale with masks

        return info_nce_loss
