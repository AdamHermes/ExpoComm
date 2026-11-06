import datetime
import os
import pprint
import time
import threading
import json

import numpy as np
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
from components.reward_scope import is_global_reward


def run(config, console_logger):

    # check args sanity
    config = args_sanity_check(config, console_logger)

    args = SN(**config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(console_logger)
    logger.metric_path = os.path.join(args.work_dir, "metrics.jsonl")

    # configure tensorboard logger
    # unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    logger.console_logger.info(f"Current directory: {os.getcwd()}.")
    logger.console_logger.info("Experiment Parameters:")
    experiment_params = pprint.pformat(config, indent=4, width=1)
    logger.console_logger.info("\n\n" + experiment_params + "\n")

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    # os._exit(os.EX_OK)


def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()

    result_path = getattr(args, "eval_result_path", None)
    if result_path:
        metrics = {
            key: float(values[-1][1])
            for key, values in runner.logger.stats.items()
            if key.startswith("test_") and values
        }
        if "test_return_mean" not in metrics:
            raise RuntimeError("Evaluation completed without test_return_mean")
        result = {
            "seed": args.seed,
            "packet_loss_percent": args.packet_loss_percent,
            "test_nepisode": args.test_nepisode,
            "checkpoint": os.path.realpath(os.path.join(args.checkpoint_path, str(runner.t_env))),
            "algorithm": args.name,
            "environment": args.env,
            "metrics": metrics,
        }
        os.makedirs(os.path.dirname(os.path.abspath(result_path)), exist_ok=True)
        temporary_path = result_path + ".tmp"
        with open(temporary_path, "w", encoding="utf-8") as output:
            json.dump(result, output, indent=2, allow_nan=False)
        os.replace(temporary_path, result_path)
        runner.logger.console_logger.info("Saved evaluation results to %s", result_path)


def _runtime_config_json_default(value):
    # Environment metadata can include NumPy integers from Gym spaces.
    if isinstance(value, np.integer):
        return int(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def run_sequential(args, logger):

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    with open(os.path.join(args.work_dir, "runtime_config.json"), "w", encoding="utf-8") as output:
        json.dump(vars(args), output, indent=2, default=_runtime_config_json_default)

    if is_global_reward(args):
        scheme = {
            "state": {"vshape": env_info["state_shape"]},
            "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
            "avail_actions": {
                "vshape": (env_info["n_actions"],),
                "group": "agents",
                "dtype": th.int,
            },
            "reward": {"vshape": (1,)},
            "terminated": {"vshape": (1,), "dtype": th.uint8},
        }
    else:

        scheme = {
            "state": {"vshape": env_info["state_shape"]},
            "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
            "avail_actions": {
                "vshape": (env_info["n_actions"],),
                "group": "agents",
                "dtype": th.int,
            },
            "reward": {"vshape": (1,), "group": "agents"},
            "terminated": {"vshape": (1,), "dtype": th.uint8},
        }
        pos_flag = getattr(args, "pos_flag", False)
        if pos_flag:
            scheme["positions"] = {"vshape": (env_info["n_agents"] * 2,)}

    if getattr(args, "use_agent_active", False):
        scheme["agent_active"] = {"vshape": (1,), "group": "agents", "dtype": th.uint8}
    if getattr(args, "use_agent_reset", False):
        scheme["agent_reset"] = {"vshape": (1,), "group": "agents", "dtype": th.uint8}
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}

    buffer = ReplayBuffer(
        scheme,
        groups,
        args.buffer_size,
        env_info["episode_limit"] + 1,
        preprocess=preprocess,
        device="cpu" if args.buffer_cpu_only else args.device,
    )

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            if getattr(args, "eval_result_path", None):
                raise FileNotFoundError(args.checkpoint_path)
            logger.console_logger.info(
                "Checkpoint directiory {} doesn't exist".format(args.checkpoint_path)
            )
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if not timesteps:
            raise ValueError("No numbered checkpoints in " + args.checkpoint_path)
        if getattr(args, "eval_result_path", None) and args.load_step and args.load_step not in timesteps:
            raise ValueError("Requested evaluation checkpoint does not exist: " + str(args.load_step))
        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            runner.log_train_stats_t = runner.t_env
            evaluate_sequential(args, runner)
            logger.log_stat("episode", runner.t_env, runner.t_env)
            logger.print_recent_stats()
            logger.console_logger.info("Finished Evaluation")
            return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    while runner.t_env <= args.t_max:

        # Run for a whole episode at a time
        episode_batch = runner.run(test_mode=False)
        buffer.insert_episode_batch(episode_batch)

        if buffer.can_sample(args.batch_size):
            episode_sample = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            learner.train(episode_sample, runner.t_env, episode)

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info(
                "t_env: {} / {}".format(runner.t_env, args.t_max)
            )
            logger.console_logger.info(
                "Estimated time left: {}. Time passed: {}".format(
                    time_left(last_time, last_test_T, runner.t_env, args.t_max),
                    time_str(time.time() - start_time),
                )
            )
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        if args.save_model and (
            runner.t_env - model_save_time >= args.save_model_interval
            or model_save_time == 0
        ):
            model_save_time = runner.t_env
            save_path = os.path.join(args.work_dir, "models", str(runner.t_env))
            # "results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    # Evaluate and save the final trained policy at the same actual step.
    for _ in range(max(1, args.test_nepisode // runner.batch_size)):
        runner.run(test_mode=True)
    if args.save_model:
        final_path = os.path.join(args.work_dir, "models", str(runner.t_env))
        os.makedirs(final_path, exist_ok=True)
        learner.save_models(final_path)
    runner.close_env()
    with open(os.path.join(args.work_dir, "completed.json"), "w", encoding="utf-8") as output:
        json.dump({"actual_steps": runner.t_env, "requested_steps": args.t_max,
                   "episodes": episode, "wall_seconds": time.time() - start_time}, output, indent=2)
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning(
            "CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!"
        )

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (
            config["test_nepisode"] // config["batch_size_run"]
        ) * config["batch_size_run"]

    return config
