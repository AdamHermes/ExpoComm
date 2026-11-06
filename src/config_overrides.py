import re


_PACKET_LOSS_MODES = {
    "off": (False, False),
    "train": (True, False),
    "test": (False, True),
    "both": (True, True),
}
_RUN_NAME_SUFFIX_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _pop_option(params, option_name):
    prefix = option_name + "="
    matches = [
        (index, value)
        for index, value in enumerate(params)
        if value.startswith(prefix)
    ]
    if len(matches) > 1:
        raise ValueError(f"{option_name} specified more than once")
    if not matches:
        return None

    index, argument = matches[0]
    del params[index]
    value = argument[len(prefix) :]
    if value == "":
        raise ValueError(f"{option_name} requires a value")
    return value


def apply_packet_loss_cli_overrides(config, params):
    """Apply the narrow CLI overrides used by packet-loss sweep jobs."""
    percent_value = _pop_option(params, "--packet-loss-percent")
    mode = _pop_option(params, "--packet-loss-mode")
    run_name_suffix = _pop_option(params, "--run-name-suffix")
    test_nepisode_value = _pop_option(params, "--test-nepisode")
    record_video_value = _pop_option(params, "--record-video")
    seed_value = _pop_option(params, "--eval-seed")
    result_path = _pop_option(params, "--eval-result-path")

    if seed_value is not None:
        try:
            seed = int(seed_value)
        except ValueError as exc:
            raise ValueError("--eval-seed must be an integer in [0, 4294967295]") from exc
        if not 0 <= seed <= 2**32 - 1:
            raise ValueError("--eval-seed must be an integer in [0, 4294967295]")
        config["seed"] = seed

    if result_path is not None:
        if not config.get("checkpoint_path"):
            raise ValueError("Evaluation requires a nonempty checkpoint_path")
        if config.get("runner") != "episode":
            raise ValueError("Sweep result export currently requires runner: episode")
        config["evaluate"] = True
        config["eval_result_path"] = result_path

    if percent_value is not None:
        try:
            percent = float(percent_value)
        except ValueError as exc:
            raise ValueError(
                "--packet-loss-percent must be a number between 0 and 100"
            ) from exc
        if not 0.0 <= percent <= 100.0:
            raise ValueError(
                "--packet-loss-percent must be between 0 and 100, "
                f"got {percent_value}"
            )
        config["packet_loss_percent"] = percent

    if mode is not None:
        if mode not in _PACKET_LOSS_MODES:
            choices = ", ".join(_PACKET_LOSS_MODES)
            raise ValueError(
                f"--packet-loss-mode must be one of {choices}, got {mode}"
            )
        config["packet_loss_train"], config["packet_loss_test"] = (
            _PACKET_LOSS_MODES[mode]
        )

    if run_name_suffix is not None:
        if not _RUN_NAME_SUFFIX_RE.fullmatch(run_name_suffix):
            raise ValueError(
                "--run-name-suffix may contain only letters, numbers, '.', '_', and '-'"
            )
        config["run_name"] = f'{config["run_name"]}_{run_name_suffix}'

    if test_nepisode_value is not None:
        try:
            test_nepisode = int(test_nepisode_value)
        except ValueError as exc:
            raise ValueError("--test-nepisode must be a positive integer") from exc
        if test_nepisode <= 0:
            raise ValueError("--test-nepisode must be a positive integer")
        config["test_nepisode"] = test_nepisode

    if record_video_value is not None:
        normalized = record_video_value.lower()
        if normalized not in {"true", "false"}:
            raise ValueError("--record-video must be either true or false")
        config["record_video"] = normalized == "true"

    return config


def apply_experiment_cli_overrides(config, params):
    """Explicit training overrides; reject typos rather than silently ignoring."""
    for option, key, minimum in (("--seed", "seed", 0), ("--t-max", "t_max", 1),
                                 ("--batch-size", "batch_size", 1),
                                 ("--buffer-size", "buffer_size", 1),
                                 ("--test-interval", "test_interval", 1)):
        value = _pop_option(params, option)
        if value is not None:
            number = int(value)
            if number < minimum or (key == "seed" and number > 2**32 - 1):
                raise ValueError(option + " outside supported range")
            config[key] = number
    mode = _pop_option(params, "--wandb-mode")
    if mode is not None:
        if mode not in ("online", "offline", "disabled"):
            raise ValueError("--wandb-mode must be online, offline or disabled")
        config["wandb_mode"] = mode
    return config
