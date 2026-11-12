import re


_PACKET_LOSS_MODES = {
    "off": (False, False),
    "train": (True, False),
    "test": (False, True),
    "both": (True, True),
}
_PACKET_LOSS_RECOVERY_MODES = {"hold", "mse_imputer"}
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
    recovery_value = _pop_option(params, "--packet-loss-recovery")
    cache_ttl_value = _pop_option(params, "--packet-cache-ttl")
    imputer_checkpoint_value = _pop_option(params, "--packet-imputer-checkpoint")

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

    if recovery_value is not None:
        if recovery_value not in _PACKET_LOSS_RECOVERY_MODES:
            choices = ", ".join(sorted(_PACKET_LOSS_RECOVERY_MODES))
            raise ValueError(
                "--packet-loss-recovery must be one of "
                f"{choices}, got {recovery_value}"
            )
        config["packet_loss_recovery"] = recovery_value

    if cache_ttl_value is not None:
        try:
            cache_ttl = int(cache_ttl_value)
        except ValueError as exc:
            raise ValueError("--packet-cache-ttl must be a non-negative integer") from exc
        if cache_ttl < 0:
            raise ValueError("--packet-cache-ttl must be a non-negative integer")
        config["packet_cache_ttl"] = cache_ttl

    if imputer_checkpoint_value is not None:
        config["packet_imputer_checkpoint"] = imputer_checkpoint_value

    return config
