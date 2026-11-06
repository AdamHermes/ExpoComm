from collections import defaultdict
import logging
import numpy as np
import json
import time
from pathlib import Path


class Logger:
    def __init__(self, console_logger):
        self.console_logger = console_logger

        self.use_tb = False
        self.use_sacred = False
        self.use_hdf = False

        self.stats = defaultdict(lambda: [])
        self.metric_path = None
        self.started_at = time.monotonic()

    def setup_tb(self, directory_name):
        raise NotImplementedError("Deprecated")

    def setup_sacred(self, sacred_run_dict):
        raise NotImplementedError("Deprecated")

    def log_stat(self, key, value, t):
        self.stats[key].append((t, value))
        if self.metric_path:
            record = dict(metric=key, value=float(value), step=int(t),
                          elapsed_seconds=time.monotonic() - self.started_at)
            with Path(self.metric_path).open("a", encoding="utf-8") as output:
                output.write(json.dumps(record, allow_nan=False) + "\n")
        # main initializes W&B only when requested. Offline tests require none.
        import sys
        wandb = sys.modules.get("wandb")
        if wandb is not None and getattr(wandb, "run", None) is not None:
            wandb.log({key: value, "timestep": t})

    def print_recent_stats(self):
        log_str = "Recent Stats | t_env: {:>10} | Episode: {:>8}\n".format(
            *self.stats["episode"][-1]
        )
        i = 0
        for k, v in sorted(self.stats.items()):
            if k == "episode":
                continue
            i += 1
            window = 5 if k != "epsilon" else 1
            try:
                item = "{:.4f}".format(np.mean([x[1] for x in self.stats[k][-window:]]))
            except:
                item = "{:.4f}".format(
                    np.mean([x[1].item() for x in self.stats[k][-window:]])
                )
            log_str += "{:<25}{:>8}".format(k + ":", item)
            log_str += "\n" if i % 4 == 0 else "\t"
        self.console_logger.info(log_str)


# set up a custom logger
def get_logger(name, log_file=None, log_level=logging.INFO):
    logger = logging.getLogger(name=name)
    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, mode="w")
        handlers.append(file_handler)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)
    logger.setLevel(log_level)

    return logger
