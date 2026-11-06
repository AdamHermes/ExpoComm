"""Aggregate completed runs only; reject incomplete/duplicate seed cohorts."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np


def summarize(root, metric, expected_seeds):
    cohorts = defaultdict(dict)
    settings = {}
    curves = []
    for config_file in sorted(root.rglob("config.json")):
        directory = config_file.parent
        completion = directory / "completed.json"
        if not completion.exists():
            continue
        config = json.loads(config_file.read_text(encoding="utf-8"))
        finished = json.loads(completion.read_text(encoding="utf-8"))
        if finished["actual_steps"] < config["t_max"]:
            raise ValueError("Incomplete step budget: " + str(directory))
        records = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
        series = [r for r in records if r["metric"] == metric]
        if not series or series[-1]["step"] != finished["actual_steps"]:
            raise ValueError("Missing final evaluation: " + str(directory))
        if not all(np.isfinite(r["value"]) for r in series):
            raise ValueError("Nonfinite metric: " + str(directory))
        # Compare identical configured tasks, budgets and eval episode counts.
        env_args = dict(config["env_args"])
        env_args.pop("seed", None)
        signature = json.dumps(env_args, sort_keys=True)
        key = (config["name"], config["env"], signature, config["t_max"], config["test_nepisode"])
        comparable = {k: v for k, v in config.items() if k not in
                      ("seed", "run_name", "work_dir", "exp_name", "env_args")}
        fingerprint = json.dumps(comparable, sort_keys=True)
        if key in settings and settings[key] != fingerprint:
            raise ValueError("Mixed hyperparameters within seed cohort: " + str(key))
        settings[key] = fingerprint
        seed = config["seed"]
        if seed in cohorts[key]:
            raise ValueError("Duplicate completed seed; select a single sweep directory: " + str(key))
        cohorts[key][seed] = (series[-1]["value"], finished["wall_seconds"])
        for row in series:
            curves.append(dict(algorithm=key[0], environment=key[1], env_args=signature,
                               seed=seed, step=row["step"], value=row["value"]))
    if not cohorts:
        raise ValueError("No completed runs found")
    result = []
    rng = np.random.default_rng(0)
    for key, runs in sorted(cohorts.items()):
        if set(runs) != set(expected_seeds):
            raise ValueError("Incomplete/unexpected seed cohort: " + str(key) + " got " + str(sorted(runs)))
        values = np.array([runs[s][0] for s in sorted(runs)])
        low = high = std = None
        if len(values) > 1:
            means = rng.choice(values, (10000, len(values)), replace=True).mean(1)
            low, high = np.quantile(means, [.025, .975]).tolist()
            std = values.std(ddof=1).item()
        result.append(dict(algorithm=key[0], environment=key[1], env_args=key[2],
                           requested_steps=key[3], evaluation_episodes=key[4], metric=metric,
                           seeds=len(values), mean=values.mean().item(), seed_std=std,
                           bootstrap95_low=low, bootstrap95_high=high,
                           wall_seconds_mean=float(np.mean([v[1] for v in runs.values()]))))
    return result, curves


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--metric", default="test_return_mean")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    p.add_argument("--output", type=Path, default=Path("results/multiview_summary.json"))
    args = p.parse_args()
    rows, curves = summarize(args.root, args.metric, args.seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2, allow_nan=False), encoding="utf-8")
    with args.output.with_suffix(".curves.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(curves[0]))
        writer.writeheader()
        writer.writerows(curves)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
