"""Summarize a completed sweep using equal weight for each evaluation seed.

Only Python's standard library is needed. Bootstrap intervals resample seed
means, not individual episodes; they describe evaluation variability for one
fixed trained checkpoint, not variability between training seeds.
"""
import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


def interval(values):
    if len(values) < 2:
        return None, None
    rng = random.Random(0)
    samples = sorted(
        statistics.mean(rng.choices(values, k=len(values))) for _ in range(10000)
    )
    return samples[249], samples[9749]


def summarize(directory):
    groups = defaultdict(list)
    missing = []
    identities = set()
    expected = set()
    with (directory / "manifest.tsv").open(encoding="utf-8") as source:
        for row in csv.DictReader(source, delimiter="\t"):
            key = float(row["loss"]), int(row["seed"])
            if key in expected:
                raise ValueError(f"Duplicate loss/seed in manifest: {key}")
            expected.add(key)
            path = directory / row["file"]
            if not path.is_file():
                missing.append(row["file"])
                continue
            result = json.loads(path.read_text(encoding="utf-8"))
            if (result["packet_loss_percent"], result["seed"]) != key:
                raise ValueError(f"Loss/seed mismatch: {path}")
            if result["test_nepisode"] != int(row["episodes"]):
                raise ValueError(f"Episode count mismatch: {path}")
            identities.add((result["checkpoint"], result["algorithm"], result["environment"], result["test_nepisode"]))
            groups[key[0]].append(result)
    if not expected:
        raise ValueError("Empty sweep manifest")
    if missing:
        raise ValueError("Sweep incomplete; missing results:\n" + "\n".join(missing))
    if len(identities) != 1:
        raise ValueError("Results use different checkpoints, algorithms, environments or episode counts")
    seed_sets = [{item["seed"] for item in runs} for runs in groups.values()]
    if any(seeds != seed_sets[0] for seeds in seed_sets):
        raise ValueError("Loss rates must use the same set of evaluation seeds")

    rows = []
    for loss, runs in sorted(groups.items()):
        metric_keys = set(runs[0]["metrics"])
        if any(set(run["metrics"]) != metric_keys for run in runs):
            raise ValueError(f"Inconsistent metrics at loss {loss}")
        for metric in sorted(metric_keys):
            values = [run["metrics"][metric] for run in runs]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"Non-finite metric: {loss}, {metric}")
            low, high = interval(values)
            baseline = groups.get(0.0)
            delta = None
            if baseline and all(metric in run["metrics"] for run in baseline):
                delta = statistics.mean(values) - statistics.mean(run["metrics"][metric] for run in baseline)
            rows.append(dict(
                loss_percent=loss, metric=metric, seeds=len(runs),
                episodes_per_seed=runs[0]["test_nepisode"],
                mean=statistics.mean(values),
                seed_std=statistics.stdev(values) if len(values) > 1 else None,
                bootstrap95_low=low, bootstrap95_high=high,
                delta_vs_loss0=delta,
            ))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args()
    try:
        rows = summarize(args.results_dir)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f"{exc}\n")
    path = args.results_dir / "summary.csv"
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Mean across evaluation seeds; 95% percentile bootstrap interval across seeds.")
    print("Intervals are approximate, especially with few seeds; one seed gives no interval.")
    print("These are evaluation seeds of ONE trained checkpoint, not independent training runs.")
    for row in rows:
        if row["metric"] in {"test_return_mean", "test_red_team_win_mean", "test_red_team_alives_mean", "test_episode_length_mean", "test_ep_length_mean"}:
            bounds = "n/a" if row["bootstrap95_low"] is None else f'[{row["bootstrap95_low"]:.4f}, {row["bootstrap95_high"]:.4f}]'
            print(f'{row["loss_percent"]:g}%  {row["metric"]}: {row["mean"]:.4f}  CI95={bounds}')
    print(f"All metrics saved to {path}")


if __name__ == "__main__":
    main()
