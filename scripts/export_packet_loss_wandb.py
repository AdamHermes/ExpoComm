"""Read packet-loss evaluation runs from W&B and save JSON, CSV and Markdown.

Requires only wandb plus the Python standard library. Reads remote data only;
does not create or modify W&B runs. Authentication uses `wandb login` or
WANDB_API_KEY. Run --help for selection options.
"""
import argparse
import csv
import datetime as dt
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


METRICS = {
    "win_rate_percent": ("test_red_team_win_mean", 100),
    "return": ("test_return_mean", 1),
    "return_episode_std": ("test_return_std", 1),
    "red_team_alives": ("test_red_team_alives_mean", 1),
    "episode_length": ("test_episode_length_mean", 1),
}


def timestamp(value):
    value = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise ValueError("Dates must include a timezone, e.g. 2026-11-03T00:00:00+07:00")
    return value


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def extract_run(run):
    config = dict(run.config)
    summary = dict(run.summary)
    match = re.search(r"_loss([0-9]+(?:p[0-9]+)?)(?:_seed([0-9]+))?(?:-|$)", run.name)
    named_loss = float(match[1].replace("p", ".")) if match else None
    named_seed = int(match[2]) if match and match[2] is not None else None
    loss = config.get("packet_loss_percent")
    problems = []
    if not numeric(loss) or not 0 <= loss <= 100:
        problems.append("invalid/missing config.packet_loss_percent")
        loss = None
    elif named_loss is not None and loss != named_loss:
        problems.append(f"name says loss={named_loss}, config says loss={loss}")
    seed = config.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        problems.append("invalid/missing config.seed")
        seed = None
    if named_seed is not None and seed != named_seed:
        problems.append(f"name says seed={named_seed}, config says seed={seed}")
    if not config.get("evaluate"):
        problems.append("config.evaluate is not enabled")
    if not config.get("packet_loss_test"):
        problems.append("config.packet_loss_test is not enabled")
    if config.get("packet_loss_train"):
        problems.append("config.packet_loss_train is enabled (expected test-only sweep)")
    row = dict(
        run_id=run.id, name=run.name, state=run.state,
        url=run.url, created_at=run.created_at,
        loss_percent=loss, config_seed=seed, name_seed=named_seed,
        test_nepisode=config.get("test_nepisode"),
        checkpoint_path=config.get("checkpoint_path"),
        load_step=config.get("load_step"), timestep=summary.get("timestep"),
        algorithm=config.get("name"), environment=config.get("env"),
        issues="; ".join(problems),
    )
    for label, (key, scale) in METRICS.items():
        value = summary.get(key)
        if label == "episode_length" and value is None:
            value = summary.get("test_ep_length_mean")
        row[label] = value * scale if numeric(value) else None
    if row["return"] is None or row["win_rate_percent"] is None:
        problems.append("missing/non-finite reward or win-rate summary")
    elif not 0 <= row["win_rate_percent"] <= 100:
        problems.append("win rate is outside [0, 100]%")
    row["issues"] = "; ".join(problems)
    return row, dict(run_id=run.id, name=run.name, config=config, summary=summary)


def aggregate(rows, rates, expected_seeds):
    groups = defaultdict(list)
    for row in rows:
        if row["loss_percent"] is not None:
            groups[row["loss_percent"]].append(row)
    # Never combine different trained checkpoints, episode counts or environments.
    identities = {
        (r["checkpoint_path"], r["load_step"], r["timestep"], r["algorithm"],
         r["environment"], r["test_nepisode"], r["comparison_config"])
        for r in rows
    }
    result, warnings = [], []
    for loss in rates:
        runs = groups[loss]
        counts = Counter(r["config_seed"] for r in runs)
        problems = []
        if not runs:
            problems.append("no runs")
        if set(counts) != set(expected_seeds):
            problems.append("actual seeds differ from expected seeds")
        if any(count > 1 for count in counts.values()):
            problems.append("duplicate actual seed; narrow selection to one sweep")
        if any(r["issues"] for r in runs):
            problems.append("run config/metrics issues; inspect runs.csv")
        if len(identities) > 1:
            problems.append("mixed checkpoint/environment/episode configuration")
        valid = bool(runs) and not problems
        item = dict(loss_percent=loss, n_runs=len(runs),
                    n_unique_actual_seeds=len(counts),
                    actual_seeds=" ".join(str(seed) for seed in counts),
                    status="ok" if valid else "; ".join(problems))
        for metric in METRICS:
            values = [r[metric] for r in runs if r[metric] is not None]
            # Preserve descriptive means even for incomplete selections. The status
            # flags make clear that these are not valid independent-seed estimates.
            compatible = len(identities) <= 1
            item[metric + "_mean"] = statistics.mean(values) if values and compatible else None
            item[metric + "_seed_std"] = (
                statistics.stdev(values)
                if valid and len(values) == len(runs) and len(values) > 1 else None
            )
        if problems:
            warnings.append(f"Loss {loss:g}%: {item['status']}")
        result.append(item)
    return result, warnings


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def format_metric(row, metric):
    mean, sd = row[metric + "_mean"], row[metric + "_seed_std"]
    if mean is None:
        return "n/a"
    return f"{mean:.4f}" if sd is None else f"{mean:.4f} ± {sd:.4f}"


def export(api, args):
    if args.run_ids:
        runs = [api.run(f"{args.project}/{run_id}") for run_id in dict.fromkeys(args.run_ids)]
    else:
        runs = api.runs(args.project, filters={"state": "finished"}, order="+created_at")
    rows, documents, histories = [], [], {}
    for run in runs:
        if run.state != "finished" or not run.name.startswith(args.name_prefix + "_loss"):
            continue
        if args.created_after and timestamp(run.created_at) < args.created_after:
            continue
        if args.created_before and timestamp(run.created_at) >= args.created_before:
            continue
        row, document = extract_run(run)
        if row["loss_percent"] not in args.loss_rates and row["loss_percent"] is not None:
            continue
        config = document["config"]
        # Include task/opponent and action-selection settings, ignoring env seed.
        comparison = {
            key: config.get(key) for key in
            ("agent", "mac", "one_peer", "topk_neighbors", "test_greedy", "evaluation_epsilon")
        }
        comparison["env_args"] = {k: v for k, v in config.get("env_args", {}).items() if k != "seed"}
        row["comparison_config"] = json.dumps(comparison, sort_keys=True)
        rows.append(row)
        documents.append(document)
        if args.history:
            # No keys filter: metrics are logged in separate sparse rows, so a
            # joint keys filter could drop rows. scan_history avoids sampling.
            histories[run.id] = list(run.scan_history())
    if not rows:
        raise ValueError("No matching finished runs. Check project, name prefix and date/run-ID filters.")
    rows.sort(key=lambda row: (row["loss_percent"] if row["loss_percent"] is not None else -1, row["run_id"]))
    summary, warnings = aggregate(rows, args.loss_rates, args.expected_seeds)
    warnings.extend(f"{row['run_id']}: {row['issues']}" for row in rows if row["issues"])
    directory = args.out
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError(f"Output directory must be empty to avoid mixing exports: {directory}")
    (directory / "runs").mkdir()
    for document in documents:
        path = directory / "runs" / (document["run_id"] + ".json")
        path.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
        if args.history:
            path.with_suffix(".history.jsonl").write_text(
                "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in histories[document["run_id"]]),
                encoding="utf-8",
            )
    write_csv(directory / "runs.csv", rows)
    write_csv(directory / "summary.csv", summary)
    table = ["| Packet loss | Runs | Unique actual seeds | Win rate (%) mean ± seed SD | Reward mean ± seed SD | Status |",
             "|---|---|---|---|---|---|"]
    for row in summary:
        table.append(f"| {row['loss_percent']:g}% | {row['n_runs']} | {row['n_unique_actual_seeds']} | "
                     f"{format_metric(row, 'win_rate_percent')} | {format_metric(row, 'return')} | {row['status']} |")
    notes = ["", "SD describes variation BETWEEN actual evaluation seeds, not episode reward spread.",
             "Means with non-ok status are descriptive selected-run means only; seed SD is omitted.",
             "These are evaluations of one trained checkpoint, not independent training runs.",
             "Raw per-episode outcomes were not logged by this code and cannot be recovered from summaries."]
    text = "\n".join(table + notes) + "\n"
    (directory / "table.md").write_text(text, encoding="utf-8")
    (directory / "warnings.txt").write_text("\n".join(warnings) + "\n", encoding="utf-8")
    (directory / "export.json").write_text(json.dumps(dict(
        project=args.project, name_prefix=args.name_prefix, run_ids=[r["run_id"] for r in rows],
        created_after=args.created_after.isoformat() if args.created_after else None,
        created_before=args.created_before.isoformat() if args.created_before else None,
        loss_rates=args.loss_rates, expected_seeds=args.expected_seeds,
        exported_at=dt.datetime.now(dt.timezone.utc).isoformat(),
    ), indent=2), encoding="utf-8")
    print(text)
    for warning in warnings:
        print("WARNING:", warning)
    print(f"Saved {len(rows)} runs to {directory.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, help="ENTITY/PROJECT, from the W&B URL")
    parser.add_argument("--name-prefix", default="ExpoComm_Battle25_eval_5009240")
    parser.add_argument("--out", type=Path, default=Path("results") / "wandb_packet_loss" / dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    parser.add_argument("--created-after", type=timestamp, help="Inclusive ISO timestamp with timezone")
    parser.add_argument("--created-before", type=timestamp, help="Exclusive ISO timestamp with timezone")
    parser.add_argument("--run-ids", nargs="+", help="Exact W&B run IDs to download instead of querying the project")
    parser.add_argument("--loss-rates", nargs="+", type=float, default=[0, 10, 20, 30, 50])
    parser.add_argument("--expected-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--history", action="store_true", help="Also download unsampled logged history as JSONL")
    args = parser.parse_args()
    if len(args.project.split("/")) != 2 or not all(args.project.split("/")):
        parser.error("--project must be ENTITY/PROJECT")
    if any(not math.isfinite(rate) or not 0 <= rate <= 100 for rate in args.loss_rates):
        parser.error("--loss-rates must be between 0 and 100")
    if len(set(args.loss_rates)) != len(args.loss_rates) or len(set(args.expected_seeds)) != len(args.expected_seeds):
        parser.error("Loss rates and expected seeds must be unique")
    if args.created_after and args.created_before and args.created_after >= args.created_before:
        parser.error("--created-after must precede --created-before")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("--out must be a new or empty directory")
    try:
        import wandb
    except ImportError:
        parser.exit(1, "Install the API client first: python -m pip install wandb\n")
    try:
        export(wandb.Api(), args)
    except (ValueError, OSError, wandb.errors.CommError) as exc:
        parser.exit(1, f"Export failed: {exc}\n")


if __name__ == "__main__":
    main()
