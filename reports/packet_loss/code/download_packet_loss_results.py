"""Download Battle/AdvPursuit evaluations (five maps by default).

For all six maps at 100% loss:
python scripts/download_packet_loss_results.py --include-battle25 --loss-rates 100

Cross-platform: uses the current Python interpreter, without shell commands.
"""
import argparse
import datetime as dt
import math
from pathlib import Path
import subprocess
import sys


EXPERIMENTS = (
    ("ExpoComm_Battle35_eval", "5008875"),
    ("ExpoComm_Battle40_eval", "5008330"),
    ("ExpoComm_AdvPursuit45_eval", "5050200"),
    ("ExpoComm_AdvPursuit60_eval", "5050200"),
    ("ExpoComm_AdvPursuit70_eval", "5050200"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default="adamhermes66-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm")
    parser.add_argument("--out-root", type=Path, help="Default: a new timestamped folder under results/")
    parser.add_argument("--created-after", help="Optional inclusive ISO timestamp with timezone")
    parser.add_argument("--created-before", help="Optional exclusive ISO timestamp with timezone")
    parser.add_argument("--expected-seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--include-battle25", action="store_true", help="Download all six maps, including Battle25")
    parser.add_argument("--loss-rates", nargs="+", type=float, default=[0, 10, 20, 30, 50])
    parser.add_argument("--history", action="store_true")
    args = parser.parse_args()
    if any(not math.isfinite(rate) or not 0 <= rate <= 100 for rate in args.loss_rates):
        parser.error("--loss-rates must be between 0 and 100")
    if len(set(args.loss_rates)) != len(args.loss_rates):
        parser.error("--loss-rates must be unique")
    if args.out_root is None:
        prefix = "packet_loss100_" if args.loss_rates == [100] else (
            "packet_loss_all_" if args.include_battle25 else "packet_loss_remaining_")
        args.out_root = Path("results") / (prefix + dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    exporter = Path(__file__).resolve().with_name("export_packet_loss_wandb.py")
    failed = []
    experiments = (("ExpoComm_Battle25_eval", "5009240"),) + EXPERIMENTS if args.include_battle25 else EXPERIMENTS
    for project, step in experiments:
        print(f"\nDownloading {project} ...", flush=True)
        command = [sys.executable, str(exporter), "--project", f"{args.entity}/{project}",
                   "--name-prefix", f"{project}_{step}",
                   "--out", str(args.out_root / project),
                   "--expected-seeds", *map(str, args.expected_seeds),
                   "--loss-rates", *map(str, args.loss_rates)]
        for option in ("created_after", "created_before"):
            value = getattr(args, option)
            if value:
                command.extend(["--" + option.replace("_", "-"), value])
        if args.history:
            command.append("--history")
        if subprocess.run(command).returncode != 0:
            failed.append(project)
    print(f"\nOutput folder: {args.out_root.resolve()}")
    if failed:
        print("Failed projects: " + ", ".join(failed), file=sys.stderr)
        return 1
    print(f"Downloaded all {len(experiments)} projects. Check warnings.txt in each folder before interpreting results.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
