"""Generate reproducible commands; execution is explicit via --execute."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def build_runs(algorithms, environments, experiments, seeds, steps, wandb_mode):
    if len(environments) != len(experiments):
        raise ValueError("Provide one experiment config per environment")
    runs = []
    for env, exp in zip(environments, experiments):
        for alg in algorithms:
            for folder, name in (("envs", env), ("exp", exp), ("algs", alg)):
                if not (ROOT / "src/config" / folder / (name + ".yaml")).is_file():
                    raise ValueError("Missing config: " + name)
            for seed in seeds:
                options = ["--config=" + alg, "--env-config=" + env, "--exp-config=" + exp,
                           "--seed=" + str(seed), "--t-max=" + str(steps),
                           "--wandb-mode=" + wandb_mode,
                           "--run-name-suffix=" + env + "_seed" + str(seed)]
                runs.append(dict(algorithm=alg, environment=env, experiment=exp,
                                 seed=seed, requested_steps=steps,
                                 command=[sys.executable, "src/main.py"] + options))
    return runs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--algorithms", nargs="+", default=["iql_expocomm_mv_no_multiview", "iql_expocomm_mv_fixed", "iql_expocomm_mv_adaptive"])
    p.add_argument("--environments", nargs="+", default=["MAgent_Battle"])
    p.add_argument("--experiments", nargs="+", default=["Multiview_Battle25_s0"])
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    p.add_argument("--steps", type=int, default=5050000)
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="disabled")
    p.add_argument("--manifest", type=Path)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--slurm", action="store_true", help="Print sbatch commands (does not submit)")
    args = p.parse_args()
    if args.steps < 1 or len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s > 2**32-1 for s in args.seeds):
        p.error("Positive steps and distinct valid seeds required")
    if args.slurm and args.execute:
        p.error("--slurm only prints commands; submit them explicitly")
    runs = build_runs(args.algorithms, args.environments, args.experiments, args.seeds, args.steps, args.wandb_mode)
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(runs, indent=2), encoding="utf-8")
    for run in runs:
        command = run["command"]
        if args.slurm:
            command = ["sbatch", "run_reproduce_expocomm.sh", run["algorithm"],
                       run["environment"], run["experiment"]] + command[5:]
        print(shlex.join(command) if args.slurm or sys.platform != "win32" else subprocess.list2cmdline(command))
        if args.execute:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
