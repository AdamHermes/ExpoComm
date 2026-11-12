# ExpoComm: Cluster Setup and Reproduction Guide

This guide describes how to install the ExpoComm environment and reproduce the
provided experiments on a Slurm-based GPU cluster.

## Prerequisites

Before starting, make sure you have:

- Access to a Slurm cluster with a GPU partition.
- A Linux login node with Git and `wget` available.
- A Weights & Biases account (optional, but recommended for viewing training
  metrics).

## 1. Install Miniconda

Run the following commands on the login node if Conda is not already
available:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

Accept Conda's Terms of Service when prompted. If needed, they can also be
accepted explicitly:

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

## 2. Clone the Repository

```bash
git clone <REPOSITORY_URL> ExpoComm
cd ExpoComm
```

All remaining commands in this guide should be run from the repository root.


## 3. Create the Conda Environment

Submit the setup job:

```bash
sbatch setup_expocomm_test.sh
```

The job creates the Conda environment, installs the Python dependencies, and
copies ExpoComm's customised MAgent scenarios into PettingZoo.

Monitor its status and logs with:

```bash
squeue -u <USERNAME>
squeue -j <JOB_ID>
tail -f setup_expocomm_<JOB_ID>.out
```

Wait until the setup job has completed successfully before submitting an
experiment. The `tail -f` command can be stopped with `Ctrl+C`.

## 4. Sign In to Weights & Biases

Activate the environment and authenticate with the API key from your W&B
account:

```bash
conda activate ExpoComm_test
wandb login
```

Training can run without using the web dashboard, but W&B is useful for
monitoring metrics and comparing runs.

## 5. Submit an Experiment

The reproduction script takes three arguments:

```bash
sbatch run_reproduce_expocomm.sh <ALGORITHM> <ENVIRONMENT> <EXPERIMENT>
```

The names correspond to configuration files under `src/config/algs`,
`src/config/envs`, and `src/config/exp`, respectively. Refer to the paper for
the experiment settings and interpretations.

### Adversarial Pursuit

```bash
# 25 agents: K = 1 and K = ceil(log2 N)
sbatch run_reproduce_expocomm.sh iql_ExpoComm_one_peer_n6 MAgent_AdvPursuit ExpoComm_AdvPursuit45_s0


sbatch run_reproduce_expocomm.sh iql_ExpoComm_static_n6   MAgent_AdvPursuit ExpoComm_AdvPursuit45_s0

# 45 agents
sbatch run_reproduce_expocomm.sh iql_ExpoComm_one_peer_n7 MAgent_AdvPursuit ExpoComm_AdvPursuit60_s0


sbatch run_reproduce_expocomm.sh iql_ExpoComm_static_n7   MAgent_AdvPursuit ExpoComm_AdvPursuit60_s0

# 61 agents
sbatch run_reproduce_expocomm.sh iql_ExpoComm_one_peer_n7 MAgent_AdvPursuit ExpoComm_AdvPursuit70_s0


sbatch run_reproduce_expocomm.sh iql_ExpoComm_static_n7   MAgent_AdvPursuit ExpoComm_AdvPursuit70_s0
```

### Battle

```bash
# 20 agents
sbatch run_reproduce_expocomm.sh iql_ExpoComm_one_peer_n6 MAgent_Battle ExpoComm_Battle25_s0


sbatch run_reproduce_expocomm.sh iql_ExpoComm_static_n6   MAgent_Battle ExpoComm_Battle25_s0

# 42 agents
sbatch run_reproduce_expocomm.sh iql_ExpoComm_one_peer_n7 MAgent_Battle ExpoComm_Battle35_s0


sbatch run_reproduce_expocomm.sh iql_ExpoComm_static_n7   MAgent_Battle ExpoComm_Battle35_s0

# 64 agents
sbatch -p gpu-long --nodelist=xgpf7 run_reproduce_expocomm.sh iql_ExpoComm_one_peer_n7 MAgent_Battle ExpoComm_Battle40_s0


sbatch run_reproduce_expocomm.sh iql_ExpoComm_static_n7   MAgent_Battle ExpoComm_Battle40_s0
```

### Choose a GPU Node (Optional)

By default, `run_reproduce_expocomm.sh` requests one GPU and pins the job to
`xgpf10`:

```bash
#SBATCH --nodelist=xgpf10
```

If another GPU node is available, replace `xgpf10` with that node's name. To
let Slurm select any eligible GPU node automatically, remove or comment out
the `#SBATCH --nodelist=xgpf10` line. You may also need to set
cluster-specific options such as `--partition` or `--account`.

Alternatively, request a node for one submission without changing the script:

```bash
sbatch --nodelist=<GPU_NODE> run_reproduce_expocomm.sh iql_ExpoComm_one_peer_n7 MAgent_Battle ExpoComm_Battle35_s0
```

## 6. Evaluate a Packet-loss Sweep

ExpoComm agents exchange tensors inside one Python process, so host-level
`tc/netem` does not affect their communication. Use the logical channel model
to drop complete agent messages independently on each remote link and timestep.

The sweep launcher evaluates an existing evaluation experiment at 0%, 10%, and
20% loss by default:

```bash
bash run_packet_loss_sweep.sh \
  iql_ExpoComm_one_peer_n6 \
  MAgent_Battle \
  ExpoComm_Battle25_eval_5009240
```

Pass explicit percentages to use a different sweep:

```bash
bash run_packet_loss_sweep.sh \
  iql_ExpoComm_one_peer_n6 \
  MAgent_Battle \
  ExpoComm_Battle25_eval_5009240 \
  0 5 10 20 30 50
```

Each submitted job enables packet loss only during evaluation, appends the
rate to `run_name`, evaluates 100 episodes, and disables GIF recording so
concurrent jobs cannot overwrite one another. Override the episode count with
`PACKET_LOSS_TEST_EPISODES`, for example:

```bash
PACKET_LOSS_TEST_EPISODES=250 bash run_packet_loss_sweep.sh \
  iql_ExpoComm_one_peer_n6 MAgent_Battle ExpoComm_Battle25_eval_5009240
```

A single run can also be submitted directly:

```bash
sbatch run_reproduce_expocomm.sh \
  iql_ExpoComm_one_peer_n6 \
  MAgent_Battle \
  ExpoComm_Battle25_eval_5009240 \
  --packet-loss-percent=20 \
  --packet-loss-mode=test \
  --run-name-suffix=loss20 \
  --test-nepisode=100 \
  --record-video=false
```

Valid modes are `off`, `train`, `test`, and `both`. Self-links are never
dropped. `packet_loss_percent` is a probability parameter, so the realized
fraction in a finite run approaches but need not equal the requested value.

## 7. Export existing packet-loss results from W&B

`run_packet_loss_sweep.sh` submits ordinary W&B runs; it does not create a native
W&B Sweep. `src/main.py` uses `exp_name` as the project, stores the full config,
and names each run `<run_name>-<algorithm>-<timestamp>`. The logger sends scalar
metrics through `wandb.log`. Download existing runs without rerunning evaluation:

```bash
python -m pip install wandb
wandb login
python scripts/export_packet_loss_wandb.py \
  --project YOUR_ENTITY/ExpoComm_Battle25_eval \
  --created-after 2026-11-03T00:00:00+07:00 \
  --created-before 2026-11-04T00:00:00+07:00 \
  --out results/battle25_packet_loss_export
```

Replace `YOUR_ENTITY` with the username or team in the W&B project URL. Set the
date range to your sweep (inclusive start, exclusive end). If there are multiple
sweeps on that date, narrow the time range or use `--run-ids ID1 ID2 ...` to select
exact W&B run IDs. `--name-prefix` defaults to `ExpoComm_Battle25_eval_5009240`;
change it for other experiments. Default loss rates are `0 10 20 30 50` and expected
actual seeds are `0 1 2 3 4`; override with `--loss-rates` / `--expected-seeds`.
Only finished runs matching the name prefix and selection are downloaded.

The output directory must be new or empty. It contains:

- `runs/<RUN_ID>.json`: full logged config and summary per run.
- `runs.csv`: per-run metrics, actual config seed, seed in the name, and issues.
- `summary.csv`: selected-run means by loss, independent-seed sample SD when valid,
  run/seed counts, and validation status. Win rates are expressed in percent.
- `table.md`: reward/win-rate comparison table for copying into a report.
- `warnings.txt`: missing rates/seeds, duplicate actual seeds, or invalid configs.
- `export.json`: selected run IDs and export filters for traceability.

Use `--history` to also download unsampled logged history to JSONL per run.
This history contains logged aggregate metrics, not unlogged per-episode outcomes.
The exporter reads W&B data only; it never creates or updates remote runs.
Summary values are taken directly from `run.summary`, not estimated from charts.

Check `status` and `warnings.txt` before interpreting the table. Seed numbers in
run names are not proof that different seeds were applied: config.seed is used as
the actual recorded seed. If names and configs disagree, or multiple runs use the
same actual seed, the exporter flags this and omits seed SD. Descriptive means
remain available for incomplete selections, but mixed checkpoint/environment/
episode configurations are not averaged. Repeating an evaluation does not create
independent training seeds.

## Troubleshooting

- **`conda: command not found`:** set `CONDA_BASE` to the Miniconda or Anaconda
  installation before submitting the setup job.
- **The training job cannot activate Conda:** update the `source .../conda.sh`
  path in `run_reproduce_expocomm.sh` to your installation.
- **The requested node is unavailable:** remove or replace the hard-coded
  `--nodelist` directive in the reproduction script.
- **Setup cannot find the repository:** ensure the setup script's `cd ~/ExpoComm`
  line matches the location where you cloned this repository.
