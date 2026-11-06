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

The sweep launcher evaluates an existing checkpoint at 0%, 10%, 20%, 30%, and 50% loss
using evaluation seeds 0, 1, 2, 3, 4 and 100 episodes per seed by default.
This submits **25 Slurm jobs (2,500 episodes total)**. Run from the repository
on the cluster. Check `checkpoint_path` and `load_step` in the experiment YAML
first; sweep jobs require the exact numbered checkpoint when `load_step` is nonzero.

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

Each job forces evaluation, enables loss only during evaluation, includes the
rate and evaluation seed in `run_name`, and disables GIF recording. Seeds change
evaluation randomness; they do not retrain the model. Override the seed list
(space-separated) and episode count with environment variables, for example:

```bash
PACKET_LOSS_SEEDS="0 1 2 3 4" PACKET_LOSS_TEST_EPISODES=250 bash run_packet_loss_sweep.sh \
  iql_ExpoComm_one_peer_n6 MAgent_Battle ExpoComm_Battle25_eval_5009240
```

A quick smoke run uses `PACKET_LOSS_SEEDS="0" PACKET_LOSS_TEST_EPISODES=10`
before the same command (5 jobs, 50 episodes total).

The script prints a unique `packet_loss_results/sweep_...` directory. It contains
`manifest.tsv` (all expected runs), `jobs.tsv` (submitted job IDs), and one JSON
file per successful run. Use `PACKET_LOSS_RESULTS_ROOT` to change the parent
directory. Metrics are still sent to W&B and printed to the Slurm log.
After all jobs finish, run the printed command, for example:

```bash
python scripts/summarize_packet_loss.py packet_loss_results/sweep_<YOUR_SWEEP>
```

The summary refuses incomplete sweeps or mixed checkpoints. It prints key metrics
and writes `summary.csv`: mean across evaluation seeds, sample standard deviation
between seed means, approximate 95% percentile bootstrap intervals (10,000 seed
resamples), and absolute difference from the 0% baseline when present. For win
rates, multiply the difference by 100 to obtain percentage points. A single seed
has no interval. `test_return_std` remains the within-run episode reward spread;
`seed_std` is the spread between seed-level metric values.

Five seeds are a starting point; intervals with few seeds may be unreliable.
These intervals describe evaluation variation for one fixed checkpoint, not
variation across independently trained policies. The same seed list is used at
each loss rate, but action trajectories and random-number consumption can diverge;
do not interpret the runs as matched episode pairs. To compare methods, evaluate
their checkpoints separately with the same rates, seeds, and episode counts.

A single run can also be submitted directly:

```bash
sbatch run_reproduce_expocomm.sh \
  iql_ExpoComm_one_peer_n6 \
  MAgent_Battle \
  ExpoComm_Battle25_eval_5009240 \
  --packet-loss-percent=20 \
  --packet-loss-mode=test \
  --run-name-suffix=loss20_seed0 \
  --eval-seed=0 \
  --eval-result-path=packet_loss_results/manual_loss20_seed0.json \
  --test-nepisode=100 \
  --record-video=false
```

Valid modes are `off`, `train`, `test`, and `both`. Self-links are never
dropped. `packet_loss_percent` is a probability parameter, so the realized
fraction in a finite run approaches but need not equal the requested value.

## Troubleshooting

- **`conda: command not found`:** set `CONDA_BASE` to the Miniconda or Anaconda
  installation before submitting the setup job.
- **The training job cannot activate Conda:** update the `source .../conda.sh`
  path in `run_reproduce_expocomm.sh` to your installation.
- **The requested node is unavailable:** remove or replace the hard-coded
  `--nodelist` directive in the reproduction script.
- **Setup cannot find the repository:** ensure the setup script's `cd ~/ExpoComm`
  line matches the location where you cloned this repository.
