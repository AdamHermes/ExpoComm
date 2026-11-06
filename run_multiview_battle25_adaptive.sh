#!/usr/bin/env bash
#SBATCH --job-name=mv_battle25_s0
#SBATCH --output=slurm/mv_battle25_s0_%j.out
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G

set -euo pipefail

# Submit from the repository root after creating the slurm log directory.
exec bash "${SLURM_SUBMIT_DIR:-$PWD}/run_reproduce_expocomm.sh" \
    iql_expocomm_mv_adaptive \
    MAgent_Battle \
    Multiview_Battle25_s0 \
    --wandb-mode=disabled
