#!/usr/bin/env bash
#SBATCH --job-name=expocomm-mse-train
#SBATCH --output=slurm/mse_imputer_train_%j.out
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G

# Train ONLY the residual message imputer; ExpoComm policy/Q weights are frozen.
# Submit from ~/ExpoComm:
#   mkdir -p slurm
#   sbatch run_mse_imputer_train.sh

set -Eeuo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONDA_ENV="${CONDA_ENV:-ExpoComm_test}"
CONDA_SH="${CONDA_SH:-/home/b/bach66/miniconda3/etc/profile.d/conda.sh}"

if [[ ! -f "${REPO_DIR}/src/main.py" || ! -f "${CONDA_SH}" ]]; then
    echo "ERROR: submit from ExpoComm root and set CONDA_SH if needed." >&2
    exit 2
fi

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${REPO_DIR}"
mkdir -p slurm

nvidia-smi -L
exec python src/main.py \
    --config=mse_ExpoImpute_n6 \
    --env-config=MAgent_Battle \
    --exp-config=ExpoComm_Battle25_mse_imputer_5009240
