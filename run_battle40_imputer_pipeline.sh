#!/usr/bin/env bash
#SBATCH --job-name=expocomm-b40-pipeline
#SBATCH --output=slurm/battle40_pipeline_%j.out
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G

# Sequential Battle40 study: hold, self-update proxy, cache, Plain-MSE, and VA.
# Submit from ~/ExpoComm: sbatch run_battle40_imputer_pipeline.sh

set -Eeuo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONDA_ENV="${CONDA_ENV:-ExpoComm_test}"
CONDA_SH="${CONDA_SH:-/home/b/bach66/miniconda3/etc/profile.d/conda.sh}"
[[ -f "${CONDA_SH}" ]] || { echo "Missing CONDA_SH: ${CONDA_SH}" >&2; exit 2; }
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${REPO_DIR}"
mkdir -p slurm

export MAP_LABEL="Battle40"
export EVAL_EXPERIMENT="ExpoComm_Battle40_eval_5008330"
export MSE_EXPERIMENT="ExpoComm_Battle40_mse_imputer_5008330"
export VA_EXPERIMENT="ExpoComm_Battle40_va_imputer_5008330"
export MSE_ROOT="${REPO_DIR}/work_dirs/ExpoComm_Battle40_mse_imputer/mse_ExpoImpute_n7/ExpoComm_Battle40_mse_imputer_5008330"
export VA_ROOT="${REPO_DIR}/work_dirs/ExpoComm_Battle40_va_imputer/va_ExpoImpute_n7/ExpoComm_Battle40_va_imputer_5008330"
exec bash "${REPO_DIR}/scripts/run_battle_imputer_pipeline.sh"
