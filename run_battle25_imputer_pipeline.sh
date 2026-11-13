#!/usr/bin/env bash
#SBATCH --job-name=expocomm-b25-pipeline
#SBATCH --output=slurm/battle25_pipeline_%j.out
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G

# Sequential Battle25 study at 70% loss: hold, self-update proxy, TTL cache,
# Plain-MSE imputer (train + eval), and VA-ExpoImpute (train + eval).
# Submit from ~/ExpoComm:
#   sbatch run_battle25_imputer_pipeline.sh
# Optional: PIPELINE_TEST_EPISODES=500 sbatch run_battle25_imputer_pipeline.sh

set -Eeuo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONDA_ENV="${CONDA_ENV:-ExpoComm_test}"
CONDA_SH="${CONDA_SH:-/home/b/bach66/miniconda3/etc/profile.d/conda.sh}"
[[ -f "${CONDA_SH}" ]] || { echo "Missing CONDA_SH: ${CONDA_SH}" >&2; exit 2; }
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${REPO_DIR}"
mkdir -p slurm

export MAP_LABEL="Battle25"
export PIPELINE_ALGORITHM="iql_ExpoComm_one_peer_n6"
export MSE_ALGORITHM="mse_ExpoImpute_n6"
export VA_ALGORITHM="va_ExpoImpute_n6"
export EVAL_EXPERIMENT="ExpoComm_Battle25_eval_5009240"
export MSE_EXPERIMENT="ExpoComm_Battle25_mse_imputer_5009240"
export VA_EXPERIMENT="ExpoComm_Battle25_va_imputer_5009240"
export MSE_ROOT="${REPO_DIR}/work_dirs/ExpoComm_Battle25_mse_imputer/mse_ExpoImpute_n6/ExpoComm_Battle25_mse_imputer_5009240"
export VA_ROOT="${REPO_DIR}/work_dirs/ExpoComm_Battle25_va_imputer/va_ExpoImpute_n6/ExpoComm_Battle25_va_imputer_5009240"
exec bash "${REPO_DIR}/scripts/run_battle_imputer_pipeline.sh"
