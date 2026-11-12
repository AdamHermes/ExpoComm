#!/usr/bin/env bash
#SBATCH --job-name=expocomm-mse-imputer
#SBATCH --output=slurm/mse_imputer_%j.out
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G

# Evaluate the one-peer residual message-imputer execution path.
#
# Submit from the ExpoComm repository root:
#   mkdir -p slurm
#   sbatch run_mse_imputer_eval.sh \
#     iql_ExpoComm_one_peer_n6 MAgent_Battle ExpoComm_Battle25_eval_5009240 50 8
#
# Arguments:
#   1 algorithm        e.g. iql_ExpoComm_one_peer_n6
#   2 environment      e.g. MAgent_Battle
#   3 experiment       e.g. ExpoComm_Battle25_eval_5009240
#   4 loss percent     optional; default 50
#   5 cache TTL        optional; default 8
#
# Optional environment variables:
#   PACKET_LOSS_TEST_EPISODES=100  (default 100)
#   CONDA_ENV=ExpoComm_test
#   CONDA_SH=/path/to/conda.sh
#   PACKET_IMPUTER_CHECKPOINT=/absolute/path/to/imputer.th
#
# Important: until an imputer checkpoint has been MSE-trained and saved, this
# runs the zero-residual imputer. It is an execution/cache smoke test, not a
# learned-predictor result.

set -Eeuo pipefail

if [[ $# -lt 3 || $# -gt 5 ]]; then
    echo "Usage: sbatch $0 <algorithm> <environment> <experiment> [loss_percent] [cache_ttl]" >&2
    exit 2
fi

ALGORITHM="$1"
ENVIRONMENT="$2"
EXPERIMENT="$3"
LOSS_PERCENT="${4:-50}"
CACHE_TTL="${5:-8}"
TEST_NEPISODE="${PACKET_LOSS_TEST_EPISODES:-100}"
REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

if [[ ! -f "${REPO_DIR}/src/main.py" || ! -f "${REPO_DIR}/run_reproduce_expocomm.sh" ]]; then
    echo "ERROR: submit this job from the ExpoComm repository root." >&2
    exit 2
fi
if [[ ! "${LOSS_PERCENT}" =~ ^[0-9]+([.][0-9]+)?$ ]] || ! awk -v p="${LOSS_PERCENT}" 'BEGIN { exit !(p >= 0 && p <= 100) }'; then
    echo "ERROR: loss_percent must be a number from 0 to 100." >&2
    exit 2
fi
if [[ ! "${CACHE_TTL}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: cache_ttl must be a non-negative integer." >&2
    exit 2
fi
if [[ ! "${TEST_NEPISODE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: PACKET_LOSS_TEST_EPISODES must be a positive integer." >&2
    exit 2
fi

cd "${REPO_DIR}"
mkdir -p slurm

echo "Algorithm: ${ALGORITHM}"
echo "Environment: ${ENVIRONMENT}"
echo "Experiment: ${EXPERIMENT}"
echo "Loss: ${LOSS_PERCENT}% | TTL: ${CACHE_TTL} | Episodes: ${TEST_NEPISODE}"
echo "Recovery: mse_imputer"

EXTRA_IMPUTER_ARG=()
if [[ -n "${PACKET_IMPUTER_CHECKPOINT:-}" ]]; then
    if [[ ! -f "${PACKET_IMPUTER_CHECKPOINT}" ]]; then
        echo "ERROR: PACKET_IMPUTER_CHECKPOINT does not exist: ${PACKET_IMPUTER_CHECKPOINT}" >&2
        exit 2
    fi
    EXTRA_IMPUTER_ARG=("--packet-imputer-checkpoint=${PACKET_IMPUTER_CHECKPOINT}")
fi

exec bash "${REPO_DIR}/run_reproduce_expocomm.sh" \
    "${ALGORITHM}" \
    "${ENVIRONMENT}" \
    "${EXPERIMENT}" \
    "--packet-loss-percent=${LOSS_PERCENT}" \
    --packet-loss-mode=test \
    --packet-loss-recovery=mse_imputer \
    "--packet-cache-ttl=${CACHE_TTL}" \
    "--run-name-suffix=mse_imputer_loss${LOSS_PERCENT}_ttl${CACHE_TTL}" \
    "--test-nepisode=${TEST_NEPISODE}" \
    --record-video=false \
    "${EXTRA_IMPUTER_ARG[@]}"
