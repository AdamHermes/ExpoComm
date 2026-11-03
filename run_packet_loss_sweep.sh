#!/usr/bin/env bash

# Submit evaluation jobs for multiple independent packet-loss rates.
# Usage:
#   bash run_packet_loss_sweep.sh <algorithm> <environment> <experiment> [loss ...]
#
# Defaults: 0%, 10%, 20%, 30%, 50%; five evaluation seeds; 100 episodes per job.

set -Eeuo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <algorithm> <environment> <experiment> [loss ...]" >&2
    exit 2
fi

ALGORITHM="$1"
ENVIRONMENT="$2"
EXPERIMENT="$3"
shift 3

if [[ $# -eq 0 ]]; then
    LOSS_RATES=(0 10 20 30 50)
else
    LOSS_RATES=("$@")
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="${SCRIPT_DIR}/run_reproduce_expocomm.sh"
TEST_NEPISODE="${PACKET_LOSS_TEST_EPISODES:-100}"
read -r -a SEEDS <<< "${PACKET_LOSS_SEEDS:-0 1 2 3 4}"

if [[ ! "${TEST_NEPISODE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "PACKET_LOSS_TEST_EPISODES must be a positive integer." >&2
    exit 2
fi

declare -A seen_seeds=() seen_rates=()
for seed in "${SEEDS[@]}"; do
    if [[ ! "${seed}" =~ ^(0|[1-9][0-9]{0,9})$ ]] ||
        (( seed > 4294967295 )) || [[ -n "${seen_seeds[$seed]:-}" ]]; then
        echo "Invalid or duplicate evaluation seed: ${seed}" >&2
        exit 2
    fi
    seen_seeds[$seed]=1
done
if [[ ${#SEEDS[@]} -eq 0 ]]; then
    echo "At least one evaluation seed is required." >&2
    exit 2
fi

# Validate the entire sweep before submitting any jobs.
for rate in "${LOSS_RATES[@]}"; do
    if [[ ! "${rate}" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
        ! awk -v rate="${rate}" 'BEGIN { exit !(rate >= 0 && rate <= 100) }'; then
        echo "Invalid packet-loss rate '${rate}'; expected a number from 0 to 100." >&2
        exit 2
    fi
    canonical_rate="$(awk -v rate="${rate}" 'BEGIN { printf "%.12g", rate }')"
    if [[ -n "${seen_rates[$canonical_rate]:-}" ]]; then
        echo "Duplicate packet-loss rate: ${rate}" >&2
        exit 2
    fi
    seen_rates[$canonical_rate]=1
done

cd "${SCRIPT_DIR}"
mkdir -p slurm
RESULTS_ROOT="${PACKET_LOSS_RESULTS_ROOT:-${SCRIPT_DIR}/packet_loss_results}"
mkdir -p "${RESULTS_ROOT}"
RESULTS_DIR="$(mktemp -d "${RESULTS_ROOT%/}/sweep_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
RESULTS_DIR="$(cd "${RESULTS_DIR}" && pwd)"
MANIFEST="${RESULTS_DIR}/manifest.tsv"
printf 'loss\tseed\tepisodes\tfile\n' > "${MANIFEST}"
for rate in "${LOSS_RATES[@]}"; do
    rate_tag="${rate//./p}"
    for seed in "${SEEDS[@]}"; do
        printf '%s\t%s\t%s\t%s\n' "${rate}" "${seed}" "${TEST_NEPISODE}" \
            "loss${rate_tag}_seed${seed}.json" >> "${MANIFEST}"
    done
done
echo "Results: ${RESULTS_DIR}"
echo "Jobs: $((${#LOSS_RATES[@]} * ${#SEEDS[@]})); episodes per job: ${TEST_NEPISODE}"
for rate in "${LOSS_RATES[@]}"; do
    rate_tag="${rate//./p}"
    for seed in "${SEEDS[@]}"; do
        echo "Submitting ${rate}% packet loss, evaluation seed ${seed}"
        job_id="$(sbatch --parsable \
        --job-name="expocomm-pl${rate_tag}-s${seed}" \
        "${LAUNCHER}" \
        "${ALGORITHM}" \
        "${ENVIRONMENT}" \
        "${EXPERIMENT}" \
        "--packet-loss-percent=${rate}" \
        --packet-loss-mode=test \
        "--run-name-suffix=loss${rate_tag}_seed${seed}" \
        "--eval-seed=${seed}" \
        "--eval-result-path=${RESULTS_DIR}/loss${rate_tag}_seed${seed}.json" \
        "--test-nepisode=${TEST_NEPISODE}" \
        --record-video=false)"
        printf '%s\t%s\t%s\n' "${rate}" "${seed}" "${job_id}" >> "${RESULTS_DIR}/jobs.tsv"
        echo "Submitted ${job_id}"
    done
done
printf 'After jobs finish, run:\npython scripts/summarize_packet_loss.py "%s"\n' "${RESULTS_DIR}"
