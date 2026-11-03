#!/usr/bin/env bash

# Submit evaluation jobs for multiple independent packet-loss rates.
# Usage:
#   bash run_packet_loss_sweep.sh <algorithm> <environment> <experiment> [loss ...]
#
# With no explicit rates, the script submits 0%, 10%, and 20%.

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
    LOSS_RATES=(0 10 20)
else
    LOSS_RATES=("$@")
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="${SCRIPT_DIR}/run_reproduce_expocomm.sh"
TEST_NEPISODE="${PACKET_LOSS_TEST_EPISODES:-100}"

if [[ ! "${TEST_NEPISODE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "PACKET_LOSS_TEST_EPISODES must be a positive integer." >&2
    exit 2
fi

for rate in "${LOSS_RATES[@]}"; do
    if [[ ! "${rate}" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
        ! awk -v rate="${rate}" 'BEGIN { exit !(rate >= 0 && rate <= 100) }'; then
        echo "Invalid packet-loss rate '${rate}'; expected a number from 0 to 100." >&2
        exit 2
    fi

    rate_tag="${rate//./p}"
    echo "Submitting ${rate}% packet loss"
    sbatch \
        --job-name="expocomm-pl${rate_tag}" \
        "${LAUNCHER}" \
        "${ALGORITHM}" \
        "${ENVIRONMENT}" \
        "${EXPERIMENT}" \
        "--packet-loss-percent=${rate}" \
        --packet-loss-mode=test \
        "--run-name-suffix=loss${rate_tag}" \
        "--test-nepisode=${TEST_NEPISODE}" \
        --record-video=false
done
