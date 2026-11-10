#!/usr/bin/env bash
#SBATCH --job-name=expocomm-loss100
#SBATCH --output=slurm/packet_loss100_%A_%a.out
#SBATCH --array=0-5
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G

# From the repository root on the cluster:
#   mkdir -p slurm
#   sbatch run_packet_loss100_all.sh
#
# Six evaluation jobs, one per map, at 100% REMOTE-message loss only.
# Existing one-peer code keeps self-links and the receiver's message memory.
# This tests a trained checkpoint without remote delivery; it does not train
# a policy from scratch without communication.
# All selected evaluation configs have actual seed 0, matching earlier runs.
# Defaults: 100 episodes per map (600 total), no GIF recording.
# Override episodes: PACKET_LOSS_TEST_EPISODES=200 sbatch run_packet_loss100_all.sh
# CONDA_ENV and CONDA_SH are forwarded to the existing reproduction launcher.
# No fixed node is requested.
#
# Optional local command preview (does not run or submit evaluation):
#   SLURM_ARRAY_TASK_ID=0 PACKET_LOSS_DRY_RUN=1 bash run_packet_loss100_all.sh
# Array indices: 0 Battle25, 1 Battle35, 2 Battle40,
#                3 AdvPursuit45, 4 AdvPursuit60, 5 AdvPursuit70.
#
# For subsequent W&B export, use --loss-rates 100 --expected-seeds 0 with
# scripts/export_packet_loss_wandb.py (the exporter otherwise defaults to 0-50).

set -Eeuo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ ! -f "${REPO_DIR}/src/main.py" || ! -f "${REPO_DIR}/run_reproduce_expocomm.sh" ]]; then
    echo "ERROR: submit from the ExpoComm repository root." >&2
    exit 2
fi
cd "${REPO_DIR}"

TASK_INDEX="${SLURM_ARRAY_TASK_ID:-}"
if [[ ! "${TASK_INDEX}" =~ ^[0-5]$ ]]; then
    echo "ERROR: expected SLURM_ARRAY_TASK_ID from 0 to 5; submit with sbatch." >&2
    exit 2
fi

TEST_NEPISODE="${PACKET_LOSS_TEST_EPISODES:-100}"
if [[ ! "${TEST_NEPISODE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: PACKET_LOSS_TEST_EPISODES must be a positive integer." >&2
    exit 2
fi

ALGORITHMS=(
    iql_ExpoComm_one_peer_n6
    iql_ExpoComm_one_peer_n7
    iql_ExpoComm_one_peer_n7
    iql_ExpoComm_one_peer_n6
    iql_ExpoComm_one_peer_n7
    iql_ExpoComm_one_peer_n7
)
ENVIRONMENTS=(
    MAgent_Battle MAgent_Battle MAgent_Battle
    MAgent_AdvPursuit MAgent_AdvPursuit MAgent_AdvPursuit
)
EXPERIMENTS=(
    ExpoComm_Battle25_eval_5009240
    ExpoComm_Battle35_eval_5008875
    ExpoComm_Battle40_eval_5008330
    ExpoComm_AdvPursuit45_eval_5050200
    ExpoComm_AdvPursuit60_eval_5050200
    ExpoComm_AdvPursuit70_eval_5050200
)

for config_path in \
    "src/config/algs/${ALGORITHMS[$TASK_INDEX]}.yaml" \
    "src/config/envs/${ENVIRONMENTS[$TASK_INDEX]}.yaml" \
    "src/config/exp/${EXPERIMENTS[$TASK_INDEX]}.yaml"; do
    if [[ ! -f "${config_path}" ]]; then
        echo "ERROR: missing configuration: ${config_path}" >&2
        exit 2
    fi
done

echo "Array task: ${TASK_INDEX}"
echo "Experiment: ${EXPERIMENTS[$TASK_INDEX]}"
echo "Remote-message loss: 100%; evaluation episodes: ${TEST_NEPISODE}; actual config seed: 0"

COMMAND=(
    bash "${REPO_DIR}/run_reproduce_expocomm.sh"
    "${ALGORITHMS[$TASK_INDEX]}"
    "${ENVIRONMENTS[$TASK_INDEX]}"
    "${EXPERIMENTS[$TASK_INDEX]}"
    --packet-loss-percent=100
    --packet-loss-mode=test
    --run-name-suffix=loss100_seed0
    "--test-nepisode=${TEST_NEPISODE}"
    --record-video=false
)

if [[ "${PACKET_LOSS_DRY_RUN:-0}" == "1" ]]; then
    printf 'Command: '
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
    exit 0
fi

exec "${COMMAND[@]}"
