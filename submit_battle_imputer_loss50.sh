#!/usr/bin/env bash
# Submit matched-loss 50% recovery pipelines for all three Battle maps.
# Each map is an independent Slurm job.  Each job evaluates Hold, self-update,
# TTL cache, Plain-MSE ExpoImpute, and VA-ExpoImpute; the two imputers are
# trained at 50% loss and then evaluated at 50% loss.
#
# Run from ~/ExpoComm:
#   bash submit_battle_imputer_loss50.sh
# Optional longer evaluation:
#   PIPELINE_TEST_EPISODES=500 bash submit_battle_imputer_loss50.sh

set -Eeuo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
[[ -f "${REPO_DIR}/run_battle25_imputer_pipeline.sh" ]] || {
    echo "ERROR: run from the ExpoComm repository root." >&2
    exit 2
}

EVAL_EPISODES="${PIPELINE_TEST_EPISODES:-100}"
for map in 25 35 40; do
    sbatch --export="ALL,PIPELINE_LOSS_PERCENT=50,PIPELINE_TRAIN_LOSS_PERCENT=50,PIPELINE_CACHE_TTL=8,PIPELINE_TEST_EPISODES=${EVAL_EPISODES}" \
        "${REPO_DIR}/run_battle${map}_imputer_pipeline.sh"
done

echo "Submitted Battle25/35/40 matched-loss pipelines: train=50%, eval=50%, TTL=8, episodes=${EVAL_EPISODES}."
