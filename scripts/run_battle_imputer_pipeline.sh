#!/usr/bin/env bash
# Common sequential pipeline for one Battle map. Called by the two Slurm
# wrappers below; do not invoke without the required exported variables.

set -Eeuo pipefail

: "${MAP_LABEL:?MAP_LABEL is required}"
: "${EVAL_EXPERIMENT:?EVAL_EXPERIMENT is required}"
: "${MSE_EXPERIMENT:?MSE_EXPERIMENT is required}"
: "${VA_EXPERIMENT:?VA_EXPERIMENT is required}"
: "${MSE_ROOT:?MSE_ROOT is required}"
: "${VA_ROOT:?VA_ROOT is required}"

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
LOSS_PERCENT="${PIPELINE_LOSS_PERCENT:-70}"
TRAIN_LOSS_PERCENT="${PIPELINE_TRAIN_LOSS_PERCENT:-${LOSS_PERCENT}}"
CACHE_TTL="${PIPELINE_CACHE_TTL:-8}"
TEST_EPISODES="${PIPELINE_TEST_EPISODES:-100}"
ALGORITHM="${PIPELINE_ALGORITHM:-iql_ExpoComm_one_peer_n7}"
MSE_ALGORITHM="${MSE_ALGORITHM:-mse_ExpoImpute_n7}"
VA_ALGORITHM="${VA_ALGORITHM:-va_ExpoImpute_n7}"
ENVIRONMENT="MAgent_Battle"
STAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="${REPO_DIR}/results/imputer_pipelines/${MAP_LABEL}_${STAMP}"
LOG_DIR="${RESULTS_DIR}/logs"
RESULTS_CSV="${RESULTS_DIR}/summary.csv"

if [[ ! -f "${REPO_DIR}/src/main.py" ]]; then
    echo "ERROR: run from the ExpoComm repository root." >&2
    exit 2
fi
if [[ ! "${LOSS_PERCENT}" =~ ^[0-9]+([.][0-9]+)?$ ]] || ! awk -v p="${LOSS_PERCENT}" 'BEGIN { exit !(p >= 0 && p <= 100) }'; then
    echo "ERROR: PIPELINE_LOSS_PERCENT must be from 0 to 100." >&2
    exit 2
fi
if [[ ! "${TRAIN_LOSS_PERCENT}" =~ ^[0-9]+([.][0-9]+)?$ ]] || ! awk -v p="${TRAIN_LOSS_PERCENT}" 'BEGIN { exit !(p >= 0 && p <= 100) }'; then
    echo "ERROR: PIPELINE_TRAIN_LOSS_PERCENT must be from 0 to 100." >&2
    exit 2
fi
if [[ ! "${CACHE_TTL}" =~ ^[0-9]+$ || ! "${TEST_EPISODES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: invalid TTL or evaluation episode count." >&2
    exit 2
fi

cd "${REPO_DIR}"
mkdir -p "${LOG_DIR}"
printf '%s\n' \
  "map=${MAP_LABEL}" \
  "algorithm=${ALGORITHM}" \
  "eval_loss_percent=${LOSS_PERCENT}" \
  "train_loss_percent=${TRAIN_LOSS_PERCENT}" \
  "cache_ttl=${CACHE_TTL}" \
  "test_episodes=${TEST_EPISODES}" \
  "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)" \
  "started_at=$(date --iso-8601=seconds)" > "${RESULTS_DIR}/manifest.txt"
printf '%s\n' \
  'method,loss_percent,cache_ttl,imputer_checkpoint,red_win_rate,return_mean,return_std,red_alive_mean,blue_alive_mean,log_path' \
  > "${RESULTS_CSV}"

cp "src/config/exp/${EVAL_EXPERIMENT}.yaml" "${RESULTS_DIR}/eval_config.yaml"
cp "src/config/exp/${MSE_EXPERIMENT}.yaml" "${RESULTS_DIR}/mse_train_config.yaml"
cp "src/config/exp/${VA_EXPERIMENT}.yaml" "${RESULTS_DIR}/va_train_config.yaml"
cp "src/config/algs/${MSE_ALGORITHM}.yaml" "${RESULTS_DIR}/mse_algorithm.yaml"
cp "src/config/algs/${VA_ALGORITHM}.yaml" "${RESULTS_DIR}/va_algorithm.yaml"
printf '%s\n' \
  "packet_loss_percent=${TRAIN_LOSS_PERCENT}" \
  "packet_loss_mode=train" \
  "packet_loss_recovery=mse_imputer" \
  "packet_cache_ttl=${CACHE_TTL}" \
  > "${RESULTS_DIR}/imputer_training_overrides.txt"

metric_from_log() {
    local metric="$1"
    local log_path="$2"
    local match
    match="$(grep -oE "${metric}:[[:space:]]*[0-9]+([.][0-9]+)?" "${log_path}" | tail -1 || true)"
    if [[ -z "${match}" ]]; then
        printf 'NA'
    else
        sed -E "s/.*:[[:space:]]*//" <<< "${match}"
    fi
}

append_result() {
    local method="$1"
    local ttl="$2"
    local checkpoint="$3"
    local log_path="$4"
    local win return_mean return_std red_alive blue_alive
    win="$(metric_from_log test_red_team_win_mean "${log_path}")"
    return_mean="$(metric_from_log test_return_mean "${log_path}")"
    return_std="$(metric_from_log test_return_std "${log_path}")"
    red_alive="$(metric_from_log test_red_team_alives_mean "${log_path}")"
    blue_alive="$(metric_from_log test_blue_team_alives_mean "${log_path}")"
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "${method}" "${LOSS_PERCENT}" "${ttl}" "${checkpoint}" \
        "${win}" "${return_mean}" "${return_std}" \
        "${red_alive}" "${blue_alive}" "${log_path}" >> "${RESULTS_CSV}"
}

run_eval() {
    local method="$1"
    local recovery="$2"
    local ttl="$3"
    local checkpoint="$4"
    local log_path="${LOG_DIR}/${method}.log"
    local cmd=(
        python src/main.py
        "--config=${ALGORITHM}"
        "--env-config=${ENVIRONMENT}"
        "--exp-config=${EVAL_EXPERIMENT}"
        "--packet-loss-percent=${LOSS_PERCENT}"
        --packet-loss-mode=test
        "--packet-loss-recovery=${recovery}"
        "--run-name-suffix=${MAP_LABEL}_${method}_loss${LOSS_PERCENT}_ttl${ttl}"
        "--test-nepisode=${TEST_EPISODES}"
        --record-video=false
    )
    if [[ "${recovery}" == "mse_imputer" ]]; then
        cmd+=("--packet-cache-ttl=${ttl}")
    fi
    if [[ -n "${checkpoint}" ]]; then
        [[ -f "${checkpoint}" ]] || { echo "Missing imputer checkpoint: ${checkpoint}" >&2; exit 1; }
        cmd+=("--packet-imputer-checkpoint=${checkpoint}")
    fi

    printf '\n=== EVALUATING %s ===\n' "${method}" | tee "${log_path}"
    printf 'Command: ' | tee -a "${log_path}"
    printf '%q ' "${cmd[@]}" | tee -a "${log_path}"
    printf '\n' | tee -a "${log_path}"
    "${cmd[@]}" 2>&1 | tee -a "${log_path}"
    append_result "${method}" "${ttl}" "${checkpoint:-none}" "${log_path}"
}

train_imputer() {
    local kind="$1"
    local algorithm="$2"
    local experiment="$3"
    local root="$4"
    local log_path="${LOG_DIR}/train_${kind}.log"
    local marker="${RESULTS_DIR}/before_${kind}_train.marker"
    touch "${marker}"
    # This function is called in command substitution. Send progress to stderr
    # so stdout contains exactly one value: the final imputer checkpoint path.
    printf '\n=== TRAINING %s IMPUTER ===\n' "${kind}" | tee "${log_path}" >&2
    python src/main.py \
        "--config=${algorithm}" \
        "--env-config=${ENVIRONMENT}" \
        "--exp-config=${experiment}" \
        "--packet-loss-percent=${TRAIN_LOSS_PERCENT}" \
        --packet-loss-mode=train \
        --packet-loss-recovery=mse_imputer \
        "--packet-cache-ttl=${CACHE_TTL}" \
        "--run-name-suffix=${MAP_LABEL}_${kind}_trainloss${TRAIN_LOSS_PERCENT}_ttl${CACHE_TTL}" \
        --record-video=false 2>&1 | tee -a "${log_path}" >&2

    local checkpoint
    checkpoint="$(find "${root}" -type f -name imputer.th -newer "${marker}" \
        -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)"
    if [[ -z "${checkpoint}" || ! -f "${checkpoint}" ]]; then
        echo "ERROR: ${kind} training finished without a newly written imputer.th" >&2
        exit 1
    fi
    printf '%s\n' "${checkpoint}" > "${RESULTS_DIR}/${kind}_imputer_checkpoint.txt"
    echo "${kind} checkpoint: ${checkpoint}" | tee -a "${log_path}" >&2
    printf '%s' "${checkpoint}"
}

echo "Results directory: ${RESULTS_DIR}"
echo "Matched-loss setting: train predictors at ${TRAIN_LOSS_PERCENT}% and evaluate at ${LOSS_PERCENT}%"

# 1--3: execution-only baselines. TTL=0 is the current self-update proxy.
run_eval hold hold 0 ""
run_eval self_update_proxy_ttl0 mse_imputer 0 ""
run_eval ttl_cache_ttl${CACHE_TTL} mse_imputer "${CACHE_TTL}" ""

# 4: Plain MSE. Train from the same frozen source policy, then evaluate it.
MSE_CHECKPOINT="$(train_imputer plain_mse "${MSE_ALGORITHM}" "${MSE_EXPERIMENT}" "${MSE_ROOT}")"
run_eval plain_mse_imputer mse_imputer "${CACHE_TTL}" "${MSE_CHECKPOINT}"

# 5: Value-aware MSE. Same data budget/config except weighting, then evaluate.
VA_CHECKPOINT="$(train_imputer value_aware "${VA_ALGORITHM}" "${VA_EXPERIMENT}" "${VA_ROOT}")"
run_eval va_expoimpute mse_imputer "${CACHE_TTL}" "${VA_CHECKPOINT}"

printf 'finished_at=%s\n' "$(date --iso-8601=seconds)" >> "${RESULTS_DIR}/manifest.txt"
echo "Pipeline complete. Formatted results: ${RESULTS_CSV}"
column -s, -t "${RESULTS_CSV}" || cat "${RESULTS_CSV}"
