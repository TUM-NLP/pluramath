#!/usr/bin/env bash
set -uo pipefail

# Usage:
#   ./run_main_experiment_transformers.sh <model_full_name> <model_short_name>
#
# Examples:
#   ./run_main_experiment_transformers.sh Qwen/Qwen2.5-7B-Instruct qwen2.5-7b-instruct
#   TORCH_DTYPE=float16 BATCH_SIZE=2 ./run_main_experiment_transformers.sh \
#     Qwen/Qwen2.5-7B-Instruct qwen2.5-7b-instruct
#   TRUST_REMOTE_CODE=false ./run_main_experiment_transformers.sh \
#     deepseek-ai/DeepSeek-R1-Distill-Qwen-7B deepseek-r1-distill-qwen-7b
#
# The script starts one detached supervisor process. The supervisor runs jobs
# sequentially by prompting strategy:
#   base over all language workbooks in final_data
#   base_encot over all language workbooks in final_data
#   bt over all language workbooks in final_data_back_translated
#
# For local Transformers, each prompting-strategy job receives all dataset paths
# at once so the model is loaded once per strategy instead of once per language.
#
# Fixed/default settings:
#   provider=transformers, temperature=0.1
#   batch_size=1 (override with BATCH_SIZE=N)
#   device_map=auto (override with DEVICE_MAP=...)
#   torch_dtype=auto (override with TORCH_DTYPE=float16|bfloat16|float32)
#   trust_remote_code=true (override with TRUST_REMOTE_CODE=false)
#   hf_token unset (override with HF_TOKEN=hf_... for gated/private models)
#
# Outputs:
#   CSVs: experiment_output/<prompting_strategy>/<lang>/<model_short_name>.csv
#   Logs: logs/main_experiment_transformers/<prompting_strategy>/<model_short_name>.log
#   Supervisor log: logs/main_experiment_transformers/supervisors/<model_short_name>_<run_id>.log
#   Supervisor PID: logs/main_experiment_transformers/pids/<model_short_name>_<run_id>_supervisor.pid
#
# Debug mode:
#   Set DEBUG=true to run only matching DEBUG_LANGS from the selected input folders.

ts() { echo "[$(date '+%H:%M:%S')] $*"; }

usage() {
  echo "Usage: DEBUG=false $0 <model_full_name> <model_short_name>"
}

if [[ "$#" -ne 2 ]]; then
  usage
  exit 2
fi

MODEL_NAME_FULL="$1"
MODEL_NAME_SHORT="$2"

PROVIDER="transformers"
TEMPERATURE="${TEMPERATURE:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
TORCH_DTYPE="${TORCH_DTYPE:-auto}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"
HF_TOKEN="${HF_TOKEN:-}"
BASE_INPUT_DIR="${BASE_INPUT_DIR:-final_data}"
BT_INPUT_DIR="${BT_INPUT_DIR:-final_data_back_translated}"
OUTPUT_DIR="${OUTPUT_DIR:-experiment_output}"
LOG_DIR="${LOG_DIR:-logs/main_experiment_transformers}"
DEBUG="${DEBUG:-false}"
DEBUG_LANGS=("en" "tt")
PROMPTING_STRATEGIES=("base" "base_encot" "bt")

lang_from_path() {
  local path="$1"
  local stem="${path##*/}"
  stem="${stem%.*}"
  echo "${stem##*_}"
}

is_debug_lang() {
  local lang="$1"
  local debug_lang
  for debug_lang in "${DEBUG_LANGS[@]}"; do
    if [[ "$lang" == "$debug_lang" ]]; then
      return 0
    fi
  done
  return 1
}

BASE_DATASETS=()
while IFS= read -r dataset_path; do
  BASE_DATASETS+=("$dataset_path")
done < <(find "$BASE_INPUT_DIR" -maxdepth 1 -type f -name '*.xlsx' | sort)

BT_DATASETS=()
while IFS= read -r dataset_path; do
  BT_DATASETS+=("$dataset_path")
done < <(find "$BT_INPUT_DIR" -maxdepth 1 -type f -name '*.xlsx' | sort)

if [[ "${DEBUG}" == "true" ]]; then
  FILTERED_BASE_DATASETS=()
  for dataset_path in "${BASE_DATASETS[@]}"; do
    lang="$(lang_from_path "$dataset_path")"
    if is_debug_lang "$lang"; then
      FILTERED_BASE_DATASETS+=("$dataset_path")
    fi
  done
  BASE_DATASETS=("${FILTERED_BASE_DATASETS[@]}")

  FILTERED_BT_DATASETS=()
  for dataset_path in "${BT_DATASETS[@]}"; do
    lang="$(lang_from_path "$dataset_path")"
    if is_debug_lang "$lang"; then
      FILTERED_BT_DATASETS+=("$dataset_path")
    fi
  done
  BT_DATASETS=("${FILTERED_BT_DATASETS[@]}")
fi

if [[ "${#BASE_DATASETS[@]}" -eq 0 ]]; then
  ts "No datasets found in ${BASE_INPUT_DIR}"
  exit 1
fi

if [[ "${#BT_DATASETS[@]}" -eq 0 ]]; then
  ts "No datasets found in ${BT_INPUT_DIR}"
  exit 1
fi

mkdir -p "$LOG_DIR"
SUPERVISOR_LOG_DIR="${LOG_DIR}/supervisors"
PID_DIR="${LOG_DIR}/pids"
RUN_ID="$(date '+%Y%m%d_%H%M%S')"
SUPERVISOR_LOG="${SUPERVISOR_LOG_DIR}/${MODEL_NAME_SHORT}_${RUN_ID}.log"
SUPERVISOR_PID_FILE="${PID_DIR}/${MODEL_NAME_SHORT}_${RUN_ID}_supervisor.pid"
BASE_DATASET_LIST_FILE="${LOG_DIR}/${MODEL_NAME_SHORT}_${RUN_ID}_base_datasets.txt"
BT_DATASET_LIST_FILE="${LOG_DIR}/${MODEL_NAME_SHORT}_${RUN_ID}_bt_datasets.txt"
mkdir -p "$SUPERVISOR_LOG_DIR" "$PID_DIR"
printf "%s\n" "${BASE_DATASETS[@]}" > "$BASE_DATASET_LIST_FILE"
printf "%s\n" "${BT_DATASETS[@]}" > "$BT_DATASET_LIST_FILE"

ts "Provider: ${PROVIDER}"
ts "Temperature: ${TEMPERATURE}"
ts "Batch size: ${BATCH_SIZE}"
ts "Device map: ${DEVICE_MAP}"
ts "Torch dtype: ${TORCH_DTYPE}"
ts "Trust remote code: ${TRUST_REMOTE_CODE}"
if [[ -n "$HF_TOKEN" ]]; then
  ts "HF token: set"
else
  ts "HF token: unset"
fi
ts "Model: ${MODEL_NAME_SHORT} (${MODEL_NAME_FULL})"
ts "Debug: ${DEBUG}"
ts "Base datasets: ${#BASE_DATASETS[@]}"
ts "BT datasets: ${#BT_DATASETS[@]}"

nohup bash -c '
  set -uo pipefail

  ts() { echo "[$(date "+%H:%M:%S")] $*"; }

  stop_child() {
    if [[ -n "${current_child_pid:-}" ]]; then
      ts "Stopping active child process ${current_child_pid}"
      kill "${current_child_pid}" 2>/dev/null || true
      wait "${current_child_pid}" 2>/dev/null || true
    fi
    exit 143
  }

  trap stop_child TERM INT

  model_name_full="$1"
  model_name_short="$2"
  provider="$3"
  temperature="$4"
  batch_size="$5"
  output_dir="$6"
  log_dir="$7"
  device_map="$8"
  torch_dtype="$9"
  trust_remote_code="${10}"
  base_dataset_list_file="${11}"
  bt_dataset_list_file="${12}"
  shift 12
  prompting_strategies=("$@")

  ts "Supervisor started: model=${model_name_short} | pid=$$"
  ts "Base dataset list: ${base_dataset_list_file}"
  ts "BT dataset list: ${bt_dataset_list_file}"

  for prompting_strategy in "${prompting_strategies[@]}"; do
    dataset_list_file="$base_dataset_list_file"
    if [[ "$prompting_strategy" == "bt" ]]; then
      dataset_list_file="$bt_dataset_list_file"
    fi

    dataset_paths=()
    while IFS= read -r dataset_path; do
      [[ -z "$dataset_path" ]] && continue
      dataset_paths+=("$dataset_path")
    done < "$dataset_list_file"
    log_path="${log_dir}/${prompting_strategy}/${model_name_short}.log"
    mkdir -p "$(dirname "$log_path")"

    ts "Starting: model=${model_name_short} | strategy=${prompting_strategy} | datasets=${#dataset_paths[@]} | log=${log_path}"
    (
      ts() { echo "[$(date "+%H:%M:%S")] $*"; }
      ts "Started: model=${model_name_short} | strategy=${prompting_strategy} | datasets=${#dataset_paths[@]}"

      cmd=(
        python run_inference.py
        --model_name_full "$model_name_full"
        --model_name_short "$model_name_short"
        --provider "$provider"
        --dataset_paths "${dataset_paths[@]}"
        --temperature "$temperature"
        --output_dir "$output_dir"
        --prompting_strategy "$prompting_strategy"
        --batch_size "$batch_size"
        --transformers_device_map "$device_map"
        --transformers_torch_dtype "$torch_dtype"
      )
      if [[ "$trust_remote_code" == "false" ]]; then
        cmd+=(--no-trust_remote_code)
      fi
      "${cmd[@]}"
      status=$?
      if [[ "$status" -eq 0 ]]; then
        ts "Done: model=${model_name_short} | strategy=${prompting_strategy}"
      else
        ts "FAILED: model=${model_name_short} | strategy=${prompting_strategy} | status=${status}"
      fi
      exit "$status"
    ) > "$log_path" 2>&1 &
    current_child_pid=$!
    wait "$current_child_pid"
    status=$?
    current_child_pid=""
    if [[ "$status" -ne 0 ]]; then
      ts "Stopping after failure. See log: ${log_path}"
      exit "$status"
    fi
  done

  ts "All sequential Transformers jobs done. Logs: ${log_dir}"
' _ "$MODEL_NAME_FULL" "$MODEL_NAME_SHORT" "$PROVIDER" "$TEMPERATURE" "$BATCH_SIZE" \
  "$OUTPUT_DIR" "$LOG_DIR" "$DEVICE_MAP" "$TORCH_DTYPE" "$TRUST_REMOTE_CODE" \
  "$BASE_DATASET_LIST_FILE" "$BT_DATASET_LIST_FILE" \
  "${PROMPTING_STRATEGIES[@]}" \
  > "$SUPERVISOR_LOG" 2>&1 &

supervisor_pid=$!
echo "$supervisor_pid" > "$SUPERVISOR_PID_FILE"

ts "Detached sequential supervisor: pid=${supervisor_pid}"
ts "Supervisor log: ${SUPERVISOR_LOG}"
ts "Supervisor PID file: ${SUPERVISOR_PID_FILE}"
