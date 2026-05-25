#!/usr/bin/env bash
set -uo pipefail

# Usage:
#   URL=host:port ./run_main_experiment_vllm.sh <model_full_name> <model_short_name>
#   URL=http://host:port/v1 ./run_main_experiment_vllm.sh <model_full_name> <model_short_name>
#
# Examples:
#   URL=http://host:port/v1 ./run_main_experiment_vllm.sh \
#     cyankiwi/command-a-reasoning-08-2025-AWQ-4bit command-a-reasoning_awq4b
#   BATCH_SIZE=20 URL=http://host:port/v1 ./run_main_experiment_vllm.sh \
#     cyankiwi/command-a-reasoning-08-2025-AWQ-4bit command-a-reasoning_awq4b
#
# The script starts one detached supervisor process. The supervisor runs
# jobs sequentially:
#   for each prompting strategy in base, base_encot, bt
#   for each language workbook in final_data for base/base_encot
#   for each language workbook in final_data_back_translated for bt
#
# Fixed settings:
#   provider=vllm_remote, temperature=0.1
#   reasoning_effort=medium (override with REASONING_EFFORT=not_selected|low|high)
#   batch_size=15 (override with BATCH_SIZE=N)
#
# Outputs:
#   CSVs: experiment_output/<prompting_strategy>/<lang>/<model_short_name>.csv
#   Logs: logs/main_experiment_vllm/<prompting_strategy>/<lang>/<model_short_name>.log
#   Supervisor log: logs/main_experiment_vllm/supervisors/<model_short_name>.log
#   Supervisor PID: logs/main_experiment_vllm/pids/<model_short_name>_supervisor.pid
#
# Debug mode:
#   Set DEBUG=true to run only matching DEBUG_LANGS from the selected input folders.

ts() { echo "[$(date '+%H:%M:%S')] $*"; }

usage() {
  echo "Usage: URL=host:port DEBUG=false $0 <model_full_name> <model_short_name>"
}

if [[ "$#" -ne 2 ]]; then
  usage
  exit 2
fi

MODEL_NAME_FULL="$1"
MODEL_NAME_SHORT="$2"

PROVIDER="vllm_remote"
URL="${URL:-}"
REASONING_EFFORT="${REASONING_EFFORT:-medium}"
TEMPERATURE="0.1"
BATCH_SIZE="${BATCH_SIZE:-15}"
BASE_INPUT_DIR="final_data"
BT_INPUT_DIR="final_data_back_translated"
OUTPUT_DIR="experiment_output"
LOG_DIR="logs/main_experiment_vllm"
DEBUG="${DEBUG:-false}"
DEBUG_LANGS=("en" "tt")
PROMPTING_STRATEGIES=("base" "base_encot" "bt")

if [[ -z "$URL" ]]; then
  echo "URL is required for vLLM remote inference, e.g. URL=http://host:port/v1 $0 <model_full_name> <model_short_name>"
  exit 2
fi

normalize_vllm_url() {
  local raw_url="$1"
  local normalized="$raw_url"
  if [[ "$normalized" != http://* && "$normalized" != https://* ]]; then
    normalized="http://${normalized}"
  fi
  normalized="${normalized%/}"
  if [[ "$normalized" != */v1 ]]; then
    normalized="${normalized}/v1"
  fi
  echo "$normalized"
}

URL="$(normalize_vllm_url "$URL")"

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
ts "URL: ${URL}"
ts "Reasoning effort: ${REASONING_EFFORT}"
ts "Temperature: ${TEMPERATURE}"
ts "Batch size: ${BATCH_SIZE}"
ts "Model: ${MODEL_NAME_SHORT} (${MODEL_NAME_FULL})"
ts "Debug: ${DEBUG}"
ts "Base datasets: ${#BASE_DATASETS[@]}"
ts "BT datasets: ${#BT_DATASETS[@]}"

nohup bash -c '
  set -uo pipefail

  ts() { echo "[$(date "+%H:%M:%S")] $*"; }

  lang_from_path() {
    local path="$1"
    local stem="${path##*/}"
    stem="${stem%.*}"
    echo "${stem##*_}"
  }

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
  url="$4"
  reasoning_effort="$5"
  temperature="$6"
  batch_size="$7"
  output_dir="$8"
  log_dir="$9"
  base_dataset_list_file="${10}"
  bt_dataset_list_file="${11}"
  shift 11
  prompting_strategies=("$@")

  ts "Supervisor started: model=${model_name_short} | pid=$$"
  ts "URL: ${url}"
  ts "Base dataset list: ${base_dataset_list_file}"
  ts "BT dataset list: ${bt_dataset_list_file}"

  for prompting_strategy in "${prompting_strategies[@]}"; do
    dataset_list_file="$base_dataset_list_file"
    if [[ "$prompting_strategy" == "bt" ]]; then
      dataset_list_file="$bt_dataset_list_file"
    fi

    while IFS= read -r dataset_path; do
      [[ -z "$dataset_path" ]] && continue
      lang="$(lang_from_path "$dataset_path")"
      log_path="${log_dir}/${prompting_strategy}/${lang}/${model_name_short}.log"
      mkdir -p "$(dirname "$log_path")"

      ts "Starting: lang=${lang} | model=${model_name_short} | strategy=${prompting_strategy} | log=${log_path}"
      (
        ts() { echo "[$(date "+%H:%M:%S")] $*"; }
        ts "Started: lang=${lang} | model=${model_name_short} | strategy=${prompting_strategy}"
        python run_inference.py \
          --model_name_full "$model_name_full" \
          --model_name_short "$model_name_short" \
          --provider "$provider" \
          --url "$url" \
          --reasoning_effort "$reasoning_effort" \
          --dataset_paths "$dataset_path" \
          --temperature "$temperature" \
          --output_dir "$output_dir" \
          --prompting_strategy "$prompting_strategy" \
          --batch_size "$batch_size"
        status=$?
        if [[ "$status" -eq 0 ]]; then
          ts "Done: lang=${lang} | model=${model_name_short} | strategy=${prompting_strategy}"
        else
          ts "FAILED: lang=${lang} | model=${model_name_short} | strategy=${prompting_strategy} | status=${status}"
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
    done < "$dataset_list_file"
  done

  ts "All sequential vLLM jobs done. Logs: ${log_dir}"
' _ "$MODEL_NAME_FULL" "$MODEL_NAME_SHORT" "$PROVIDER" "$URL" "$REASONING_EFFORT" \
  "$TEMPERATURE" "$BATCH_SIZE" "$OUTPUT_DIR" "$LOG_DIR" "$BASE_DATASET_LIST_FILE" \
  "$BT_DATASET_LIST_FILE" \
  "${PROMPTING_STRATEGIES[@]}" \
  > "$SUPERVISOR_LOG" 2>&1 &

supervisor_pid=$!
echo "$supervisor_pid" > "$SUPERVISOR_PID_FILE"

ts "Detached sequential supervisor: pid=${supervisor_pid}"
ts "Supervisor log: ${SUPERVISOR_LOG}"
ts "Supervisor PID file: ${SUPERVISOR_PID_FILE}"
