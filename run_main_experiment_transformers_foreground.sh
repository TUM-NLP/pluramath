#!/usr/bin/env bash
set -uo pipefail

# Usage:
#   ./run_main_experiment_transformers_foreground.sh <model_full_name> <model_short_name>
#   PROMPTING_STRATEGY=base ./run_main_experiment_transformers_foreground.sh <model_full_name> <model_short_name>
#
# This is the sbatch-friendly alternative to run_main_experiment_transformers.sh.
# It does not use nohup, does not detach, and exits only after all inference jobs
# have completed or after the first failure.
#
# For local Transformers, each prompting-strategy job receives all dataset paths
# at once so the model is loaded once per strategy instead of once per language.
#
# Fixed/default settings:
#   provider=transformers, temperature=0.1
#   prompting_strategy=all (override with PROMPTING_STRATEGY=base|base_encot|bt|all)
#   batch_size=1 (override with BATCH_SIZE=N)
#   device_map=auto (override with DEVICE_MAP=...)
#   torch_dtype=auto (override with TORCH_DTYPE=float16|bfloat16|float32)
#   trust_remote_code=true (override with TRUST_REMOTE_CODE=false)
#   hf_token unset (override with HF_TOKEN=hf_... for gated/private models)
#
# Outputs:
#   CSVs: experiment_output/<prompting_strategy>/<lang>/<model_short_name>.csv
#   Logs: logs/main_experiment_transformers_foreground/<prompting_strategy>/<model_short_name>.log
#
# Debug mode:
#   Set DEBUG=true to run only matching DEBUG_LANGS from the selected input folders.

ts() { echo "[$(date '+%H:%M:%S')] $*"; }

usage() {
  echo "Usage: DEBUG=false PROMPTING_STRATEGY=all $0 <model_full_name> <model_short_name>"
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
LOG_DIR="${LOG_DIR:-logs/main_experiment_transformers_foreground}"
DEBUG="${DEBUG:-false}"
DEBUG_LANGS=("en" "tt")
PROMPTING_STRATEGY="${PROMPTING_STRATEGY:-all}"

if [[ "$PROMPTING_STRATEGY" == "all" ]]; then
  PROMPTING_STRATEGIES=("base" "base_encot" "bt")
elif [[ "$PROMPTING_STRATEGY" == "base" || "$PROMPTING_STRATEGY" == "base_encot" || "$PROMPTING_STRATEGY" == "bt" ]]; then
  PROMPTING_STRATEGIES=("$PROMPTING_STRATEGY")
else
  echo "Invalid PROMPTING_STRATEGY: ${PROMPTING_STRATEGY}"
  echo "Expected one of: base, base_encot, bt, all"
  exit 2
fi

needs_base_datasets=false
needs_bt_datasets=false
for prompting_strategy in "${PROMPTING_STRATEGIES[@]}"; do
  if [[ "$prompting_strategy" == "bt" ]]; then
    needs_bt_datasets=true
  else
    needs_base_datasets=true
  fi
done

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

if [[ "$needs_base_datasets" == "true" && "${#BASE_DATASETS[@]}" -eq 0 ]]; then
  ts "No datasets found in ${BASE_INPUT_DIR}"
  exit 1
fi

if [[ "$needs_bt_datasets" == "true" && "${#BT_DATASETS[@]}" -eq 0 ]]; then
  ts "No datasets found in ${BT_INPUT_DIR}"
  exit 1
fi

mkdir -p "$LOG_DIR"

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
ts "Prompting strategy: ${PROMPTING_STRATEGY}"
ts "Base datasets: ${#BASE_DATASETS[@]}"
ts "BT datasets: ${#BT_DATASETS[@]}"

for prompting_strategy in "${PROMPTING_STRATEGIES[@]}"; do
  dataset_paths=("${BASE_DATASETS[@]}")
  if [[ "$prompting_strategy" == "bt" ]]; then
    dataset_paths=("${BT_DATASETS[@]}")
  fi

  log_path="${LOG_DIR}/${prompting_strategy}/${MODEL_NAME_SHORT}.log"
  mkdir -p "$(dirname "$log_path")"

  ts "Starting: model=${MODEL_NAME_SHORT} | strategy=${prompting_strategy} | datasets=${#dataset_paths[@]} | log=${log_path}"

  cmd=(
    python run_inference.py
    --model_name_full "$MODEL_NAME_FULL"
    --model_name_short "$MODEL_NAME_SHORT"
    --provider "$PROVIDER"
    --dataset_paths "${dataset_paths[@]}"
    --temperature "$TEMPERATURE"
    --output_dir "$OUTPUT_DIR"
    --prompting_strategy "$prompting_strategy"
    --batch_size "$BATCH_SIZE"
    --transformers_device_map "$DEVICE_MAP"
    --transformers_torch_dtype "$TORCH_DTYPE"
  )
  if [[ "$TRUST_REMOTE_CODE" == "false" ]]; then
    cmd+=(--no-trust_remote_code)
  fi
  "${cmd[@]}" 2>&1 | tee "$log_path"
  status=${PIPESTATUS[0]}
  if [[ "$status" -eq 0 ]]; then
    ts "Done: model=${MODEL_NAME_SHORT} | strategy=${prompting_strategy}"
  else
    ts "FAILED: model=${MODEL_NAME_SHORT} | strategy=${prompting_strategy} | status=${status}"
    ts "Stopping after failure. See log: ${log_path}"
    exit "$status"
  fi
done

ts "All foreground Transformers jobs done. Logs: ${LOG_DIR}"
