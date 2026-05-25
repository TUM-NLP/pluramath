#!/usr/bin/env bash
set -uo pipefail

# Usage:
#   ./run_main_experiment_api.sh <model_full_name> <model_short_name> <prompting_strategy>
#
# Examples:
#   ./run_main_experiment_api.sh openai/gpt-oss-120b gpt-oss-120b base
#   DEBUG=true ./run_main_experiment_api.sh openai/gpt-oss-120b gpt-oss-120b base_encot
#   PROVIDER=bedrock ./run_main_experiment_api.sh <bedrock_model_id> <model_short_name> base
#   PROVIDER=bedrock_boto3 ./run_main_experiment_api.sh <bedrock_model_id> <model_short_name> base
#
# Arguments:
#   model_full_name     Full model name passed to DeepInfra, e.g. openai/gpt-oss-120b
#   model_short_name    Short model tag used in output filenames and logs
#   prompting_strategy  One of: base, base_encot, bt, all
#
# Fixed settings:
#   provider defaults to deepinfra, reasoning_effort=medium, temperature=0.1,
#   batch_size=10. reasoning_effort is not passed for bedrock_boto3.
#
# Optional overrides:
#   BATCH_SIZE=20 ./run_main_experiment_api.sh ...
#   PROVIDER=bedrock_boto3 BEDROCK_REGION=eu-west-1 ./run_main_experiment_api.sh ...
#   MAX_UNFINISHED_FILES=5 ./run_main_experiment_api.sh ...
#
# Outputs:
#   CSVs: experiment_output/<prompting_strategy>/<lang>/<model_short_name>.csv
#   Logs: logs/main_experiment_api/<prompting_strategy>/<lang>/<model_short_name>.log
#   PIDs: logs/main_experiment_api/<prompting_strategy>/pids/<lang>_<model_short_name>.pid
#
# The script launches detached jobs with nohup and exits immediately after all
# language jobs are started. Check the per-language logs for progress.
#
# Debug mode:
#   Set DEBUG=true to run only matching DEBUG_LANGS from the selected input folder.

ts() { echo "[$(date '+%H:%M:%S')] $*"; }

usage() {
  echo "Usage: DEBUG=false $0 <model_full_name> <model_short_name> <prompting_strategy>"
  echo "Example: DEBUG=true $0 openai/gpt-oss-120b gpt-oss-120b base_encot"
}

if [[ "$#" -ne 3 ]]; then
  usage
  exit 2
fi

MODEL_NAME_FULL="$1"
MODEL_NAME_SHORT="$2"
PROMPTING_STRATEGY="$3"

if [[ "$PROMPTING_STRATEGY" != "base" && "$PROMPTING_STRATEGY" != "base_encot" && "$PROMPTING_STRATEGY" != "bt" && "$PROMPTING_STRATEGY" != "all" ]]; then
  echo "Invalid prompting_strategy: ${PROMPTING_STRATEGY}"
  usage
  exit 2
fi

if [[ "$PROMPTING_STRATEGY" == "all" ]]; then
  PROMPTING_STRATEGIES=("base" "base_encot" "bt")
else
  PROMPTING_STRATEGIES=("$PROMPTING_STRATEGY")
fi

PROVIDER="${PROVIDER:-deepinfra}"
REASONING_EFFORT="medium"
TEMPERATURE="0.1"
BATCH_SIZE="${BATCH_SIZE:-10}"
BEDROCK_REGION="${BEDROCK_REGION:-}"
MAX_UNFINISHED_FILES="${MAX_UNFINISHED_FILES:-}"
BASE_INPUT_DIR="${BASE_INPUT_DIR:-final_data}"
BT_INPUT_DIR="${BT_INPUT_DIR:-final_data_back_translated}"
OUTPUT_DIR="experiment_output"
LOG_DIR="logs/main_experiment_api"
DEBUG="${DEBUG:-false}"
DEBUG_LANGS=("en" "tt")

if [[ "$PROVIDER" != "deepinfra" && "$PROVIDER" != "bedrock" && "$PROVIDER" != "bedrock_boto3" ]]; then
  echo "Invalid PROVIDER: ${PROVIDER}. Expected one of: deepinfra, bedrock, bedrock_boto3"
  exit 2
fi

if [[ -n "$MAX_UNFINISHED_FILES" && ! "$MAX_UNFINISHED_FILES" =~ ^[0-9]+$ ]]; then
  echo "Invalid MAX_UNFINISHED_FILES: ${MAX_UNFINISHED_FILES}. Expected a non-negative integer."
  exit 2
fi

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

is_finished_output() {
  local output_path="$1"
  if [[ ! -f "$output_path" ]]; then
    return 1
  fi

  python - "$output_path" <<'PY'
import csv
import sys

path = sys.argv[1]

try:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            sys.exit(1)
        if "response" not in reader.fieldnames and "internal_reasoning" not in reader.fieldnames:
            sys.exit(1)
        for row in reader:
            response = row.get("response")
            reasoning = row.get("internal_reasoning")
            if not ((response and response.strip()) or (reasoning and reasoning.strip())):
                sys.exit(1)
except Exception:
    sys.exit(1)

sys.exit(0)
PY
}

filter_unfinished_datasets() {
  local prompting_strategy="$1"
  local dataset_path
  local lang
  local output_path
  local selected=0

  shift
  if [[ "$MAX_UNFINISHED_FILES" == "0" ]]; then
    return 0
  fi

  for dataset_path in "$@"; do
    lang="$(lang_from_path "$dataset_path")"
    output_path="${OUTPUT_DIR}/${prompting_strategy}/${lang}/${MODEL_NAME_SHORT}.csv"
    if is_finished_output "$output_path"; then
      continue
    fi

    printf "%s\n" "$dataset_path"
    selected=$((selected + 1))
    if [[ -n "$MAX_UNFINISHED_FILES" && "$selected" -ge "$MAX_UNFINISHED_FILES" ]]; then
      break
    fi
  done
}

BASE_DATASETS=()
BT_DATASETS=()
while IFS= read -r dataset_path; do
  BASE_DATASETS+=("$dataset_path")
done < <(find "$BASE_INPUT_DIR" -maxdepth 1 -type f -name '*.xlsx' | sort)

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

NEED_BASE=false
NEED_BT=false
for prompting_strategy in "${PROMPTING_STRATEGIES[@]}"; do
  if [[ "$prompting_strategy" == "bt" ]]; then
    NEED_BT=true
  else
    NEED_BASE=true
  fi
done

if [[ "$NEED_BASE" == "true" && "${#BASE_DATASETS[@]}" -eq 0 ]]; then
  ts "No datasets found in ${BASE_INPUT_DIR}"
  exit 1
fi

if [[ "$NEED_BT" == "true" && "${#BT_DATASETS[@]}" -eq 0 ]]; then
  ts "No datasets found in ${BT_INPUT_DIR}"
  exit 1
fi

mkdir -p "$LOG_DIR"

ts "Provider: ${PROVIDER}"
if [[ "$PROVIDER" == "bedrock_boto3" ]]; then
  ts "Reasoning effort: not passed for bedrock_boto3"
  if [[ -n "$BEDROCK_REGION" ]]; then
    ts "Bedrock region: ${BEDROCK_REGION}"
  fi
else
  ts "Reasoning effort: ${REASONING_EFFORT}"
fi
ts "Temperature: ${TEMPERATURE}"
ts "Batch size: ${BATCH_SIZE}"
ts "Model: ${MODEL_NAME_SHORT} (${MODEL_NAME_FULL})"
ts "Prompting strategy: ${PROMPTING_STRATEGY}"
if [[ -n "$MAX_UNFINISHED_FILES" ]]; then
  ts "Max unfinished files per strategy: ${MAX_UNFINISHED_FILES}"
else
  ts "Max unfinished files per strategy: unlimited"
fi
ts "Debug: ${DEBUG}"
ts "Base datasets: ${#BASE_DATASETS[@]}"
ts "BT datasets: ${#BT_DATASETS[@]}"

total_jobs=0
manifest_paths=()

for prompting_strategy in "${PROMPTING_STRATEGIES[@]}"; do
  if [[ "$prompting_strategy" == "bt" ]]; then
    DATASETS=("${BT_DATASETS[@]}")
  else
    DATASETS=("${BASE_DATASETS[@]}")
  fi

  if [[ -n "$MAX_UNFINISHED_FILES" ]]; then
    FILTERED_DATASETS=()
    while IFS= read -r dataset_path; do
      FILTERED_DATASETS+=("$dataset_path")
    done < <(filter_unfinished_datasets "$prompting_strategy" "${DATASETS[@]}")
    ts "Strategy ${prompting_strategy}: selected ${#FILTERED_DATASETS[@]} unfinished dataset(s) out of ${#DATASETS[@]}"
    DATASETS=("${FILTERED_DATASETS[@]}")
  fi

  PID_DIR="${LOG_DIR}/${prompting_strategy}/pids"
  LAUNCH_MANIFEST="${LOG_DIR}/${prompting_strategy}/${MODEL_NAME_SHORT}_launch_manifest.tsv"
  mkdir -p "$PID_DIR"
  printf "pid\tlang\tmodel_short\tprompting_strategy\tlog_path\tdataset_path\n" > "$LAUNCH_MANIFEST"
  manifest_paths+=("$LAUNCH_MANIFEST")

  for dataset_path in "${DATASETS[@]}"; do
    lang="$(lang_from_path "$dataset_path")"
    log_path="${LOG_DIR}/${prompting_strategy}/${lang}/${MODEL_NAME_SHORT}.log"
    mkdir -p "$(dirname "$log_path")"

    ts "Launching: lang=${lang} | model=${MODEL_NAME_SHORT} | strategy=${prompting_strategy}"
    nohup bash -c '
      ts() { echo "[$(date "+%H:%M:%S")] $*"; }
      model_name_full="$1"
      model_name_short="$2"
      prompting_strategy="$3"
      dataset_path="$4"
      provider="$5"
      reasoning_effort="$6"
      temperature="$7"
      batch_size="$8"
      output_dir="$9"
      bedrock_region="${10}"
      lang="${11}"

      ts "Started: lang=${lang} | model=${model_name_short} | strategy=${prompting_strategy} | pid=$$"
      command=(
        python run_inference.py
        --model_name_full "$model_name_full"
        --model_name_short "$model_name_short"
        --provider "$provider"
        --dataset_paths "$dataset_path"
        --temperature "$temperature"
        --output_dir "$output_dir"
        --prompting_strategy "$prompting_strategy"
        --batch_size "$batch_size"
      )
      if [[ "$provider" != "bedrock_boto3" ]]; then
        command+=(--reasoning_effort "$reasoning_effort")
      elif [[ -n "$bedrock_region" ]]; then
        command+=(--bedrock_region "$bedrock_region")
      fi
      "${command[@]}"
      status=$?
      if [[ "$status" -eq 0 ]]; then
        ts "Done: lang=${lang} | model=${model_name_short} | strategy=${prompting_strategy}"
      else
        ts "FAILED: lang=${lang} | model=${model_name_short} | strategy=${prompting_strategy} | status=${status}"
      fi
      exit "$status"
    ' _ "$MODEL_NAME_FULL" "$MODEL_NAME_SHORT" "$prompting_strategy" "$dataset_path" \
      "$PROVIDER" "$REASONING_EFFORT" "$TEMPERATURE" "$BATCH_SIZE" "$OUTPUT_DIR" "$BEDROCK_REGION" "$lang" \
      > "$log_path" 2>&1 &
    pid=$!
    echo "$pid" > "${PID_DIR}/${lang}_${MODEL_NAME_SHORT}.pid"
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$pid" "$lang" "$MODEL_NAME_SHORT" "$prompting_strategy" "$log_path" "$dataset_path" \
      >> "$LAUNCH_MANIFEST"
    total_jobs=$((total_jobs + 1))
  done
done

ts "Detached ${total_jobs} job(s). Logs: ${LOG_DIR}"
for manifest_path in "${manifest_paths[@]}"; do
  ts "Launch manifest: ${manifest_path}"
done
