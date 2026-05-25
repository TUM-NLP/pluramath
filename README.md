# PluraMath inference experiment code

This repository open-sources the code used to run the inference experiments for
**PluraMath: Extending Mathematical Reasoning Evaluation Beyond High-Resource
Languages**. The paper introduces the benchmark and reports the experimental
findings; this repository focuses only on the operational code used to query
models under different inference modes.

The experiments evaluated whether current inference models can solve
multilingual mathematical reasoning tasks. Each language workbook contains
problem sheets by difficulty. The inference runner reads those workbooks,
constructs prompts with the language-specific final-answer instruction, sends
the prompts through the selected provider, and writes one CSV per model,
language, and prompting strategy.

## Repository contents

- `run_inference.py` - main Python entrypoint for one model/provider run.
- `inference_class/` - provider wrappers for OpenAI-compatible APIs, AWS
  Bedrock via boto3, and local Hugging Face Transformers generation.
- `run_main_experiment_api.sh` - launcher for API-backed inference
  (`deepinfra`, Bedrock OpenAI-compatible endpoint, or Bedrock boto3).
- `run_main_experiment_vllm.sh` - launcher for an OpenAI-compatible remote
  vLLM server. In the reported experiments, vLLM was served through Docker
  using vLLM `v0.20.0`.
- `run_main_experiment_transformers.sh` - detached local Transformers launcher.
- `run_main_experiment_transformers_foreground.sh` - foreground Transformers
  launcher intended for Slurm/sbatch usage.
- `run_sbatch_transformers.sh` - example Slurm submission script for running
  the foreground Transformers launcher on an A40 GPU partition.
- `instructions_prompts.py` - language-specific instruction suffixes.
- `backtranslation_preprocessing/language_to_source_language_map.json` - source
  language map used by the back-translation prompting strategy.
- `final_data/` - PluraMath workbooks used for the base and English-chain-of-
  thought prompting settings.
- `final_data_back_translated/` - workbooks used for the back-translated
  prompting setting.

Generated files are written under `experiment_output/` and `logs/`; those
directories are intentionally ignored by git.

## Setup

Create an environment with the shared dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install provider-specific extras as needed:

```bash
pip install openai python-dotenv pandas openpyxl tqdm
pip install boto3                 # only for --provider bedrock_boto3
pip install "transformers>=4.44" accelerate torch  # local model inference
```

Set the relevant credentials before API runs:

```bash
export DEEPINFRA_API_KEY=...
export AWS_BEARER_TOKEN_BEDROCK=...
export AWS_REGION=eu-west-1
```

For gated Hugging Face models, set `HF_TOKEN=...` in the environment. The
runner also accepts `--hf_token`, but environment variables avoid putting the
token in the process command line.

## Prompting strategies

`run_inference.py` supports three experiment modes:

- `base` - prompt in the dataset language with the language-specific boxed-answer
  instruction.
- `base_encot` - same user prompt, plus an English system prompt asking the model
  to reason step by step in English before giving the boxed answer.
- `bt` - use `final_data_back_translated/`, the
  `questions_back_translated_nllb` column, and the mapped source-language final
  answer instruction.

Use `--prompting_strategy all` or launcher argument `all` to run all three.

## Direct runner usage

Run a single API-backed model over selected datasets:

```bash
python run_inference.py \
  --model_name_full openai/gpt-oss-120b \
  --model_name_short gpt-oss-120b \
  --provider deepinfra \
  --dataset_paths final_data/pluramath_en.xlsx final_data/pluramath_de.xlsx \
  --prompting_strategy base \
  --reasoning_effort medium \
  --temperature 0.1 \
  --batch_size 10
```

Run against a remote vLLM OpenAI-compatible endpoint:

```bash
python run_inference.py \
  --model_name_full Qwen/Qwen2.5-7B-Instruct \
  --model_name_short qwen2.5-7b-instruct \
  --provider vllm_remote \
  --url http://host:port/v1 \
  --dataset_paths final_data/pluramath_en.xlsx \
  --prompting_strategy base \
  --batch_size 15
```

Run local Transformers inference:

```bash
python run_inference.py \
  --model_name_full Qwen/Qwen2.5-7B-Instruct \
  --model_name_short qwen2.5-7b-instruct \
  --provider transformers \
  --dataset_paths final_data/pluramath_en.xlsx \
  --prompting_strategy base \
  --batch_size 1 \
  --transformers_device_map auto \
  --transformers_torch_dtype auto
```

## Main experiment launchers

API-backed providers:

```bash
./run_main_experiment_api.sh openai/gpt-oss-120b gpt-oss-120b all
PROVIDER=bedrock_boto3 BEDROCK_REGION=eu-west-1 \
  ./run_main_experiment_api.sh us.amazon.nova-pro-v1:0 nova-pro-v1 base
```

Remote vLLM:

```bash
URL=http://host:port/v1 \
  ./run_main_experiment_vllm.sh Qwen/Qwen2.5-7B-Instruct qwen2.5-7b-instruct
```

Local Transformers, detached:

```bash
TORCH_DTYPE=bfloat16 BATCH_SIZE=1 \
  ./run_main_experiment_transformers.sh Qwen/Qwen2.5-7B-Instruct qwen2.5-7b
```

Local Transformers, foreground/sbatch-friendly:

```bash
PROMPTING_STRATEGY=base TORCH_DTYPE=bfloat16 BATCH_SIZE=1 \
  ./run_main_experiment_transformers_foreground.sh \
  Qwen/Qwen2.5-7B-Instruct qwen2.5-7b
```

Slurm example:

```bash
sbatch run_sbatch_transformers.sh
```

## Outputs and resumability

Outputs are written as:

```text
experiment_output/<prompting_strategy>/<language>/<model_short_name>.csv
```

Each run also writes checkpoint CSVs while it is active:

```text
experiment_output/<prompting_strategy>/<language>/<model_short_name>.checkpoint.csv
```

If a final CSV or checkpoint already exists, the runner checks whether rows have
either `response` or `internal_reasoning` populated and resumes missing rows.
The runner stops a dataset if at least 5% of rows fail.

Launcher logs are written under `logs/`, with one log per model/language or per
model/prompting strategy depending on the launcher.
