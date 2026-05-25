import argparse
import json
import logging
import os
from pathlib import Path
from typing import Iterable

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from inference_class.bedrock_boto3_inferencer import BedrockBoto3Inferencer
from inference_class.openai_inferencer import OpenaiInferencer
from inference_class.transformers_inferencer import TransformersInferencer
from instructions_prompts import INSTRUCTION

MAX_COMPLETION_TOKENS = 2000
TEMPERATURE = 0.1
TIMEOUT_SEC = 500
BEDROCK_OPENAI_BASE_URL = "https://bedrock-runtime.eu-west-1.amazonaws.com/openai/v1"
BASE_ENCOT_SYSTEM_PROMPT = (
    "You are solving a mathematical problem. Reason step by step in English, "
    "thenwrite the final answer in $\\boxed{}$"
)
BT_LANGUAGE_MAP_PATH = (
    Path(__file__).parent
    / "backtranslation_preprocessing"
    / "language_to_source_language_map.json"
)
BT_QUESTION_COLUMN = "questions_back_translated_nllb"
LANG_FILENAME_ALIASES = {
    "chv": "cv",
}
PROMPTING_STRATEGIES = ("base", "base_encot", "bt")
PROMPTING_STRATEGY_CHOICES = (*PROMPTING_STRATEGIES, "all")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DeepInfra inference over multilingual Excel datasets."
    )
    parser.add_argument(
        "--model_name_full",
        required=True,
        help="Full model name sent to API, e.g. openai/gpt-oss-120b",
    )
    parser.add_argument(
        "--model_name_short",
        required=True,
        help="Short model tag used in output filenames",
    )
    parser.add_argument(
        "--provider",
        required=True,
        choices=[
            "deepinfra",
            "bedrock",
            "bedrock_boto3",
            "vllm_remote",
            "transformers",
        ],
        help=(
            "Inference provider: 'deepinfra', 'bedrock', 'bedrock_boto3', "
            "'vllm_remote', or 'transformers'"
        ),
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Base URL for vllm_remote provider, e.g. http://host:port/v1",
    )

    parser.add_argument(
        "--dataset_paths",
        nargs="+",
        help=(
            "One or more dataset paths. If omitted, final_data/*.xlsx and "
            "final_data/*.xlsw are used for base/base_encot, and "
            "final_data_back_translated/*.xlsx is used for bt."
        ),
    )
    parser.add_argument(
        "--input_dir",
        default=None,
        help=(
            "Input directory used when --dataset_paths is omitted. Defaults to "
            "final_data, or final_data_back_translated for --prompting_strategy bt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default="experiment_output",
        help="Output base directory. The prompting strategy is appended under this directory.",
    )
    parser.add_argument(
        "--prompting_strategy",
        default="base",
        choices=PROMPTING_STRATEGY_CHOICES,
        help=(
            "Prompting strategy. 'base' sends only the dataset-language user prompt. "
            "'base_encot' also sends the English system prompt. "
            "'bt' means backtranslation: read final_data_back_translated by default, "
            "use questions_back_translated_nllb, and use the source-language closing instruction. "
            "'all' runs base, base_encot, and bt independently."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Number of concurrent requests to process per batch",
    )
    parser.add_argument(
        "--reasoning_effort",
        default=None,
        choices=["low", "medium", "high", "not_selected"],
        help="Reasoning effort level passed to OpenAI-compatible providers.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=TEMPERATURE,
        help="Sampling temperature (default: 0.1)",
    )
    parser.add_argument(
        "--bedrock_region",
        default=None,
        help=(
            "AWS region for --provider bedrock_boto3. Defaults to AWS_REGION, "
            "AWS_DEFAULT_REGION, or eu-west-1."
        ),
    )
    parser.add_argument(
        "--transformers_device_map",
        default="auto",
        help="Device map passed to Transformers from_pretrained for --provider transformers.",
    )
    parser.add_argument(
        "--transformers_torch_dtype",
        default="auto",
        help=(
            "Torch dtype for local Transformers inference: auto, float16, bfloat16, "
            "float32, etc."
        ),
    )
    parser.add_argument(
        "--trust_remote_code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to pass trust_remote_code when loading local Transformers models.",
    )
    parser.add_argument(
        "--hf_token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face token for gated/private models when using --provider transformers.",
    )
    return parser.parse_args()


def selected_prompting_strategies(prompting_strategy: str) -> list[str]:
    if prompting_strategy == "all":
        return list(PROMPTING_STRATEGIES)
    return [prompting_strategy]


def find_dataset_by_lang(input_dir: Path, lang: str) -> Path:
    for path in sorted(list(input_dir.glob("*.xlsx")) + list(input_dir.glob("*.xlsw"))):
        if not path.is_file() or path.name.startswith("~$"):
            continue
        try:
            if infer_lang_from_filename(path) == lang:
                return path
        except ValueError:
            continue
    raise FileNotFoundError(f"No dataset found for lang={lang} in {input_dir}")


def resolve_datasets(args: argparse.Namespace, prompting_strategy: str) -> list[Path]:
    if args.dataset_paths and not (
        args.prompting_strategy == "all" and prompting_strategy == "bt"
    ):
        datasets = [Path(p) for p in args.dataset_paths]
    elif args.dataset_paths:
        bt_input_dir = Path(args.input_dir or "final_data_back_translated")
        datasets = [
            find_dataset_by_lang(bt_input_dir, infer_lang_from_filename(Path(p)))
            for p in args.dataset_paths
        ]
    else:
        default_input_dir = (
            "final_data_back_translated"
            if prompting_strategy == "bt"
            else "final_data"
        )
        base = Path(args.input_dir or default_input_dir)
        datasets = sorted(list(base.glob("*.xlsx")) + list(base.glob("*.xlsw")))

    filtered = [
        p
        for p in datasets
        if p.is_file()
        and p.suffix.lower() in {".xlsx", ".xlsw"}
        and not p.name.startswith("~$")
    ]
    if not filtered:
        raise FileNotFoundError(
            "No dataset files found. Pass --dataset_paths or place .xlsx/.xlsw files under input_dir."
        )
    return filtered


def infer_lang_from_filename(path: Path) -> str:
    stem = path.stem.lower()
    for lang in INSTRUCTION:
        if stem.endswith(f"_{lang}") or f"_{lang}_" in stem:
            return lang
    for filename_lang in LANG_FILENAME_ALIASES:
        if stem.endswith(f"_{filename_lang}") or f"_{filename_lang}_" in stem:
            return filename_lang
    raise ValueError(
        f"Could not infer language from filename '{path.name}'. Expected suffix like *_uk or *_cs."
    )


def instruction_lang_for_dataset_lang(lang: str) -> str:
    instruction_lang = LANG_FILENAME_ALIASES.get(lang, lang)
    if instruction_lang not in INSTRUCTION:
        raise KeyError(
            f"Missing instruction prompt for language '{lang}' "
            f"(resolved instruction key: '{instruction_lang}')"
        )
    return instruction_lang


def load_bt_language_map() -> dict[str, str]:
    with BT_LANGUAGE_MAP_PATH.open(encoding="utf-8") as handle:
        language_map = json.load(handle)
    if not isinstance(language_map, dict):
        raise TypeError(f"Expected JSON object in {BT_LANGUAGE_MAP_PATH}")
    return {str(key): str(value) for key, value in language_map.items()}


def instruction_lang_for_prompting_strategy(
    dataset_lang: str, prompting_strategy: str, bt_language_map: dict[str, str]
) -> str:
    if prompting_strategy == "bt":
        source_lang = bt_language_map.get(dataset_lang)
        if source_lang is None:
            raise KeyError(
                f"Missing backtranslation source language for '{dataset_lang}' "
                f"in {BT_LANGUAGE_MAP_PATH}"
            )
        return instruction_lang_for_dataset_lang(source_lang)
    return instruction_lang_for_dataset_lang(dataset_lang)


def resolve_question_column(
    df: pd.DataFrame, prompting_strategy: str = "base"
) -> str:
    if prompting_strategy == "bt":
        candidates = [BT_QUESTION_COLUMN]
    else:
        candidates = ["questions_translated", "question_translated"]
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"Missing question column. Expected one of: {candidates}")


def iter_rows(df: pd.DataFrame) -> Iterable[tuple[int, pd.Series]]:
    for idx, row in df.iterrows():
        yield idx, row


def build_user_prompt(row: pd.Series, question_col: str, lang: str) -> str:
    prompt = f"{row[question_col]}\n\n" f"{INSTRUCTION[lang]}"
    return prompt


def resolve_system_prompt(prompting_strategy: str) -> str | None:
    if prompting_strategy in {"base", "bt"}:
        return None
    if prompting_strategy == "base_encot":
        return BASE_ENCOT_SYSTEM_PROMPT
    raise ValueError(f"Unknown prompting strategy: {prompting_strategy}")


def batched(
    items: list[tuple[int, str, str | None]], batch_size: int
) -> Iterable[list[tuple[int, str, str | None]]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def has_value(value: object) -> bool:
    if pd.isna(value):
        return False
    return bool(str(value).strip())


def has_completed_result(row: pd.Series) -> bool:
    has_response = "response" in row and has_value(row["response"])
    has_internal_reasoning = "internal_reasoning" in row and has_value(
        row["internal_reasoning"]
    )
    return has_response or has_internal_reasoning


def count_missing_results(df: pd.DataFrame) -> int:
    if "response" not in df.columns and "internal_reasoning" not in df.columns:
        return len(df)
    return int((~df.apply(has_completed_result, axis=1)).sum())


def run_inference_for_file(
    inferencer: OpenaiInferencer,
    dataset_path: Path,
    model_name_short: str,
    output_dir: Path,
    batch_size: int,
    prompting_strategy: str,
) -> Path:
    lang = infer_lang_from_filename(dataset_path)
    lang_output_dir = output_dir / lang
    lang_output_dir.mkdir(parents=True, exist_ok=True)

    output_name = f"{model_name_short}.csv"
    output_path = lang_output_dir / output_name
    checkpoint_path = lang_output_dir / f"{model_name_short}.checkpoint.csv"

    tasks_levels = ["top", "high", "medium", "low"]
    df_dict = pd.read_excel(dataset_path, sheet_name=tasks_levels)
    df_list = []
    for level in tasks_levels:
        df_level = df_dict[level]
        df_level["difficulty"] = level
        df_list.append(df_level)
    df = pd.concat(df_list, ignore_index=True)

    # df = df[:20]
    question_col = resolve_question_column(df, prompting_strategy=prompting_strategy)
    bt_language_map = load_bt_language_map() if prompting_strategy == "bt" else {}
    instruction_lang = instruction_lang_for_prompting_strategy(
        dataset_lang=lang,
        prompting_strategy=prompting_strategy,
        bt_language_map=bt_language_map,
    )

    if output_path.exists():
        existing_output_df = pd.read_csv(output_path)
        missing_results = count_missing_results(existing_output_df)
        if missing_results == 0:
            print(f"Skipping complete existing output: {output_path}")
            return output_path
        print(
            f"Existing output is incomplete: {output_path} "
            f"({missing_results} row(s) missing both response and internal_reasoning); resuming"
        )
        out_df = existing_output_df
        out_df.index = df.index
    elif checkpoint_path.exists():
        print(f"Resuming from checkpoint: {checkpoint_path}")
        out_df = pd.read_csv(checkpoint_path)
        out_df.index = df.index
    else:
        out_df = df.copy()
        out_df["user_prompt"] = None
        out_df["system_prompt"] = None
        out_df["prompting_strategy"] = None
        out_df["response"] = None
        out_df["internal_reasoning"] = None

    for column in [
        "user_prompt",
        "system_prompt",
        "prompting_strategy",
        "response",
        "internal_reasoning",
    ]:
        if column not in out_df.columns:
            out_df[column] = None

    system_prompt = resolve_system_prompt(prompting_strategy)
    row_prompts = [
        (
            idx,
            build_user_prompt(row, question_col=question_col, lang=instruction_lang),
            system_prompt,
        )
        for idx, row in iter_rows(df)
    ]
    for idx, prompt, row_system_prompt in row_prompts:
        out_df.at[idx, "user_prompt"] = prompt
        out_df.at[idx, "system_prompt"] = row_system_prompt
        out_df.at[idx, "prompting_strategy"] = prompting_strategy

    # Skip rows that already have a response (from checkpoint)
    pending_prompts = [
        (idx, prompt, row_system_prompt)
        for idx, prompt, row_system_prompt in row_prompts
        if not has_completed_result(out_df.loc[idx])
    ]
    already_done = len(row_prompts) - len(pending_prompts)
    if already_done > 0:
        print(
            f"[{dataset_path.name}] Skipping {already_done} already-completed rows, {len(pending_prompts)} remaining"
        )

    failed_count = 0

    with tqdm(
        total=len(pending_prompts), desc=dataset_path.name, unit="row"
    ) as progress:
        for prompt_batch in batched(pending_prompts, batch_size):
            successes, failures = inferencer.infer_batch(prompt_batch)

            for idx, (response, internal_reasoning) in successes.items():
                out_df.at[idx, "response"] = response
                out_df.at[idx, "internal_reasoning"] = internal_reasoning

            for idx, exc in failures.items():
                failed_count += 1
                current_fail_rate = failed_count / len(row_prompts)
                print(
                    f"[{dataset_path.name}] ERROR row_index={idx}: "
                    f"{type(exc).__name__}: {exc}"
                )
                if current_fail_rate >= 0.05:
                    raise RuntimeError(
                        f"Failure threshold reached for {dataset_path.name}: "
                        f"{failed_count}/{len(row_prompts)} "
                        f"({current_fail_rate:.2%})"
                    ) from exc

            progress.update(len(prompt_batch))

            # Periodic checkpoint after each batch
            out_df.to_csv(checkpoint_path, index=False)

    total_rows = len(row_prompts)
    fail_rate = (failed_count / total_rows) if total_rows else 0.0
    print(
        f"[{dataset_path.name}] failures: {failed_count}/{total_rows} ({fail_rate:.2%})"
    )
    if fail_rate >= 0.05:
        raise RuntimeError(
            f"Too many failed inferences for {dataset_path.name}: "
            f"{failed_count}/{total_rows} ({fail_rate:.2%})"
        )

    out_df.to_csv(output_path, index=False)
    checkpoint_path.unlink(missing_ok=True)
    return output_path


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    load_dotenv()
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")

    if args.reasoning_effort == "not_selected":
        args.reasoning_effort = None

    if args.provider == "vllm_remote" and not args.url:
        raise ValueError("--url is required for the 'vllm_remote' provider")

    if args.provider == "deepinfra":
        api_key = os.environ.get("DEEPINFRA_API_KEY")
        if not api_key:
            raise EnvironmentError("DEEPINFRA_API_KEY is not set")
        inferencer = OpenaiInferencer(
            model_name_full=args.model_name_full,
            api_key=api_key,
            base_url="https://api.deepinfra.com/v1/openai",
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            temperature=args.temperature,
            timeout_sec=TIMEOUT_SEC,
            reasoning_effort=args.reasoning_effort,
        )
    elif args.provider == "bedrock":
        api_key = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
        if not api_key:
            raise EnvironmentError("AWS_BEARER_TOKEN_BEDROCK is not set")
        inferencer = OpenaiInferencer(
            model_name_full=args.model_name_full,
            api_key=api_key,
            base_url=BEDROCK_OPENAI_BASE_URL,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            temperature=args.temperature,
            timeout_sec=TIMEOUT_SEC,
            reasoning_effort=args.reasoning_effort,
        )
    elif args.provider == "bedrock_boto3":
        region_name = (
            args.bedrock_region
            or os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
            or "eu-west-1"
        )
        inferencer = BedrockBoto3Inferencer(
            model_name_full=args.model_name_full,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            temperature=args.temperature,
            region_name=region_name,
        )
    elif args.provider == "vllm_remote":
        inferencer = OpenaiInferencer(
            model_name_full=args.model_name_full,
            api_key="EMPTY",
            base_url=args.url,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            temperature=args.temperature,
            timeout_sec=TIMEOUT_SEC,
            reasoning_effort=args.reasoning_effort,
        )
    elif args.provider == "transformers":
        inferencer = TransformersInferencer(
            model_name_full=args.model_name_full,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            temperature=args.temperature,
            device_map=args.transformers_device_map,
            torch_dtype=args.transformers_torch_dtype,
            trust_remote_code=args.trust_remote_code,
            hf_token=args.hf_token,
        )
    else:
        raise ValueError(f"Unknown provider: {args.provider}")

    for prompting_strategy in selected_prompting_strategies(args.prompting_strategy):
        datasets = resolve_datasets(args, prompting_strategy=prompting_strategy)
        output_dir = Path(args.output_dir) / prompting_strategy

        print(f"[{prompting_strategy}] Found {len(datasets)} dataset(s)")
        for dataset_path in datasets:
            print(f"[{prompting_strategy}] Processing: {dataset_path}")
            output_path = run_inference_for_file(
                inferencer=inferencer,
                dataset_path=dataset_path,
                model_name_short=args.model_name_short,
                # provider=args.provider,
                output_dir=output_dir,
                batch_size=args.batch_size,
                prompting_strategy=prompting_strategy,
            )
            print(f"[{prompting_strategy}] Saved: {output_path}")


if __name__ == "__main__":
    main()
