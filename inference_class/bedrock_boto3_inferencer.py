from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from typing import Optional


class BedrockBoto3Inferencer:
    """Wraps AWS Bedrock Runtime Converse via boto3."""

    def __init__(
        self,
        model_name_full: str,
        max_completion_tokens: int,
        temperature: Optional[float],
        region_name: str,
    ) -> None:
        try:
            import boto3
        except ImportError as exc:
            raise ImportError(
                "Install boto3 to use --provider bedrock_boto3."
            ) from exc

        self.model_name_full = model_name_full
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self.client = boto3.client("bedrock-runtime", region_name=region_name)

    def infer_single(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> tuple[Optional[str], Optional[str]]:
        messages = [{"role": "user", "content": [{"text": prompt}]}]
        converse_kwargs = {
            "modelId": self.model_name_full,
            "messages": messages,
        }
        inference_config = {}
        if self.temperature is not None:
            inference_config["temperature"] = self.temperature
        if self.max_completion_tokens is not None:
            inference_config["maxTokens"] = self.max_completion_tokens
        if inference_config:
            converse_kwargs["inferenceConfig"] = inference_config
        if system_prompt is not None:
            converse_kwargs["system"] = [{"text": system_prompt}]

        response = self.client.converse(**converse_kwargs)
        content_blocks = response["output"]["message"]["content"]
        response_text, internal_reasoning, tool_uses = self._extract_response_results(
            content_blocks
        )

        if not response_text and tool_uses:
            tool_input = tool_uses[0].get("input")
            if isinstance(tool_input, dict):
                response_text = json.dumps(tool_input, indent=2, ensure_ascii=False)
            elif isinstance(tool_input, str):
                response_text = tool_input

        return response_text or "", internal_reasoning or None

    def infer_batch(
        self,
        indexed_prompts: list[tuple[int, str, Optional[str]]],
    ) -> tuple[
        dict[int, tuple[Optional[str], Optional[str]]],
        dict[int, Exception],
    ]:
        successes: dict[int, tuple[Optional[str], Optional[str]]] = {}
        failures: dict[int, Exception] = {}
        with ThreadPoolExecutor(max_workers=len(indexed_prompts)) as executor:
            future_to_idx = {
                executor.submit(self.infer_single, prompt, system_prompt): idx
                for idx, prompt, system_prompt in indexed_prompts
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    successes[idx] = future.result()
                except Exception as exc:
                    failures[idx] = exc
        return successes, failures

    @staticmethod
    def _extract_response_results(
        content_blocks: list[dict],
    ) -> tuple[str, str, list[dict]]:
        text_parts = []
        reasoning_parts = []
        tool_uses = []

        for block in content_blocks:
            if not isinstance(block, dict):
                continue

            text_value = block.get("text")
            if isinstance(text_value, str):
                text_parts.append(text_value)
                continue

            reasoning = block.get("reasoningContent")
            if isinstance(reasoning, dict):
                reasoning_text = reasoning.get("reasoningText", {}).get("text")
                if isinstance(reasoning_text, str):
                    reasoning_parts.append(reasoning_text)
                continue

            tool_use = block.get("toolUse")
            if isinstance(tool_use, dict):
                tool_uses.append(tool_use)

        return (
            "\n".join(text_parts).strip(),
            "\n".join(reasoning_parts).strip(),
            tool_uses,
        )
