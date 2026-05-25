from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from typing import Optional

from openai import APITimeoutError, OpenAI


def _extract_text_parts(message: object, key: str) -> Optional[str]:
    try:
        payload = message.model_dump() if hasattr(message, "model_dump") else message
    except Exception:
        payload = message
    if not isinstance(payload, dict):
        return None

    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts) if parts else None
    return str(value)


class OpenaiInferencer:
    """Wraps an OpenAI-compatible client and provides single and batched inference."""

    def __init__(
        self,
        model_name_full: str,
        api_key: str,
        base_url: str,
        max_completion_tokens: int,
        temperature: float,
        timeout_sec: float,
        reasoning_effort: Optional[str] = None,
    ) -> None:
        self.model_name_full = model_name_full
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_sec)

    def infer_single(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> tuple[Optional[str], Optional[str]]:
        """Single API call. APITimeoutError is captured and returned as a sentinel."""
        extra = {}
        if self.reasoning_effort is not None:
            extra["reasoning_effort"] = self.reasoning_effort
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name_full,
                messages=messages,
                temperature=self.temperature,
                max_completion_tokens=self.max_completion_tokens,
                **extra,
            )
        except APITimeoutError as exc:
            logging.warning(f"API call timed out for prompt: {prompt[:50]}... Error: {exc}")
            return None, f"<APITimeoutError: {exc}>"
        message = completion.choices[0].message
        response = message.content or ""
        internal_reasoning = (
            _extract_text_parts(message, "reasoning_content")
            or _extract_text_parts(message, "reasoning")
            or _extract_text_parts(message, "thinking")
        )
        return response, internal_reasoning

    def infer_batch(
        self,
        indexed_prompts: list[tuple[int, str, Optional[str]]],
    ) -> tuple[
        dict[int, tuple[Optional[str], Optional[str]]],
        dict[int, Exception],
    ]:
        """
        Concurrently infer over a list of (idx, prompt) pairs.
        Returns (successes, failures):
          successes: {idx: (response, internal_reasoning)}
          failures:  {idx: exception}
        """
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
