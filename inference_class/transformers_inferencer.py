import logging
from typing import Optional

logger = logging.getLogger(__name__)


class TransformersInferencer:
    """Runs local Hugging Face Transformers text generation."""

    def __init__(
        self,
        model_name_full: str,
        max_completion_tokens: int,
        temperature: float,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        trust_remote_code: bool = True,
        hf_token: Optional[str] = None,
    ) -> None:
        logger.info(
            "Initializing TransformersInferencer: model=%s device_map=%s "
            "torch_dtype=%s trust_remote_code=%s hf_token=%s",
            model_name_full,
            device_map,
            torch_dtype,
            trust_remote_code,
            "set" if hf_token else "unset",
        )
        try:
            logger.info("Importing torch and transformers")
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoModelForImageTextToText,
                AutoProcessor,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise ImportError(
                "Install local inference dependencies first: "
                "pip install 'transformers>=4.44' accelerate torch"
            ) from exc

        self.model_name_full = model_name_full
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature

        dtype = torch_dtype
        if torch_dtype != "auto":
            logger.info("Resolving torch dtype: torch.%s", torch_dtype)
            dtype = getattr(torch, torch_dtype)

        logger.info("Loading model weights for %s", model_name_full)

        model_name_lower = model_name_full.lower()
        use_image_text_loader = (
            "ministral" in model_name_lower or "mistral3" in model_name_lower
        )
        if use_image_text_loader:
            logger.info(
                "Using AutoProcessor and AutoModelForImageTextToText for Mistral3/Ministral model"
            )
            self.processor = AutoProcessor.from_pretrained(
                model_name_full,
                trust_remote_code=trust_remote_code,
                token=hf_token,
            )
            self.tokenizer = self.processor.tokenizer
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name_full,
                device_map=device_map,
                torch_dtype=dtype,
                trust_remote_code=trust_remote_code,
                token=hf_token,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_full,
                device_map=device_map,
                torch_dtype=dtype,
                trust_remote_code=trust_remote_code,
                token=hf_token,
            )

            logger.info("Loading tokenizer for %s", model_name_full)
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_full,
                trust_remote_code=trust_remote_code,
                token=hf_token,
            )

        if self.tokenizer.pad_token_id is None:
            logger.info("Tokenizer has no pad_token_id; using eos_token as pad_token")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        logger.info(
            "Tokenizer loaded: vocab_size=%s pad_token_id=%s eos_token_id=%s",
            getattr(self.tokenizer, "vocab_size", "unknown"),
            self.tokenizer.pad_token_id,
            self.tokenizer.eos_token_id,
        )

        self.model.eval()
        self.torch = torch
        logger.info(
            "Model loaded: class=%s device=%s dtype=%s",
            self.model.__class__.__name__,
            getattr(self.model, "device", "unknown"),
            getattr(self.model, "dtype", "unknown"),
        )

    def _format_prompt(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as exc:
            logger.warning(
                "Tokenizer chat template failed for %s; falling back to plain prompt: %s",
                self.model_name_full,
                exc,
            )
            if system_prompt is None:
                return prompt
            return f"{system_prompt}\n\n{prompt}"

    def infer_single(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> tuple[Optional[str], Optional[str]]:
        successes, failures = self.infer_batch([(0, prompt, system_prompt)])
        if failures:
            raise failures[0]
        return successes[0]

    def infer_batch(
        self,
        indexed_prompts: list[tuple[int, str, Optional[str]]],
    ) -> tuple[
        dict[int, tuple[Optional[str], Optional[str]]],
        dict[int, Exception],
    ]:
        if not indexed_prompts:
            return {}, {}

        logger.info(
            "Running Transformers generation batch: batch_size=%d max_new_tokens=%d "
            "temperature=%s",
            len(indexed_prompts),
            self.max_completion_tokens,
            self.temperature,
        )
        formatted_prompts = [
            self._format_prompt(prompt, system_prompt)
            for _, prompt, system_prompt in indexed_prompts
        ]

        try:
            inputs = self.tokenizer(
                formatted_prompts,
                return_tensors="pt",
                padding=True,
            )
            inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
            logger.info(
                "Tokenized batch: input_shape=%s attention_mask_shape=%s device=%s",
                tuple(inputs["input_ids"].shape),
                (
                    tuple(inputs["attention_mask"].shape)
                    if "attention_mask" in inputs
                    else None
                ),
                getattr(self.model, "device", "unknown"),
            )

            generation_kwargs = {
                "max_new_tokens": self.max_completion_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
            }
            if self.temperature > 0:
                generation_kwargs["do_sample"] = True
                generation_kwargs["temperature"] = self.temperature
            else:
                generation_kwargs["do_sample"] = False

            with self.torch.inference_mode():
                outputs = self.model.generate(**inputs, **generation_kwargs)
            logger.info("Generation finished: output_shape=%s", tuple(outputs.shape))

            prompt_width = inputs["input_ids"].shape[1]
            successes = {}
            for output_idx, (row_idx, _, _) in enumerate(indexed_prompts):
                generated_ids = outputs[output_idx][prompt_width:]
                response = self.tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                ).strip()
                successes[row_idx] = (response, None)
            return successes, {}
        except Exception as exc:
            logger.exception(
                "Transformers generation failed for batch_size=%d",
                len(indexed_prompts),
            )
            return {}, {idx: exc for idx, _, _ in indexed_prompts}
