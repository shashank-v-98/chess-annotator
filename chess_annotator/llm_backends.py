"""Pluggable generation backends.

Compute access wasn't confirmed up front, so this ships three interchangeable
backends behind one interface -- pick whichever matches what's available on
the day:

  - GroqBackend:  free, no-GPU, fast (OpenAI-compatible /chat/completions).
                  Good default if there's no local GPU. Needs GROQ_API_KEY.
  - LocalHFBackend: runs a small open-weight instruct model locally via
                  transformers (e.g. Qwen2.5-1.5B-Instruct, SmolLM3-3B,
                  Llama-3.2-3B-Instruct). Works on CPU (slow) or GPU (fast).
  - MockBackend:  deterministic template-based output, no network/model
                  needed. Used for dry runs and for testing the pipeline
                  wiring itself.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod


class LLMBackend(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 200) -> str:
        ...


class RateLimitError(Exception):
    """Raised on HTTP 429. Carries the server's own `retry-after` (seconds)
    when present, so callers can wait the actual required time instead of
    guessing with exponential backoff. Groq's per-model daily caps (RPD/TPD)
    mean retry-after can sometimes be very long (until the daily reset) --
    callers should treat a large value as "switch model / try again later",
    not just "sleep and retry"."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class GroqBackend(LLMBackend):
    """Uses Groq's OpenAI-compatible chat completions endpoint.

    Free tier as of mid-2026: no credit card, rate-limited, hosts several
    strong small/medium open models (e.g. llama-3.1-8b-instant, qwen3-32b).
    Get a key at https://console.groq.com and set GROQ_API_KEY.
    """

    def __init__(self, model: str = "llama-3.1-8b-instant", api_key: str | None = None, temperature: float = 0.4):
        import requests  # local import so the module loads even without `requests`

        self._requests = requests
        self.model = model
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise RuntimeError("Set GROQ_API_KEY or pass api_key= explicitly.")
        self.temperature = temperature
        self.url = "https://api.groq.com/openai/v1/chat/completions"

    def _payload(self, system_prompt: str, user_prompt: str, max_tokens: int) -> dict:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        }
        # Reasoning models (e.g. openai/gpt-oss-20b, the replacement for the
        # decommissioned llama-3.1-8b-instant) spend hidden reasoning tokens out
        # of max_tokens before writing the answer. With a small budget they can
        # return an empty string. Keep reasoning short so the answer fits.
        if "gpt-oss" in self.model:
            payload["reasoning_effort"] = getattr(self, "reasoning_effort", "low")
        return payload

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 200) -> str:
        resp = self._requests.post(
            self.url,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            data=json.dumps(self._payload(system_prompt, user_prompt, max_tokens)),
            timeout=30,
        )
        if resp.status_code == 429:
            retry_after = resp.headers.get("retry-after")
            raise RateLimitError(
                f"429 rate limited on model={self.model}",
                retry_after=float(retry_after) if retry_after else None,
            )
        resp.raise_for_status()
        return (resp.json()["choices"][0]["message"].get("content") or "").strip()


class LocalHFBackend(LLMBackend):
    """Runs a small open-weight instruct model locally via transformers.

    Suggested checkpoints (July 2026 landscape):
      - "Qwen/Qwen2.5-1.5B-Instruct"  -- the base model of the distilled student
      - "HuggingFaceTB/SmolLM3-3B"    -- strongest small open model, needs a GPU
                                          or patience on CPU
      - "meta-llama/Llama-3.2-3B-Instruct"

    Pass adapter_path to load a LoRA adapter trained with
    distill/train_v3_on_kaggle.ipynb on top of the base model, e.g.:
        annotate.py --backend local --model Qwen/Qwen2.5-1.5B-Instruct \\
            --adapter path/to/chess-annotator-lora-v3
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        adapter_path: str | None = None,
        device: str | None = None,
        temperature: float = 0.4,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self.torch = torch
        # If a LoRA adapter is given, prefer the tokenizer saved alongside it
        # (adapter training may have added special tokens) and fall back to
        # the base model's tokenizer otherwise.
        tokenizer_source = adapter_path or model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.temperature = temperature

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 200) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=self.temperature > 0,
                temperature=max(self.temperature, 1e-4),
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return text.strip()


class MockBackend(LLMBackend):
    """Deterministic, model-free backend for dry runs / pipeline testing."""

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 200) -> str:
        try:
            payload = json.loads(user_prompt.split("Move to annotate (JSON):\n", 1)[1].split("\n\nPrevious")[0])
            label = payload.get("quality_label", "move")
            san = payload.get("san", "?")
            best = payload.get("best_move_san")
            if label in ("blunder", "mistake") and best:
                return f"[MOCK] {san} is a {label}; {best} was stronger here."
            return f"[MOCK] {san} is a {label} move."
        except Exception:
            return "[MOCK] annotation placeholder."


def get_backend(name: str, **kwargs) -> LLMBackend:
    name = name.lower()
    if name == "groq":
        return GroqBackend(**kwargs)
    if name in ("local", "hf", "local_hf"):
        return LocalHFBackend(**kwargs)
    if name == "mock":
        return MockBackend()
    raise ValueError(f"Unknown backend: {name}")
