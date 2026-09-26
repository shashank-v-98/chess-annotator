"""Settings for the web app, read from environment variables (see webapp/README.md)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default, cast=str):
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    if cast is bool:
        return v.lower() in ("1", "true", "yes", "on")
    return cast(v)


# USD per million tokens (input, output). Verify against Groq's pricing page before
# relying on the daily budget; override with PRICE_IN_PER_M / PRICE_OUT_PER_M.
DEFAULT_PRICES = {
    "openai/gpt-oss-120b": (0.15, 0.75),
    "openai/gpt-oss-20b": (0.075, 0.30),
}


@dataclass
class Settings:
    groq_api_key: str = field(default_factory=lambda: _env("GROQ_API_KEY", ""))
    commentator_backend: str = field(default_factory=lambda: _env("COMMENTATOR_BACKEND", "groq"))  # groq | student | mock
    # Distilled student (COMMENTATOR_BACKEND=student): base model + LoRA adapter, run locally.
    student_base_model: str = field(default_factory=lambda: _env("STUDENT_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"))
    student_adapter: str = field(default_factory=lambda: _env("STUDENT_ADAPTER", "chess-annotator-lora-v3"))
    student_max_tokens: int = field(default_factory=lambda: _env("STUDENT_MAX_TOKENS", 260, int))
    # Quantized student via llama.cpp (COMMENTATOR_BACKEND=student_gguf): either a GGUF file the app
    # serves with its own llama-server, or the URL of a llama-server you started yourself.
    student_gguf: str = field(default_factory=lambda: _env("STUDENT_GGUF", "student-v3-q8_0.gguf"))
    student_gguf_url: str = field(default_factory=lambda: _env("STUDENT_GGUF_URL", ""))
    llama_server: str = field(default_factory=lambda: _env("LLAMA_SERVER", "llama-server"))
    llama_port: int = field(default_factory=lambda: _env("LLAMA_PORT", 8081, int))
    student_threads: int = field(default_factory=lambda: _env("STUDENT_THREADS", 0, int))  # 0 = llama.cpp default
    model: str = field(default_factory=lambda: _env("COMMENTATOR_MODEL", "openai/gpt-oss-120b"))
    reasoning_effort: str = field(default_factory=lambda: _env("REASONING_EFFORT", "medium"))
    factcheck_default: bool = field(default_factory=lambda: _env("FACTCHECK_DEFAULT", True, bool))
    factcheck_model: str = field(default_factory=lambda: _env("FACTCHECK_MODEL", "openai/gpt-oss-120b"))

    stockfish_path: str = field(default_factory=lambda: _env("STOCKFISH_PATH", "stockfish"))
    engine_depth: int = field(default_factory=lambda: _env("ENGINE_DEPTH", 14, int))

    max_pgn_chars: int = field(default_factory=lambda: _env("MAX_PGN_CHARS", 30000, int))
    max_plies: int = field(default_factory=lambda: _env("MAX_PLIES", 200, int))

    max_concurrent_games: int = field(default_factory=lambda: _env("MAX_CONCURRENT_GAMES", 2, int))
    queue_wait_s: float = field(default_factory=lambda: _env("QUEUE_WAIT_S", 30.0, float))
    llm_concurrency: int = field(default_factory=lambda: _env("LLM_CONCURRENCY", 6, int))
    games_per_ip_per_hour: int = field(default_factory=lambda: _env("GAMES_PER_IP_PER_HOUR", 10, int))
    trust_proxy_headers: bool = field(default_factory=lambda: _env("TRUST_PROXY_HEADERS", False, bool))

    daily_budget_usd: float = field(default_factory=lambda: _env("DAILY_BUDGET_USD", 2.0, float))
    price_in_per_m: float = field(default_factory=lambda: _env("PRICE_IN_PER_M", -1.0, float))
    price_out_per_m: float = field(default_factory=lambda: _env("PRICE_OUT_PER_M", -1.0, float))

    db_path: str = field(default_factory=lambda: _env("DB_PATH", "webapp_data.sqlite3"))

    def prices(self, model: str) -> tuple:
        if self.price_in_per_m >= 0 and self.price_out_per_m >= 0:
            return self.price_in_per_m, self.price_out_per_m
        return DEFAULT_PRICES.get(model, (0.5, 1.5))  # unknown model: assume expensive

    def cost(self, model: str, usage: dict) -> float:
        pin, pout = self.prices(model)
        return (usage.get("prompt_tokens", 0) * pin + usage.get("completion_tokens", 0) * pout) / 1e6


# Bump when prompts/insights change so cached commentary from older versions isn't reused.
PIPELINE_VERSION = "app-1"
