"""
Token budget tracking and context management for LLM calls.

Tracks cumulative token usage per provider/model, enforces budgets,
and provides context compression utilities.
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional

from loguru import logger

# Estimated token-to-character ratio (conservative, for non-OpenAI providers)
_CHARS_PER_TOKEN_ESTIMATE = 4


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class BudgetConfig:
    max_total_tokens: int = 100_000
    max_prompt_tokens: int = 80_000
    max_completion_tokens: int = 20_000
    warning_threshold: float = 0.8

    @classmethod
    def from_dict(cls, d: dict) -> "BudgetConfig":
        return cls(
            max_total_tokens=d.get("max_total_tokens", 100_000),
            max_prompt_tokens=d.get("max_prompt_tokens", 80_000),
            max_completion_tokens=d.get("max_completion_tokens", 20_000),
            warning_threshold=d.get("warning_threshold", 0.8),
        )


class TokenBudgetTracker:
    """Per-session token budget tracker for LLM calls."""

    def __init__(self, config: Optional[BudgetConfig] = None):
        self.config = config or BudgetConfig()
        self._usage: Dict[str, TokenUsage] = defaultdict(TokenUsage)
        self._session_start = time.time()

    def record(self, provider: str, prompt: str, completion: str = ""):
        usage = self._usage[provider]
        usage.prompt_tokens += self._estimate_tokens(prompt)
        usage.completion_tokens += self._estimate_tokens(completion)
        self._maybe_warn(provider, usage)

    def record_usage(self, provider: str, prompt_tokens: int, completion_tokens: int):
        usage = self._usage[provider]
        usage.prompt_tokens += prompt_tokens
        usage.completion_tokens += completion_tokens
        self._maybe_warn(provider, usage)

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // _CHARS_PER_TOKEN_ESTIMATE)

    def _maybe_warn(self, provider: str, usage: TokenUsage):
        budget = self.config
        if usage.total >= budget.max_total_tokens * budget.warning_threshold:
            logger.warning(
                f"[{provider}] token usage at {usage.total}/{budget.max_total_tokens} "
                f"({usage.total / budget.max_total_tokens:.0%}) — approaching budget limit"
            )

    def can_proceed(self, provider: str, estimated_prompt: int = 0) -> bool:
        usage = self._usage.get(provider, TokenUsage())
        budget = self.config
        if usage.total + estimated_prompt >= budget.max_total_tokens:
            logger.warning(f"[{provider}] budget exhausted, blocking further calls")
            return False
        return True

    def summary(self, provider: str = "") -> str:
        if provider:
            u = self._usage.get(provider, TokenUsage())
            return f"[{provider}] prompt={u.prompt_tokens}, completion={u.completion_tokens}, total={u.total}"
        return "; ".join(
            f"[{p}] total={u.total}" for p, u in self._usage.items()
        )

    def reset(self):
        self._usage.clear()
        self._session_start = time.time()


def compress_prompt(prompt: str, max_chars: int = 8000) -> str:
    """Compress a prompt to fit within a character budget.

    Uses a decay strategy: truncate middle sections first, keeping
    the beginning (instructions) and end (context).
    """
    if len(prompt) <= max_chars:
        return prompt

    logger.warning(f"compressing prompt from {len(prompt)} to {max_chars} chars")
    header_end = prompt.find("## Context") if "## Context" in prompt else min(2000, len(prompt) // 4)
    header = prompt[:header_end]
    remaining_budget = max_chars - len(header) - 100
    if remaining_budget <= 0:
        return prompt[:max_chars]

    trailer_start = max(header_end, len(prompt) - remaining_budget)
    trailer = prompt[trailer_start:]
    return f"{header}\n\n[content truncated due to length]\n\n{trailer}"


def sliding_window_context(contexts: list[str], max_total_chars: int) -> list[str]:
    """Keep the most recent contexts, dropping oldest when over budget."""
    total = sum(len(c) for c in contexts)
    if total <= max_total_chars:
        return contexts

    budget = max_total_chars
    result: list[str] = []
    for ctx in reversed(contexts):
        if len(ctx) <= budget:
            result.insert(0, ctx)
            budget -= len(ctx)
        else:
            result.insert(0, ctx[:budget])
            break
    return result


# Global singleton
global_budget = TokenBudgetTracker()
