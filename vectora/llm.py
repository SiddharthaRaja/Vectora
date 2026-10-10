"""
The one place that talks to OpenAI.

What it adds over calling the SDK directly:

* **Timeouts and bounded retries** on transient failures only — a 400 for a
  malformed request is not retried, a 429 or a 503 is, with exponential
  backoff and jitter.
* **A concurrency gate**, so fanning out three rewrites plus HyDE cannot
  stampede the rate limit.
* **Token and cost accounting** per call, aggregated per request and against
  a daily ceiling. Crossing the ceiling disables the optional utility calls
  rather than the answer itself.
* **Structured output** helpers that validate into a pydantic model, so the
  routing and citation paths stop depending on the model emitting
  well-formed JSON in prose.
* **Graceful degradation**: ``utility`` calls return ``None`` on failure and
  the caller carries on without that stage. Only the answer call raises.
"""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from vectora.cache import Cache, make_key
from vectora.config import Settings
from vectora.models import Usage
from vectora.observability import LLM_CALLS, LLM_COST, get_logger

log = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

#: USD per 1M tokens, (input, output). Used for budgeting and the cost line
#: in diagnostics — informational, not billing. Update when prices change.
PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "o4-mini": (1.10, 4.40),
}
_DEFAULT_PRICE = (0.15, 0.60)


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    inp, out = PRICING.get(model, _DEFAULT_PRICE)
    return (prompt_tokens * inp + completion_tokens * out) / 1_000_000


class LLMError(RuntimeError):
    """Raised only for failures the caller must not paper over."""


@dataclass
class Budget:
    """Running spend for the process, reset at UTC midnight."""

    ceiling_usd: float
    spent_usd: float = 0.0
    day: date = field(default_factory=date.today)

    def record(self, cost: float) -> None:
        today = date.today()
        if today != self.day:
            self.day, self.spent_usd = today, 0.0
        self.spent_usd += cost

    @property
    def exhausted(self) -> bool:
        return self.ceiling_usd > 0 and self.spent_usd >= self.ceiling_usd

    @property
    def remaining(self) -> float:
        return max(self.ceiling_usd - self.spent_usd, 0.0) if self.ceiling_usd else float("inf")


_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def _is_retryable(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int):
        return status in _RETRYABLE_STATUS
    # Network-level problems carry no status.
    return isinstance(exc, (asyncio.TimeoutError, ConnectionError, TimeoutError)) or (
        type(exc).__name__
        in {"APIConnectionError", "APITimeoutError", "InternalServerError", "RateLimitError"}
    )


class LLMClient:
    """Async OpenAI wrapper. One instance per process."""

    def __init__(self, settings: Settings, cache: Cache | None = None) -> None:
        from openai import AsyncOpenAI

        self.settings = settings
        self.cache = cache
        self.budget = Budget(ceiling_usd=settings.daily_cost_ceiling_usd)
        self._sem = asyncio.Semaphore(settings.llm_max_concurrency)
        self._client = AsyncOpenAI(
            api_key=settings.require_openai(),
            base_url=settings.openai_base_url,
            timeout=settings.llm_timeout_seconds,
            max_retries=0,  # we own the retry policy
        )

    # ── core ──────────────────────────────────────────────────────────────

    async def _call(
        self,
        *,
        purpose: str,
        model: str,
        messages: Sequence[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        usage: Usage | None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        last: BaseException | None = None
        for attempt in range(1, self.settings.llm_max_retries + 1):
            try:
                async with self._sem:
                    kwargs: dict[str, Any] = {
                        "model": model,
                        "messages": list(messages),
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    }
                    if response_format:
                        kwargs["response_format"] = response_format
                    resp = await self._client.chat.completions.create(**kwargs)

                prompt_tokens = getattr(resp.usage, "prompt_tokens", 0) or 0
                completion_tokens = getattr(resp.usage, "completion_tokens", 0) or 0
                cost = estimate_cost(model, prompt_tokens, completion_tokens)
                self.budget.record(cost)
                if usage is not None:
                    usage.add(prompt_tokens, completion_tokens, cost)

                LLM_CALLS.labels(purpose=purpose, model=model, outcome="ok").inc()
                LLM_COST.labels(purpose=purpose, model=model).inc(cost)
                log.debug(
                    "llm call ok",
                    extra={
                        "purpose": purpose,
                        "model": model,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "cost_usd": round(cost, 6),
                        "attempt": attempt,
                    },
                )
                return (resp.choices[0].message.content or "").strip()

            except Exception as exc:
                last = exc
                if not _is_retryable(exc) or attempt == self.settings.llm_max_retries:
                    LLM_CALLS.labels(purpose=purpose, model=model, outcome="error").inc()
                    log.warning(
                        "llm call failed",
                        extra={
                            "purpose": purpose,
                            "model": model,
                            "attempt": attempt,
                            "err": f"{type(exc).__name__}: {exc}",
                        },
                    )
                    break
                backoff = min(2 ** (attempt - 1), 8) + random.uniform(0, 0.4)
                LLM_CALLS.labels(purpose=purpose, model=model, outcome="retry").inc()
                log.info(
                    "llm retry",
                    extra={"purpose": purpose, "attempt": attempt, "sleep_s": round(backoff, 2)},
                )
                await asyncio.sleep(backoff)

        raise LLMError(f"{purpose} failed after {self.settings.llm_max_retries} attempts: {last}")

    # ── public surface ────────────────────────────────────────────────────

    async def answer(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        usage: Usage | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """The user-visible generation call. Raises on failure."""
        return await self._call(
            purpose="answer",
            model=self.settings.answer_model,
            messages=messages,
            max_tokens=max_tokens or self.settings.answer_max_tokens,
            temperature=(
                self.settings.answer_temperature if temperature is None else temperature
            ),
            usage=usage,
        )

    async def stream_answer(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        usage: Usage | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Token stream for the answer call.

        Not retried: once the first token has been sent to the client a retry
        would duplicate output. A failure before any token is surfaced to the
        caller, which falls back to the non-streaming path.
        """
        model = self.settings.answer_model
        prompt_tokens = sum(len(str(m.get("content", ""))) for m in messages) // 4
        completion_chars = 0
        async with self._sem:
            stream = await self._client.chat.completions.create(
                model=model,
                messages=list(messages),
                max_tokens=max_tokens or self.settings.answer_max_tokens,
                temperature=self.settings.answer_temperature,
                stream=True,
                stream_options={"include_usage": True},
            )
            final_usage = None
            async for event in stream:
                if getattr(event, "usage", None):
                    final_usage = event.usage
                if not event.choices:
                    continue
                piece = event.choices[0].delta.content
                if piece:
                    completion_chars += len(piece)
                    yield piece

        prompt_tokens = getattr(final_usage, "prompt_tokens", prompt_tokens) or prompt_tokens
        completion_tokens = (
            getattr(final_usage, "completion_tokens", None) or completion_chars // 4
        )
        cost = estimate_cost(model, prompt_tokens, completion_tokens)
        self.budget.record(cost)
        if usage is not None:
            usage.add(prompt_tokens, completion_tokens, cost)
        LLM_CALLS.labels(purpose="answer_stream", model=model, outcome="ok").inc()
        LLM_COST.labels(purpose="answer_stream", model=model).inc(cost)

    async def utility(
        self,
        *,
        purpose: str,
        system: str,
        user: str,
        max_tokens: int = 300,
        temperature: float | None = None,
        usage: Usage | None = None,
        cache_ttl: int | None = None,
    ) -> str | None:
        """A cheap, optional call. Returns ``None`` instead of raising.

        Cached on (purpose, model, system, user) because these calls are
        deterministic enough that repeating a question should not repeat the
        spend.
        """
        if self.budget.exhausted:
            log.warning("budget exhausted, skipping utility call", extra={"purpose": purpose})
            return None

        model = self.settings.utility_model
        key = make_key("llm", purpose, model, system, user)
        if self.cache:
            hit = self.cache.get(key)
            if hit is not None:
                log.debug("llm cache hit", extra={"purpose": purpose})
                return str(hit)

        try:
            out = await self._call(
                purpose=purpose,
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=(
                    self.settings.utility_temperature if temperature is None else temperature
                ),
                usage=usage,
            )
        except LLMError:
            return None

        if self.cache and out:
            self.cache.set(key, out, ttl_seconds=cache_ttl)
        return out

    async def structured(
        self,
        *,
        purpose: str,
        system: str,
        user: str,
        schema: type[T],
        max_tokens: int = 800,
        temperature: float = 0.0,
        usage: Usage | None = None,
        required: bool = False,
    ) -> T | None:
        """Call the model and validate the reply into ``schema``.

        Uses JSON mode and validates with pydantic. A reply that does not
        validate is a miss, not a crash — the caller falls back to whatever
        it does when the stage is unavailable.
        """
        if self.budget.exhausted and not required:
            log.warning("budget exhausted, skipping structured call", extra={"purpose": purpose})
            return None

        model = self.settings.utility_model
        instructions = (
            f"{system}\n\nReply with a single JSON object and nothing else. "
            f"It must conform to this JSON Schema:\n"
            f"{json.dumps(schema.model_json_schema(), separators=(',', ':'))}"
        )
        key = make_key("structured", purpose, model, instructions, user)
        if self.cache:
            hit = self.cache.get(key)
            if hit is not None:
                try:
                    return schema.model_validate(hit)
                except ValidationError:
                    self.cache.delete(key)

        try:
            raw = await self._call(
                purpose=purpose,
                model=model,
                messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                usage=usage,
                response_format={"type": "json_object"},
            )
        except LLMError:
            if required:
                raise
            return None

        try:
            data = json.loads(_strip_fences(raw))
            parsed = schema.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as exc:
            log.warning(
                "structured output did not validate",
                extra={"purpose": purpose, "err": str(exc)[:200]},
            )
            if required:
                raise LLMError(f"{purpose} returned unusable JSON") from exc
            return None

        if self.cache:
            self.cache.set(key, data)
        return parsed

    async def close(self) -> None:
        await self._client.close()

    def stats(self) -> dict[str, Any]:
        return {
            "answer_model": self.settings.answer_model,
            "utility_model": self.settings.utility_model,
            "spent_usd_today": round(self.budget.spent_usd, 4),
            "ceiling_usd": self.budget.ceiling_usd,
            "budget_exhausted": self.budget.exhausted,
        }


def _strip_fences(text: str) -> str:
    """Tolerate a model that wraps JSON in a markdown fence anyway."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1] if "\n" in t else t
        if t.lstrip().startswith("json"):
            t = t.lstrip()[4:]
        if "```" in t:
            t = t[: t.rindex("```")]
    return t.strip()
