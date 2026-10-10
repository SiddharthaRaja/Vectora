"""
Query expansion: rewriting, HyDE and conversational contextualisation.

Changes from the previous implementation:

* **The calls run concurrently.** Rewriting and HyDE were serial, and HyDE
  ran once *per rewritten variant* — so ``deep`` mode made one routing call,
  one rewrite call and three HyDE calls before retrieval even started, all
  one after another. They now go out together with ``asyncio.gather``, which
  turns five round-trips into roughly one.
* **One HyDE document, not one per variant.** The hypothetical answer is a
  function of the question, not of its phrasings; generating three was
  paying three times for near-identical text.
* **No module-global mutation.** The variant count used to be applied by
  assigning to ``retrieval.REWRITE_N`` and restoring it afterwards, which
  races under a threaded server. It is a parameter now.
* **Structured output** for the rewrites, so a chatty model cannot smuggle
  a preamble into the variant list.
* **Follow-up questions are resolved against the conversation** before
  retrieval, so "and in serum?" becomes a standalone query instead of being
  naively prefixed with the previous question.

Every stage degrades to a no-op if the LLM is unavailable; the caller gets
the original query back and the diagnostics record what was skipped.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, Field

from vectora.config import Settings
from vectora.embeddings import Encoder
from vectora.llm import LLMClient
from vectora.models import Usage
from vectora.observability import get_logger

log = get_logger(__name__)


class _Rewrites(BaseModel):
    variants: list[str] = Field(default_factory=list, max_length=6)


class _Standalone(BaseModel):
    question: str = ""


_REWRITE_SYSTEM = (
    "You rewrite questions for a search engine over biomedical and analytical "
    "chemistry papers. Produce alternative phrasings of the user's question "
    "that use formal domain terminology, so that a lexical search can match "
    "the wording a paper would actually use. Each variant must be a complete "
    "question about the same topic, approaching it from a different angle: "
    "one may use the technical term for the concept, another the common name, "
    "another the measurement or method involved. Never answer the question. "
    "Never add topics the user did not ask about."
)

_HYDE_SYSTEM = (
    "You write a short passage in the register of a scientific paper. Given a "
    "research question, write one dense paragraph of three to five sentences "
    "that reads like an excerpt from the results or methods section of a paper "
    "answering it. Use formal academic language, domain terminology, plausible "
    "units and the names of relevant techniques. Factual accuracy is not the "
    "goal — vocabulary coverage is. Do not hedge, do not mention that this is "
    "hypothetical, and do not address the reader."
)

_STANDALONE_SYSTEM = (
    "You rewrite a follow-up question into a standalone question. Use the "
    "conversation to resolve pronouns and ellipsis, so the result makes sense "
    "with no prior context. Keep the user's intent and scope exactly: do not "
    "broaden it, do not answer it, and do not add detail that was not implied. "
    "If the question already stands alone, return it unchanged."
)


@dataclass
class Expansion:
    """Everything the retriever needs, after expansion."""

    original: str
    search_query: str
    """The query used for lexical search and for reranking: the standalone
    form of the question, never the hypothetical document."""
    variants: list[str] = field(default_factory=list)
    """Alternative phrasings, excluding ``search_query``."""
    dense_vectors: list[tuple[str, np.ndarray]] = field(default_factory=list)
    """``(label, vector)`` pairs to run dense search with."""
    hyde_document: str = ""
    hyde_used: bool = False
    degraded: list[str] = field(default_factory=list)

    @property
    def all_queries(self) -> list[str]:
        return [self.search_query, *self.variants]


async def contextualise(
    query: str,
    history: list[dict[str, str]],
    *,
    llm: LLMClient | None,
    settings: Settings,
    usage: Usage | None = None,
) -> tuple[str, bool]:
    """Turn a follow-up into a standalone question.

    Only runs for short questions in a conversation that has history — a
    fully-formed question does not need it, and the call is not free.
    Returns ``(question, was_rewritten)``.
    """
    if not history or len(query.split()) >= settings.contextualise_below_words:
        return query, False
    if llm is None:
        return query, False

    transcript = _render_history(history, limit=4, chars=600)
    result = await llm.structured(
        purpose="contextualise",
        system=_STANDALONE_SYSTEM,
        user=f"Conversation so far:\n{transcript}\n\nFollow-up question: {query}",
        schema=_Standalone,
        max_tokens=150,
        usage=usage,
    )
    if result and result.question.strip():
        rewritten = result.question.strip()
        # Guard against the model answering instead of rewriting.
        if len(rewritten) <= max(len(query) * 6, 240):
            log.info("follow-up resolved", extra={"from": query[:60], "to": rewritten[:80]})
            return rewritten, True
    return query, False


async def rewrite(
    query: str,
    *,
    llm: LLMClient,
    n: int,
    usage: Usage | None = None,
) -> list[str]:
    """``n`` alternative phrasings. Empty list on any failure."""
    if n <= 0:
        return []
    result = await llm.structured(
        purpose="rewrite",
        system=_REWRITE_SYSTEM,
        user=f"Produce exactly {n} alternative phrasings of this question: {query}",
        schema=_Rewrites,
        max_tokens=100 + 60 * n,
        temperature=0.4,
        usage=usage,
    )
    if not result:
        return []

    seen = {query.strip().lower()}
    out: list[str] = []
    for variant in result.variants:
        text = variant.strip().lstrip("-•0123456789. ").strip()
        if 8 <= len(text) <= 300 and text.lower() not in seen:
            seen.add(text.lower())
            out.append(text)
    return out[:n]


async def hypothetical_document(
    query: str,
    *,
    llm: LLMClient,
    usage: Usage | None = None,
) -> str:
    """One HyDE passage for the question. Empty string on failure."""
    out = await llm.utility(
        purpose="hyde",
        system=_HYDE_SYSTEM,
        user=f"Research question: {query}",
        max_tokens=260,
        temperature=0.7,
        usage=usage,
    )
    return (out or "").strip()


def blend(query_vector: np.ndarray, hyde_vector: np.ndarray, weight: float) -> np.ndarray:
    """Interpolate query and HyDE embeddings, re-normalised.

    Blending rather than replacing matters: a pure HyDE embedding drifts
    toward whatever the model invented, which retrieves fluent but
    off-target passages. Keeping half the literal question anchors it.
    """
    mixed = (1.0 - weight) * query_vector + weight * hyde_vector
    norm = float(np.linalg.norm(mixed))
    return mixed / norm if norm > 0 else query_vector


async def expand(
    query: str,
    *,
    settings: Settings,
    encoder: Encoder,
    llm: LLMClient | None,
    rewrite_variants: int,
    use_hyde: bool,
    usage: Usage | None = None,
) -> Expansion:
    """Run every expansion step that the mode asks for, concurrently."""
    exp = Expansion(original=query, search_query=query)

    if llm is None or (rewrite_variants <= 0 and not use_hyde):
        if llm is None and (rewrite_variants > 0 or use_hyde):
            exp.degraded.append("expansion_unavailable")
        exp.dense_vectors = [("original", encoder.encode_query(query))]
        return exp

    tasks: dict[str, asyncio.Task] = {}
    if rewrite_variants > 0:
        tasks["rewrite"] = asyncio.create_task(
            rewrite(query, llm=llm, n=rewrite_variants, usage=usage)
        )
    if use_hyde:
        tasks["hyde"] = asyncio.create_task(
            hypothetical_document(query, llm=llm, usage=usage)
        )

    results: dict[str, object] = {}
    if tasks:
        gathered = await asyncio.gather(*tasks.values(), return_exceptions=True)
        for name, value in zip(tasks.keys(), gathered, strict=True):
            if isinstance(value, BaseException):
                log.warning("expansion stage failed", extra={"stage": name, "err": str(value)[:120]})
                exp.degraded.append(name)
            else:
                results[name] = value

    variants = list(results.get("rewrite") or [])
    if rewrite_variants > 0 and not variants and "rewrite" not in exp.degraded:
        exp.degraded.append("rewrite_empty")
    exp.variants = variants

    # One encode call for the question and every variant.
    labels = ["original", *[f"variant{i + 1}" for i in range(len(variants))]]
    texts = [query, *variants]
    matrix = encoder.encode(texts)
    vectors: list[tuple[str, np.ndarray]] = list(
        zip(labels, list(matrix), strict=True)
    )

    hyde_doc = str(results.get("hyde") or "")
    if hyde_doc:
        exp.hyde_document = hyde_doc
        exp.hyde_used = True
        hyde_vector = encoder.encode([hyde_doc])[0]
        # The HyDE vector is added as an extra search direction *and* blended
        # with the literal question, which covers both the case where the
        # hypothetical is on target and the case where it drifted.
        vectors.append(("hyde", hyde_vector))
        vectors.append(("hyde_blend", blend(matrix[0], hyde_vector, settings.hyde_weight)))
    elif use_hyde and "hyde" not in exp.degraded:
        exp.degraded.append("hyde_empty")

    exp.dense_vectors = vectors
    log.info(
        "expanded query",
        extra={
            "variants": len(variants),
            "hyde": exp.hyde_used,
            "vectors": len(vectors),
            "degraded": ",".join(exp.degraded) or "-",
        },
    )
    return exp


def _render_history(history: list[dict[str, str]], *, limit: int, chars: int) -> str:
    """Compact transcript for the contextualiser."""
    lines: list[str] = []
    for turn in history[-limit:]:
        role = "User" if turn.get("role") == "user" else "Assistant"
        content = (turn.get("content") or "").strip().replace("\n", " ")
        if len(content) > chars:
            content = content[:chars] + "…"
        lines.append(f"{role}: {content}")
    return "\n".join(lines)
