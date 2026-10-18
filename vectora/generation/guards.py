"""
Answer guards and the grounded fallback.

Two guards decide whether retrieval found enough to answer from. The
previous implementation had one, and it was broken: the threshold
``-8.0`` was calibrated for cross-encoder logits but was compared against
whatever ``score`` happened to hold, which in ``fast`` mode was a
``hybrid_score`` in [0, 1]. The guard therefore never fired in the one mode
where weak retrieval is most likely.

Here the threshold is chosen by *which score is actually present*: a
reranker logit is compared against the logit floor, a fused RRF score
against the RRF floor. Because RRF scores do not depend on the batch they
were computed in, that floor means the same thing from one query to the
next — which is the property the old min-max normalisation destroyed.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from vectora.config import Settings
from vectora.generation.prompts import FALLBACK_SYSTEM, fallback_user_message
from vectora.llm import LLMClient
from vectora.models import Fallback, RetrievedChunk, Usage
from vectora.observability import get_logger

log = get_logger(__name__)

#: Titles and sections that are extraction debris rather than subject matter.
#: Without this the fallback suggested topics like "GModel ARTICLE IN PRESS —
#: Review", which tells the user nothing about what the corpus covers.
_NOISY_TOPIC = re.compile(
    r"https?://|www\.|\d{4}[-/]\d{2}|"
    r"^(type|received|accepted|revised|published|doi|issn|isbn|pmid)\b|"
    r"\b(article in press|g\s*model|journal homepage|contents lists|"
    r"all rights reserved|sciencedirect|crossmark|no\.?\s*of\s*pages)\b|"
    r"^[a-z]{2,6}-\d{3,6}\b|"
    r"^(library|abstract|references|acknowledge?ments?|appendix|front matter|"
    r"review|article|untitled)$",
    re.I,
)


def check(
    chunks: Sequence[RetrievedChunk], settings: Settings
) -> tuple[bool, str]:
    """Decide whether retrieval found enough to answer from.

    Returns ``(ok, reason)``; ``reason`` is the fallback reason when not ok.

    The signal is chosen by what is actually available, strongest first:

    1. the cross-encoder logit, when the reranker ran — the best absolute
       judgement of (query, passage) relevance we have;
    2. otherwise the dense cosine of the best chunk, which is absolute and
       comparable across queries;
    3. only if neither exists (lexical-only retrieval), the RRF score, which
       can tell us that *something* ranked but not whether it was relevant.

    Mixing these scales is exactly the bug this replaces: a cross-encoder
    logit floor was compared against whatever ``score`` happened to hold,
    which in fast mode was a 0-1 similarity, so the guard never fired in the
    mode where weak retrieval is most likely.
    """
    if not chunks or len(chunks) < settings.min_chunks_to_answer:
        return False, "no_results"

    reranked = [rc for rc in chunks if rc.is_reranked]
    if reranked:
        value = max(rc.rerank_score or 0.0 for rc in reranked)
        floor, scale = settings.min_rerank_score, "rerank"
    else:
        dense = [rc.dense_score for rc in chunks if rc.dense_score is not None]
        if dense:
            value, floor, scale = max(dense), settings.min_dense_score, "dense"
        else:
            value = max(rc.fused_score for rc in chunks)
            floor, scale = settings.min_fused_score, "fused"

    if value < floor:
        log.info(
            "low confidence",
            extra={"scale": scale, "top_score": round(value, 5), "floor": floor},
        )
        return False, "low_confidence"
    return True, ""


def related_topics(chunks: Sequence[RetrievedChunk], limit: int = 6) -> list[str]:
    """Readable "what is in the collection" hints from weak chunks.

    Filters the layout debris that made the old version suggest topics like
    ``"GModel ARTICLE IN PRESS — Review"``. Prefers high-quality chunks,
    since those are the ones whose titles and sections are trustworthy.
    """
    out: list[str] = []
    seen: set[str] = set()
    ordered = sorted(chunks, key=lambda rc: rc.chunk.quality, reverse=True)

    for rc in ordered:
        chunk = rc.chunk
        title = (chunk.title or "").strip()
        section = (chunk.section or "").strip()
        if not title or len(title) < 8 or len(title) > 120:
            continue
        if _NOISY_TOPIC.search(title) or (section and _NOISY_TOPIC.search(section)):
            continue
        letters = sum(c.isalpha() for c in title)
        if letters / max(len(title), 1) < 0.6:
            continue
        label = f"{title} — {section}" if section else title
        key = label.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(label)
        if len(out) >= limit:
            break
    return out


async def build_fallback(
    query: str,
    chunks: Sequence[RetrievedChunk],
    *,
    reason: str,
    llm: LLMClient | None,
    usage: Usage | None = None,
) -> Fallback:
    """Compose the response shown when no answer can be grounded.

    The suggestions are steered toward topics observed in the retrieved
    (but too weak) chunks, so the user is not told to rephrase toward
    subjects the corpus also lacks. On LLM failure the suggestions are
    derived from those topics directly.
    """
    topics = related_topics(chunks)

    suggestions: list[str] = []
    if llm is not None:
        raw = await llm.utility(
            purpose="fallback",
            system=FALLBACK_SYSTEM,
            user=fallback_user_message(query, topics),
            max_tokens=220,
            temperature=0.4,
            usage=usage,
        )
        if raw:
            for line in raw.splitlines():
                text = line.strip().lstrip("-•*0123456789. ").strip()
                if 12 <= len(text) <= 220 and "?" in text:
                    suggestions.append(text)
            suggestions = suggestions[:3]

    if not suggestions:
        suggestions = [
            f"What do the papers say about {t.split(' — ')[0][:70]}?" for t in topics[:3]
        ] or [
            f"What methods are used to study {query[:60]}?",
            f"What are the main findings on {query[:60]}?",
        ]

    message = {
        "no_corpus": (
            "The paper index is empty, so there is nothing to search yet. Run "
            "`vectora index` to build it from chunks.jsonl."
        ),
        "no_results": (
            "Nothing in the paper collection matches this question. The topic may "
            "not be covered by these papers — or it may be there under different "
            "terminology, in which case one of the rephrasings below should find it."
        ),
        "low_confidence": (
            "Found some loosely related passages, but none close enough to answer "
            "from without guessing. Rather than give you something that reads "
            "confident and is not grounded, here is what the collection does cover."
        ),
    }[reason]

    log.info("fallback", extra={"reason": reason, "topics": len(topics)})
    return Fallback(
        reason=reason,  # type: ignore[arg-type]
        message=message,
        suggestions=suggestions,
        related_topics=topics,
    )
