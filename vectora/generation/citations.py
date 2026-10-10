"""
Citation extraction and verification.

The old approach asked the model to append a JSON block listing the paper
titles it used, then matched those strings back to chunks with a
40-character substring comparison. That fails three ways: the block is
sometimes missing or malformed, titles in this corpus are often wrong or
duplicated across papers so the match is ambiguous, and a hallucinated
title silently produced no citation at all while the answer still read as
cited.

What replaces it:

1. **The model cites by source number** (``[1]``, ``[2][4]``), which is
   unambiguous and parseable with a regex. Numbers are resolved against the
   exact list of chunks that were in the prompt, so a citation can only
   point at something the model was actually shown.
2. **Invented numbers are stripped** from the answer text rather than left
   to look authoritative.
3. **Quotes are verified.** When the attribution pass supplies a supporting
   quote, it is checked against the chunk it claims to come from; a quote
   that is not in the text is dropped and logged, because a fabricated
   quote is worse than no quote.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from difflib import SequenceMatcher

from pydantic import BaseModel, Field

from vectora.generation.prompts import CITATION_SYSTEM
from vectora.llm import LLMClient
from vectora.models import Citation, RetrievedChunk, Usage
from vectora.observability import get_logger

log = get_logger(__name__)

#: ``[1]``, ``[1, 3]``, ``[1][3]``, ``[1-3]``
_MARKER = re.compile(r"\[(\d{1,2}(?:\s*[,\-–]\s*\d{1,2})*)\]")
_RANGE = re.compile(r"^(\d{1,2})\s*[\-–]\s*(\d{1,2})$")
_WHITESPACE = re.compile(r"\s+")

#: A quote must be at least this similar to a window of the source to count.
_QUOTE_MATCH_RATIO = 0.82


class _CitedSource(BaseModel):
    source: int
    quote: str = ""


class _Attribution(BaseModel):
    sources: list[_CitedSource] = Field(default_factory=list, max_length=20)


def parse_markers(text: str) -> list[int]:
    """Every source number referenced in the answer, in order of appearance."""
    found: list[int] = []
    for match in _MARKER.finditer(text):
        for part in re.split(r"[,]", match.group(1)):
            part = part.strip()
            rng = _RANGE.match(part)
            if rng:
                start, end = int(rng.group(1)), int(rng.group(2))
                if 0 < end - start < 12:
                    found.extend(range(start, end + 1))
                continue
            if part.isdigit():
                found.append(int(part))
    seen: set[int] = set()
    return [n for n in found if not (n in seen or seen.add(n))]


def strip_invalid_markers(text: str, valid: set[int]) -> tuple[str, list[int]]:
    """Remove citation markers that point at nothing.

    A model occasionally cites ``[7]`` when it was given six sources. Left
    in place, that reads as a reference the user cannot follow; removed, the
    sentence is merely uncited, which is honest.
    """
    removed: list[int] = []

    def replace(match: re.Match[str]) -> str:
        numbers: list[int] = []
        for part in re.split(r"[,]", match.group(1)):
            part = part.strip()
            rng = _RANGE.match(part)
            if rng:
                numbers.extend(range(int(rng.group(1)), int(rng.group(2)) + 1))
            elif part.isdigit():
                numbers.append(int(part))
        keep = [n for n in numbers if n in valid]
        removed.extend(n for n in numbers if n not in valid)
        if not keep:
            return ""
        return "".join(f"[{n}]" for n in keep)

    cleaned = _MARKER.sub(replace, text)
    # Tidy the spacing a removed marker leaves behind.
    cleaned = re.sub(r" +([.,;:])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip(), removed


def _normalise(text: str) -> str:
    return _WHITESPACE.sub(" ", text.lower()).strip()


def verify_quote(quote: str, source_text: str) -> str:
    """Return the quote if it really appears in ``source_text``, else "".

    Exact match first; then a fuzzy window comparison, because PDF text
    carries whitespace and ligature variation that a model will normalise
    when copying. Anything below the similarity floor is treated as
    fabricated.
    """
    q = _normalise(quote)
    if len(q) < 12:
        return ""
    source = _normalise(source_text)
    if q in source:
        return quote.strip()

    window = len(q)
    if window > len(source):
        return ""
    best = 0.0
    step = max(1, window // 4)
    for start in range(0, len(source) - window + 1, step):
        ratio = SequenceMatcher(None, q, source[start : start + window]).ratio()
        if ratio > best:
            best = ratio
            if best >= _QUOTE_MATCH_RATIO:
                return quote.strip()
    return ""


def citations_from_markers(
    answer: str, chunks: Sequence[RetrievedChunk]
) -> tuple[str, list[Citation]]:
    """Build citations from the markers in the answer.

    This is the primary path and needs no extra LLM call. ``chunks`` must be
    in the same order they were numbered in the prompt, one-indexed.
    """
    valid = set(range(1, len(chunks) + 1))
    cleaned, removed = strip_invalid_markers(answer, valid)
    if removed:
        log.warning(
            "answer cited sources that were not provided",
            extra={"invalid": sorted(set(removed)), "provided": len(chunks)},
        )

    cited: list[Citation] = []
    for number in parse_markers(cleaned):
        cited.append(Citation.from_retrieved(chunks[number - 1], number))
    return cleaned, cited


async def attribute(
    answer: str,
    chunks: Sequence[RetrievedChunk],
    *,
    llm: LLMClient | None,
    usage: Usage | None = None,
) -> list[Citation]:
    """Second-pass attribution with verified supporting quotes.

    Optional: used for ``deep`` mode and for the eval harness, where knowing
    *which sentence* supports a claim is worth an extra call. Returns an
    empty list on failure, and the marker-derived citations stand.
    """
    if llm is None or not chunks or not answer.strip():
        return []

    numbered = "\n\n".join(rc.to_context_block(i + 1) for i, rc in enumerate(chunks))
    result = await llm.structured(
        purpose="attribute",
        system=CITATION_SYSTEM,
        user=f"ANSWER:\n{answer}\n\nSOURCES:\n{numbered}",
        schema=_Attribution,
        max_tokens=900,
        usage=usage,
    )
    if result is None:
        return []

    out: list[Citation] = []
    seen: set[int] = set()
    for item in result.sources:
        if not (1 <= item.source <= len(chunks)) or item.source in seen:
            continue
        seen.add(item.source)
        rc = chunks[item.source - 1]
        quote = verify_quote(item.quote, rc.chunk.content) if item.quote else ""
        if item.quote and not quote:
            log.warning(
                "unverifiable quote discarded",
                extra={"source": item.source, "quote": item.quote[:80]},
            )
        out.append(Citation.from_retrieved(rc, item.source, quote=quote))
    return out


def merge_citations(
    primary: Sequence[Citation], enriched: Sequence[Citation]
) -> list[Citation]:
    """Keep the marker-derived set, adding verified quotes where available.

    The markers are authoritative about *what the answer cited*; the
    attribution pass is only trusted to add evidence for those, never to
    introduce a citation the answer did not make.
    """
    quotes = {c.index: c.quote for c in enriched if c.quote}
    out: list[Citation] = []
    for citation in primary:
        quote = quotes.get(citation.index, "")
        out.append(citation.model_copy(update={"quote": quote}) if quote else citation)
    return out


def dedupe_by_paper(citations: Sequence[Citation]) -> list[Citation]:
    """One reference per paper for display, keeping the first occurrence.

    The full per-chunk list stays available; this is only for the reference
    block, where three chunks from one paper should not look like three
    independent sources.
    """
    seen: set[str] = set()
    out: list[Citation] = []
    for citation in citations:
        key = citation.paper_id or citation.title
        if key in seen:
            continue
        seen.add(key)
        out.append(citation)
    return out
