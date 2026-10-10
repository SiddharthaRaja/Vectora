"""
Grounding check — does the answer's content actually come from the sources?

This is the old "faithfulness" checker, with three changes:

* It is **consistent**. Previously the non-conversational path always ran it
  and the conversational path only ran it in ``deep`` mode, so the same
  question gave different output depending on which entry point was used.
  Now one code path decides, from the mode profile.
* It is **validated**, through the structured-output helper, instead of
  parsing JSON out of prose and guessing when that fails.
* It does **not see the question**. Showing the checker what the answer was
  supposed to say biases it toward accepting plausible-sounding claims; it
  gets only the answer and the sources, which is the actual job.

A cheap mechanical pre-check runs first: an answer with no numbers and no
citations that is also short cannot be meaningfully graded, and an answer
whose every sentence is cited and whose numbers all appear in the sources
is very likely grounded. That catches the easy cases without a call.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import BaseModel, Field

from vectora.generation.prompts import GROUNDING_SYSTEM
from vectora.llm import LLMClient
from vectora.models import Grounding, RetrievedChunk, Usage
from vectora.observability import get_logger

log = get_logger(__name__)

_NUMBER = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?(?:\s*[-–]\s*\d+(?:[.,]\d+)?)?")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


class _GroundingReply(BaseModel):
    score: int = Field(ge=0, le=100)
    supported_claims: list[str] = Field(default_factory=list, max_length=20)
    unsupported_claims: list[str] = Field(default_factory=list, max_length=20)
    summary: str = ""


def unsupported_numbers(answer: str, chunks: Sequence[RetrievedChunk]) -> list[str]:
    """Numbers in the answer that appear in none of the sources.

    A fabricated figure is the most consequential and most detectable kind
    of ungrounded claim in this domain, and finding it costs nothing.
    Single digits and years are ignored — they are too common to be
    evidence either way.
    """
    corpus = " ".join(rc.chunk.content for rc in chunks)
    corpus_numbers = {n.replace(",", ".") for n in _NUMBER.findall(corpus)}
    missing: list[str] = []
    for raw in _NUMBER.findall(answer):
        value = raw.replace(",", ".")
        if len(value.strip()) <= 1:
            continue
        if re.fullmatch(r"(?:19|20)\d{2}", value.strip()):
            continue
        if value not in corpus_numbers and value.rstrip("0").rstrip(".") not in {
            c.rstrip("0").rstrip(".") for c in corpus_numbers
        }:
            missing.append(raw.strip())
    seen: set[str] = set()
    return [m for m in missing if not (m in seen or seen.add(m))]


def citation_coverage(answer: str) -> float:
    """Fraction of sentences carrying at least one citation marker."""
    sentences = [s for s in _SENTENCE.split(answer.strip()) if len(s.strip()) > 25]
    if not sentences:
        return 1.0
    cited = sum(1 for s in sentences if re.search(r"\[\d{1,2}\]", s))
    return cited / len(sentences)


async def check(
    answer: str,
    chunks: Sequence[RetrievedChunk],
    *,
    llm: LLMClient | None,
    usage: Usage | None = None,
) -> Grounding | None:
    """Grade how well the answer is supported. ``None`` if not checkable."""
    if not answer.strip() or not chunks:
        return None

    bad_numbers = unsupported_numbers(answer, chunks)
    coverage = citation_coverage(answer)

    if llm is None:
        # Mechanical-only verdict, clearly labelled as such.
        score = 90
        notes = []
        if bad_numbers:
            score -= min(15 * len(bad_numbers), 50)
            notes.append(f"figures not found in the sources: {', '.join(bad_numbers[:4])}")
        if coverage < 0.6:
            score -= 20
            notes.append(f"only {coverage:.0%} of sentences carry a citation")
        return Grounding.from_score(
            max(score, 10),
            unsupported_claims=notes,
            summary="Checked mechanically only; no model was available for a claim-level review.",
        )

    numbered = "\n\n".join(rc.to_context_block(i + 1) for i, rc in enumerate(chunks))
    reply = await llm.structured(
        purpose="grounding",
        system=GROUNDING_SYSTEM,
        user=f"ANSWER TO CHECK:\n{answer}\n\nSOURCES:\n{numbered}",
        schema=_GroundingReply,
        max_tokens=700,
        temperature=0.0,
        usage=usage,
    )
    if reply is None:
        return Grounding.from_score(
            50,
            summary="The grounding check could not be completed for this answer.",
        )

    score = reply.score
    unsupported = list(reply.unsupported_claims)
    # The mechanical findings are additive: a number absent from every source
    # is a fact, not an opinion, so it caps the score regardless of the
    # model's verdict.
    if bad_numbers:
        unsupported.append(
            f"Figures not present in any source: {', '.join(bad_numbers[:5])}"
        )
        score = min(score, 72)
    if coverage < 0.5:
        unsupported.append(f"Only {coverage:.0%} of sentences carry a citation.")
        score = min(score, 80)

    grounding = Grounding.from_score(
        max(0, min(100, score)),
        supported_claims=reply.supported_claims,
        unsupported_claims=unsupported,
        summary=reply.summary or "",
    )
    log.info(
        "grounding checked",
        extra={
            "score": grounding.score,
            "verdict": grounding.verdict,
            "bad_numbers": len(bad_numbers),
            "citation_coverage": round(coverage, 2),
        },
    )
    return grounding
