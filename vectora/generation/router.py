"""
Query routing.

Classifies a question so retrieval and the prompt can be shaped for it.

Two changes from the previous implementation. First, a rule pass runs
*before* the LLM: most questions are classifiable from their interrogative
and a handful of cue words, and skipping a network round-trip on those
removes a step from the critical path for roughly half of real traffic.
Second, when the LLM is used the result is validated into an enum rather
than string-matched, and a failure falls back to the rules rather than
defaulting straight to ``exploratory`` — which was the previous behaviour
and is the worst default, since it pulls the largest and most diverse
candidate set.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from vectora.config import QueryType, Settings
from vectora.generation.prompts import ROUTER_SYSTEM
from vectora.llm import LLMClient
from vectora.models import Usage
from vectora.observability import get_logger

log = get_logger(__name__)


class _Classification(BaseModel):
    category: QueryType


#: Ordered, and the order is the whole design: the first pattern to match
#: wins, so the most specific cue must be tried first. ``DEFINITIONAL`` is
#: last among the "what is…" patterns because its shape — "what is <noun>?" —
#: also matches "what is the concentration threshold?" (factual) and "what
#: are the effects of X?" (causal), which are narrower readings and belong to
#: those types. Each pattern keys on how questions are phrased rather than on
#: topic vocabulary, so the rules stay corpus-independent.
_RULES: tuple[tuple[QueryType, re.Pattern[str]], ...] = (
    (
        QueryType.COMPARATIVE,
        re.compile(
            r"\b(compare|comparison|contrast|versus|vs\.?|difference(s)? between|"
            r"better than|worse than|which (is|are|one|method|assay|technique)|"
            r"advantages? and disadvantages?|trade[- ]?offs?|relative to)\b",
            re.I,
        ),
    ),
    (
        QueryType.METHODOLOGICAL,
        re.compile(
            r"\b(how (was|were|is|are|do|does|did) .{0,40}(measur|assay|quantif|"
            r"determin|detect|prepar|calibrat|validat|analys|analyz|synthesis|"
            r"fabricat|test)|what (method|methods|technique|techniques|protocol|"
            r"procedure|instrument|assay|approach)|methodolog|experimental "
            r"(design|setup|procedure)|sample preparation|statistical (test|analysis)|"
            r"limit of (detection|quantification))",
            re.I,
        ),
    ),
    (
        QueryType.CAUSAL,
        re.compile(
            r"\b(why\b|what causes?\b|caused by\b|what (is|are) the (effects?|impact|"
            r"consequences?)\b|lead(s)? to\b|result(s)? in\b|mechanism (of|by which)\b|"
            r"influence(s)? (on|of)\b|associated with\b|risk factors?\b|"
            r"what happens (if|when)\b)",
            re.I,
        ),
    ),
    (
        QueryType.EXPLORATORY,
        re.compile(
            r"\b(overview|summar(y|ise|ize)|review|state of the (art|field)|"
            r"tell me about|what do (we|the papers|the studies) know|"
            r"main (findings|themes|conclusions)|current (research|understanding|"
            r"approaches)|open questions|landscape|what research exists|"
            r"describe\b)",
            re.I,
        ),
    ),
    (
        QueryType.FACTUAL,
        re.compile(
            r"^\s*(what (is|are|was|were) the (value|level|concentration|range|"
            r"threshold|cut[- ]?off|prevalence|percentage|number|dose|dosage|"
            r"sensitivity|specificity|accuracy|precision|recovery)|"
            r"how (much|many|long|often)\b|when (was|were|did)\b|"
            r"who (was|were|first)\b|is it true\b|do(es)? .{0,30}\?$)",
            re.I,
        ),
    ),
    (
        QueryType.DEFINITIONAL,
        re.compile(
            r"^\s*(what (is|are) (a|an|the)?\s*[\w\s/()-]{1,40}\??$|"
            r"define\b|definition of\b|what does \S+ (stand for|mean)\b|"
            r"meaning of\b|what is meant by\b)",
            re.I,
        ),
    ),
)


def classify_by_rules(query: str) -> QueryType | None:
    """Rule-based classification, or ``None`` when no rule is confident."""
    text = query.strip()
    if not text:
        return None
    for query_type, pattern in _RULES:
        if pattern.search(text):
            return query_type
    return None


async def classify(
    query: str,
    *,
    llm: LLMClient | None,
    settings: Settings,
    usage: Usage | None = None,
) -> tuple[QueryType, str]:
    """Classify a question.

    Returns the type and how it was decided (``"rules"``, ``"llm"`` or
    ``"default"``), which goes into diagnostics.
    """
    by_rule = classify_by_rules(query)
    if by_rule is not None:
        log.debug("routed by rule", extra={"query_type": by_rule.value})
        return by_rule, "rules"

    if llm is not None:
        result = await llm.structured(
            purpose="route",
            system=ROUTER_SYSTEM,
            user=query,
            schema=_Classification,
            max_tokens=20,
            temperature=0.0,
            usage=usage,
        )
        if result is not None:
            log.debug("routed by llm", extra={"query_type": result.category.value})
            return result.category, "llm"

    # Nothing matched and the model was unavailable. Methodological is the
    # safer default than exploratory for this corpus — a middling top_k and
    # a middling lambda, rather than the widest, most diffuse retrieval.
    log.debug("routing fell back to default")
    return QueryType.METHODOLOGICAL, "default"
