"""
Answer-quality evaluation.

Retrieval metrics measure whether the right passage was found. They say
nothing about whether the answer used it, cited it honestly, or addressed
the question — and those are what a reader actually experiences. This scores
the generated answer.

Five measures, deliberately split into two kinds:

**Mechanical** — computed from the text, no model involved, so they cannot
drift or flatter:

* ``citation_coverage`` — share of substantive sentences carrying a marker.
  Low coverage means claims are being made without attribution.
* ``raw_citation_validity`` — share of the source numbers *the model wrote*
  that were actually in the prompt, measured before the pipeline strips the
  invalid ones. This is the hallucinated-citation rate.
* ``citation_validity`` — the same check on the answer the user receives.
  The pipeline strips invalid markers, so this is 1.0 by construction; it is
  kept as a regression check on the stripping, not as a quality score.
* ``quote_verifiability`` — supporting quotes are verified at generation
  time and unverifiable ones dropped, so this too is 1.0 by construction.
  The interesting number is ``quotes_dropped``.
* ``token_f1`` / ``token_recall`` — overlap with the annotators' reference
  answers, SQuAD-normalised, best over references (see
  :mod:`evaluation.textmatch`). Only cases with references are averaged.
* ``anchor_recall`` — share of the golden case's expected phrases that
  appear in the answer. A weak proxy for correctness, but a real one: if the
  case says the answer must mention a figure and it does not, something is
  wrong regardless of how fluent the prose is.

**Judged** — one model call per answer:

* ``groundedness`` — is every claim supported by the sources?
* ``relevance`` — does the answer address the question that was asked?

A note on the judged pair, because this is the part of RAG evaluation most
often oversold: using a language model to grade a language model shares the
failure modes of both. The judge is systematically kinder to fluent, hedged,
well-structured answers than a careful human would be, and it cannot catch a
confident error on a subject it is also wrong about. They are reported
separately from the mechanical scores, and the mechanical scores are the
ones to trust when the two disagree. The judge uses a different prompt and
temperature 0, and is never shown the expected answer, so it cannot simply
pattern-match to a target.
"""

from __future__ import annotations

import asyncio
import json
import re
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from evaluation.textmatch import token_scores
from vectora.config import SearchMode, Settings
from vectora.observability import configure_logging, get_logger, request_context

log = get_logger(__name__)

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_MARKER = re.compile(r"\[(\d{1,2})(?:\s*[,\-–]\s*\d{1,2})*\]")


class _RelevanceVerdict(BaseModel):
    addresses_question: int = Field(ge=0, le=100)
    completeness: int = Field(ge=0, le=100)
    reasoning: str = ""


_RELEVANCE_SYSTEM = """You judge whether an answer addresses the question it was asked. You are not judging whether it is factually correct, well written, or complete relative to any ideal answer.

Score two things 0-100:

addresses_question — does the answer respond to what was actually asked? An answer about a neighbouring topic scores low even if it is accurate. An answer that correctly states the sources do not cover the question scores HIGH: declining is a valid response and must not be penalised.

completeness — does it cover the parts of the question that were asked? A question with two parts answered in one part scores around 50. Do not penalise brevity when the question was narrow.

Keep the reasoning to one sentence."""


@dataclass
class AnswerScore:
    """Per-case answer measurements."""

    case_id: str
    query: str
    answered: bool
    refused: bool
    expected_refusal: bool = False

    citation_coverage: float = 0.0
    citation_validity: float = 1.0
    quote_verifiability: float = 1.0
    anchor_recall: float = 0.0
    citations: int = 0
    invalid_citations: int = 0
    raw_citation_validity: float | None = None
    raw_markers: int = 0
    hallucinated_markers: int = 0
    token_f1: float | None = None
    token_recall: float | None = None

    groundedness: int | None = None
    relevance: int | None = None
    completeness: int | None = None

    answer_chars: int = 0
    #: Kept so answers can be hand-labelled (evaluation/judge_agreement.py).
    answer_text: str = ""
    cited_chunk_ids: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    llm_calls: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def correct_refusal_behaviour(self) -> bool:
        """Did it answer when it should and decline when it should?"""
        return self.refused == self.expected_refusal


@dataclass
class AnswerReport:
    corpus: str = ""
    mode: str = ""
    scores: list[AnswerScore] = field(default_factory=list)

    def _mean(self, key: str, *, only_answered: bool = True) -> float:
        pool = [s for s in self.scores if not s.refused] if only_answered else self.scores
        values = [getattr(s, key) for s in pool if getattr(s, key) is not None]
        return sum(values) / len(values) if values else 0.0

    @property
    def answered(self) -> list[AnswerScore]:
        return [s for s in self.scores if not s.refused]

    @property
    def refusal_accuracy(self) -> float:
        if not self.scores:
            return 0.0
        return sum(1 for s in self.scores if s.correct_refusal_behaviour) / len(
            self.scores
        )

    @property
    def hallucinated_rate(self) -> float:
        """Invalid markers / all markers the model wrote, pooled over answers."""
        raw = sum(s.raw_markers for s in self.answered)
        return sum(s.hallucinated_markers for s in self.answered) / raw if raw else 0.0

    @property
    def total_cost(self) -> float:
        return sum(s.cost_usd for s in self.scores)

    @property
    def median_latency(self) -> float:
        values = [s.latency_ms for s in self.scores if s.latency_ms]
        return statistics.median(values) if values else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "corpus": self.corpus,
            "mode": self.mode,
            "cases": len(self.scores),
            "answered": len(self.answered),
            "refused": len(self.scores) - len(self.answered),
            "mechanical": {
                "citation_coverage": round(self._mean("citation_coverage"), 4),
                "raw_citation_validity": round(self._mean("raw_citation_validity"), 4),
                "hallucinated_citation_rate": round(self.hallucinated_rate, 4),
                "answers_with_hallucinated_citation": sum(
                    1 for s in self.answered if s.hallucinated_markers
                ),
                "citation_validity": round(self._mean("citation_validity"), 4),
                "token_f1": round(self._mean("token_f1"), 4),
                "token_recall": round(self._mean("token_recall"), 4),
                "cases_with_reference": sum(
                    1 for s in self.answered if s.token_f1 is not None
                ),
                "quote_verifiability": round(self._mean("quote_verifiability"), 4),
                "anchor_recall": round(self._mean("anchor_recall"), 4),
                "refusal_accuracy": round(self.refusal_accuracy, 4),
            },
            "judged": {
                "groundedness": round(self._mean("groundedness"), 1),
                "relevance": round(self._mean("relevance"), 1),
                "completeness": round(self._mean("completeness"), 1),
            },
            "cost": {
                "total_usd": round(self.total_cost, 5),
                "per_question_usd": round(
                    self.total_cost / len(self.scores) if self.scores else 0.0, 5
                ),
                "median_latency_ms": round(self.median_latency, 0),
            },
            "per_case": [asdict(s) for s in self.scores],
        }

    def summary(self) -> str:
        n = len(self.scores)
        answered = len(self.answered)
        lines = [
            "",
            "=" * 72,
            f"  ANSWER QUALITY — {self.corpus}, {self.mode} mode",
            "=" * 72,
            f"  cases {n}   answered {answered}   refused {n - answered}",
            "",
            "  Mechanical (computed from the text — trust these first)",
            f"    citation coverage      {self._mean('citation_coverage'):.1%}"
            "   share of substantive sentences with a marker",
            f"    raw citation validity  {self._mean('raw_citation_validity'):.1%}"
            "   markers the model wrote that pointed at a real source",
            f"    hallucinated markers   {self.hallucinated_rate:.1%}"
            "   (stripped before the user sees the answer)",
            f"    citation validity      {self._mean('citation_validity'):.1%}"
            "   after stripping — 100% by construction",
            f"    answer token F1        {self._mean('token_f1'):.1%}"
            "   vs annotators' answers (best reference)",
            f"    answer token recall    {self._mean('token_recall'):.1%}"
            "   share of the reference the answer contains",
            f"    anchor recall          {self._mean('anchor_recall'):.1%}"
            "   expected phrases present in the answer",
            f"    refusal accuracy       {self.refusal_accuracy:.1%}"
            "   answered when it should, declined when it should",
            "",
            "  Judged by a model (indicative — see the module docstring)",
            f"    groundedness           {self._mean('groundedness'):.0f}/100",
            f"    relevance              {self._mean('relevance'):.0f}/100",
            f"    completeness           {self._mean('completeness'):.0f}/100",
            "",
            "  Cost",
            f"    total                  ${self.total_cost:.4f}"
            f"  (${self.total_cost / n if n else 0:.5f} per question)",
            f"    median latency         {self.median_latency:.0f} ms",
        ]

        invalid = [s for s in self.scores if s.hallucinated_markers or s.invalid_citations]
        if invalid:
            lines += [
                "",
                "  Answers where the model cited a source that was not provided:",
            ]
            for s in invalid[:6]:
                lines.append(
                    f"    {s.hallucinated_markers or s.invalid_citations} invalid in: {s.query[:50]}"
                )

        wrong = [s for s in self.scores if not s.correct_refusal_behaviour]
        if wrong:
            lines += ["", "  Wrong answer/refusal decision:"]
            for s in wrong[:8]:
                what = "refused a fair question" if s.refused else "answered an off-corpus question"
                lines.append(f"    {what}: {s.query[:52]}")

        lines += ["=" * 72, ""]
        return "\n".join(lines)


def citation_coverage(answer: str) -> float:
    """Share of substantive sentences carrying at least one citation."""
    sentences = [s for s in _SENTENCE.split(answer.strip()) if len(s.strip()) > 30]
    if not sentences:
        return 1.0
    return sum(1 for s in sentences if _MARKER.search(s)) / len(sentences)


def anchor_recall(answer: str, anchors: list[str]) -> float:
    """Share of the case's expected phrases present in the answer.

    Substring matching on normalised text. Crude, and that is deliberate:
    anything cleverer would start accepting paraphrases, and the point of
    this measure is to be the one score that cannot be argued with.
    """
    if not anchors:
        return 1.0
    haystack = re.sub(r"\s+", " ", answer.lower())
    hits = sum(1 for a in anchors if re.sub(r"\s+", " ", a.lower()) in haystack)
    return hits / len(anchors)


async def _judge_relevance(
    query: str, answer: str, *, llm: Any
) -> _RelevanceVerdict | None:
    return await llm.structured(
        purpose="judge_relevance",
        system=_RELEVANCE_SYSTEM,
        user=f"QUESTION:\n{query}\n\nANSWER:\n{answer}",
        schema=_RelevanceVerdict,
        max_tokens=250,
        temperature=0.0,
    )


async def score_case(
    container: Any,
    *,
    case_id: str,
    query: str,
    anchors: list[str],
    expected_refusal: bool,
    mode: SearchMode,
    judge: bool,
    references: list[str] | None = None,
) -> AnswerScore:
    """Answer one question and score the result."""
    import time

    score = AnswerScore(
        case_id=case_id,
        query=query,
        answered=False,
        refused=False,
        expected_refusal=expected_refusal,
    )

    start = time.perf_counter()
    try:
        with request_context():
            result = await container.engine.answer(query, mode=mode)
    except Exception as exc:
        score.notes.append(f"answer failed: {exc}")
        log.warning("answer failed", extra={"case": case_id, "err": str(exc)[:140]})
        return score
    score.latency_ms = (time.perf_counter() - start) * 1000

    score.answered = True
    score.refused = result.fallback is not None
    score.answer_chars = len(result.answer)
    score.answer_text = result.answer
    score.cited_chunk_ids = [c.chunk_id for c in result.citations]
    score.cost_usd = result.usage.cost_usd
    score.llm_calls = result.usage.calls

    if score.refused:
        # A refusal is scored on whether it was the right call, not on
        # citation metrics it cannot have.
        score.citation_coverage = 1.0
        score.anchor_recall = 1.0 if expected_refusal else 0.0
        if judge and container.llm is not None:
            verdict = await _judge_relevance(query, result.answer, llm=container.llm)
            if verdict:
                score.relevance = verdict.addresses_question
                score.completeness = verdict.completeness
        return score

    score.citations = len(result.citations)
    score.citation_coverage = citation_coverage(result.answer)
    score.anchor_recall = anchor_recall(result.answer, anchors)

    if references:
        score.token_f1, score.token_recall = token_scores(result.answer, references)

    # Pre-strip validity: what the model actually wrote.
    raw = list(getattr(result, "raw_citation_markers", []) or [])
    bad = list(getattr(result, "invalid_citation_markers", []) or [])
    score.raw_markers = len(raw)
    score.hallucinated_markers = len(bad)
    score.raw_citation_validity = 1.0 - len(bad) / len(raw) if raw else 1.0
    if bad:
        score.notes.append(f"model cited absent sources (stripped): {sorted(bad)}")

    # Citation validity: the pipeline strips invalid markers before the
    # answer is returned, so a surviving marker pointing nowhere would be a
    # pipeline bug. Checking anyway is how we would find out.
    diagnostics = result.diagnostics
    provided = diagnostics.returned if diagnostics else len(result.citations)
    cited_numbers = {
        int(m.group(1)) for m in _MARKER.finditer(result.answer) if m.group(1).isdigit()
    }
    invalid = {n for n in cited_numbers if n < 1 or (provided and n > provided)}
    score.invalid_citations = len(invalid)
    score.citation_validity = (
        1.0 - len(invalid) / len(cited_numbers) if cited_numbers else 1.0
    )
    if invalid:
        score.notes.append(f"cited absent sources: {sorted(invalid)}")

    quoted = [c for c in result.citations if c.quote]
    if quoted:
        # Quotes are verified at generation time, so every quote present
        # here has already passed. This records the share that survived.
        score.quote_verifiability = 1.0
        score.notes.append(f"{len(quoted)} verified quote(s)")

    if result.grounding:
        score.groundedness = result.grounding.score

    if judge and container.llm is not None:
        verdict = await _judge_relevance(query, result.answer, llm=container.llm)
        if verdict:
            score.relevance = verdict.addresses_question
            score.completeness = verdict.completeness

    return score


async def run(
    settings: Settings,
    *,
    mode: SearchMode = SearchMode.DEEP,
    limit: int | None = None,
    judge: bool = True,
    corpus: str | None = None,
    output: Path | None = None,
) -> AnswerReport:
    """Score generated answers over the golden set.

    Defaults to ``deep`` mode because that is the configuration with the
    grounding check and verified quotes — the things this is measuring. It
    makes one generation call per case plus one judge call, so it costs real
    money: ``--limit`` exists for that reason.
    """
    configure_logging(settings)
    from evaluation.run_eval import load_dataset
    from vectora.api.deps import build_container

    container = await asyncio.to_thread(
        build_container, settings, warm=True, corpus=corpus
    )
    try:
        ctx = container.corpus
        if container.llm is None:
            raise RuntimeError(
                "Answer evaluation needs an OpenAI key: set OPENAI_API_KEY."
            )
        if not container.store.exists():
            raise RuntimeError("Nothing is indexed. Run `vectora index` first.")
        if not ctx.golden.exists():
            raise RuntimeError(
                f"No golden set at {ctx.golden}. Draft one with "
                f"`vectora golden generate`."
            )

        cases, negatives = load_dataset(ctx.golden)
        if limit:
            cases = cases[:limit]
            negatives = negatives[: max(1, limit // 4)]

        total = len(cases) + len(negatives)
        estimate = total * (0.0012 if mode is SearchMode.DEEP else 0.0007)
        print(
            f"\nScoring {len(cases)} questions and {len(negatives)} negatives "
            f"in {mode.value} mode."
        )
        print(
            f"This makes {total * (3 if judge else 2)}-ish OpenAI calls, "
            f"roughly ${estimate:.3f} on gpt-4o-mini.\n"
        )

        report = AnswerReport(corpus=ctx.name, mode=mode.value)
        for i, case in enumerate(cases, start=1):
            print(f"  [{i}/{total}] {case.query[:62]}")
            report.scores.append(
                await score_case(
                    container,
                    case_id=case.id,
                    query=case.query,
                    anchors=list(case.must_contain),
                    expected_refusal=False,
                    mode=mode,
                    judge=judge,
                    references=list(case.reference_answers) or None,
                )
            )
        for j, negative in enumerate(negatives, start=len(cases) + 1):
            print(f"  [{j}/{total}] (negative) {negative.query[:52]}")
            report.scores.append(
                await score_case(
                    container,
                    case_id=negative.id,
                    query=negative.query,
                    anchors=[],
                    expected_refusal=True,
                    mode=mode,
                    judge=judge,
                )
            )

        print(report.summary())

        ctx.reports.mkdir(parents=True, exist_ok=True)
        target = output or ctx.reports / f"answers-{mode.value}.json"
        payload = report.as_dict()
        payload["generated_at"] = datetime.now(timezone.utc).isoformat()
        payload["judged_by"] = settings.utility_model if judge else None
        target.write_text(json.dumps(payload, indent=2), "utf-8")
        print(f"Wrote {target}\n")
        return report
    finally:
        await container.aclose()
