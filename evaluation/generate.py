"""
Draft a golden set from an indexed corpus.

Writing fifty evaluation cases by hand is the step that stops people
evaluating their RAG system at all, so it gets skipped and every tuning
decision afterwards is guesswork. This drafts the set instead: sample
chunks from the corpus, ask a model what question each one answers, then
**verify by retrieval** that the question actually finds the chunk it came
from.

That verification step is the whole point. A question generated from a
passage is not automatically a good test case — it may be answerable from
twenty other chunks, or so specific that it quotes the passage back. Each
candidate is run through the real retrieval pipeline and labelled:

* ``verified``   — retrieval finds the source chunk in the top k. Keep.
* ``weak``       — found, but ranked low. Keep and flag; these are the
                   interesting ones, because they are near the system's edge.
* ``unverified`` — retrieval does not find it at all. Either the question is
                   bad or retrieval is bad, and you cannot tell which from
                   here, so it is written out commented and left for review.

Negative cases are generated differently: there is no passage to anchor
them, so the model is asked for questions about subjects *adjacent* to the
corpus's own topics, then each is checked to confirm the guard actually
refuses it. One that gets answered is reported loudly — it is a real finding
about the system, not a bad test case.

Nothing here is trusted blindly. The output is a file for a human to read,
with every case's provenance recorded, which is why it writes YAML comments
rather than a tidy machine-generated blob.
"""

from __future__ import annotations

import asyncio
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from vectora.config import QueryType, SearchMode, Settings
from vectora.models import Chunk
from vectora.observability import configure_logging, get_logger, request_context

log = get_logger(__name__)

#: Chunks below this quality are not worth building a test case from: a
#: question generated from layout debris tests nothing.
MIN_QUALITY = 0.7
#: A chunk needs enough substance to support a specific question.
MIN_TOKENS = 60


class _GeneratedCase(BaseModel):
    question: str = ""
    answer_phrase: str = Field(
        default="",
        description="A short verbatim phrase from the passage that answers it",
    )
    query_type: str = ""


class _GeneratedNegatives(BaseModel):
    questions: list[str] = Field(default_factory=list, max_length=12)


class _CorpusTopics(BaseModel):
    topics: list[str] = Field(default_factory=list, max_length=20)
    adjacent_topics: list[str] = Field(default_factory=list, max_length=20)


_CASE_SYSTEM = """You write evaluation questions for a document retrieval system.

Given one passage from a document, write the single most natural question that a reader of this collection would ask, and that THIS passage specifically answers.

Rules:
- The question must be answerable from the passage alone.
- It must be specific enough that a generic passage on the topic would not answer it.
- It must NOT quote the passage. Someone who has not read it should be able to ask it.
- Do not reference "the passage", "the document", "this study" or "the authors" — ask about the subject matter directly.
- Also return a short verbatim phrase, 3 to 12 words, copied exactly from the passage, that contains the answer.
- Classify the question as one of: factual, definitional, methodological, causal, comparative, exploratory."""


_TOPIC_SYSTEM = """You summarise what a document collection covers.

From the sample of passages given, list:

- topics: up to 12 specific subjects this collection genuinely covers, each 2-6 words
- adjacent_topics: up to 8 subjects from the SAME broad field that this sample shows NO evidence of covering. These should be plausible things someone might wrongly assume are in the collection — near misses, not absurd ones.

Judge only from the sample. Do not infer that a subject is covered because it is related to one that is."""


_NEGATIVE_SYSTEM = """You write questions that a document collection CANNOT answer, to test whether a retrieval system correctly declines rather than inventing an answer.

You will be given the topics a collection covers and topics it does not. Write questions that:
- are specific and well-formed, as a real user would ask them
- concern the listed adjacent topics, or are plainly from another field entirely
- are NOT answerable from the covered topics, even partially

Mix difficulty: some obviously unrelated, most from the adjacent list — those are the ones that actually test the system."""


@dataclass
class Candidate:
    """One drafted case plus what retrieval said about it."""

    case_id: str
    query: str
    query_type: str
    must_contain: list[str]
    source_paper: str
    source_chunk: str
    source_section: str
    status: str = "unverified"
    rank: int | None = None
    top_score: float | None = None
    note: str = ""

    @property
    def keep(self) -> bool:
        return self.status in {"verified", "weak"}


@dataclass
class NegativeCandidate:
    case_id: str
    query: str
    refused: bool = False
    top_score: float | None = None

    @property
    def keep(self) -> bool:
        return self.refused


@dataclass
class GenerationReport:
    sampled: int = 0
    drafted: int = 0
    verified: int = 0
    weak: int = 0
    unverified: int = 0
    negatives_drafted: int = 0
    negatives_refused: int = 0
    negatives_answered: list[str] = field(default_factory=list)
    by_type: Counter[str] = field(default_factory=Counter)

    def summary(self) -> str:
        lines = [
            "",
            "=" * 66,
            "  GOLDEN SET DRAFT",
            "=" * 66,
            f"  chunks sampled        {self.sampled}",
            f"  questions drafted     {self.drafted}",
            f"    verified            {self.verified}  (source chunk in top k)",
            f"    weak                {self.weak}  (found, ranked low — kept and flagged)",
            f"    unverified          {self.unverified}  (not found — written commented out)",
            f"  negatives drafted     {self.negatives_drafted}",
            f"    correctly refused   {self.negatives_refused}",
        ]
        if self.by_type:
            lines.append(
                "  by query type         "
                + ", ".join(f"{k} {v}" for k, v in self.by_type.most_common())
            )
        if self.negatives_answered:
            lines += [
                "",
                "  These off-topic questions were ANSWERED rather than refused.",
                "  That is a finding about the system, not a bad test case —",
                "  run `vectora calibrate` and reconsider the refusal floors:",
            ]
            for q in self.negatives_answered[:8]:
                lines.append(f"    {q[:70]}")
        lines += ["=" * 66, ""]
        return "\n".join(lines)


def _slug(text: str, used: set[str]) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    base = "-".join(base.split("-")[:5])[:40] or "case"
    candidate = base
    n = 2
    while candidate in used:
        candidate = f"{base}-{n}"
        n += 1
    used.add(candidate)
    return candidate


def sample_chunks(chunks: list[Chunk], n: int, seed: int = 7) -> list[Chunk]:
    """Pick ``n`` chunks worth generating from, spread across papers.

    Stratified by paper so a long document cannot dominate the eval set,
    which would make the metrics measure one paper's retrievability rather
    than the corpus's.
    """
    eligible = [
        c
        for c in chunks
        if c.quality >= MIN_QUALITY
        and c.token_estimate >= MIN_TOKENS
        and c.chunk_type == "text"
    ]
    if not eligible:
        return []

    by_paper: dict[str, list[Chunk]] = {}
    for chunk in eligible:
        by_paper.setdefault(chunk.paper_id, []).append(chunk)

    rng = random.Random(seed)
    for group in by_paper.values():
        rng.shuffle(group)

    picked: list[Chunk] = []
    papers = sorted(by_paper)
    rng.shuffle(papers)
    round_index = 0
    while len(picked) < n:
        added = False
        for paper in papers:
            group = by_paper[paper]
            if round_index < len(group):
                picked.append(group[round_index])
                added = True
                if len(picked) >= n:
                    break
        if not added:
            break
        round_index += 1
    return picked[:n]


async def _draft_case(
    chunk: Chunk, *, llm: Any, used_ids: set[str]
) -> Candidate | None:
    passage = chunk.content[:2400]
    result = await llm.structured(
        purpose="golden_case",
        system=_CASE_SYSTEM,
        user=(
            f"Document title: {chunk.title}\n"
            f"Section: {chunk.section or 'unknown'}\n\n"
            f"PASSAGE:\n{passage}"
        ),
        schema=_GeneratedCase,
        max_tokens=250,
        temperature=0.4,
    )
    if result is None or not result.question.strip():
        return None

    question = result.question.strip()
    if len(question) < 15 or "?" not in question:
        return None

    # The anchor phrase must really be in the passage; a model that
    # paraphrased it would produce a case that can never verify.
    phrase = result.answer_phrase.strip().strip('"')
    anchors: list[str] = []
    if phrase and _normalise(phrase) in _normalise(chunk.content):
        anchors.append(phrase)
    else:
        anchors.append(_salient_term(chunk.content))

    qtype = result.query_type.strip().lower()
    if qtype not in {q.value for q in QueryType}:
        qtype = ""

    return Candidate(
        case_id=_slug(question, used_ids),
        query=question,
        query_type=qtype,
        must_contain=[a for a in anchors if a],
        source_paper=chunk.paper_id,
        source_chunk=chunk.chunk_id,
        source_section=chunk.section,
    )


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _salient_term(text: str) -> str:
    """Fallback anchor: the longest distinctive token in the passage.

    Used when the model's quote could not be verified. Weaker than a real
    phrase, which is why the case still has to pass retrieval verification.
    """
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9/\-]{5,}", text)
    stop = {"however", "therefore", "although", "because", "between", "respectively"}
    candidates = [t for t in tokens if t.lower() not in stop]
    return max(candidates, key=len)[:40] if candidates else ""


async def generate(
    settings: Settings,
    *,
    count: int = 40,
    negatives: int = 10,
    mode: SearchMode = SearchMode.BALANCED,
    top_k: int = 5,
    output: Path | None = None,
    corpus: str | None = None,
    seed: int = 7,
) -> GenerationReport:
    """Draft and verify a golden set for the active corpus."""
    configure_logging(settings)
    from vectora.api.deps import build_container
    from vectora.generation.guards import check as guard_check
    from vectora.ingestion.pipeline import read_chunks

    container = await asyncio.to_thread(
        build_container, settings, warm=True, corpus=corpus
    )
    report = GenerationReport()
    try:
        if container.llm is None:
            raise RuntimeError(
                "Generating a golden set needs an OpenAI key: set OPENAI_API_KEY."
            )
        if not container.store.exists():
            raise RuntimeError(
                "The corpus is not indexed. Run `vectora index` first — the "
                "generator verifies each question against real retrieval."
            )

        ctx = container.corpus
        target = output or ctx.golden
        chunks = list(read_chunks(ctx.chunks))  # type: ignore[arg-type]
        if not chunks:
            raise RuntimeError(f"No chunks at {ctx.chunks}")

        picked = sample_chunks(chunks, count, seed=seed)
        report.sampled = len(picked)
        if not picked:
            raise RuntimeError(
                f"No chunks met the quality floor ({MIN_QUALITY}) and size floor "
                f"({MIN_TOKENS} tokens). Check the ingestion report."
            )

        print(f"\nDrafting {len(picked)} questions from {ctx.name}…")
        used_ids: set[str] = set()
        drafted = await asyncio.gather(
            *(_draft_case(c, llm=container.llm, used_ids=used_ids) for c in picked)
        )
        candidates = [c for c in drafted if c is not None]
        report.drafted = len(candidates)

        print(f"Verifying {len(candidates)} against retrieval…")
        for candidate in candidates:
            with request_context():
                found, _ = await container.engine.retriever.retrieve(
                    candidate.query,
                    mode=mode,
                    query_type=(
                        QueryType(candidate.query_type)
                        if candidate.query_type
                        else QueryType.EXPLORATORY
                    ),
                    top_k=max(top_k, 8),
                )
            position = next(
                (
                    i
                    for i, rc in enumerate(found, start=1)
                    if rc.chunk.chunk_id == candidate.source_chunk
                ),
                None,
            )
            candidate.top_score = round(found[0].score, 4) if found else None
            if position is None:
                candidate.status = "unverified"
                candidate.note = "source chunk not retrieved for this question"
                report.unverified += 1
            elif position <= top_k:
                candidate.status = "verified"
                candidate.rank = position
                report.verified += 1
            else:
                candidate.status = "weak"
                candidate.rank = position
                candidate.note = f"source chunk only at rank {position}"
                report.weak += 1
            if candidate.keep and candidate.query_type:
                report.by_type[candidate.query_type] += 1

        negative_cases: list[NegativeCandidate] = []
        if negatives > 0:
            negative_cases = await _generate_negatives(
                container,
                picked,
                wanted=negatives,
                mode=mode,
                report=report,
                guard_check=guard_check,
            )

        _write_yaml(target, ctx, candidates, negative_cases, mode=mode, top_k=top_k)
        print(report.summary())
        print(f"Wrote {target}")
        print(
            "Read it before trusting it. The cases marked `weak` sit at the edge "
            "of what retrieval finds and are the most informative ones to keep; "
            "the commented-out block at the end needs a human decision.\n"
        )
        return report
    finally:
        await container.aclose()


async def _generate_negatives(
    container: Any,
    sampled: list[Chunk],
    *,
    wanted: int,
    mode: SearchMode,
    report: GenerationReport,
    guard_check: Any,
) -> list[NegativeCandidate]:
    """Draft off-corpus questions and confirm the guard refuses them."""
    excerpt = "\n\n".join(
        f"[{c.title[:60]}] {c.content[:300]}" for c in sampled[:14]
    )
    topics = await container.llm.structured(
        purpose="golden_topics",
        system=_TOPIC_SYSTEM,
        user=f"Sample of passages from the collection:\n\n{excerpt}",
        schema=_CorpusTopics,
        max_tokens=600,
        temperature=0.3,
    )
    if topics is None:
        log.warning("could not infer corpus topics; skipping negatives")
        return []

    drafted = await container.llm.structured(
        purpose="golden_negatives",
        system=_NEGATIVE_SYSTEM,
        user=(
            "Covered topics:\n"
            + "\n".join(f"- {t}" for t in topics.topics)
            + "\n\nAdjacent topics NOT covered:\n"
            + "\n".join(f"- {t}" for t in topics.adjacent_topics)
            + f"\n\nWrite {wanted} questions this collection cannot answer."
        ),
        schema=_GeneratedNegatives,
        max_tokens=140 * wanted,
        temperature=0.6,
    )
    if drafted is None:
        return []

    used: set[str] = set()
    out: list[NegativeCandidate] = []
    for question in drafted.questions[:wanted]:
        question = question.strip()
        if len(question) < 15 or "?" not in question:
            continue
        candidate = NegativeCandidate(
            case_id=_slug(question, used), query=question
        )
        with request_context():
            found, _ = await container.engine.retriever.retrieve(
                question, mode=mode, query_type=QueryType.EXPLORATORY
            )
        ok, _ = guard_check(found, container.settings)
        candidate.refused = not ok
        candidate.top_score = round(found[0].score, 4) if found else None
        report.negatives_drafted += 1
        if candidate.refused:
            report.negatives_refused += 1
        else:
            report.negatives_answered.append(question)
        out.append(candidate)
    return out


def _esc(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _write_yaml(
    path: Path,
    ctx: Any,
    candidates: list[Candidate],
    negatives: list[NegativeCandidate],
    *,
    mode: SearchMode,
    top_k: int,
) -> None:
    """Write the drafted set, keeping provenance as comments.

    Hand-written rather than dumped through a YAML library so the comments
    survive: a generated eval set that does not say where each case came
    from, or how confident it is, invites being trusted more than it should
    be.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    kept = [c for c in candidates if c.keep]
    rejected = [c for c in candidates if not c.keep]

    lines: list[str] = [
        f"# Golden set for the `{ctx.name}` corpus — DRAFTED, needs review.",
        "#",
        f"# Generated by `vectora golden generate` against {ctx.collection},",
        f"# verified in {mode.value} mode at top_k={top_k}.",
        "#",
        "# Each case was drafted from one passage and then checked by running it",
        "# through real retrieval. `rank` is where the source passage came back.",
        "# Cases that could not be verified are at the end, commented out.",
        "#",
        "# Review this file. Delete questions that are not natural, tighten",
        "# must_contain where it is too loose, and add your own cases whenever",
        "# you find a question the system gets wrong — those are worth more than",
        "# anything generated here.",
        "",
        "cases:",
    ]

    for c in kept:
        lines.append(f"  - id: {c.case_id}")
        lines.append(f'    query: "{_esc(c.query)}"')
        if c.query_type:
            lines.append(f"    query_type: {c.query_type}")
        lines.append("    must_contain:")
        for anchor in c.must_contain:
            lines.append(f'      - "{_esc(anchor)}"')
        lines.append(f"    relevant_chunks: [{c.source_chunk}]")
        detail = f"from {c.source_paper}"
        if c.source_section:
            detail += f" ({c.source_section})"
        detail += f"; status={c.status}"
        if c.rank:
            detail += f", source at rank {c.rank}"
        lines.append(f'    notes: "{_esc(detail)}"')
        lines.append("")

    lines.append("negative_cases:")
    if negatives:
        for n in negatives:
            prefix = "  " if n.keep else "  # "
            if not n.keep:
                lines.append(
                    f"  # NOT REFUSED (top score {n.top_score}) — the system answered"
                )
                lines.append(
                    "  # this. Fix the thresholds rather than deleting the case."
                )
            lines.append(f"{prefix}- id: {n.case_id}")
            lines.append(f'{prefix}  query: "{_esc(n.query)}"')
            lines.append(f"{prefix}  expect: fallback")
            lines.append("")
    else:
        lines += [
            "  # None generated. Add questions your corpus cannot answer —",
            "  # the ranking metrics cannot see whether the system declines.",
            "",
        ]

    if rejected:
        lines += [
            "",
            "# ─── UNVERIFIED ───────────────────────────────────────────────────",
            "#",
            "# Retrieval did not surface the passage each of these was written",
            "# from. That means either the question is bad or retrieval is — and",
            "# you cannot tell which from here. Read the question against its",
            "# source passage and decide: a fair question that retrieval misses",
            "# is the most valuable case in the whole file.",
            "#",
        ]
        for c in rejected:
            lines.append(f"#  - id: {c.case_id}")
            lines.append(f'#    query: "{_esc(c.query)}"')
            if c.query_type:
                lines.append(f"#    query_type: {c.query_type}")
            lines.append("#    must_contain:")
            for anchor in c.must_contain:
                lines.append(f'#      - "{_esc(anchor)}"')
            lines.append(f"#    relevant_chunks: [{c.source_chunk}]")
            lines.append(
                f'#    notes: "from {_esc(c.source_paper)}; {_esc(c.note)}"'
            )
            lines.append("#")

    path.write_text("\n".join(lines) + "\n", "utf-8")
