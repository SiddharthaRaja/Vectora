"""
Retrieval evaluation.

Answers the question "did that change help?" with a number instead of a
feeling. Without this, every tuning decision — the RRF constant, the
reranker pool size, the MMR lambda, the chunk size — is guesswork, and
regressions are only noticed when a citation looks wrong weeks later.

Metrics, per mode:

* **recall@k** — did any relevant chunk make the top k? The headline number:
  if the right passage is not retrieved, no prompt can save the answer.
* **precision@k** — what fraction of what we returned was relevant? Low
  precision wastes context and invites the model to cite the wrong source.
* **MRR** — reciprocal rank of the first relevant chunk. Sensitive to
  whether the best chunk is at rank 1 or rank 5, which recall ignores.
* **nDCG@k** — rank-weighted, so improving the ordering shows up even when
  the set of retrieved chunks is unchanged.

What counts as "relevant" is the strictest thing the case specifies. A case
with ``relevant_evidence`` (the paragraphs annotators marked as supporting
the answer, as QASPER provides) only counts a chunk that contains one of
those paragraphs — finding the right paper but the wrong section is a miss.
``paper_recall`` is reported alongside, so the gap between "found the paper"
and "found the passage" is visible rather than hidden in one number.

Negative cases are scored separately: a system that answers "who is the
president" from a vitamin D corpus is broken no matter how good its recall
is, and that failure is invisible to the metrics above.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evaluation.textmatch import overlap, shingles
from vectora.config import QueryType, SearchMode, Settings
from vectora.models import RetrievedChunk
from vectora.observability import configure_logging, get_logger, request_context

log = get_logger(__name__)

#: Only a fallback: the golden set normally lives with its corpus, since an
#: eval set written for one corpus measures nothing on another.
DEFAULT_DATASET = Path(__file__).resolve().parent / "golden.template.yaml"


@dataclass
class Case:
    id: str
    query: str
    query_type: QueryType | None = None
    must_contain: list[str] = field(default_factory=list)
    relevant_papers: list[str] = field(default_factory=list)
    relevant_chunks: list[str] = field(default_factory=list)
    #: Annotator-marked evidence paragraphs. When present, a chunk is only
    #: relevant if it contains one of them (see :mod:`evaluation.textmatch`).
    relevant_evidence: list[str] = field(default_factory=list)
    #: Reference answers, for answer token F1. All annotators' answers.
    reference_answers: list[str] = field(default_factory=list)
    notes: str = ""
    _evidence_shingles: list[set] | None = field(default=None, repr=False, compare=False)

    def evidence_hit(self, rc: RetrievedChunk, threshold: float = 0.5) -> bool:
        if self._evidence_shingles is None:
            self._evidence_shingles = [shingles(e) for e in self.relevant_evidence if e.strip()]
        chunk = shingles(rc.chunk.content)
        return any(overlap(chunk, ev) >= threshold for ev in self._evidence_shingles)

    def paper_hit(self, rc: RetrievedChunk) -> bool:
        return bool(self.relevant_papers) and rc.chunk.paper_id in self.relevant_papers

    def is_relevant(self, rc: RetrievedChunk) -> bool:
        """Whether one retrieved chunk counts as relevant for this case.

        Evidence, when given, is the whole test: a chunk from the right paper
        that does not contain the evidence is not relevant.
        """
        if self.relevant_chunks and rc.chunk.chunk_id in self.relevant_chunks:
            return True
        if self.relevant_evidence:
            return self.evidence_hit(rc)
        if self.paper_hit(rc):
            return True
        if self.must_contain:
            haystack = f"{rc.chunk.content}\n{rc.chunk.title}\n{rc.chunk.section}".lower()
            return all(needle.lower() in haystack for needle in self.must_contain)
        return False


@dataclass
class NegativeCase:
    id: str
    query: str
    expect: str = "fallback"
    notes: str = ""


@dataclass
class CaseResult:
    case_id: str
    query: str
    returned: int
    relevant_positions: list[int]
    elapsed_ms: float
    top_score: float | None
    degraded: list[str] = field(default_factory=list)
    titles: list[str] = field(default_factory=list)
    #: Positions of chunks from a relevant paper, regardless of evidence.
    #: ``None`` when the case names no papers.
    paper_positions: list[int] | None = None

    @property
    def hit(self) -> bool:
        return bool(self.relevant_positions)

    @property
    def reciprocal_rank(self) -> float:
        return 1.0 / self.relevant_positions[0] if self.relevant_positions else 0.0

    @property
    def precision(self) -> float:
        return len(self.relevant_positions) / self.returned if self.returned else 0.0

    def ndcg(self) -> float:
        """Binary-gain nDCG. Ideal is every relevant chunk at the top."""
        if not self.relevant_positions:
            return 0.0
        dcg = sum(1.0 / math.log2(p + 1) for p in self.relevant_positions)
        ideal = sum(
            1.0 / math.log2(i + 2) for i in range(len(self.relevant_positions))
        )
        return dcg / ideal if ideal else 0.0


@dataclass
class ModeReport:
    mode: SearchMode
    results: list[CaseResult] = field(default_factory=list)
    negatives_correct: int = 0
    negatives_total: int = 0

    def _mean(self, values: Sequence[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    @property
    def recall(self) -> float:
        return self._mean([1.0 if r.hit else 0.0 for r in self.results])

    @property
    def precision(self) -> float:
        return self._mean([r.precision for r in self.results])

    @property
    def paper_recall(self) -> float | None:
        """Share of cases where any chunk from the right paper came back."""
        scored = [r for r in self.results if r.paper_positions is not None]
        if not scored:
            return None
        return self._mean([1.0 if r.paper_positions else 0.0 for r in scored])

    @property
    def mrr(self) -> float:
        return self._mean([r.reciprocal_rank for r in self.results])

    @property
    def ndcg(self) -> float:
        return self._mean([r.ndcg() for r in self.results])

    @property
    def median_latency_ms(self) -> float:
        if not self.results:
            return 0.0
        ordered = sorted(r.elapsed_ms for r in self.results)
        mid = len(ordered) // 2
        return (
            ordered[mid]
            if len(ordered) % 2
            else (ordered[mid - 1] + ordered[mid]) / 2
        )

    @property
    def p95_latency_ms(self) -> float:
        if not self.results:
            return 0.0
        ordered = sorted(r.elapsed_ms for r in self.results)
        return ordered[min(int(len(ordered) * 0.95), len(ordered) - 1)]

    @property
    def rejection_rate(self) -> float:
        return (
            self.negatives_correct / self.negatives_total
            if self.negatives_total
            else 0.0
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "cases": len(self.results),
            "recall": round(self.recall, 4),
            "paper_recall": (
                round(self.paper_recall, 4) if self.paper_recall is not None else None
            ),
            "precision": round(self.precision, 4),
            "mrr": round(self.mrr, 4),
            "ndcg": round(self.ndcg, 4),
            "median_latency_ms": round(self.median_latency_ms, 1),
            "p95_latency_ms": round(self.p95_latency_ms, 1),
            "negative_rejection_rate": round(self.rejection_rate, 4),
            "misses": [r.case_id for r in self.results if not r.hit],
        }


def load_dataset(path: Path) -> tuple[list[Case], list[NegativeCase]]:
    """Read the golden set. YAML if PyYAML is installed, else JSON."""
    text = path.read_text("utf-8")
    if path.suffix in {".yaml", ".yml"}:
        try:
            import yaml

            data = yaml.safe_load(text)
        except ImportError as exc:
            raise RuntimeError(
                "Reading a YAML golden set needs PyYAML: pip install pyyaml "
                "(or convert the file to JSON)."
            ) from exc
    else:
        data = json.loads(text)

    cases = [
        Case(
            id=str(c["id"]),
            query=str(c["query"]),
            query_type=QueryType(c["query_type"]) if c.get("query_type") else None,
            must_contain=list(c.get("must_contain") or []),
            relevant_papers=list(c.get("relevant_papers") or []),
            relevant_chunks=[str(x) for x in c.get("relevant_chunks") or []],
            relevant_evidence=[str(x) for x in c.get("relevant_evidence") or []],
            reference_answers=[str(x) for x in c.get("reference_answers") or []],
            notes=str(c.get("notes") or ""),
        )
        for c in data.get("cases", [])
    ]
    negatives = [
        NegativeCase(
            id=str(c["id"]),
            query=str(c["query"]),
            expect=str(c.get("expect") or "fallback"),
            notes=str(c.get("notes") or ""),
        )
        for c in data.get("negative_cases", [])
    ]
    return cases, negatives


async def evaluate_mode(
    container: Any,
    mode: SearchMode,
    cases: Sequence[Case],
    negatives: Sequence[NegativeCase],
) -> ModeReport:
    from vectora.generation.guards import check as guard_check
    from vectora.generation.router import classify

    report = ModeReport(mode=mode)

    for case in cases:
        query_type = case.query_type
        if query_type is None:
            query_type, _ = await classify(
                case.query, llm=container.llm, settings=container.settings
            )
        start = time.perf_counter()
        with request_context():
            chunks, diag = await container.engine.retriever.retrieve(
                case.query, mode=mode, query_type=query_type
            )
        elapsed = (time.perf_counter() - start) * 1000

        positions = [i for i, rc in enumerate(chunks, start=1) if case.is_relevant(rc)]
        paper_positions = (
            [i for i, rc in enumerate(chunks, start=1) if case.paper_hit(rc)]
            if case.relevant_papers
            else None
        )
        report.results.append(
            CaseResult(
                case_id=case.id,
                query=case.query,
                returned=len(chunks),
                relevant_positions=positions,
                elapsed_ms=elapsed,
                top_score=diag.top_score,
                degraded=diag.degraded,
                titles=[rc.chunk.title[:60] for rc in chunks[:3]],
                paper_positions=paper_positions,
            )
        )

    # Negatives: retrieval must come back weak enough that the guard refuses.
    for negative in negatives:
        report.negatives_total += 1
        query_type, _ = await classify(
            negative.query, llm=container.llm, settings=container.settings
        )
        with request_context():
            chunks, _ = await container.engine.retriever.retrieve(
                negative.query, mode=mode, query_type=query_type
            )
        ok, _ = guard_check(chunks, container.settings)
        refused = not ok
        if refused == (negative.expect == "fallback"):
            report.negatives_correct += 1
        else:
            log.warning(
                "negative case behaved wrongly",
                extra={"case": negative.id, "answered": ok},
            )

    return report


def render(reports: Sequence[ModeReport]) -> str:
    lines = ["", "=" * 86, "  RETRIEVAL EVALUATION", "=" * 86, ""]
    header = (
        f"  {'mode':<10} {'recall':>7} {'paper':>7} {'prec':>7} {'MRR':>7} {'nDCG':>7} "
        f"{'p50 ms':>8} {'p95 ms':>8} {'neg ok':>7}"
    )
    lines += [header, "  " + "-" * 82]
    for r in reports:
        paper = f"{r.paper_recall:>7.3f}" if r.paper_recall is not None else f"{'-':>7}"
        lines.append(
            f"  {r.mode.value:<10} {r.recall:>7.3f} {paper} {r.precision:>7.3f} "
            f"{r.mrr:>7.3f} {r.ndcg:>7.3f} {r.median_latency_ms:>8.0f} "
            f"{r.p95_latency_ms:>8.0f} {r.rejection_rate:>7.0%}"
        )

    for r in reports:
        misses = [x for x in r.results if not x.hit]
        if not misses:
            continue
        lines += ["", f"  {r.mode.value} missed {len(misses)}:"]
        for miss in misses:
            lines.append(f"    {miss.case_id}: {miss.query[:62]}")
            if miss.titles:
                lines.append(f"      got: {miss.titles[0]}")

    degraded = {d for r in reports for x in r.results for d in x.degraded}
    if degraded:
        lines += ["", "  stages degraded during this run: " + ", ".join(sorted(degraded))]

    lines += ["", "=" * 86, ""]
    return "\n".join(lines)


async def run(
    settings: Settings,
    *,
    dataset: Path | None = None,
    only_mode: SearchMode | None = None,
    output: Path | None = None,
    corpus: str | None = None,
) -> list[ModeReport]:
    configure_logging(settings)
    from vectora.api.deps import build_container

    container = await asyncio.to_thread(
        build_container, settings, warm=True, corpus=corpus
    )
    try:
        if not container.store.exists():
            raise RuntimeError(
                "Nothing is indexed. Run `vectora index` before evaluating."
            )
        # The golden set belongs to the corpus, not to the project: an eval
        # set written for one corpus measures nothing on another.
        path = dataset or container.corpus.golden
        if not path.exists():
            raise RuntimeError(
                f"No golden set at {path}. Draft one with `vectora golden "
                f"generate`, or copy evaluation/golden.template.yaml there."
            )
        cases, negatives = load_dataset(path)
        print(
            f"\nLoaded {len(cases)} cases and {len(negatives)} negatives "
            f"from {path.name} (corpus '{container.corpus.name}')"
        )
        modes = [only_mode] if only_mode else list(SearchMode)
        reports: list[ModeReport] = []
        for mode in modes:
            print(f"  evaluating {mode.value}…")
            reports.append(await evaluate_mode(container, mode, cases, negatives))
    finally:
        await container.aclose()

    print(render(reports))
    if output:
        payload = {
            "corpus": container.corpus.name,
            "dataset": str(path),
            "settings": {
                "embedding_model": settings.embedding_model,
                "reranker_model": settings.reranker_model,
                "rrf_k": settings.rrf_k,
                "dense_weight": settings.dense_weight,
                "sparse_weight": settings.sparse_weight,
            },
            "reports": [r.as_dict() for r in reports],
        }
        output.write_text(json.dumps(payload, indent=2), "utf-8")
        print(f"Wrote {output}\n")
    return reports


if __name__ == "__main__":
    from vectora.config import get_settings

    asyncio.run(run(get_settings()))
