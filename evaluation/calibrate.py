"""
Threshold calibration.

``min_rerank_score`` and ``min_dense_score`` decide when Vectora refuses to
answer. Their right values depend on the corpus and the models, so shipping
a guess is wrong: too low and off-topic questions get fluent ungrounded
answers, too high and real questions get refused.

This measures the two score distributions — over the golden set's real
questions and over its negative cases — and reports the floor that separates
them best, along with what that floor costs in refused good questions.

    vectora calibrate

Run it after any change to the corpus, the embedding model or the reranker.
Nothing is written automatically: it prints the values to put in ``.env``,
because silently retuning a safety threshold is not a thing software should
do to you.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from vectora.config import QueryType, SearchMode, Settings
from vectora.observability import configure_logging, get_logger, request_context

log = get_logger(__name__)


@dataclass
class Distribution:
    """Top scores observed for one class of question."""

    name: str
    rerank: list[float] = field(default_factory=list)
    dense: list[float] = field(default_factory=list)

    def summary(self, values: Sequence[float]) -> str:
        if not values:
            return "no data"
        ordered = sorted(values)
        n = len(ordered)

        def q(p: float) -> float:
            return ordered[min(int(p * n), n - 1)]

        return (
            f"n={n}  min={ordered[0]:+.3f}  p10={q(0.10):+.3f}  "
            f"median={q(0.50):+.3f}  p90={q(0.90):+.3f}  max={ordered[-1]:+.3f}"
        )


def suggest_floor(
    positives: Sequence[float],
    negatives: Sequence[float],
    *,
    margin: float = 0.15,
) -> tuple[float | None, str]:
    """A floor that admits the positives and rejects the negatives.

    Picks the midpoint of the gap when the two distributions separate
    cleanly. When they overlap there is no threshold that gets everything
    right, so it favours **not refusing real questions** — a wrong answer
    carries a citation the user can check, while a refusal on a good
    question is a dead end — and reports the overlap honestly.
    """
    if not positives:
        return None, "no positive samples"
    if not negatives:
        return None, "no negative samples; cannot separate"

    lowest_good = min(positives)
    highest_bad = max(negatives)

    if lowest_good > highest_bad:
        floor = highest_bad + (lowest_good - highest_bad) * 0.5
        return round(floor, 3), (
            f"clean separation (worst good {lowest_good:+.3f} > "
            f"best bad {highest_bad:+.3f})"
        )

    # Overlapping. Choose the floor that rejects the most negatives while
    # keeping every positive but the worst few.
    ordered_good = sorted(positives)
    keep_index = max(0, int(len(ordered_good) * 0.05))
    floor = ordered_good[keep_index] - abs(ordered_good[keep_index]) * margin * 0.1
    refused_good = sum(1 for v in positives if v < floor)
    rejected_bad = sum(1 for v in negatives if v < floor)
    return round(floor, 3), (
        f"distributions overlap — this floor refuses {refused_good}/{len(positives)} "
        f"real questions and catches {rejected_bad}/{len(negatives)} off-topic ones"
    )


async def run(
    settings: Settings,
    *,
    dataset: Path | None = None,
    mode: SearchMode = SearchMode.BALANCED,
    corpus: str | None = None,
) -> None:
    configure_logging(settings)
    from evaluation.run_eval import load_dataset
    from vectora.api.deps import build_container
    from vectora.generation.router import classify

    container_settings = settings
    probe = await asyncio.to_thread(
        build_container, container_settings, warm=False, corpus=corpus
    )
    path = dataset or probe.corpus.golden
    await probe.aclose()
    if not path.exists():
        print(f"\nNo golden set at {path}. Draft one: vectora golden generate\n")
        return
    cases, negatives = load_dataset(path)
    if not negatives:
        print(
            "\nThis golden set has no negative_cases. Calibration needs questions "
            "the corpus genuinely cannot answer — add a few to "
            f"{path.name} and run this again.\n"
        )
        return

    container = await asyncio.to_thread(
        build_container, settings, warm=True, corpus=corpus
    )
    try:
        if not container.store.exists():
            raise RuntimeError("Nothing is indexed. Run `vectora index` first.")

        print(f"\nCalibrating in {mode.value} mode over {len(cases)} real and "
              f"{len(negatives)} off-topic questions…\n")

        good = Distribution("on-topic")
        bad = Distribution("off-topic")

        for case, bucket in [(c, good) for c in cases] + [
            (n, bad) for n in negatives
        ]:
            query_type = getattr(case, "query_type", None)
            if query_type is None:
                query_type, _ = await classify(
                    case.query, llm=container.llm, settings=container.settings
                )
            with request_context():
                chunks, _ = await container.engine.retriever.retrieve(
                    case.query, mode=mode, query_type=query_type or QueryType.EXPLORATORY
                )
            if not chunks:
                continue
            rerank = [c.rerank_score for c in chunks if c.rerank_score is not None]
            dense = [c.dense_score for c in chunks if c.dense_score is not None]
            if rerank:
                bucket.rerank.append(max(rerank))
            if dense:
                bucket.dense.append(max(dense))

        print("Cross-encoder logit of the best chunk")
        print(f"  on-topic   {good.summary(good.rerank)}")
        print(f"  off-topic  {bad.summary(bad.rerank)}")
        rerank_floor, rerank_note = suggest_floor(good.rerank, bad.rerank)
        if rerank_floor is not None:
            print(f"  suggested  VECTORA_MIN_RERANK_SCORE={rerank_floor}")
            print(f"             {rerank_note}")
        print(f"  current    {settings.min_rerank_score}")

        print("\nDense cosine of the best chunk")
        print(f"  on-topic   {good.summary(good.dense)}")
        print(f"  off-topic  {bad.summary(bad.dense)}")
        dense_floor, dense_note = suggest_floor(good.dense, bad.dense)
        if dense_floor is not None:
            print(f"  suggested  VECTORA_MIN_DENSE_SCORE={dense_floor}")
            print(f"             {dense_note}")
        print(f"  current    {settings.min_dense_score}")

        print(
            "\nNothing has been changed. Put the suggested values in .env if they "
            "look right, then re-run `vectora eval` to confirm both the recall and "
            "the rejection rate moved the way you expected.\n"
        )
    finally:
        await container.aclose()
