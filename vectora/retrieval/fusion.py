"""
Reciprocal rank fusion.

Dense cosine scores live in roughly [0, 1] and BM25 scores are unbounded
and corpus-dependent. The old code combined them with a weighted sum of a
cosine score and a min-max-normalised BM25 score, which has two problems:
the normalisation depends on which candidates happened to be in the batch,
so the same chunk gets a different score depending on what it is ranked
against; and the weights have no stable meaning across queries.

RRF sidesteps both by discarding the magnitudes and fusing *ranks*:

    score(d) = Σ_lists  weight_list / (k + rank_list(d))

A document ranked first by either retriever scores well; a document ranked
well by both scores better. ``k`` damps the contribution of top ranks so
one retriever cannot dominate — 60 is the value from Cormack, Clarke &
Buettcher (2009) and is a sane default.

The fused score is comparable across queries, which min-max normalisation
over a batch is not. Note what it still cannot tell you: RRF measures
*rank*, so a chunk placed first in a hopeless pool scores exactly as well as
one placed first in a good pool. Deciding whether anything relevant was
found at all needs an absolute signal — a cross-encoder logit or a raw
cosine — and that is what :mod:`vectora.generation.guards` gates on.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from vectora.models import Chunk, RetrievedChunk


@dataclass
class RankedList:
    """One retriever's output, best first."""

    name: str
    chunks: Sequence[Chunk]
    scores: Sequence[float]
    weight: float = 1.0
    variant: str = ""
    """Which query phrasing produced this list, for diagnostics."""

    def __post_init__(self) -> None:
        if len(self.chunks) != len(self.scores):
            raise ValueError(f"{self.name}: {len(self.chunks)} chunks, {len(self.scores)} scores")


def reciprocal_rank_fusion(
    lists: Iterable[RankedList],
    *,
    k: int = 60,
    limit: int | None = None,
) -> list[RetrievedChunk]:
    """Fuse ranked lists into one ordering.

    Chunks are identified by ``chunk_id``, so the same passage found by
    several query variants or by both retrievers is one entry whose score
    accumulates — which is the behaviour we want, since independent
    agreement is evidence.
    """
    merged: dict[str, RetrievedChunk] = {}
    raw: dict[str, float] = {}

    for ranked in lists:
        for position, (chunk, score) in enumerate(
            zip(ranked.chunks, ranked.scores, strict=True), start=1
        ):
            key = chunk.chunk_id
            contribution = ranked.weight / (k + position)
            raw[key] = raw.get(key, 0.0) + contribution

            entry = merged.get(key)
            if entry is None:
                entry = RetrievedChunk(chunk=chunk)
                merged[key] = entry

            if ranked.name == "dense":
                if entry.dense_score is None or score > entry.dense_score:
                    entry.dense_score = float(score)
                if entry.dense_rank is None or position < entry.dense_rank:
                    entry.dense_rank = position
            elif ranked.name == "sparse":
                if entry.sparse_score is None or score > entry.sparse_score:
                    entry.sparse_score = float(score)
                if entry.sparse_rank is None or position < entry.sparse_rank:
                    entry.sparse_rank = position

            label = ranked.variant or ranked.name
            if label and label not in entry.matched_variants:
                entry.matched_variants.append(label)

    for key, entry in merged.items():
        entry.fused_score = round(raw[key], 9)

    ordered = sorted(
        merged.values(),
        key=lambda rc: (rc.fused_score, rc.chunk.quality),
        reverse=True,
    )
    return ordered[:limit] if limit else ordered


def normalise_scores(values: Sequence[float]) -> list[float]:
    """Min-max to [0, 1]. Only for display; never for ranking decisions.

    Kept in one place and clearly labelled, because batch-relative
    normalisation used for ranking was the original bug this module
    replaces.
    """
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]
