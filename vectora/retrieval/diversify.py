"""
Maximal marginal relevance and near-duplicate removal.

Two bugs fixed from the previous implementation:

1. **MMR frequently did nothing.** It returned early whenever
   ``len(candidates) <= top_k``, and the stage before it handed over
   ``min(top_k * 3, dense_k)`` candidates — so for the ``comparative`` and
   ``exploratory`` routes, which ask for 8 chunks, diversification almost
   never ran. The early return is now only for the genuinely trivial case,
   and the candidate pool is sized so there is something to choose from.

2. **Deduplication embedded every chunk twice** — once to compare against
   what was kept, once to store for the next comparison — and ran the
   encoder inside the loop, one text at a time. It is now a single batched
   encode, with an exact-text pass first so identical chunks never reach
   the encoder at all.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

import numpy as np

from vectora.models import RetrievedChunk
from vectora.observability import get_logger

log = get_logger(__name__)

_WHITESPACE = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")

Encode = Callable[[Sequence[str]], np.ndarray]


def _fingerprint(text: str) -> str:
    """Normalised form for exact-duplicate detection."""
    return _WHITESPACE.sub(" ", _PUNCT.sub("", text.lower())).strip()


def mmr(
    candidates: Sequence[RetrievedChunk],
    query: str,
    *,
    encode: Encode,
    top_k: int,
    lambda_: float,
) -> list[RetrievedChunk]:
    """Select ``top_k`` chunks balancing relevance against redundancy.

        score(d) = λ · sim(d, query) − (1 − λ) · max sim(d, already chosen)

    ``λ`` near 1 maximises relevance (right for a definition), near 0
    maximises coverage (right for a comparison or an overview).

    Relevance uses the pipeline's own score where available rather than
    re-deriving it from embeddings, because by this point the cross-encoder
    has usually produced a better relevance estimate than cosine similarity
    can.
    """
    if top_k <= 0:
        return []
    if len(candidates) <= 1 or (len(candidates) <= top_k and len(candidates) <= 2):
        return list(candidates[:top_k])

    texts = [rc.chunk.content for rc in candidates]
    try:
        doc_vectors = encode(texts)
        query_vector = encode([query])[0]
    except Exception as exc:
        log.warning("mmr skipped, encoder failed", extra={"err": str(exc)[:120]})
        return list(candidates[:top_k])

    if len(doc_vectors) != len(candidates):
        return list(candidates[:top_k])

    # Relevance on a common 0-1 scale so λ means the same thing whether or
    # not the reranker ran.
    pipeline_scores = np.array([rc.score for rc in candidates], dtype=np.float32)
    lo, hi = float(pipeline_scores.min()), float(pipeline_scores.max())
    if hi - lo > 1e-9:
        relevance = (pipeline_scores - lo) / (hi - lo)
    else:
        relevance = np.clip(doc_vectors @ query_vector, 0.0, 1.0)

    similarity = np.clip(doc_vectors @ doc_vectors.T, -1.0, 1.0)

    selected: list[int] = [int(np.argmax(relevance))]
    remaining = [i for i in range(len(candidates)) if i not in selected]

    while len(selected) < min(top_k, len(candidates)) and remaining:
        redundancy = similarity[np.ix_(remaining, selected)].max(axis=1)
        objective = lambda_ * relevance[remaining] - (1.0 - lambda_) * redundancy
        pick = remaining[int(np.argmax(objective))]
        selected.append(pick)
        remaining.remove(pick)

    return [candidates[i] for i in selected]


def deduplicate(
    candidates: Sequence[RetrievedChunk],
    *,
    encode: Encode | None = None,
    threshold: float = 0.93,
    max_semantic: int = 60,
) -> list[RetrievedChunk]:
    """Drop repeated passages, keeping the best-scoring copy of each.

    Three passes, cheapest first:

    1. by ``chunk_id`` — the same passage reached us twice;
    2. by normalised text — the same prose from two papers, or the same
       boilerplate, with different ids;
    3. by embedding similarity — near-duplicates, e.g. an abstract and the
       conclusion that restates it.

    Pass 3 is skipped above ``max_semantic`` candidates, where an N² matrix
    stops being worth it and passes 1 and 2 have already done most of the
    work.
    """
    by_id: dict[str, RetrievedChunk] = {}
    for rc in candidates:
        current = by_id.get(rc.chunk.chunk_id)
        if current is None or rc.score > current.score:
            by_id[rc.chunk.chunk_id] = rc
    stage = sorted(by_id.values(), key=lambda rc: rc.score, reverse=True)

    seen_text: dict[str, RetrievedChunk] = {}
    for rc in stage:
        key = _fingerprint(rc.chunk.content)
        if key not in seen_text:
            seen_text[key] = rc
    stage = list(seen_text.values())

    if encode is None or len(stage) < 2 or len(stage) > max_semantic:
        return stage

    try:
        vectors = encode([rc.chunk.content for rc in stage])
    except Exception as exc:
        log.warning("semantic dedup skipped", extra={"err": str(exc)[:120]})
        return stage
    if len(vectors) != len(stage):
        return stage

    kept_indices: list[int] = []
    for i in range(len(stage)):
        if kept_indices:
            sims = vectors[i] @ vectors[kept_indices].T
            if float(np.max(sims)) > threshold:
                continue
        kept_indices.append(i)

    if len(kept_indices) < len(stage):
        log.debug(
            "semantic dedup", extra={"before": len(stage), "after": len(kept_indices)}
        )
    return [stage[i] for i in kept_indices]


def cap_per_paper(
    candidates: Sequence[RetrievedChunk], *, max_per_paper: int
) -> list[RetrievedChunk]:
    """Limit how many chunks one paper may contribute.

    Without this, a single long and lexically rich paper can fill the whole
    context window and the answer silently becomes a summary of one source
    presented as a survey of many. Order is otherwise preserved.
    """
    if max_per_paper <= 0:
        return list(candidates)
    counts: dict[str, int] = {}
    out: list[RetrievedChunk] = []
    overflow: list[RetrievedChunk] = []
    for rc in candidates:
        pid = rc.chunk.paper_id
        if counts.get(pid, 0) < max_per_paper:
            counts[pid] = counts.get(pid, 0) + 1
            out.append(rc)
        else:
            overflow.append(rc)
    # Overflow is kept at the tail rather than discarded: if the pool is
    # thin, a second chunk from the best paper beats nothing at all.
    return out + overflow
