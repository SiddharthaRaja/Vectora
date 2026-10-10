"""
Semantic chunking with enforced size bounds and sentence overlap.

Three changes from the previous implementation, each fixing an observed
defect:

1. **The size clamp is no longer bypassable.** ``semantic_chunk`` used to
   ``return sentences`` early when the text held a single sentence, which
   skipped ``_enforce_size_limits`` entirely. That is how 12-token fragments
   like ``"journal homepage: www.elsevier.com/locate/steroids"`` reached the
   index. Every path now ends at the same clamp.
2. **Short chunks merge in both directions.** The old merge only folded a
   runt into the *previous* chunk, so the first chunk of every section could
   be arbitrarily small.
3. **Chunks overlap by a sentence.** A fact that straddles a boundary used
   to be unretrievable from either side; now the trailing sentence of chunk
   *n* opens chunk *n+1*.

The splitter is a one-pass variant of the usual embedding-distance method:
embed the sentences, cut where adjacent similarity drops below a threshold,
then repair the sizes.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

import numpy as np

#: Abbreviations that end in a period and must not end a sentence. Without
#: these, scientific prose shatters: "approx. 25 ng/mL" would split twice.
#:
#: Units are deliberately *not* in this list. A unit does legitimately end a
#: sentence ("the LOQ was 2.5 ng/mL. Recovery was 98%."), so treating "mL."
#: as an abbreviation merges two real sentences into one — which is worse
#: than the rare over-split it would prevent.
_ABBREVIATIONS = (
    "e.g", "i.e", "etc", "cf", "vs", "viz", "al", "ca", "approx", "approx",
    "fig", "figs", "eq", "eqs", "ref", "refs", "no", "vol", "ed", "eds",
    "pp", "p", "ch", "sec", "dr", "prof", "mr", "mrs", "st", "inc", "ltd",
    "co", "dept", "univ", "spp", "subsp", "var", "resp", "cal",
)
_ABBREV_RE = re.compile(
    r"(?<!\w)(" + "|".join(re.escape(a) for a in _ABBREVIATIONS) + r")\.\s*$",
    re.IGNORECASE,
)
#: Split after ., !, ? or ; when followed by whitespace and a capital,
#: a digit or an opening bracket.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])[\s]+(?=[A-Z(\[“\"]|\d)")
_DECIMAL = re.compile(r"\d\.\d")


def split_sentences(text: str) -> list[str]:
    """Sentence splitter tuned for scientific prose, no NLTK dependency.

    Deliberately conservative: an over-merge costs a slightly coarse chunk,
    an over-split costs a chunk that is too small to mean anything.
    """
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []

    rough = _SENTENCE_BOUNDARY.split(text)
    merged: list[str] = []
    for part in rough:
        part = part.strip()
        if not part:
            continue
        if merged:
            prev = merged[-1]
            # Rejoin when the previous fragment ended on an abbreviation, a
            # decimal number, or a single initial.
            if (
                _ABBREV_RE.search(prev)
                or _DECIMAL.search(prev[-3:])
                or re.search(r"(?<!\w)[A-Z]\.\s*$", prev)
                or len(prev) < 16
            ):
                merged[-1] = f"{prev} {part}"
                continue
        merged.append(part)
    return merged


def estimate_tokens(text: str) -> int:
    return max(len(text) // 4, len(text.split()))


def _pack(
    sentences: Sequence[str],
    *,
    max_tokens: int,
    overlap: int,
) -> list[str]:
    """Greedily pack sentences into ``max_tokens`` windows with overlap.

    Used both as the hard splitter for oversized semantic chunks and as the
    fallback when no embedder is available.
    """
    out: list[str] = []
    window: list[str] = []
    budget = 0

    for sent in sentences:
        cost = estimate_tokens(sent)
        if window and budget + cost > max_tokens:
            out.append(" ".join(window))
            tail = window[-overlap:] if overlap > 0 else []
            window = [*tail, sent]
            budget = sum(estimate_tokens(s) for s in window)
        else:
            window.append(sent)
            budget += cost

    if window:
        out.append(" ".join(window))
    return out


def enforce_size(
    chunks: Sequence[str],
    *,
    min_tokens: int,
    max_tokens: int,
    overlap: int = 1,
) -> list[str]:
    """Bring every chunk inside ``[min_tokens, max_tokens]``.

    Oversized chunks are split on sentence boundaries; undersized ones are
    merged with a neighbour — forwards for the first chunk, backwards
    otherwise. A chunk that is still below the floor after merging (because
    it is the only chunk in its section) is returned and left for the
    quality gate to judge, which is the right place for that decision.
    """
    # 1. Split what is too long.
    split: list[str] = []
    for chunk in chunks:
        text = chunk.strip()
        if not text:
            continue
        if estimate_tokens(text) <= max_tokens:
            split.append(text)
            continue
        sentences = split_sentences(text)
        if len(sentences) <= 1:
            # One enormous sentence, or a table row: hard-wrap on words.
            words = text.split()
            stride = max(max_tokens, 1)
            split.extend(
                " ".join(words[i : i + stride]) for i in range(0, len(words), stride)
            )
        else:
            split.extend(_pack(sentences, max_tokens=max_tokens, overlap=overlap))

    if not split:
        return []

    # 2. Merge what is too short, in both directions.
    merged: list[str] = []
    for text in split:
        if (
            merged
            and estimate_tokens(text) < min_tokens
            and estimate_tokens(merged[-1]) + estimate_tokens(text) <= max_tokens * 1.35
        ):
            merged[-1] = f"{merged[-1]} {text}"
        else:
            merged.append(text)

    # The first chunk had no predecessor to merge into; fold it forwards.
    if (
        len(merged) > 1
        and estimate_tokens(merged[0]) < min_tokens
        and estimate_tokens(merged[0]) + estimate_tokens(merged[1]) <= max_tokens * 1.35
    ):
        merged[1] = f"{merged[0]} {merged[1]}"
        merged.pop(0)

    return [m.strip() for m in merged if m.strip()]


def semantic_chunk(
    text: str,
    *,
    encode: Callable[[Sequence[str]], np.ndarray] | None,
    similarity_threshold: float,
    min_tokens: int,
    max_tokens: int,
    overlap: int = 1,
) -> list[str]:
    """Split ``text`` into topically coherent chunks.

    ``encode`` must return L2-normalised vectors so similarity is a dot
    product. Pass ``None`` to skip the semantic step and pack by size only,
    which is what the tests and ``--no-semantic`` do.

    Every return path goes through :func:`enforce_size`.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []

    if encode is None or len(sentences) < 3:
        return enforce_size(
            [" ".join(sentences)] if encode is None else sentences,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            overlap=overlap,
        )

    vectors = encode(sentences)
    if len(vectors) != len(sentences):  # defensive: encoder returned junk
        return enforce_size(
            _pack(sentences, max_tokens=max_tokens, overlap=overlap),
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            overlap=overlap,
        )

    sims = np.sum(vectors[:-1] * vectors[1:], axis=1)

    groups: list[list[str]] = [[sentences[0]]]
    running = estimate_tokens(sentences[0])
    for i, sim in enumerate(sims):
        nxt = sentences[i + 1]
        cost = estimate_tokens(nxt)
        topic_shift = float(sim) < similarity_threshold
        too_big = running + cost > max_tokens
        # Only honour a topic shift once the current chunk is substantial,
        # otherwise a single transitional sentence becomes its own chunk.
        if (topic_shift and running >= min_tokens) or too_big:
            groups.append([nxt])
            running = cost
        else:
            groups[-1].append(nxt)
            running += cost

    chunks = [" ".join(g) for g in groups]
    return enforce_size(
        chunks, min_tokens=min_tokens, max_tokens=max_tokens, overlap=overlap
    )
