"""
Text matching for evaluation: answer token F1 and evidence overlap.

Two questions the rest of the harness could not answer before:

* **Is the answer right?** ``token_f1`` compares the generated answer with
  the annotators' reference answers using the SQuAD/QASPER recipe —
  lower-case, drop punctuation and articles, split on whitespace, then F1
  over the bag of tokens, taking the best match across references. QASPER
  references are short spans and Vectora writes cited prose, so precision is
  structurally low; ``token_recall`` (how much of the reference the answer
  contains) is reported next to F1 because it is the fairer number here.

* **Did retrieval find the passage, not just the paper?** ``evidence_match``
  checks whether a retrieved chunk contains an annotator-marked evidence
  paragraph. Chunks and paragraphs have different boundaries, so this uses
  word-shingle overlap rather than equality: a chunk matches when it holds at
  least ``threshold`` of the shingles of the smaller of the two texts.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Iterable, Sequence

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)
_CITE = re.compile(r"\[\d{1,2}(?:\s*[,\-–]\s*\d{1,2})*\]")

#: Words per shingle for evidence matching. Six is long enough that two
#: unrelated paragraphs almost never share one, short enough to survive the
#: odd re-wrapped line or cleaned placeholder.
SHINGLE = 6


def normalise(text: str) -> str:
    text = _CITE.sub(" ", text.lower())
    text = text.translate(_PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def tokens(text: str) -> list[str]:
    return normalise(text).split()


def _prf(pred: list[str], ref: list[str]) -> tuple[float, float, float]:
    if not pred or not ref:
        same = float(pred == ref)
        return same, same, same
    common = Counter(pred) & Counter(ref)
    overlap = sum(common.values())
    if not overlap:
        return 0.0, 0.0, 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return precision, recall, 2 * precision * recall / (precision + recall)


def token_scores(prediction: str, references: Sequence[str]) -> tuple[float, float]:
    """(best F1, best recall) of ``prediction`` against any reference.

    Returns (0, 0) when there are no usable references, so callers should
    only average over cases that have them.
    """
    refs = [r for r in references if r and r.strip()]
    if not refs:
        return 0.0, 0.0
    pred = tokens(prediction)
    best_f1 = best_recall = 0.0
    for ref in refs:
        _, recall, f1 = _prf(pred, tokens(ref))
        best_f1 = max(best_f1, f1)
        best_recall = max(best_recall, recall)
    return best_f1, best_recall


def token_f1(prediction: str, references: Sequence[str]) -> float:
    return token_scores(prediction, references)[0]


def shingles(text: str, size: int = SHINGLE) -> set[tuple[str, ...]]:
    words = normalise(text).split()
    if len(words) < size:
        return {tuple(words)} if words else set()
    return {tuple(words[i : i + size]) for i in range(len(words) - size + 1)}


def overlap(a: set[tuple[str, ...]], b: set[tuple[str, ...]]) -> float:
    """|a ∩ b| / min(|a|, |b|): 1.0 when the smaller text is fully contained."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def evidence_match(
    chunk_text: str, evidence: Iterable[str], *, threshold: float = 0.5
) -> bool:
    """Whether ``chunk_text`` contains (most of) any evidence paragraph."""
    chunk = shingles(chunk_text)
    return any(overlap(chunk, shingles(e)) >= threshold for e in evidence if e.strip())
