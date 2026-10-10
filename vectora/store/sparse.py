"""
Persistent whole-corpus BM25 index.

This is the fix for the biggest conceptual problem in the old retrieval
path. BM25 was run over the *dense candidate pool*, which means it could
only re-rank what vector search had already found. The entire reason to run
a lexical retriever alongside a dense one is to catch what the dense one
misses — an exact assay name, a rare acronym, a specific instrument model,
a number. Re-scoring 20 dense hits cannot do that by construction.

So the sparse index now covers the whole collection, is queried
independently, and its ranking is fused with the dense ranking by
reciprocal rank fusion. For a 9k-chunk corpus an in-process BM25 is the
right tool: it builds in a couple of seconds, needs no extra service, and
the index pickles to a few megabytes. Beyond a few hundred thousand chunks
this should move to Qdrant's native sparse vectors or an OpenSearch
cluster, and the interface here is deliberately narrow enough that swapping
it is a contained change.
"""

from __future__ import annotations

import pickle
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from vectora.config import Settings
from vectora.models import Chunk
from vectora.observability import get_logger

log = get_logger(__name__)

_INDEX_VERSION = 2

_TOKEN = re.compile(r"[a-z0-9]+(?:[-/.][a-z0-9]+)*")
#: Kept deliberately small. Scientific queries lean on words a generic stop
#: list would remove ("a" in "vitamin A"), and BM25 already discounts terms
#: that appear everywhere.
_STOPWORDS = frozenset(
    ["the", "of", "and", "to", "in", "is", "are", "was", "were", "be", "been", "being", "for", "on", "with", "as", "by", "that", "this", "these", "those", "it", "its", "from", "at", "an", "or", "if", "then", "than", "so", "such", "we", "our", "they", "their", "there", "here", "which", "who", "whom", "what", "when", "where", "how", "about", "into", "over", "under"]
)
#: Hyphenated and slashed scientific terms are also indexed as their parts,
#: so "LC-MS/MS" is findable as "lc", "ms", "lcmsms" and the whole token.
_SPLITTABLE = re.compile(r"[-/.]")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, with scientific compounds expanded.

    Must be identical at index and query time or the term statistics do not
    line up, which is why it lives here rather than in the retriever.
    """
    lowered = text.lower()
    tokens: list[str] = []
    for match in _TOKEN.finditer(lowered):
        token = match.group(0)
        if token in _STOPWORDS or len(token) < 2:
            continue
        tokens.append(token)
        if _SPLITTABLE.search(token):
            parts = [p for p in _SPLITTABLE.split(token) if len(p) >= 2]
            if len(parts) > 1:
                tokens.extend(parts)
                tokens.append("".join(parts))
    return tokens


@dataclass
class SparseHit:
    chunk: Chunk
    score: float
    rank: int


class SparseIndex:
    """BM25 over every chunk in the corpus.

    Built from the vector store (preferred, because it reflects what is
    actually searchable) or from ``chunks.jsonl``.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.path = settings.bm25_index_path
        self._bm25 = None
        self._chunks: list[Chunk] = []
        self.built_at: float | None = None

    # ── lifecycle ─────────────────────────────────────────────────────────

    @property
    def ready(self) -> bool:
        return self._bm25 is not None and bool(self._chunks)

    @property
    def size(self) -> int:
        return len(self._chunks)

    def build(self, chunks: Iterable[Chunk]) -> int:
        """Build the index in memory from ``chunks``."""
        from rank_bm25 import BM25Okapi

        start = time.perf_counter()
        self._chunks = [c for c in chunks if c.content.strip()]
        if not self._chunks:
            self._bm25 = None
            log.warning("sparse index built empty")
            return 0

        corpus = [tokenize(c.content) for c in self._chunks]
        self._bm25 = BM25Okapi(
            corpus, k1=self.settings.bm25_k1, b=self.settings.bm25_b
        )
        self.built_at = time.time()
        log.info(
            "sparse index built",
            extra={
                "chunks": len(self._chunks),
                "ms": round((time.perf_counter() - start) * 1000, 1),
            },
        )
        return len(self._chunks)

    def save(self, path: Path | None = None) -> Path:
        """Persist to disk, atomically."""
        target = path or self.path
        target.parent.mkdir(parents=True, exist_ok=True)
        blob = {
            "version": _INDEX_VERSION,
            "built_at": self.built_at,
            "k1": self.settings.bm25_k1,
            "b": self.settings.bm25_b,
            # The BM25 object is cheap to rebuild and its internals are not a
            # stable format, so only the corpus is persisted.
            "chunks": [c.payload() for c in self._chunks],
        }
        tmp = target.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(blob, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(target)
        log.info("sparse index saved", extra={"path": str(target), "chunks": self.size})
        return target

    def load(self, path: Path | None = None) -> bool:
        """Load from disk. Returns False when absent or stale."""
        target = path or self.path
        if not target.exists():
            return False
        try:
            with open(target, "rb") as f:
                blob = pickle.load(f)
        except Exception as exc:
            log.warning("sparse index unreadable", extra={"err": str(exc)[:140]})
            return False

        if blob.get("version") != _INDEX_VERSION:
            log.info("sparse index version changed; rebuild needed")
            return False

        chunks: list[Chunk] = []
        for payload in blob.get("chunks", []):
            try:
                chunks.append(Chunk.model_validate(payload))
            except Exception:
                continue
        if not chunks:
            return False

        self.built_at = blob.get("built_at")
        self.build(chunks)
        return self.ready

    def ensure(self, *, store: object | None = None, chunks_path: Path | None = None) -> bool:
        """Load the index, building it from a source if there is none.

        Called at startup. A missing sparse index degrades retrieval to
        dense-only rather than failing the request, and that degradation is
        reported in the diagnostics.
        """
        if self.ready:
            return True
        if self.load():
            return True

        sourced: list[Chunk] = []
        if store is not None:
            try:
                sourced = list(store.iter_all())  # type: ignore[attr-defined]
                log.info("sparse index sourced from vector store", extra={"chunks": len(sourced)})
            except Exception as exc:
                log.warning("could not read vector store", extra={"err": str(exc)[:140]})

        if not sourced:
            path = chunks_path or self.settings.chunks_path
            if path.exists():
                from vectora.ingestion.pipeline import read_chunks

                sourced = list(read_chunks(path))  # type: ignore[arg-type]
                log.info("sparse index sourced from jsonl", extra={"chunks": len(sourced)})

        if not sourced:
            log.warning("no source for sparse index; lexical retrieval disabled")
            return False

        self.build(sourced)
        if self.ready:
            self.save()
        return self.ready

    def invalidate(self) -> None:
        """Drop the on-disk index, so the next start rebuilds it."""
        self._bm25 = None
        self._chunks = []
        self.path.unlink(missing_ok=True)

    # ── query ─────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        *,
        limit: int,
        filters: dict[str, object] | None = None,
    ) -> list[SparseHit]:
        """Top lexical matches.

        Filters are applied post-scoring. For this corpus size that is
        cheaper than maintaining per-field postings, but it does mean a very
        narrow filter can return fewer than ``limit`` hits; the caller treats
        that as a normal short result.
        """
        if not self.ready:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)  # type: ignore[union-attr]
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        hits: list[SparseHit] = []
        for i in order:
            if scores[i] <= 0:
                break
            chunk = self._chunks[i]
            if filters and not _matches(chunk, filters):
                continue
            hits.append(SparseHit(chunk=chunk, score=float(scores[i]), rank=len(hits) + 1))
            if len(hits) >= limit:
                break
        return hits

    def stats(self) -> dict[str, object]:
        return {
            "ready": self.ready,
            "chunks": self.size,
            "built_at": self.built_at,
            "path": str(self.path),
            "exists_on_disk": self.path.exists(),
        }


def _matches(chunk: Chunk, filters: dict[str, object]) -> bool:
    for key, value in filters.items():
        if value in (None, ""):
            continue
        actual = str(getattr(chunk, key, "") or "")
        if isinstance(value, (list, tuple, set)):
            if actual not in {str(v) for v in value}:
                return False
        elif isinstance(value, dict):
            try:
                number = int(actual)
            except (TypeError, ValueError):
                return False
            if "gte" in value and number < int(value["gte"]):  # type: ignore[arg-type]
                return False
            if "lte" in value and number > int(value["lte"]):  # type: ignore[arg-type]
                return False
        elif actual != str(value):
            return False
    return True
