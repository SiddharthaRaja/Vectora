"""
Vector store adapter.

The important change from the previous implementation is **deterministic
point ids**. Points used to be numbered with ``enumerate`` over the batch
being uploaded, so a second incremental run started again at 0 and
overwrote the vectors of whichever paper was uploaded first. Ids now come
from a UUID5 of ``paper_id:chunk_id``, which makes every upsert idempotent:
re-running the embedder updates vectors in place and cannot clobber an
unrelated paper.

Also here: a single place that knows whether we are talking to a server or
to the embedded on-disk engine, so nothing else in the codebase has a
hardcoded ``localhost:6333``.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from vectora.config import Settings
from vectora.models import Chunk
from vectora.observability import get_logger

log = get_logger(__name__)

#: Payload fields that get an index, so filtering by them is cheap.
_INDEXED_FIELDS = ("paper_id", "chunk_type", "section", "year")


@dataclass
class CollectionStats:
    name: str
    vectors: int
    papers: int
    dimensions: int
    distance: str
    indexed: bool

    def summary(self) -> str:
        return (
            f"collection={self.name} vectors={self.vectors} papers={self.papers} "
            f"dim={self.dimensions} distance={self.distance}"
        )


class QdrantStore:
    """Thin, explicit wrapper over the Qdrant client."""

    def __init__(
        self,
        settings: Settings,
        client: Any | None = None,
        *,
        collection: str | None = None,
    ) -> None:
        self.settings = settings
        # The collection comes from the active corpus, not from settings, so
        # two corpora cannot end up sharing one collection.
        self.collection = collection or settings.qdrant_collection or "vectora_default"
        self._client = client
        self._ensured = False
        # Embedded (local-mode) Qdrant is not safe to call from several threads
        # at once: its cosine search normalises the stored vector matrix *in
        # place* on every query. Retrieval runs one dense search per query
        # variant concurrently via asyncio.to_thread, so in balanced/deep mode
        # those in-place writes race. Run 2 showed the symptom: cosine scores
        # of ~1e13, numpy overflow warnings from qdrant_client/local, and an
        # ablation baseline far below the identical eval run. A server
        # handles its own concurrency, so the lock is only taken in embedded mode.
        self._lock = threading.RLock()

    def _guard(self) -> contextlib.AbstractContextManager:
        return self._lock if self.settings.qdrant_path else contextlib.nullcontext()

    # ── connection ────────────────────────────────────────────────────────

    @property
    def client(self) -> Any:
        if self._client is None:
            from qdrant_client import QdrantClient

            if self.settings.qdrant_path:
                log.info("qdrant embedded", extra={"path": self.settings.qdrant_path})
                self._client = QdrantClient(path=self.settings.qdrant_path)
            else:
                log.info("qdrant server", extra={"url": self.settings.qdrant_url})
                self._client = QdrantClient(
                    url=self.settings.qdrant_url,
                    api_key=(
                        self.settings.qdrant_api_key.get_secret_value()
                        if self.settings.qdrant_api_key
                        else None
                    ),
                    timeout=int(self.settings.qdrant_timeout),
                    prefer_grpc=False,
                )
        return self._client

    def healthy(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception as exc:
            log.warning("qdrant unreachable", extra={"err": str(exc)[:160]})
            return False

    def exists(self) -> bool:
        try:
            names = {c.name for c in self.client.get_collections().collections}
            return self.collection in names
        except Exception:
            return False

    # ── schema ────────────────────────────────────────────────────────────

    def ensure_collection(self, *, recreate: bool = False) -> None:
        """Create the collection and its payload indexes if needed.

        Validates the dimension of an existing collection against the
        configured embedding model: a mismatch means the index was built
        with a different encoder and would return nonsense, so it fails
        loudly rather than silently retrieving garbage.
        """
        from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

        if recreate and self.exists():
            log.warning("dropping collection", extra={"collection": self.collection})
            self.client.delete_collection(self.collection)

        if self.exists():
            info = self.client.get_collection(self.collection)
            params = info.config.params.vectors
            dim = getattr(params, "size", None)
            if dim is not None and dim != self.settings.embedding_dim:
                raise RuntimeError(
                    f"Collection '{self.collection}' has {dim}-dim vectors but "
                    f"{self.settings.embedding_model} produces "
                    f"{self.settings.embedding_dim}. Re-index with "
                    f"`vectora index --recreate`, or use a different corpus."
                )
            self._check_embedding_model()
            self._ensured = True
            return

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(
                size=self.settings.embedding_dim, distance=Distance.COSINE
            ),
        )
        for field in _INDEXED_FIELDS:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception as exc:  # index already present, or unsupported
                log.debug("payload index skipped", extra={"field": field, "err": str(exc)[:80]})
        log.info(
            "collection created",
            extra={"collection": self.collection, "dim": self.settings.embedding_dim},
        )
        self._ensured = True

    # ── writes ────────────────────────────────────────────────────────────

    def _check_embedding_model(self) -> None:
        """Refuse an index built by a different encoder of the same dimension.

        The dimension check above cannot tell all-MiniLM-L6-v2 from
        bge-small-en-v1.5 (both 384-d): querying one model's index with the
        other's vectors raises nothing and returns plausible-looking garbage.
        Every point records the model that embedded it, so compare one.
        """
        with self._guard():
            points, _ = self.client.scroll(
                collection_name=self.collection, limit=1, with_payload=True, with_vectors=False
            )
        if not points:
            return
        built_with = (points[0].payload or {}).get("embedding_model")
        if built_with is None:
            log.warning(
                "index does not record its embedding model; if VECTORA_EMBEDDING_MODEL "
                "changed since it was built, re-index with `vectora index --recreate`",
                extra={"collection": self.collection},
            )
        elif built_with != self.settings.embedding_model:
            raise RuntimeError(
                f"Collection '{self.collection}' was embedded with {built_with}, but "
                f"VECTORA_EMBEDDING_MODEL is {self.settings.embedding_model}. Same "
                f"dimension, different vector space: results would be nonsense. "
                f"Re-index with `vectora index --recreate`, or set the model back."
            )

    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> int:
        """Idempotently store chunks and their vectors.

        Safe to call repeatedly with the same input: point ids are derived
        from content, so a repeat is an update, not a duplicate.
        """
        from qdrant_client.models import PointStruct

        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        if not chunks:
            return 0
        if not self._ensured:
            self.ensure_collection()

        total = 0
        batch_size = self.settings.qdrant_upsert_batch
        for start in range(0, len(chunks), batch_size):
            window = chunks[start : start + batch_size]
            points = [
                PointStruct(
                    id=chunk.point_id,
                    vector=vectors[start + i].tolist(),
                    payload={**chunk.payload(), "embedding_model": self.settings.embedding_model},
                )
                for i, chunk in enumerate(window)
            ]
            with self._guard():
                self.client.upsert(collection_name=self.collection, points=points, wait=True)
            total += len(points)
        log.info("upserted", extra={"points": total, "collection": self.collection})
        return total

    def delete_paper(self, paper_id: str) -> None:
        """Remove every vector belonging to one paper."""
        from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

        self.client.delete(
            collection_name=self.collection,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id))]
                )
            ),
            wait=True,
        )
        log.info("paper deleted from index", extra={"paper_id": paper_id})

    # ── reads ─────────────────────────────────────────────────────────────

    def build_filter(self, filters: dict[str, Any] | None) -> Any | None:
        """Translate a plain dict into a Qdrant filter.

        A list value becomes a 'any of' match; ``year`` accepts a
        ``{"gte": 2015}`` style range.
        """
        if not filters:
            return None
        from qdrant_client.models import (
            FieldCondition,
            Filter,
            MatchAny,
            MatchValue,
            Range,
        )

        must = []
        for key, value in filters.items():
            if value is None or value == "":
                continue
            if isinstance(value, (list, tuple, set)):
                items = [str(v) for v in value if v not in (None, "")]
                if items:
                    must.append(FieldCondition(key=key, match=MatchAny(any=items)))
            elif isinstance(value, dict):
                must.append(FieldCondition(key=key, range=Range(**value)))
            else:
                must.append(FieldCondition(key=key, match=MatchValue(value=str(value))))
        return Filter(must=must) if must else None

    def search(
        self,
        vector: np.ndarray | Sequence[float],
        *,
        limit: int,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Nearest neighbours, as validated chunks with their cosine score."""
        if not self._ensured:
            self.ensure_collection()
        query = vector.tolist() if isinstance(vector, np.ndarray) else list(vector)

        with self._guard():
            response = self.client.query_points(
                collection_name=self.collection,
                query=query,
                limit=limit,
                with_payload=True,
                with_vectors=False,
                query_filter=self.build_filter(filters),
                score_threshold=score_threshold,
            )
        out: list[tuple[Chunk, float]] = []
        for point in response.points:
            chunk = _to_chunk(point.payload)
            if chunk is not None:
                out.append((chunk, float(point.score)))
        return out

    def iter_all(
        self, *, batch_size: int = 512, filters: dict[str, Any] | None = None
    ) -> Iterator[Chunk]:
        """Stream every chunk in the collection.

        Used to build the sparse index from whatever is actually indexed,
        rather than trusting that ``chunks.jsonl`` still matches.
        """
        if not self.exists():
            return
        offset = None
        qfilter = self.build_filter(filters)
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=qfilter,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                chunk = _to_chunk(point.payload)
                if chunk is not None:
                    yield chunk
            if offset is None:
                break

    def indexed_paper_ids(self) -> set[str]:
        """Paper ids already present, for incremental embedding."""
        if not self.exists():
            return set()
        found: set[str] = set()
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                limit=1024,
                offset=offset,
                with_payload=["paper_id"],
                with_vectors=False,
            )
            for point in points:
                pid = (point.payload or {}).get("paper_id")
                if pid:
                    found.add(str(pid))
            if offset is None:
                break
        return found

    def stats(self) -> CollectionStats:
        if not self.exists():
            return CollectionStats(self.collection, 0, 0, self.settings.embedding_dim, "cosine", False)
        info = self.client.get_collection(self.collection)
        count = self.client.count(self.collection, exact=True).count
        params = info.config.params.vectors
        return CollectionStats(
            name=self.collection,
            vectors=count,
            papers=len(self.indexed_paper_ids()),
            dimensions=getattr(params, "size", self.settings.embedding_dim),
            distance=str(getattr(params, "distance", "cosine")).split(".")[-1].lower(),
            indexed=True,
        )

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None


def _to_chunk(payload: dict[str, Any] | None) -> Chunk | None:
    if not payload:
        return None
    try:
        return Chunk.model_validate(payload)
    except Exception as exc:
        log.warning("unreadable payload in index", extra={"err": str(exc)[:120]})
        return None


def embed_and_store(
    chunks: Sequence[Chunk],
    store: QdrantStore,
    encoder: Any,
    *,
    batch_size: int = 512,
    progress: bool = True,
) -> int:
    """Encode and upsert in windows, so peak memory stays bounded.

    Embedding 9,101 chunks at once materialises every vector before the
    first write; windowing keeps memory flat and makes an interrupted run
    resumable, since completed windows are already committed.
    """
    if not chunks:
        return 0
    store.ensure_collection()

    windows = range(0, len(chunks), batch_size)
    try:
        from tqdm import tqdm

        iterator: Iterable[int] = tqdm(windows, desc="Indexing", unit="batch") if progress else windows
    except ImportError:
        iterator = windows

    total = 0
    for start in iterator:
        window = list(chunks[start : start + batch_size])
        vectors = encoder.encode([c.content for c in window], progress=False)
        total += store.upsert(window, vectors)
    return total
