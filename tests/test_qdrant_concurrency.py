"""
Regression tests for the embedded-Qdrant race (fixed in 50c59bc).

The bug: qdrant-client's local mode normalises the stored vector matrix *in
place* on every cosine search. Retrieval runs one dense search per query
variant concurrently (``asyncio.to_thread``), so in balanced/deep mode those
in-place writes overlapped. Run 2 saw cosine scores of ~2.4e13 and an
ablation baseline of 0.239 against the identical eval's 0.591.

The race itself is timing-dependent and does not reproduce on demand, so the
first test checks the property that prevents it — no two calls are ever
inside the embedded client at once — with a fake client that makes overlap
certain if the lock is missing. The second runs real embedded Qdrant under
concurrent load and checks every score is a valid cosine.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from vectora.store.qdrant_store import QdrantStore


class _Response:
    points: list = []


class OverlapCountingClient:
    """Records the most calls that were ever inside it simultaneously."""

    def __init__(self, hold: float = 0.02) -> None:
        self.hold = hold
        self.inside = 0
        self.max_inside = 0
        self._count = threading.Lock()

    def _enter(self) -> None:
        with self._count:
            self.inside += 1
            self.max_inside = max(self.max_inside, self.inside)
        time.sleep(self.hold)  # wide window: without a lock, overlap is certain
        with self._count:
            self.inside -= 1

    def query_points(self, **_: object) -> _Response:
        self._enter()
        return _Response()

    def upsert(self, **_: object) -> None:
        self._enter()


def _hammer(store: QdrantStore, calls: int = 16) -> None:
    store._ensured = True  # skip collection checks: the fake has none
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: store.search(np.ones(4), limit=3), range(calls)))


def test_embedded_mode_never_enters_client_concurrently(settings):
    client = OverlapCountingClient()
    store = QdrantStore(settings, client=client, collection="t")
    _hammer(store)
    assert client.max_inside == 1


def test_server_mode_is_not_serialised(settings):
    """A Qdrant server handles its own concurrency; locking would only cost latency."""
    server = settings.model_copy(update={"qdrant_path": None, "qdrant_url": "http://x:6333"})
    client = OverlapCountingClient()
    store = QdrantStore(server, client=client, collection="t")
    _hammer(store)
    assert client.max_inside > 1


def test_concurrent_embedded_searches_return_valid_cosines(settings, chunk_factory):
    """Real local-mode Qdrant, many threads: every score must stay in [-1, 1]."""
    pytest.importorskip("qdrant_client")
    rng = np.random.default_rng(0)
    dim = settings.embedding_dim
    chunks = [chunk_factory(f"chunk number {i} text", paper_id=f"p{i % 7}") for i in range(400)]
    # Deliberately not unit length, so any read of a half-normalised matrix
    # shows up as an out-of-range score.
    vectors = rng.normal(size=(len(chunks), dim)).astype(np.float32) * 5

    store = QdrantStore(settings, collection="race")
    store.ensure_collection(recreate=True)
    store.upsert(chunks, vectors)

    queries = rng.normal(size=(64, dim)).astype(np.float32)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda q: store.search(q, limit=10), queries))
    store.close()

    scores = [s for hits in results for _, s in hits]
    assert len(scores) == 64 * 10
    assert max(abs(s) for s in scores) <= 1.0 + 1e-4


def test_index_built_with_another_same_size_model_is_refused(settings, chunk_factory):
    """MiniLM and bge-small are both 384-d: the dimension check cannot tell
    them apart, so the model name stored with each point must."""
    pytest.importorskip("qdrant_client")
    rng = np.random.default_rng(1)
    chunks = [chunk_factory(f"chunk {i} text here", paper_id="p1") for i in range(5)]
    vectors = rng.normal(size=(5, settings.embedding_dim)).astype(np.float32)

    built = settings.model_copy(update={"embedding_model": "sentence-transformers/all-MiniLM-L6-v2"})
    store = QdrantStore(built, collection="models")
    store.ensure_collection(recreate=True)
    store.upsert(chunks, vectors)
    store.close()

    same = QdrantStore(built, collection="models")
    same.ensure_collection()  # same model: fine
    same.close()

    other = settings.model_copy(update={"embedding_model": "BAAI/bge-small-en-v1.5"})
    reopened = QdrantStore(other, collection="models")
    with pytest.raises(RuntimeError, match="embedded with"):
        reopened.ensure_collection()
    reopened.close()
