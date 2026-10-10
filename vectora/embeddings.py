"""
Encoder and reranker loading, with a query-embedding cache.

The two models are loaded lazily and once per process. Loading them at import
time would make the CLI's ``--help`` take ten seconds and would download
~150MB during a unit test run.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence

import numpy as np

from vectora.cache import Cache, make_key
from vectora.config import Settings
from vectora.observability import get_logger

log = get_logger(__name__)


class Encoder:
    """Dense bi-encoder. Returns L2-normalised vectors.

    Normalising at encode time means cosine similarity is a plain dot
    product everywhere downstream, which both Qdrant and the MMR code rely
    on.
    """

    def __init__(self, settings: Settings, cache: Cache | None = None) -> None:
        self.settings = settings
        self.cache = cache
        self._model = None
        self._lock = threading.Lock()

    @property
    def model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    device = self.settings.resolved_device
                    log.info(
                        "loading encoder",
                        extra={"model": self.settings.embedding_model, "device": device},
                    )
                    self._model = SentenceTransformer(
                        self.settings.embedding_model, device=device
                    )
                    actual = self._model.get_sentence_embedding_dimension()
                    if actual != self.settings.embedding_dim:
                        raise RuntimeError(
                            f"embedding_dim is {self.settings.embedding_dim} but "
                            f"{self.settings.embedding_model} produces {actual}. "
                            "Fix VECTORA_EMBEDDING_DIM and rebuild the index."
                        )
        return self._model

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
        progress: bool = False,
    ) -> np.ndarray:
        """Encode a batch. Shape ``(len(texts), embedding_dim)``, float32."""
        if not texts:
            return np.zeros((0, self.settings.embedding_dim), dtype=np.float32)
        vecs = self.model.encode(
            list(texts),
            batch_size=batch_size or self.settings.embedding_batch_size,
            show_progress_bar=progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(vecs, dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        """Encode one query, memoised.

        Query embeddings are cached because the same question is often asked
        again — by a retry, by the eval harness, or by a user refining a
        conversation — and encoding is the slowest step in ``fast`` mode.
        """
        key = make_key("qvec", self.settings.embedding_model, text)
        if self.cache:
            hit = self.cache.get(key)
            if hit is not None:
                return np.asarray(hit, dtype=np.float32)
        vec = self.encode([text])[0]
        if self.cache:
            self.cache.set(key, vec.tolist())
        return vec

    def warm(self) -> None:
        """Force model load, so the first real request is not the slow one."""
        self.encode(["warmup"])


class Reranker:
    """Cross-encoder. Scores are unbounded logits, higher is better.

    Slower than the bi-encoder by roughly the candidate count, so it only
    ever sees a small pool, never the collection.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = None
        self._lock = threading.Lock()

    @property
    def model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import CrossEncoder

                    log.info("loading reranker", extra={"model": self.settings.reranker_model})
                    self._model = CrossEncoder(
                        self.settings.reranker_model,
                        max_length=self.settings.reranker_max_length,
                        device=self.settings.resolved_device,
                    )
        return self._model

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        scores = self.model.predict(
            [(query, p) for p in passages],
            batch_size=self.settings.reranker_batch_size,
            show_progress_bar=False,
        )
        return [float(s) for s in scores]

    def warm(self) -> None:
        self.score("warmup", ["warmup passage"])


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity for already-normalised matrices."""
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    return np.clip(a @ b.T, -1.0, 1.0)
