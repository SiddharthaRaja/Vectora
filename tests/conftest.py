"""Shared fixtures. No network, no model downloads, no Qdrant."""

from __future__ import annotations

import numpy as np
import pytest

from vectora.config import Settings, settings_for_test
from vectora.models import Chunk


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Isolated settings: every path inside tmp_path, embedded Qdrant."""
    return settings_for_test(
        qdrant_url=None,
        qdrant_path=str(tmp_path / "qdrant"),
        sqlite_path=tmp_path / "chats.db",
        cache_path=tmp_path / "cache.db",
        chunks_path=tmp_path / "chunks.jsonl",
        manifest_path=tmp_path / "manifest.json",
        images_dir=tmp_path / "images",
        bm25_index_path=tmp_path / "bm25.pkl",
        papers_dir=tmp_path / "papers",
        enable_cache=False,
        min_chunk_tokens=20,
        max_chunk_tokens=120,
    )


@pytest.fixture
def fake_encode():
    """Deterministic, normalised embeddings derived from the text.

    Hash-based rather than random so a test that depends on two texts being
    similar is reproducible, and no model is ever downloaded.
    """

    def encode(texts):
        out = []
        for text in texts:
            rng = np.random.default_rng(abs(hash(text.lower().strip())) % (2**32))
            vector = rng.normal(size=16).astype(np.float32)
            # Lexical overlap nudges vectors together, so similarity is not
            # pure noise for the tests that assert on it.
            for i, token in enumerate(sorted(set(text.lower().split()))[:8]):
                vector[i % 16] += (abs(hash(token)) % 100) / 50.0
            out.append(vector / np.linalg.norm(vector))
        return np.vstack(out) if out else np.zeros((0, 16), dtype=np.float32)

    return encode


def make_chunk(
    content: str,
    *,
    paper_id: str = "paper_a",
    title: str = "A Study of Something",
    section: str = "Methods",
    page: int = 3,
    chunk_type: str = "text",
    quality: float = 1.0,
    year: str = "2020",
    authors: str = "A.M. Wallace, S. Gibson",
) -> Chunk:
    return Chunk.build(
        content=content,
        paper_id=paper_id,
        title=title,
        section=section,
        page=page,
        chunk_type=chunk_type,  # type: ignore[arg-type]
        quality=quality,
        year=year,
        authors=authors,
    )


@pytest.fixture
def chunk_factory():
    return make_chunk


PROSE = (
    "Serum 25-hydroxyvitamin D was quantified by liquid chromatography tandem "
    "mass spectrometry using a deuterated internal standard. Calibration was "
    "performed against NIST SRM 972a reference material across the range 5 to "
    "150 ng/mL. Inter-assay coefficients of variation were below 8 percent at "
    "all three quality control levels, and the lower limit of quantification "
    "was established at 2.5 ng/mL."
)


@pytest.fixture
def prose() -> str:
    return PROSE
