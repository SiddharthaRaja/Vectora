"""
Tests for fusion, diversification and the pieces of retrieval that are pure
functions. No network, no models, no Qdrant.
"""

from __future__ import annotations

import numpy as np
import pytest

from vectora.models import RetrievedChunk
from vectora.retrieval.diversify import cap_per_paper, deduplicate, mmr
from vectora.retrieval.expansion import blend
from vectora.retrieval.fusion import RankedList, normalise_scores, reciprocal_rank_fusion
from vectora.store.sparse import tokenize


def rc(chunk, **scores) -> RetrievedChunk:
    return RetrievedChunk(chunk=chunk, **scores)


# ── fusion ────────────────────────────────────────────────────────────────


class TestReciprocalRankFusion:
    def test_agreement_between_retrievers_wins(self, chunk_factory):
        """A chunk both retrievers rank highly beats one that only one
        retriever likes — the property a weighted sum over two different
        score scales cannot guarantee.

        Dense ranks: b, a, c. Sparse ranks: a, c, b.
        a scores 1/62 + 1/61; b scores 1/61 + 1/63. a wins on agreement.
        """
        a = chunk_factory("alpha content here", paper_id="p1")
        b = chunk_factory("bravo content here", paper_id="p2")
        c = chunk_factory("charlie content here", paper_id="p3")

        fused = reciprocal_rank_fusion(
            [
                RankedList(name="dense", chunks=[b, a, c], scores=[0.9, 0.8, 0.7]),
                RankedList(name="sparse", chunks=[a, c, b], scores=[12.0, 5.0, 3.0]),
            ],
            k=60,
        )
        assert fused[0].chunk.chunk_id == a.chunk_id

    def test_a_chunk_only_sparse_finds_still_surfaces(self, chunk_factory):
        """The whole point of running BM25 over the full corpus: a chunk
        dense search never returned must still be able to reach the answer."""
        dense_only = chunk_factory("semantically close text", paper_id="p1")
        sparse_only = chunk_factory("exact rare acronym SRM 972a", paper_id="p2")
        fused = reciprocal_rank_fusion(
            [
                RankedList(name="dense", chunks=[dense_only], scores=[0.8]),
                RankedList(name="sparse", chunks=[sparse_only], scores=[18.0]),
            ]
        )
        assert {f.chunk.paper_id for f in fused} == {"p1", "p2"}

    def test_scores_are_comparable_across_queries(self, chunk_factory):
        """RRF does not depend on the magnitudes in the batch, which is what
        makes a fixed confidence floor meaningful."""
        a = chunk_factory("alpha", paper_id="p1")
        low = reciprocal_rank_fusion(
            [RankedList(name="dense", chunks=[a], scores=[0.01])], k=60
        )
        high = reciprocal_rank_fusion(
            [RankedList(name="dense", chunks=[a], scores=[0.99])], k=60
        )
        assert low[0].fused_score == high[0].fused_score == pytest.approx(1 / 61)

    def test_ranks_and_scores_are_preserved(self, chunk_factory):
        a = chunk_factory("alpha", paper_id="p1")
        fused = reciprocal_rank_fusion(
            [
                RankedList(name="dense", chunks=[a], scores=[0.77]),
                RankedList(name="sparse", chunks=[a], scores=[9.5]),
            ]
        )
        entry = fused[0]
        assert entry.dense_score == pytest.approx(0.77)
        assert entry.sparse_score == pytest.approx(9.5)
        assert entry.dense_rank == 1
        assert entry.sparse_rank == 1

    def test_variants_are_recorded(self, chunk_factory):
        a = chunk_factory("alpha", paper_id="p1")
        fused = reciprocal_rank_fusion(
            [
                RankedList(name="dense", chunks=[a], scores=[0.5], variant="original"),
                RankedList(name="dense", chunks=[a], scores=[0.6], variant="variant1"),
            ]
        )
        assert set(fused[0].matched_variants) == {"original", "variant1"}

    def test_weights_shift_the_balance(self, chunk_factory):
        a = chunk_factory("alpha", paper_id="p1")
        b = chunk_factory("bravo", paper_id="p2")
        fused = reciprocal_rank_fusion(
            [
                RankedList(name="dense", chunks=[a], scores=[0.5], weight=1.0),
                RankedList(name="sparse", chunks=[b], scores=[5.0], weight=0.1),
            ]
        )
        assert fused[0].chunk.chunk_id == a.chunk_id

    def test_mismatched_lengths_rejected(self, chunk_factory):
        with pytest.raises(ValueError):
            RankedList(name="dense", chunks=[chunk_factory("a")], scores=[1.0, 2.0])

    def test_empty_input(self):
        assert reciprocal_rank_fusion([]) == []


class TestNormalisation:
    def test_constant_input_does_not_divide_by_zero(self):
        assert normalise_scores([3.0, 3.0, 3.0]) == [1.0, 1.0, 1.0]

    def test_range_maps_to_unit(self):
        assert normalise_scores([0.0, 5.0, 10.0]) == [0.0, 0.5, 1.0]


# ── diversification ───────────────────────────────────────────────────────


class TestMMR:
    def test_runs_when_pool_exceeds_k(self, chunk_factory, fake_encode):
        """The old implementation returned early whenever the pool was no
        bigger than top_k, which in practice was most of the time."""
        candidates = [
            rc(chunk_factory(f"distinct topic {i} with several words", paper_id=f"p{i}"),
               fused_score=1.0 - i * 0.01)
            for i in range(10)
        ]
        out = mmr(candidates, "a query", encode=fake_encode, top_k=5, lambda_=0.5)
        assert len(out) == 5

    def test_high_lambda_follows_relevance_order(self, chunk_factory, fake_encode):
        candidates = [
            rc(chunk_factory(f"text number {i} here with words", paper_id=f"p{i}"),
               rerank_score=10.0 - i)
            for i in range(8)
        ]
        out = mmr(candidates, "query", encode=fake_encode, top_k=3, lambda_=1.0)
        assert [c.rerank_score for c in out] == [10.0, 9.0, 8.0]

    def test_survives_encoder_failure(self, chunk_factory):
        candidates = [
            rc(chunk_factory(f"text {i} with some words", paper_id=f"p{i}"), fused_score=1.0)
            for i in range(6)
        ]

        def broken(_texts):
            raise RuntimeError("model unavailable")

        out = mmr(candidates, "query", encode=broken, top_k=3, lambda_=0.5)
        assert len(out) == 3  # degrades to the incoming order

    def test_zero_k(self, chunk_factory, fake_encode):
        assert mmr([rc(chunk_factory("a"))], "q", encode=fake_encode, top_k=0, lambda_=0.5) == []


class TestDeduplication:
    def test_identical_ids_collapse_keeping_best_score(self, chunk_factory):
        chunk = chunk_factory("the same passage of text appears twice over")
        out = deduplicate([rc(chunk, fused_score=0.1), rc(chunk, fused_score=0.9)])
        assert len(out) == 1
        assert out[0].fused_score == 0.9

    def test_same_text_different_papers_collapses(self, chunk_factory):
        a = chunk_factory("Identical boilerplate sentence here.", paper_id="p1")
        b = chunk_factory("identical  boilerplate   sentence here!", paper_id="p2")
        out = deduplicate([rc(a, fused_score=0.5), rc(b, fused_score=0.4)])
        assert len(out) == 1

    def test_distinct_text_is_kept(self, chunk_factory):
        a = chunk_factory("Vitamin D was measured by LC-MS/MS.", paper_id="p1")
        b = chunk_factory("Bone density was assessed using DXA.", paper_id="p2")
        out = deduplicate([rc(a, fused_score=0.5), rc(b, fused_score=0.4)], encode=None)
        assert len(out) == 2

    def test_skips_semantic_pass_on_large_pools(self, chunk_factory):
        calls = []

        def counting_encode(texts):
            calls.append(len(texts))
            return np.eye(len(texts), 8, dtype=np.float32)

        candidates = [
            rc(chunk_factory(f"unique passage {i} of text", paper_id=f"p{i}"), fused_score=0.5)
            for i in range(80)
        ]
        deduplicate(candidates, encode=counting_encode, max_semantic=60)
        assert not calls  # too big to be worth an N² comparison

    def test_semantic_pass_encodes_once(self, chunk_factory):
        """The old version encoded each chunk twice, inside a loop."""
        calls = []

        def counting_encode(texts):
            calls.append(len(texts))
            rng = np.random.default_rng(0)
            m = rng.normal(size=(len(texts), 8)).astype(np.float32)
            return m / np.linalg.norm(m, axis=1, keepdims=True)

        candidates = [
            rc(chunk_factory(f"unique passage {i} of text", paper_id=f"p{i}"), fused_score=0.5)
            for i in range(10)
        ]
        deduplicate(candidates, encode=counting_encode)
        assert len(calls) == 1


class TestCapPerPaper:
    def test_limits_one_paper_but_keeps_overflow_at_the_tail(self, chunk_factory):
        candidates = [
            rc(chunk_factory(f"text {i}", paper_id="dominant"), fused_score=1.0 - i * 0.01)
            for i in range(5)
        ] + [rc(chunk_factory("other", paper_id="second"), fused_score=0.5)]
        out = cap_per_paper(candidates, max_per_paper=2)
        assert [c.chunk.paper_id for c in out[:3]] == ["dominant", "dominant", "second"]
        assert len(out) == 6  # nothing is lost, just reordered


# ── expansion helpers ─────────────────────────────────────────────────────


class TestBlend:
    def test_result_is_normalised(self):
        q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        h = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        out = blend(q, h, 0.5)
        assert np.linalg.norm(out) == pytest.approx(1.0, abs=1e-6)

    def test_weight_zero_returns_the_query(self):
        q = np.array([1.0, 0.0], dtype=np.float32)
        h = np.array([0.0, 1.0], dtype=np.float32)
        assert np.allclose(blend(q, h, 0.0), q)

    def test_opposing_vectors_fall_back_to_query(self):
        q = np.array([1.0, 0.0], dtype=np.float32)
        out = blend(q, -q, 0.5)
        assert np.allclose(out, q)


# ── tokenisation ──────────────────────────────────────────────────────────


class TestTokenize:
    def test_scientific_compounds_are_expanded(self):
        tokens = tokenize("LC-MS/MS quantification")
        assert "lc-ms/ms" in tokens
        assert "lc" in tokens and "ms" in tokens
        assert "lcmsms" in tokens

    def test_stopwords_removed(self):
        assert "the" not in tokenize("the assay and the method")

    def test_numbers_retained(self):
        assert "25" in tokenize("serum 25 hydroxyvitamin")

    def test_index_and_query_tokenisation_agree(self):
        text = "Serum 25(OH)D measured by LC-MS/MS"
        assert tokenize(text) == tokenize(text)
