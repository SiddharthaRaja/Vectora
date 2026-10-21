"""
Tests for routing, citation resolution, guards and the grounding
pre-checks. No OpenAI calls: every LLM-dependent path is exercised with the
client set to ``None``, which is also the degraded path that must work in
production when the API is down.
"""

from __future__ import annotations

import asyncio

import pytest

from vectora.config import QueryType
from vectora.generation import citations as cite
from vectora.generation import grounding, guards
from vectora.generation.pipeline import build_context, clean_history
from vectora.generation.router import classify, classify_by_rules
from vectora.models import RetrievedChunk


def rc(chunk, **scores) -> RetrievedChunk:
    return RetrievedChunk(chunk=chunk, **scores)


# ── routing ───────────────────────────────────────────────────────────────


class TestRuleRouting:
    @pytest.mark.parametrize(
        "query,expected",
        [
            ("Compare LC-MS/MS and immunoassay for 25(OH)D", QueryType.COMPARATIVE),
            ("What is the difference between D2 and D3?", QueryType.COMPARATIVE),
            ("What is calcitriol?", QueryType.DEFINITIONAL),
            ("Define vitamin D deficiency", QueryType.DEFINITIONAL),
            ("What does LOD stand for?", QueryType.DEFINITIONAL),
            ("How was serum 25(OH)D quantified?", QueryType.METHODOLOGICAL),
            ("What methods were used for calibration?", QueryType.METHODOLOGICAL),
            ("Why does vitamin D deficiency cause rickets?", QueryType.CAUSAL),
            ("What are the effects of supplementation?", QueryType.CAUSAL),
            ("Give me an overview of vitamin D assays", QueryType.EXPLORATORY),
            ("Summarise the main findings", QueryType.EXPLORATORY),
            ("What is the concentration threshold for deficiency?", QueryType.FACTUAL),
        ],
    )
    def test_common_phrasings_need_no_llm_call(self, query, expected):
        assert classify_by_rules(query) == expected

    def test_unclear_query_defers(self):
        assert classify_by_rules("vitamin D") is None

    def test_default_is_not_exploratory(self, settings):
        """Exploratory pulls the widest, most diffuse candidate set, so it is
        the worst thing to fall back to."""
        query_type, how = asyncio.run(
            classify("vitamin D", llm=None, settings=settings)
        )
        assert how == "default"
        assert query_type is QueryType.METHODOLOGICAL


# ── citations ─────────────────────────────────────────────────────────────


class TestMarkerParsing:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Claim [1] and another [3].", [1, 3]),
            ("Both agree [1][2].", [1, 2]),
            ("Listed [1, 3, 5].", [1, 3, 5]),
            ("A range [1-3].", [1, 2, 3]),
            ("Repeated [2] then [2] again.", [2]),
            ("No citations here.", []),
        ],
    )
    def test_parsing(self, text, expected):
        assert cite.parse_markers(text) == expected


class TestInvalidMarkers:
    def test_hallucinated_numbers_are_removed(self):
        """A model given 3 sources sometimes cites [7]; left in place that
        reads as a reference the reader cannot follow."""
        cleaned, removed = cite.strip_invalid_markers(
            "Supported [1] but also [7].", valid={1, 2, 3}
        )
        assert "[7]" not in cleaned
        assert "[1]" in cleaned
        assert removed == [7]

    def test_partially_valid_group_keeps_the_valid_part(self):
        cleaned, removed = cite.strip_invalid_markers("Claim [1][9].", valid={1})
        assert cleaned == "Claim [1]."
        assert removed == [9]

    def test_spacing_is_tidied(self):
        cleaned, _ = cite.strip_invalid_markers("A claim [9] .", valid={1})
        assert "  " not in cleaned


class TestCitationResolution:
    def test_citations_resolve_to_the_chunks_shown(self, chunk_factory):
        chunks = [
            rc(chunk_factory("first source text", paper_id="p1", title="Paper One")),
            rc(chunk_factory("second source text", paper_id="p2", title="Paper Two")),
        ]
        answer, citations = cite.citations_from_markers(
            "The assay is sensitive [2] and reproducible [1].", chunks
        )
        assert [c.index for c in citations] == [2, 1]
        assert citations[0].title == "Paper Two"
        # Valid markers survive untouched; only invalid ones are stripped.
        assert "[2]" in answer and "[1]" in answer
        assert citations[1].paper_id == "p1"

    def test_uncited_answer_gets_no_citations(self, chunk_factory):
        chunks = [rc(chunk_factory("source text here"))]
        _, citations = cite.citations_from_markers("An uncited claim.", chunks)
        assert citations == []

    def test_dedupe_by_paper_keeps_one_reference_each(self, chunk_factory):
        chunks = [
            rc(chunk_factory("a", paper_id="p1", title="One")),
            rc(chunk_factory("b", paper_id="p1", title="One")),
            rc(chunk_factory("c", paper_id="p2", title="Two")),
        ]
        _, citations = cite.citations_from_markers("Claims [1][2][3].", chunks)
        assert len(cite.dedupe_by_paper(citations)) == 2


class TestQuoteVerification:
    def test_exact_quote_accepted(self, prose):
        assert cite.verify_quote("deuterated internal standard", prose)

    def test_whitespace_variation_accepted(self, prose):
        assert cite.verify_quote("deuterated   internal\nstandard", prose)

    def test_fabricated_quote_rejected(self, prose):
        assert cite.verify_quote("the results were conclusively negative", prose) == ""

    def test_too_short_to_verify(self, prose):
        assert cite.verify_quote("the", prose) == ""


class TestMergeCitations:
    def test_quotes_are_added_but_citations_are_not_invented(self, chunk_factory):
        chunks = [rc(chunk_factory("some source text here with content"))]
        _, primary = cite.citations_from_markers("A claim [1].", chunks)
        enriched = [
            cite.Citation.from_retrieved(chunks[0], 1, quote="source text"),
            cite.Citation.from_retrieved(chunks[0], 2, quote="never cited"),
        ]
        merged = cite.merge_citations(primary, enriched)
        assert len(merged) == 1
        assert merged[0].quote == "source text"


# ── guards ────────────────────────────────────────────────────────────────


class TestGuards:
    def test_empty_retrieval_blocks(self, settings):
        ok, reason = guards.check([], settings)
        assert not ok and reason == "no_results"

    def test_reranked_score_uses_the_logit_floor(self, settings, chunk_factory):
        weak = [rc(chunk_factory("x" * 300), rerank_score=-9.0, fused_score=0.02)]
        ok, reason = guards.check(weak, settings)
        assert not ok and reason == "low_confidence"

        strong = [rc(chunk_factory("x" * 300), rerank_score=2.0, fused_score=0.02)]
        assert guards.check(strong, settings)[0]

    def test_unreranked_falls_back_to_the_dense_cosine_floor(self, settings, chunk_factory):
        """The bug this covers: the logit floor was compared against a 0-1
        similarity in fast mode, so the guard could never fire there."""
        weak = [rc(chunk_factory("x" * 300), dense_score=0.09, fused_score=0.0164)]
        ok, reason = guards.check(weak, settings)
        assert not ok and reason == "low_confidence"

        strong = [rc(chunk_factory("x" * 300), dense_score=0.85, fused_score=0.0164)]
        assert guards.check(strong, settings)[0]

    def test_rrf_alone_cannot_gate(self, settings, chunk_factory):
        """RRF ranks, it does not measure relevance: a chunk placed first in
        a hopeless pool scores identically to one placed first in a good
        pool. So an absolute score must win when one is present."""
        top_of_a_bad_pool = [
            rc(chunk_factory("x" * 300), dense_score=0.04, fused_score=1 / 61)
        ]
        ok, reason = guards.check(top_of_a_bad_pool, settings)
        assert not ok and reason == "low_confidence"

    def test_lexical_only_retrieval_still_gates_on_rrf(self, settings, chunk_factory):
        """With no dense score and no rerank, RRF is all there is."""
        nothing = [rc(chunk_factory("x" * 300), sparse_score=2.0, fused_score=0.0001)]
        assert not guards.check(nothing, settings)[0]

        something = [rc(chunk_factory("x" * 300), sparse_score=9.0, fused_score=0.03)]
        assert guards.check(something, settings)[0]

    def test_rerank_wins_over_dense_when_both_present(self, settings, chunk_factory):
        """A high cosine with a strongly negative cross-encoder logit is the
        classic false positive the reranker exists to catch."""
        misleading = [
            rc(chunk_factory("x" * 300), dense_score=0.61, rerank_score=-9.5)
        ]
        ok, reason = guards.check(misleading, settings)
        assert not ok and reason == "low_confidence"

    def test_related_topics_filter_out_debris(self, chunk_factory):
        chunks = [
            rc(chunk_factory("x" * 300, title="GModel ARTICLE IN PRESS", section="Review")),
            rc(chunk_factory("y" * 300, title="journal homepage: www.elsevier.com", section="")),
            rc(
                chunk_factory(
                    "z" * 300,
                    title="Measurement of 25-hydroxyvitamin D in the clinical laboratory",
                    section="Methods",
                )
            ),
        ]
        topics = guards.related_topics(chunks)
        assert len(topics) == 1
        assert "25-hydroxyvitamin" in topics[0]

    def test_fallback_works_without_an_llm(self, chunk_factory):
        chunks = [rc(chunk_factory("x" * 300, title="A Real Paper Title Here", section="Methods"))]
        fallback = asyncio.run(
            guards.build_fallback("some query", chunks, reason="low_confidence", llm=None)
        )
        assert fallback.triggered
        assert fallback.suggestions
        assert fallback.reason == "low_confidence"


# ── grounding pre-checks ──────────────────────────────────────────────────


class TestUnsupportedNumbers:
    def test_fabricated_figure_is_caught(self, chunk_factory, prose):
        chunks = [rc(chunk_factory(prose))]
        missing = grounding.unsupported_numbers(
            "The limit of quantification was 47.3 ng/mL.", chunks
        )
        assert "47.3" in missing

    def test_figure_present_in_sources_passes(self, chunk_factory, prose):
        chunks = [rc(chunk_factory(prose))]
        assert grounding.unsupported_numbers("The LOQ was 2.5 ng/mL.", chunks) == []

    def test_years_are_ignored(self, chunk_factory, prose):
        chunks = [rc(chunk_factory(prose))]
        assert grounding.unsupported_numbers("Reported in 1998.", chunks) == []


class TestCitationCoverage:
    def test_fully_cited(self):
        text = "First claim here with length [1]. Second claim also long enough [2]."
        assert grounding.citation_coverage(text) == 1.0

    def test_partially_cited(self):
        text = "First claim here with enough length [1]. Second claim with no citation at all."
        assert grounding.citation_coverage(text) == 0.5

    def test_short_answer_is_not_penalised(self):
        assert grounding.citation_coverage("Yes.") == 1.0


class TestMechanicalGrounding:
    def test_works_with_no_llm(self, chunk_factory, prose):
        chunks = [rc(chunk_factory(prose))]
        result = asyncio.run(
            grounding.check("The LOQ was 99.9 ng/mL.", chunks, llm=None)
        )
        assert result is not None
        assert result.score < 90
        assert result.unsupported_claims


# ── context and history ───────────────────────────────────────────────────


class TestContext:
    def test_sources_are_one_indexed_in_order(self, chunk_factory):
        chunks = [
            rc(chunk_factory("alpha text", title="A")),
            rc(chunk_factory("bravo text", title="B")),
        ]
        context = build_context(chunks)
        assert "SOURCE [1]" in context and "SOURCE [2]" in context
        assert context.index("SOURCE [1]") < context.index("SOURCE [2]")


class TestHistoryCleaning:
    def test_legacy_citation_blocks_are_stripped(self):
        history = [
            {"role": "user", "content": "a question"},
            {
                "role": "assistant",
                "content": 'An answer.\n```json\n{"citations_used": ["x"]}\n```',
            },
        ]
        out = clean_history(history, limit=10)
        assert "```json" not in out[1]["content"]
        assert out[1]["content"] == "An answer."

    def test_leaked_sources_are_stripped(self):
        history = [
            {"role": "assistant", "content": "An answer.\nSOURCES:\nSOURCE [1]\nTitle: x"}
        ]
        assert clean_history(history, limit=10)[0]["content"] == "An answer."

    def test_limit_is_applied(self):
        history = [{"role": "user", "content": f"q{i}"} for i in range(30)]
        assert len(clean_history(history, limit=5)) == 5

    def test_blank_and_unknown_roles_dropped(self):
        history = [
            {"role": "user", "content": "  "},
            {"role": "system", "content": "ignore me"},
            {"role": "user", "content": "real"},
        ]
        out = clean_history(history, limit=10)
        assert len(out) == 1 and out[0]["content"] == "real"
