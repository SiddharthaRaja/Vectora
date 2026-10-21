"""
Tests for the measurement tooling: ablation statistics, answer metrics and
the MCP surface.

The statistics get the most attention here. An ablation harness whose
confidence interval is wrong produces confident nonsense, which is worse
than no harness at all — it would let a change that does nothing look like
an improvement.
"""

from __future__ import annotations

import asyncio
import json
import random

import pytest

from evaluation.ablate import (
    Variant,
    _ndcg,
    paired_bootstrap,
    standard_ablations,
    weight_sweep,
)
from evaluation.answers import anchor_recall, citation_coverage
from evaluation.generate import sample_chunks
from vectora.config import MODE_PROFILES, SearchMode
from vectora.mcp_server import MCPServer
from vectora.tracing import (
    configure_tracing,
    is_enabled,
    question_span,
    set_attributes,
    shutdown_tracing,
    span,
)

# ── ablation statistics ───────────────────────────────────────────────────


class TestPairedBootstrap:
    def test_identical_vectors_show_no_difference(self):
        values = [1, 0, 1, 1, 0, 1, 1, 0, 1, 1.0]
        diff, low, high, significant = paired_bootstrap(values, values)
        assert diff == 0.0
        assert not significant
        assert low == high == 0.0

    def test_a_uniform_improvement_is_significant(self):
        diff, low, _, significant = paired_bootstrap([0.0] * 20, [1.0] * 20)
        assert diff == 1.0
        assert significant
        assert low > 0

    def test_noise_on_few_cases_is_not_significant(self):
        """The guard against overclaiming: with 15 cases a difference of a
        few points must not be reported as real."""
        rng = random.Random(3)
        base = [rng.random() for _ in range(15)]
        noisy = [b + rng.gauss(0, 0.3) for b in base]
        _, _, _, significant = paired_bootstrap(base, noisy)
        assert not significant

    def test_a_consistent_small_gain_becomes_significant_with_enough_cases(self):
        _, _, _, significant = paired_bootstrap([0.5] * 200, [0.56] * 200)
        assert significant

    def test_the_same_gain_is_not_significant_with_five_noisy_cases(self):
        _, _, _, significant = paired_bootstrap([0, 1, 0, 1, 0.0], [1, 0, 1, 0, 1.0])
        assert not significant

    def test_empty_input_is_handled(self):
        assert paired_bootstrap([], []) == (0.0, 0.0, 0.0, False)

    def test_is_deterministic(self):
        base = [0.1, 0.5, 0.9, 0.2, 0.7]
        variant = [0.3, 0.4, 0.95, 0.25, 0.8]
        assert paired_bootstrap(base, variant) == paired_bootstrap(base, variant)


class TestNdcg:
    def test_rank_one_is_perfect(self):
        assert _ndcg([1]) == pytest.approx(1.0)

    def test_a_full_prefix_is_perfect(self):
        assert _ndcg([1, 2]) == pytest.approx(1.0)
        assert _ndcg([1, 2, 3]) == pytest.approx(1.0)

    def test_later_ranks_score_lower(self):
        assert _ndcg([1]) > _ndcg([3]) > _ndcg([8]) > 0

    def test_ordering_matters(self):
        assert _ndcg([1, 2]) > _ndcg([4, 5])

    def test_nothing_relevant_is_zero(self):
        assert _ndcg([]) == 0.0


class TestVariants:
    def test_each_single_variant_changes_exactly_one_thing(self, settings):
        """An ablation that changes two things cannot attribute the result,
        so single variants are checked to differ in exactly one respect."""
        base_profile = MODE_PROFILES[SearchMode.BALANCED]
        for variant in standard_ablations():
            if variant.is_baseline or variant.combined:
                continue
            tuned_settings, tuned_profile = variant.apply(settings, SearchMode.BALANCED)
            changes = 0
            for field in ("dense_weight", "sparse_weight", "rrf_k"):
                if getattr(tuned_settings, field) != getattr(settings, field):
                    changes += 1
            for field in ("use_reranker", "use_mmr", "use_hyde", "rewrite_variants"):
                if getattr(tuned_profile, field) != getattr(base_profile, field):
                    changes += 1
            assert changes == 1, f"{variant.name} changed {changes} things"

    def test_applying_a_variant_does_not_mutate_the_originals(self, settings):
        base_profile = MODE_PROFILES[SearchMode.BALANCED]
        original_sparse = settings.sparse_weight
        original_rerank = base_profile.use_reranker
        for variant in standard_ablations():
            variant.apply(settings, SearchMode.BALANCED)
        assert settings.sparse_weight == original_sparse
        assert MODE_PROFILES[SearchMode.BALANCED].use_reranker == original_rerank

    def test_combined_variants_are_labelled(self):
        combined = [v for v in standard_ablations() if v.combined]
        assert combined, "the multi-change variants must be marked as such"
        for variant in combined:
            assert variant.name in {"no-expansion", "retrieval-only"}

    def test_exactly_one_baseline(self):
        assert sum(1 for v in standard_ablations() if v.is_baseline) == 1

    def test_sweep_covers_the_range_including_zero(self):
        variants = weight_sweep(4)
        weights = [v.sparse_weight for v in variants if v.sparse_weight is not None]
        assert 0.0 in weights, "sparse off must be one of the points"
        assert max(weights) > 1.0
        assert sum(1 for v in variants if v.is_baseline) == 1

    def test_a_variant_with_no_overrides_returns_the_originals(self, settings):
        tuned_settings, tuned_profile = Variant(
            name="noop", description=""
        ).apply(settings, SearchMode.BALANCED)
        assert tuned_settings is settings
        assert tuned_profile is MODE_PROFILES[SearchMode.BALANCED]


# ── answer metrics ────────────────────────────────────────────────────────


class TestAnswerMetrics:
    def test_citation_coverage(self):
        both = "A long enough first claim here [1]. A second long enough claim too [2]."
        half = "A long enough first claim here [1]. A second long enough claim with none."
        assert citation_coverage(both) == 1.0
        assert citation_coverage(half) == 0.5

    def test_short_answers_are_not_penalised(self):
        assert citation_coverage("Yes.") == 1.0
        assert citation_coverage("") == 1.0

    def test_anchor_recall_counts_expected_phrases(self):
        answer = "The limit of quantification was 2.5 ng/mL as measured."
        assert anchor_recall(answer, ["2.5", "ng/mL"]) == 1.0
        assert anchor_recall(answer, ["2.5", "absent phrase"]) == 0.5
        assert anchor_recall(answer, []) == 1.0

    def test_anchor_recall_normalises_whitespace_and_case(self):
        assert anchor_recall("A  B   C", ["a b c"]) == 1.0


class TestGoldenSampling:
    def test_stratifies_across_documents(self, chunk_factory):
        """Sampling without stratification lets one long document dominate,
        which makes the metrics measure that document's retrievability."""
        chunks = []
        for i in range(40):
            chunks.append(chunk_factory("x " * 120, paper_id="dominant", page=i))
        for i in range(3):
            chunks.append(chunk_factory("y " * 120, paper_id=f"small{i}", page=i))

        picked = sample_chunks(chunks, 6)
        papers = {c.paper_id for c in picked}
        assert len(papers) >= 3, f"spread across documents, got {papers}"

    def test_skips_low_quality_and_short_chunks(self, chunk_factory):
        chunks = [
            chunk_factory("x " * 120, paper_id="good", quality=1.0),
            chunk_factory("y " * 120, paper_id="poor", quality=0.2),
            chunk_factory("tiny", paper_id="short", quality=1.0),
        ]
        picked = sample_chunks(chunks, 10)
        assert {c.paper_id for c in picked} == {"good"}

    def test_is_reproducible_for_a_seed(self, chunk_factory):
        chunks = [
            chunk_factory("x " * 120, paper_id=f"p{i % 5}", page=i) for i in range(30)
        ]
        first = [c.chunk_id for c in sample_chunks(chunks, 8, seed=42)]
        second = [c.chunk_id for c in sample_chunks(chunks, 8, seed=42)]
        assert first == second

    def test_empty_input(self):
        assert sample_chunks([], 5) == []


# ── MCP ───────────────────────────────────────────────────────────────────


@pytest.fixture
def mcp(settings):
    return MCPServer(settings)


def run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


class TestMCPProtocol:
    def test_initialize_advertises_tools(self, mcp):
        reply = run(
            mcp.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        )
        result = reply["result"]
        assert result["serverInfo"]["name"] == "vectora"
        assert "tools" in result["capabilities"]
        assert result["instructions"]

    def test_notifications_are_not_answered(self, mcp):
        """A JSON-RPC notification has no id; replying to one is a protocol
        violation that some clients treat as fatal."""
        assert run(
            mcp.handle({"jsonrpc": "2.0", "method": "notifications/initialized"})
        ) is None

    def test_tool_schemas_are_valid(self, mcp):
        reply = run(mcp.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}))
        tools = reply["result"]["tools"]
        names = {t["name"] for t in tools}
        assert names == {
            "search_corpus",
            "answer_question",
            "list_corpora",
            "corpus_info",
        }
        for tool in tools:
            schema = tool["inputSchema"]
            assert schema["type"] == "object"
            assert tool["description"]
            for prop in schema.get("properties", {}).values():
                assert "type" in prop

    def test_unknown_method_is_a_protocol_error(self, mcp):
        reply = run(mcp.handle({"jsonrpc": "2.0", "id": 3, "method": "nope"}))
        assert reply["error"]["code"] == -32601

    def test_unknown_tool_is_a_tool_error_not_a_protocol_error(self, mcp):
        """A client should be able to act on a bad tool name rather than see
        the transport fail."""
        reply = run(
            mcp.handle(
                {
                    "jsonrpc": "2.0",
                    "id": 4,
                    "method": "tools/call",
                    "params": {"name": "ghost", "arguments": {}},
                }
            )
        )
        assert "error" not in reply
        assert reply["result"]["isError"] is True

    def test_missing_required_argument_is_a_tool_error(self, mcp):
        reply = run(
            mcp.handle(
                {
                    "jsonrpc": "2.0",
                    "id": 5,
                    "method": "tools/call",
                    "params": {"name": "search_corpus", "arguments": {}},
                }
            )
        )
        assert reply["result"]["isError"] is True

    def test_list_corpora_needs_no_index(self, mcp):
        reply = run(
            mcp.handle(
                {
                    "jsonrpc": "2.0",
                    "id": 6,
                    "method": "tools/call",
                    "params": {"name": "list_corpora", "arguments": {}},
                }
            )
        )
        assert reply["result"]["isError"] is False
        body = json.loads(reply["result"]["content"][0]["text"])
        assert "corpora" in body and "count" in body

    def test_ping(self, mcp):
        reply = run(mcp.handle({"jsonrpc": "2.0", "id": 7, "method": "ping"}))
        assert reply["result"] == {}


# ── tracing ───────────────────────────────────────────────────────────────


class TestTracingIsOptional:
    def test_disabled_tracing_is_a_no_op(self, settings):
        assert configure_tracing(settings) is False
        assert is_enabled() is False
        with span("retrieval.dense", candidates=40) as current:
            set_attributes(current, hits=12)
            assert current is None

    def test_question_span_works_disabled(self, settings):
        configure_tracing(settings)
        with question_span("balanced", "factual", "docs") as root:
            set_attributes(root, cost_usd=0.001)
            assert root is None
        shutdown_tracing()

    def test_exceptions_propagate_through_a_no_op_span(self, settings):
        configure_tracing(settings)
        with pytest.raises(ValueError, match="propagates"), span("boom"):
            raise ValueError("propagates")

    def test_set_attributes_tolerates_none(self):
        set_attributes(None, anything=1)  # must not raise


def test_eval_no_llm_flag_drops_the_llm(monkeypatch, settings):
    """`vectora eval --no-llm` must hand the harness settings with no LLM key."""
    from pydantic import SecretStr
    from typer.testing import CliRunner

    import evaluation.run_eval as run_eval_mod
    from vectora import cli

    keyed = settings.model_copy(update={"openai_api_key": SecretStr("sk-test")})
    seen = {}

    async def fake_run(s, **kwargs):
        seen["has_llm"] = s.has_openai
        return []

    monkeypatch.setattr(cli, "_settings", lambda: keyed)
    monkeypatch.setattr(run_eval_mod, "run", fake_run)
    result = CliRunner().invoke(cli.app, ["eval", "--no-llm", "--mode", "fast"])
    assert result.exit_code == 0, result.output
    assert seen["has_llm"] is False


def test_ablate_no_llm_skips_expansion_variants_and_drops_the_llm(monkeypatch, settings):
    from pydantic import SecretStr
    from typer.testing import CliRunner

    import evaluation.ablate as ablate_mod
    from vectora import cli

    keyed = settings.model_copy(update={"openai_api_key": SecretStr("sk-test")})
    seen = {}

    async def fake_run(s, **kwargs):
        seen["has_llm"] = s.has_openai
        seen["no_llm"] = kwargs.get("no_llm")
        return []

    monkeypatch.setattr(cli, "_settings", lambda: keyed)
    monkeypatch.setattr(ablate_mod, "run", fake_run)
    result = CliRunner().invoke(cli.app, ["ablate", "--no-llm"])
    assert result.exit_code == 0, result.output
    assert seen == {"has_llm": False, "no_llm": True}


def test_use_mmr_override_applies_to_every_mode(settings):
    from vectora.config import SearchMode

    # Off by default since 2026-10-21; None restores each mode's own choice.
    assert settings.profile(SearchMode.BALANCED).use_mmr is False
    per_mode = settings.model_copy(update={"use_mmr": None})
    assert per_mode.profile(SearchMode.BALANCED).use_mmr is True
    assert per_mode.profile(SearchMode.FAST).use_mmr is False
    off = settings.model_copy(update={"use_mmr": False})
    assert off.profile(SearchMode.BALANCED).use_mmr is False
    assert off.profile(SearchMode.DEEP).use_mmr is False
    # Everything else in the profile is untouched.
    assert off.profile(SearchMode.BALANCED).use_reranker is True


def test_calibrate_no_llm_drops_the_llm(monkeypatch, settings):
    from pydantic import SecretStr
    from typer.testing import CliRunner

    import evaluation.calibrate as calibrate_mod
    from vectora import cli

    keyed = settings.model_copy(update={"openai_api_key": SecretStr("sk-test")})
    seen = {}

    async def fake_run(s, **kwargs):
        seen["has_llm"] = s.has_openai
        return None

    monkeypatch.setattr(cli, "_settings", lambda: keyed)
    monkeypatch.setattr(calibrate_mod, "run", fake_run)
    result = CliRunner().invoke(cli.app, ["calibrate", "--no-llm", "--mode", "fast"])
    assert result.exit_code == 0, result.output
    assert seen["has_llm"] is False
