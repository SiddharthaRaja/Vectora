"""
Ablations and parameter sweeps.

The README says hybrid retrieval beats dense-only and that reranking helps.
Those are claims. This turns them into a table.

An ablation removes one component and re-measures, which is the only way to
know whether a stage earns its latency. The interesting outcome is a stage
that turns out not to matter: that is a stage to delete, and finding one is
a better result than confirming everything helps.

Two things this is careful about, because an ablation harness that gets them
wrong produces confident nonsense:

**It changes one thing at a time.** Each variant differs from the baseline
in exactly one respect, so a difference is attributable. The combined
variants are labelled as such.

**It reports whether a difference means anything.** With 20 or 50 cases, a
recall difference of 0.02 is noise. Each comparison against the baseline
gets a paired bootstrap confidence interval, and the report says plainly
when a difference is inside it. The alternative — a table of three-decimal
numbers with no uncertainty — invites exactly the overclaiming that makes
benchmark tables worthless.
"""

from __future__ import annotations

import asyncio
import json
import random
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vectora.config import MODE_PROFILES, QueryType, SearchMode, Settings
from vectora.observability import configure_logging, get_logger, request_context

log = get_logger(__name__)

BOOTSTRAP_SAMPLES = 2000
CONFIDENCE = 0.95


@dataclass
class Variant:
    """One retrieval configuration to measure."""

    name: str
    description: str
    dense_weight: float | None = None
    sparse_weight: float | None = None
    use_reranker: bool | None = None
    use_mmr: bool | None = None
    use_hyde: bool | None = None
    rewrite_variants: int | None = None
    rrf_k: int | None = None
    is_baseline: bool = False
    combined: bool = False
    """True when this variant changes more than one thing, so the report can
    say its effect is not attributable to a single component."""

    def apply(self, settings: Settings, mode: SearchMode) -> tuple[Settings, Any]:
        """Return a settings copy and mode profile with this variant applied.

        Both are copied rather than mutated: the previous implementation of
        mode switching mutated a module global, and an ablation runner that
        leaks configuration between variants measures nothing.
        """
        overrides: dict[str, Any] = {}
        if self.dense_weight is not None:
            overrides["dense_weight"] = self.dense_weight
        if self.sparse_weight is not None:
            overrides["sparse_weight"] = self.sparse_weight
        if self.rrf_k is not None:
            overrides["rrf_k"] = self.rrf_k
        tuned_settings = (
            settings.model_copy(update=overrides) if overrides else settings
        )

        base = MODE_PROFILES[mode]
        profile_overrides: dict[str, Any] = {}
        if self.use_reranker is not None:
            profile_overrides["use_reranker"] = self.use_reranker
        if self.use_mmr is not None:
            profile_overrides["use_mmr"] = self.use_mmr
        if self.use_hyde is not None:
            profile_overrides["use_hyde"] = self.use_hyde
        if self.rewrite_variants is not None:
            profile_overrides["rewrite_variants"] = self.rewrite_variants
        tuned_profile = (
            base.model_copy(update=profile_overrides) if profile_overrides else base
        )
        return tuned_settings, tuned_profile


def standard_ablations() -> list[Variant]:
    """The set worth running by default.

    Each removes one stage from the balanced configuration, plus two
    weight-extreme variants that isolate the retrievers completely.
    """
    return [
        Variant(
            name="hybrid+rerank+mmr",
            description="The shipped balanced configuration",
            is_baseline=True,
        ),
        Variant(
            name="dense-only",
            description="Lexical retrieval off (sparse weight 0)",
            sparse_weight=0.0,
        ),
        Variant(
            name="sparse-only",
            description="Dense retrieval off (dense weight 0)",
            dense_weight=0.0,
        ),
        Variant(
            name="no-rerank",
            description="Cross-encoder off; fusion order kept",
            use_reranker=False,
        ),
        Variant(
            name="no-mmr",
            description="Diversification off; top-k by score",
            use_mmr=False,
        ),
        Variant(
            name="no-hyde",
            description="Hypothetical document embedding off",
            use_hyde=False,
        ),
        Variant(
            name="no-rewrite",
            description="Query rewriting off; original phrasing only",
            rewrite_variants=0,
        ),
        Variant(
            name="no-expansion",
            description="Neither HyDE nor rewriting (all LLM expansion off)",
            use_hyde=False,
            rewrite_variants=0,
            combined=True,
        ),
        Variant(
            name="retrieval-only",
            description="No expansion, no rerank, no MMR — raw hybrid fusion",
            use_hyde=False,
            rewrite_variants=0,
            use_reranker=False,
            use_mmr=False,
            combined=True,
        ),
    ]


def weight_sweep(steps: int = 5) -> list[Variant]:
    """Fusion weight variants, sparse from 0 to 1.6 with dense pinned at 1.0.

    Pinning dense and varying sparse is the same family of ratios as varying
    both, with half the runs and an interpretable axis.
    """
    out = [
        Variant(
            name="sweep-baseline",
            description="Shipped weights (dense 1.0, sparse 0.8)",
            is_baseline=True,
        )
    ]
    for i in range(steps + 1):
        value = round(i * (1.6 / steps), 2)
        out.append(
            Variant(
                name=f"sparse-{value}",
                description=f"dense 1.0, sparse {value}",
                dense_weight=1.0,
                sparse_weight=value,
            )
        )
    return out


@dataclass
class CaseOutcome:
    case_id: str
    hit: bool
    reciprocal_rank: float
    precision: float
    ndcg: float
    elapsed_ms: float


@dataclass
class VariantResult:
    variant: Variant
    outcomes: list[CaseOutcome] = field(default_factory=list)
    negatives_total: int = 0
    negatives_refused: int = 0
    errors: int = 0

    def _mean(self, key: str) -> float:
        values = [getattr(o, key) for o in self.outcomes]
        return sum(values) / len(values) if values else 0.0

    @property
    def recall(self) -> float:
        return (
            sum(1 for o in self.outcomes if o.hit) / len(self.outcomes)
            if self.outcomes
            else 0.0
        )

    @property
    def mrr(self) -> float:
        return self._mean("reciprocal_rank")

    @property
    def precision(self) -> float:
        return self._mean("precision")

    @property
    def ndcg(self) -> float:
        return self._mean("ndcg")

    @property
    def median_ms(self) -> float:
        return (
            statistics.median(o.elapsed_ms for o in self.outcomes)
            if self.outcomes
            else 0.0
        )

    @property
    def p95_ms(self) -> float:
        if not self.outcomes:
            return 0.0
        ordered = sorted(o.elapsed_ms for o in self.outcomes)
        return ordered[min(int(len(ordered) * 0.95), len(ordered) - 1)]

    @property
    def refusal_rate(self) -> float:
        return (
            self.negatives_refused / self.negatives_total
            if self.negatives_total
            else 0.0
        )

    def vector(self, key: str) -> list[float]:
        """Per-case values, for a paired comparison."""
        if key == "recall":
            return [1.0 if o.hit else 0.0 for o in self.outcomes]
        return [getattr(o, key) for o in self.outcomes]

    def as_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant.name,
            "description": self.variant.description,
            "combined": self.variant.combined,
            "cases": len(self.outcomes),
            "recall": round(self.recall, 4),
            "precision": round(self.precision, 4),
            "mrr": round(self.mrr, 4),
            "ndcg": round(self.ndcg, 4),
            "median_ms": round(self.median_ms, 1),
            "p95_ms": round(self.p95_ms, 1),
            "refusal_rate": round(self.refusal_rate, 4),
            "errors": self.errors,
        }


def paired_bootstrap(
    baseline: list[float],
    variant: list[float],
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    confidence: float = CONFIDENCE,
    seed: int = 11,
) -> tuple[float, float, float, bool]:
    """Confidence interval for the mean paired difference.

    Paired because both configurations ran the same cases, so the pairing
    removes case difficulty from the comparison and makes a small difference
    detectable with few cases. Returns
    ``(difference, low, high, significant)``, where ``significant`` means the
    interval excludes zero.

    A bootstrap rather than a t-test: recall per case is a 0/1 variable and
    reciprocal rank is badly non-normal, so the normality assumption does
    not hold at these sample sizes.
    """
    n = min(len(baseline), len(variant))
    if n == 0:
        return 0.0, 0.0, 0.0, False
    deltas = [variant[i] - baseline[i] for i in range(n)]
    observed = sum(deltas) / n

    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(samples):
        total = 0.0
        for _ in range(n):
            total += deltas[rng.randrange(n)]
        means.append(total / n)
    means.sort()

    tail = (1.0 - confidence) / 2
    low = means[int(tail * samples)]
    high = means[min(int((1 - tail) * samples), samples - 1)]
    return observed, low, high, not (low <= 0.0 <= high)


async def _run_variant(
    container: Any,
    variant: Variant,
    cases: list[Any],
    negatives: list[Any],
    *,
    mode: SearchMode,
    top_k: int,
) -> VariantResult:
    """Measure one configuration over the whole golden set."""
    from vectora.generation.guards import check as guard_check
    from vectora.generation.router import classify
    from vectora.retrieval.pipeline import Retriever

    tuned_settings, tuned_profile = variant.apply(container.settings, mode)

    # A retriever built per variant, sharing the loaded models. Rebuilding
    # rather than mutating keeps variants independent; sharing the encoder
    # and reranker keeps the run from reloading 150MB of weights nine times.
    retriever = Retriever(
        tuned_settings,
        store=container.store,
        sparse=container.sparse,
        encoder=container.encoder,
        reranker=container.reranker,
        llm=container.llm,
        corpus=container.corpus.profile,
    )
    result = VariantResult(variant=variant)
    for case in cases:
        query_type = getattr(case, "query_type", None)
        if query_type is None:
            query_type, _ = await classify(
                case.query, llm=container.llm, settings=tuned_settings
            )
        start = time.perf_counter()
        try:
            with request_context():
                found, _ = await retriever.retrieve(
                    case.query,
                    mode=mode,
                    query_type=query_type,
                    top_k=top_k,
                    profile_override=tuned_profile,
                )
        except Exception as exc:
            log.warning(
                "case failed",
                extra={"variant": variant.name, "case": case.id, "err": str(exc)[:120]},
            )
            result.errors += 1
            continue
        elapsed = (time.perf_counter() - start) * 1000

        positions = [
            i for i, rc in enumerate(found, start=1) if case.is_relevant(rc)
        ]
        result.outcomes.append(
            CaseOutcome(
                case_id=case.id,
                hit=bool(positions),
                reciprocal_rank=1.0 / positions[0] if positions else 0.0,
                precision=len(positions) / len(found) if found else 0.0,
                ndcg=_ndcg(positions),
                elapsed_ms=elapsed,
            )
        )

    for negative in negatives:
        result.negatives_total += 1
        try:
            with request_context():
                found, _ = await retriever.retrieve(
                    negative.query,
                    mode=mode,
                    query_type=QueryType.EXPLORATORY,
                    top_k=top_k,
                    profile_override=tuned_profile,
                )
        except Exception:
            result.errors += 1
            continue
        ok, _ = guard_check(found, tuned_settings)
        if not ok:
            result.negatives_refused += 1

    return result


def _ndcg(positions: list[int]) -> float:
    import math

    if not positions:
        return 0.0
    dcg = sum(1.0 / math.log2(p + 1) for p in positions)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(len(positions)))
    return dcg / ideal if ideal else 0.0


# ── reporting ─────────────────────────────────────────────────────────────


def render_table(results: list[VariantResult], metric: str = "recall") -> str:
    baseline = next((r for r in results if r.variant.is_baseline), results[0])
    base_vector = baseline.vector(metric)

    lines = [
        "",
        "=" * 104,
        f"  ABLATION — sorted by {metric}, compared against "
        f"'{baseline.variant.name}'",
        "=" * 104,
        "",
        f"  {'variant':<20} {'recall':>7} {'MRR':>7} {'nDCG':>7} {'prec':>7} "
        f"{'p50ms':>7} {'p95ms':>7} {'neg':>5}  {'Δ' + metric:>9}  verdict",
        "  " + "-" * 100,
    ]

    ordered = sorted(results, key=lambda r: getattr(r, metric), reverse=True)
    for r in ordered:
        if r is baseline:
            delta_text, verdict = "  baseline", ""
        else:
            diff, low, high, significant = paired_bootstrap(
                base_vector, r.vector(metric)
            )
            delta_text = f"{diff:+.3f}"
            if significant:
                verdict = "worse" if diff < 0 else "better"
                verdict += f" (CI {low:+.3f}..{high:+.3f})"
            else:
                verdict = f"no sig. diff (CI {low:+.3f}..{high:+.3f})"
        flag = "*" if r.variant.combined else " "
        lines.append(
            f"  {r.variant.name:<19}{flag} {r.recall:>7.3f} {r.mrr:>7.3f} "
            f"{r.ndcg:>7.3f} {r.precision:>7.3f} {r.median_ms:>7.0f} "
            f"{r.p95_ms:>7.0f} {r.refusal_rate:>5.0%}  {delta_text:>9}  {verdict}"
        )

    lines += [
        "  " + "-" * 100,
        "",
        f"  n = {len(baseline.outcomes)} cases, {baseline.negatives_total} negatives.",
        f"  Δ is the mean paired difference with a {int(CONFIDENCE * 100)}% bootstrap CI.",
        "  'no sig. diff' means the interval includes zero — at this sample size",
        "  the difference is not distinguishable from noise. Add cases to resolve it.",
        "  '*' marks a variant that changes more than one thing, so its effect",
        "  cannot be attributed to a single component.",
        "  'neg' is the share of off-corpus questions correctly refused.",
        "=" * 104,
        "",
    ]
    return "\n".join(lines)


def render_markdown(
    results: list[VariantResult], ctx: Any, *, mode: SearchMode, metric: str
) -> str:
    baseline = next((r for r in results if r.variant.is_baseline), results[0])
    base_vector = baseline.vector(metric)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    rows = ["| variant | recall | MRR | nDCG | p50 ms | refused | Δrecall vs baseline |",
            "|---|---:|---:|---:|---:|---:|---|"]
    for r in sorted(results, key=lambda r: getattr(r, metric), reverse=True):
        if r is baseline:
            delta = "_baseline_"
        else:
            diff, low, high, significant = paired_bootstrap(
                base_vector, r.vector(metric)
            )
            delta = (
                f"**{diff:+.3f}** ({low:+.3f}..{high:+.3f})"
                if significant
                else f"{diff:+.3f} (n.s.)"
            )
        name = r.variant.name + ("\\*" if r.variant.combined else "")
        rows.append(
            f"| {name} | {r.recall:.3f} | {r.mrr:.3f} | {r.ndcg:.3f} | "
            f"{r.median_ms:.0f} | {r.refusal_rate:.0%} | {delta} |"
        )

    return "\n".join(
        [
            f"# Ablation — `{ctx.name}` corpus",
            "",
            f"{stamp} · {mode.value} mode · {len(baseline.outcomes)} cases, "
            f"{baseline.negatives_total} negatives",
            "",
            *rows,
            "",
            "Δ is the mean paired difference in recall against the shipped",
            f"configuration, with a {int(CONFIDENCE * 100)}% bootstrap confidence",
            "interval. Bold means the interval excludes zero; `n.s.` means it does",
            "not, so the difference is not distinguishable from noise at this",
            "sample size. `*` marks variants that change more than one thing.",
            "",
            "Reproduce with:",
            "",
            "```bash",
            f"vectora ablate --corpus {ctx.name} --mode {mode.value}",
            "```",
            "",
        ]
    )


async def run(
    settings: Settings,
    *,
    mode: SearchMode = SearchMode.BALANCED,
    top_k: int = 5,
    metric: str = "recall",
    sweep: bool = False,
    sweep_steps: int = 5,
    corpus: str | None = None,
    output: Path | None = None,
    no_llm: bool = False,
) -> list[VariantResult]:
    """Run the ablation set (or a weight sweep) over the golden set.

    ``no_llm`` drops the variants that only switch LLM query expansion on or
    off: with no LLM they are identical to the baseline, so running them
    would spend time to report a difference of exactly zero.
    """
    configure_logging(settings)
    from evaluation.run_eval import load_dataset
    from vectora.api.deps import build_container

    container = await asyncio.to_thread(
        build_container, settings, warm=True, corpus=corpus
    )
    try:
        ctx = container.corpus
        if not container.store.exists():
            raise RuntimeError("Nothing is indexed. Run `vectora index` first.")
        if not ctx.golden.exists():
            raise RuntimeError(
                f"No golden set at {ctx.golden}. Draft one with "
                f"`vectora golden generate`, or copy evaluation/golden.template.yaml."
            )

        cases, negatives = load_dataset(ctx.golden)
        if not cases:
            raise RuntimeError(f"{ctx.golden} has no cases.")
        if len(cases) < 12:
            print(
                f"\n  Note: {len(cases)} cases is few for an ablation. Differences "
                f"under ~0.1 recall will not be distinguishable from noise.\n"
            )

        variants = weight_sweep(sweep_steps) if sweep else standard_ablations()
        if no_llm and not sweep:
            expansion_only = {"no-hyde", "no-rewrite", "no-expansion"}
            variants = [v for v in variants if v.name not in expansion_only]
        print(
            f"\nRunning {len(variants)} configurations over {len(cases)} cases "
            f"and {len(negatives)} negatives, in {mode.value} mode.\n"
        )

        results: list[VariantResult] = []
        for i, variant in enumerate(variants, start=1):
            print(f"  [{i}/{len(variants)}] {variant.name} — {variant.description}")
            results.append(
                await _run_variant(
                    container,
                    variant,
                    cases,
                    negatives,
                    mode=mode,
                    top_k=top_k,
                )
            )

        print(render_table(results, metric=metric))

        reports = ctx.reports
        reports.mkdir(parents=True, exist_ok=True)
        kind = "sweep" if sweep else "ablation"
        json_path = output or reports / f"{kind}-{mode.value}.json"
        json_path.write_text(
            json.dumps(
                {
                    "corpus": ctx.name,
                    "collection": ctx.collection,
                    "mode": mode.value,
                    "top_k": top_k,
                    "metric": metric,
                    "cases": len(cases),
                    "negatives": len(negatives),
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "settings": {
                        "embedding_model": settings.embedding_model,
                        "reranker_model": settings.reranker_model,
                        "rrf_k": settings.rrf_k,
                    },
                    "variants": [
                        {**r.as_dict(), "config": asdict(r.variant)} for r in results
                    ],
                },
                indent=2,
            ),
            "utf-8",
        )
        md_path = reports / f"{kind}-{mode.value}.md"
        md_path.write_text(
            render_markdown(results, ctx, mode=mode, metric=metric), "utf-8"
        )
        print(f"Wrote {json_path}\n      {md_path}\n")
        return results
    finally:
        await container.aclose()
