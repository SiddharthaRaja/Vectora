"""
Check the LLM faithfulness judge against human labels.

An LLM judge is only worth reporting if it agrees with a careful human.
This makes that measurable in two steps:

1. ``export`` — take an answers report (``vectora answers --mode deep``, the
   mode that runs the grounding judge) and write a CSV of N answers with the
   sources they cite. Open it in Excel / Sheets and fill ``human_faithful``
   with 1 (every claim is supported by the cited sources) or 0 (at least one
   claim is not). Do this *without* looking at the judge columns — hide them.

2. ``score`` — read the labelled CSV back and report raw agreement, Cohen's
   kappa and the confusion matrix, treating a judge score of
   ``--threshold`` (default 70, "mostly grounded") or above as faithful.

    python -m evaluation.judge_agreement export --answers corpora/qasper-300/reports/answers-deep.json
    python -m evaluation.judge_agreement score --labels corpora/qasper-300/reports/judge_labels.csv

Kappa reads roughly: < 0.2 slight, 0.2–0.4 fair, 0.4–0.6 moderate,
0.6–0.8 substantial, > 0.8 near-perfect. Thirty labels give a wide
confidence interval, so report the count with the number.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

COLUMNS = [
    "case_id", "query", "answer", "sources",
    "human_faithful", "human_notes",
    "judge_score", "judge_faithful",
]


def _load_chunks(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            out[row["chunk_id"]] = f"{row.get('title', '')}\n{row.get('content', '')}"
    return out


def export(answers: Path, out: Path, n: int, chunks_path: Path, threshold: int) -> int:
    report = json.loads(answers.read_text("utf-8"))
    pool = [
        s for s in report["per_case"]
        if not s["refused"] and s.get("groundedness") is not None and s.get("answer_text")
    ]
    if not pool:
        raise SystemExit(
            "No judged answers with text in that report. Re-run "
            "`vectora answers --mode deep` with this version of Vectora."
        )
    # Stable pseudo-random sample, so re-exporting gives the same rows.
    pool.sort(key=lambda s: hashlib.sha256(s["case_id"].encode()).hexdigest())
    rows = pool[:n]
    chunks = _load_chunks(chunks_path)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for s in rows:
            sources = "\n\n".join(
                f"[{cid[:8]}] {chunks.get(cid, '(chunk text not found)')[:1500]}"
                for cid in s.get("cited_chunk_ids", [])
            )
            w.writerow({
                "case_id": s["case_id"],
                "query": s["query"],
                "answer": s["answer_text"],
                "sources": sources or "(no citations)",
                "human_faithful": "",
                "human_notes": "",
                "judge_score": s["groundedness"],
                "judge_faithful": int(s["groundedness"] >= threshold),
            })
    print(f"Wrote {len(rows)} rows to {out}")
    print("Fill human_faithful with 1 or 0. Hide the judge_* columns while labelling.")
    return len(rows)


def cohen_kappa(a: list[int], b: list[int]) -> float:
    n = len(a)
    if not n:
        return 0.0
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def score(labels: Path, threshold: int) -> dict:
    human, judge = [], []
    with labels.open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            h = (row.get("human_faithful") or "").strip()
            if h not in {"0", "1"}:
                continue
            human.append(int(h))
            judge.append(int(float(row["judge_score"]) >= threshold))
    if not human:
        raise SystemExit("No labelled rows (human_faithful must be 0 or 1).")
    n = len(human)
    tp = sum(h and j for h, j in zip(human, judge, strict=True))
    tn = sum((not h) and (not j) for h, j in zip(human, judge, strict=True))
    fp = sum((not h) and j for h, j in zip(human, judge, strict=True))  # judge passed, human failed
    fn = sum(h and (not j) for h, j in zip(human, judge, strict=True))
    result = {
        "labelled": n,
        "threshold": threshold,
        "agreement": round((tp + tn) / n, 4),
        "cohen_kappa": round(cohen_kappa(human, judge), 4),
        "human_faithful_rate": round(sum(human) / n, 4),
        "judge_faithful_rate": round(sum(judge) / n, 4),
        "judge_passed_unfaithful": fp,
        "judge_failed_faithful": fn,
        "confusion": {"both_faithful": tp, "both_unfaithful": tn,
                      "judge_yes_human_no": fp, "judge_no_human_yes": fn},
    }
    print(json.dumps(result, indent=2))
    if fp:
        print(f"\nThe judge passed {fp} answer(s) a human found unfaithful — "
              "the failure that matters most.")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--answers", type=Path, required=True)
    e.add_argument("--out", type=Path, default=None)
    e.add_argument("--n", type=int, default=30)
    e.add_argument("--chunks", type=Path, default=None,
                   help="chunks.jsonl for source text (default: next to the report's corpus)")
    s = sub.add_parser("score")
    s.add_argument("--labels", type=Path, required=True)
    s.add_argument("--out", type=Path, default=None, help="Write the result JSON here")
    for p in (e, s):
        p.add_argument("--threshold", type=int, default=70)
    args = ap.parse_args()

    if args.cmd == "export":
        reports = args.answers.resolve().parent
        out = args.out or reports / "judge_labels.csv"
        chunks = args.chunks or reports.parent / "chunks.jsonl"
        export(args.answers, out, args.n, chunks, args.threshold)
    else:
        result = score(args.labels, args.threshold)
        target = args.out or args.labels.with_name("judge_agreement.json")
        target.write_text(json.dumps(result, indent=2), "utf-8")
        print(f"Wrote {target}")


if __name__ == "__main__":
    main()
