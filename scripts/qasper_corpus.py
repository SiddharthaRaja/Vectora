"""
Build a Vectora corpus and golden set from QASPER.

QASPER (Dasigi et al., 2021; CC BY 4.0) is NLP papers plus questions written by
NLP practitioners who had read only the title and abstract, each answered by
other annotators who read the full paper and marked the evidence. That makes it
a ready-made evaluation set for retrieval over research papers.

What this script writes:

    corpora/<name>/corpus.json     the corpus profile (via Vectora's registry)
    corpora/<name>/papers/*.md     one Markdown file per paper, from QASPER's text
    corpora/<name>/golden.yaml     questions with the paper that answers them

Two decisions worth knowing about:

* Papers are written as Markdown from QASPER's own parsed text, not downloaded
  as PDFs. That tests retrieval and generation on clean text. To test PDF
  parsing as well, pass --pdf: the arXiv PDF of each paper is downloaded
  instead (slower, and some papers will fail to parse — which is the point).
* QASPER questions assume you know which paper they are about ("What datasets
  do they use?"). Across a 40-paper corpus that is ambiguous, so every query is
  prefixed with the paper's title. Retrieval is then scored on whether chunks
  from that paper come back (relevant_papers). The gold answer is kept in
  `notes` so you can read it next to Vectora's answer.

Questions every annotator marked unanswerable become negative cases: the paper
is in the corpus, but it does not contain the answer, so Vectora should decline.
That is the hardest kind of negative — "in-domain but unasked".

Usage (from the repo root, with the venv active):

    python scripts/qasper_corpus.py                      # 40 papers, dev split
    python scripts/qasper_corpus.py --papers 20 --questions-per-paper 2
    python scripts/qasper_corpus.py --source path/to/qasper-train-dev-v0.3.tgz
    python scripts/qasper_corpus.py --pdf                # arXiv PDFs instead of .md
    python scripts/qasper_corpus.py --split all --list-topics
    python scripts/qasper_corpus.py --split all --topic "question answering, reading comprehension"

Then set VECTORA_CORPUS=qasper in .env and run ingest / index / calibrate / eval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

DATASET_URL = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"
SPLIT_FILES = {"dev": "qasper-dev-v0.3.json", "train": "qasper-train-v0.3.json"}
SPLITS = [*SPLIT_FILES, "all"]
ARXIV_PDF = "https://arxiv.org/pdf/{id}"

DESCRIPTION = "QASPER: NLP research papers with practitioner-written questions"
KIND = "NLP research papers"
TERMS = (
    "dataset, baseline, benchmark, annotation, model, embedding, attention, "
    "fine-tuning, evaluation, F1, accuracy, corpus"
)


# ── loading ─────────────────────────────────────────────────────────────────


def load_split(source: Path | None, cache_dir: Path, split: str) -> dict:
    """Return the QASPER split as {paper_id: paper}. Downloads once, then caches.

    ``all`` merges train and dev: about 1,170 papers instead of 281, which a
    narrow --topic usually needs.
    """
    members = list(SPLIT_FILES.values()) if split == "all" else [SPLIT_FILES[split]]

    if source and source.suffix == ".json":
        return json.loads(source.read_text("utf-8"))

    archive = source
    if archive is None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        archive = cache_dir / "qasper-train-dev-v0.3.tgz"
        if not archive.exists():
            print(f"Downloading QASPER (~10 MB) from\n  {DATASET_URL}")
            try:
                urllib.request.urlretrieve(DATASET_URL, archive)
            except Exception as exc:  # noqa: BLE001
                archive.unlink(missing_ok=True)
                sys.exit(
                    f"\nDownload failed: {exc}\n"
                    "Download the archive yourself from the QASPER page "
                    "(allenai.org/data/qasper) and pass it with --source."
                )

    merged: dict = {}
    with tarfile.open(archive, "r:gz") as tar:
        for member in members:
            found = [i for i in tar.getmembers() if i.name.endswith(member)]
            if not found:
                sys.exit(f"{member} not found inside {archive}")
            fh = tar.extractfile(found[0])
            assert fh is not None
            merged.update(json.loads(fh.read().decode("utf-8")))
    return merged


def matches_topic(paper: dict, keywords: list[str]) -> bool:
    """True if any keyword appears in the title or abstract (whole word, any case)."""
    if not keywords:
        return True
    text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
    return any(re.search(rf"\b{re.escape(k)}", text) for k in keywords)


# ── answers ─────────────────────────────────────────────────────────────────


def gold_answer(qa: dict) -> tuple[str | None, bool]:
    """(answer text, all_unanswerable). First usable annotation wins."""
    answers = [a.get("answer", {}) for a in qa.get("answers", [])]
    if not answers:
        return None, False
    if all(a.get("unanswerable") for a in answers):
        return None, True
    for a in answers:
        if a.get("unanswerable"):
            continue
        if a.get("free_form_answer"):
            return a["free_form_answer"].strip(), False
        if a.get("extractive_spans"):
            return " | ".join(s.strip() for s in a["extractive_spans"] if s.strip()), False
        if a.get("yes_no") is not None:
            return ("Yes" if a["yes_no"] else "No"), False
    return None, False


def has_text_evidence(qa: dict) -> bool:
    """Skip questions whose only evidence is a figure or table: Vectora indexes
    text and tables from PDFs, but QASPER's Markdown has no tables in it."""
    for a in qa.get("answers", []):
        ev = a.get("answer", {}).get("evidence") or []
        if any(not e.startswith("FLOAT SELECTED") for e in ev):
            return True
    return False


# ── writing ─────────────────────────────────────────────────────────────────


def stable_rank(key: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()


def safe_stem(paper_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", paper_id)


def yaml_str(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def paper_markdown(pid: str, paper: dict) -> str:
    title = (paper.get("title") or pid).strip()
    out = ["---", f"title: {title}", f"description: arXiv {pid}", "---", "", f"# {title}", ""]
    if paper.get("abstract"):
        out += ["## Abstract", "", paper["abstract"].strip(), ""]
    for section in paper.get("full_text", []):
        name = (section.get("section_name") or "").replace(" ::: ", " — ").strip()
        paras = [p.strip() for p in section.get("paragraphs", []) if p and p.strip()]
        if not paras:
            continue
        if name:
            out += [f"## {name}", ""]
        for p in paras:
            out += [p, ""]
    return "\n".join(out)


def download_pdf(pid: str, dest: Path) -> bool:
    if dest.exists():
        return True
    req = urllib.request.Request(
        ARXIV_PDF.format(id=pid), headers={"User-Agent": "vectora-qasper/1.0"}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
    except Exception as exc:  # noqa: BLE001
        print(f"  ! {pid}: {exc}")
        return False
    if not data.startswith(b"%PDF"):
        print(f"  ! {pid}: response was not a PDF")
        return False
    dest.write_bytes(data)
    time.sleep(3)  # arXiv asks for one request every three seconds
    return True


def write_golden(path: Path, cases: list[dict], negatives: list[dict], split: str) -> None:
    lines = [
        "# Golden set generated from QASPER (CC BY 4.0) by scripts/qasper_corpus.py.",
        f"# Split: {split}. Each query is prefixed with its paper's title because QASPER",
        "# questions assume the reader knows which paper they are about.",
        "#",
        "# Retrieval is scored with relevant_papers. `notes` holds the annotators' answer",
        "# so you can compare it with Vectora's. Negative cases are questions every",
        "# annotator marked unanswerable from that paper: Vectora should decline them.",
        "",
        "cases:",
    ]
    for c in cases:
        lines += [
            f"  - id: {c['id']}",
            f"    query: {yaml_str(c['query'])}",
            "    relevant_papers:",
            f"      - {yaml_str(c['paper'])}",
            f"    notes: {yaml_str(c['notes'])}",
            "",
        ]
    lines += ["negative_cases:"]
    for n in negatives:
        lines += [
            f"  - id: {n['id']}",
            f"    query: {yaml_str(n['query'])}",
            "    expect: fallback",
            f"    notes: {yaml_str(n['notes'])}",
            "",
        ]
    path.write_text("\n".join(lines) + "\n", "utf-8")


# ── main ────────────────────────────────────────────────────────────────────


STOPWORDS = set("""a an and the of for in on to with from by via at as is are using
towards toward based learning neural model models network networks approach task tasks
language languages natural nlp deep data new study analysis method methods""".split())


def list_topics(data: dict) -> None:
    from collections import Counter

    counts: Counter[str] = Counter()
    for p in data.values():
        words = re.findall(r"[a-z][a-z-]+", (p.get("title") or "").lower())
        counts.update(w for w in set(words) if w not in STOPWORDS and len(w) > 2)
    print(f"Most common title words across {len(data)} papers:\n")
    for word, n in counts.most_common(40):
        print(f"  {n:4d}  {word}")
    print('\nPick a few related ones: --topic "word one, word two"')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus", default="qasper", help="Corpus name (default: qasper)")
    ap.add_argument("--papers", type=int, default=40, help="How many papers (default 40)")
    ap.add_argument("--questions-per-paper", type=int, default=3)
    ap.add_argument("--max-negatives", type=int, default=12)
    ap.add_argument("--split", choices=SPLITS, default="dev",
                    help="dev (281 papers), train, or all (~1,170)")
    ap.add_argument("--topic", default="",
                    help='Comma-separated keywords matched against title and abstract, '
                         'e.g. "question answering, reading comprehension"')
    ap.add_argument("--list-topics", action="store_true",
                    help="Print the most common title words and exit")
    ap.add_argument("--seed", type=int, default=0, help="Changes which papers are picked")
    ap.add_argument("--source", type=Path, help="Local QASPER .tgz or split .json")
    ap.add_argument("--pdf", action="store_true", help="Download arXiv PDFs instead of .md")
    ap.add_argument("--force", action="store_true", help="Overwrite an existing golden.yaml")
    args = ap.parse_args()
    keywords = [k.strip().lower() for k in args.topic.split(",") if k.strip()]

    # Works whether or not Vectora is pip-installed: the repo root is one up.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from vectora.config import get_settings
    from vectora.corpus import CorpusRegistry

    settings = get_settings()
    registry = CorpusRegistry(settings.corpora_dir)
    if args.list_topics:
        repo_root = Path(__file__).resolve().parent.parent
        data = load_split(args.source, repo_root / "data" / "qasper", args.split)
        list_topics(data)
        return

    profile = registry.create(
        args.corpus,
        description=DESCRIPTION + (f" — topic: {args.topic}" if keywords else ""),
        document_kind=KIND,
        domain_terms=[t.strip() for t in TERMS.split(",")],
        source=f"QASPER v0.3 {args.split} split",
        exist_ok=True,
    )
    profile.ensure_dirs()
    if profile.golden_path.exists() and not args.force:
        sys.exit(f"{profile.golden_path} already exists. Pass --force to overwrite it.")

    repo_root = Path(__file__).resolve().parent.parent
    # data/ is gitignored, so the downloaded archive never gets committed.
    data = load_split(args.source, repo_root / "data" / "qasper", args.split)
    print(f"Loaded {len(data)} papers from the QASPER {args.split} split")

    # Papers with at least one answerable, text-evidenced question, in a stable
    # pseudo-random order so --seed picks a different but reproducible sample.
    usable = [
        pid for pid, p in data.items()
        if matches_topic(p, keywords)
        and any(gold_answer(q)[0] and has_text_evidence(q) for q in p.get("qas", []))
    ]
    if keywords:
        print(f"{len(usable)} papers match the topic {args.topic!r}")
        if len(usable) < args.papers:
            print(f"  fewer than --papers {args.papers}; using all of them."
                  + ("" if args.split == "all" else " Try --split all for more."))
        if not usable:
            sys.exit("No papers matched. Broaden the keywords, or run --list-topics.")
    usable.sort(key=lambda pid: stable_rank(pid, args.seed))

    cases: list[dict] = []
    negatives: list[dict] = []
    written = 0
    for pid in usable:
        if written >= args.papers:
            break
        paper = data[pid]
        title = (paper.get("title") or pid).strip()
        stem = safe_stem(pid)

        if args.pdf:
            if not download_pdf(pid, profile.papers_dir / f"{stem}.pdf"):
                continue
        else:
            (profile.papers_dir / f"{stem}.md").write_text(paper_markdown(pid, paper), "utf-8")
        written += 1

        taken = 0
        for qa in paper.get("qas", []):
            answer, unanswerable = gold_answer(qa)
            question = qa["question"].strip()
            query = f'In the paper "{title}": {question}'
            qid = qa.get("question_id", stable_rank(question, 0))[:10]
            if unanswerable:
                if len(negatives) < args.max_negatives:
                    negatives.append({
                        "id": f"neg-{qid}",
                        "query": query,
                        "notes": f"Paper {pid} is in the corpus but does not answer this.",
                    })
                continue
            if answer and has_text_evidence(qa) and taken < args.questions_per_paper:
                cases.append({
                    "id": f"q-{qid}",
                    "query": query,
                    "paper": stem,
                    "notes": f"Gold answer: {answer[:400]}",
                })
                taken += 1

    negatives.append({
        "id": "off-topic",
        "query": "What is the boiling point of water at the top of Mount Everest?",
        "notes": "Plainly outside the corpus.",
    })
    write_golden(profile.golden_path, cases, negatives, args.split)
    registry.refresh_counts(profile)

    print(f"\nCorpus '{profile.name}':")
    print(f"  documents  {written} {'PDFs' if args.pdf else 'Markdown files'} in {profile.papers_dir}")
    print(f"  golden set {len(cases)} questions + {len(negatives)} negatives in {profile.golden_path}")
    print("\nNext:")
    print(f"  set VECTORA_CORPUS={profile.name} in .env, then")
    print("  vectora ingest\n  vectora index\n  vectora calibrate\n  vectora eval")


if __name__ == "__main__":
    main()
