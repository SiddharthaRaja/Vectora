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
* Queries are title-free by default: users do not type the paper's title, and
  Vectora run 2 showed the title prefix doing most of retrieval's work (recall
  1.00 with it, 0.55 without). Many QASPER questions only make sense with the
  title ("What is the baseline?"), so questions with fewer than three specific
  words are dropped. Pass --title-prefix for the old behaviour.
* Each case carries `relevant_evidence` (the paragraphs the annotators marked,
  figure/table-only evidence skipped) so retrieval is scored on finding the
  passage, not just the paper, and `reference_answers` (every annotator's
  answer) for answer token F1. `relevant_papers` is still written, so
  paper-level recall is reported alongside.
* QASPER's placeholder tokens (BIBREF12, INLINEFORM0, TABREF3, ...) are cleaned
  out of the papers and the evidence alike. --keep-placeholders turns that off.

With --title-prefix, questions every annotator marked unanswerable become
negative cases: the paper is in the corpus but does not contain the answer.
Without the title those questions are ambiguous — another paper may well answer
"what datasets do they use" — so title-free sets use fixed off-corpus negatives.

Usage (from the repo root, with the venv active):

    python scripts/qasper_corpus.py                      # 40 papers, dev split
    python scripts/qasper_corpus.py --corpus qasper-300 --split all --papers 300
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


# ── cleaning ────────────────────────────────────────────────────────────────

# No \b: tokens are often glued to the word before ("worldwideBIBREF0").
PLACEHOLDERS = [
    (re.compile(r"\s*BIBREF\d+(\s*,\s*BIBREF\d+)*"), ""),
    (re.compile(r"(INLINEFORM|DISPLAYFORM)\d+"), "[formula]"),
    (re.compile(r"TABREF\d+"), "the table"),
    (re.compile(r"FIGREF\d+"), "the figure"),
    (re.compile(r"SECREF\d+"), "a later section"),
]
CLEAN = True


def clean(text: str) -> str:
    if not CLEAN:
        return text
    for rx, rep in PLACEHOLDERS:
        text = rx.sub(rep, text)
    return text


# Words that say nothing about *which* paper a question is about.
GENERIC = set("""a an and the of for in on to with from by at as is are was were be been
what which who whom whose how why when where do does did done they their them it its
this that these those there here you your we our use used uses using than then
paper papers author authors proposed propose approach approaches method methods model
models system systems baseline baselines dataset datasets data result results
performance perform performs work task tasks experiment experiments evaluate evaluated
evaluation compare compared comparison improve improvement improvements state art
previous existing best better new size big large many much number kind kinds type
types other others some any all only also well same different between over under
show shown much more most less used test testing train training set sets metric
metrics score scores accuracy f1 language languages english text corpus corpora
""".split())  # noqa: SIM905


def specific_words(question: str) -> int:
    words = re.findall(r"[a-z][a-z0-9-]+", question.lower())
    return sum(1 for w in words if w not in GENERIC)


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


def all_answers(qa: dict) -> list[str]:
    """Every annotator's answer, deduplicated: references for token F1."""
    out: list[str] = []
    for a in qa.get("answers", []):
        a = a.get("answer", {})
        if a.get("unanswerable"):
            continue
        if a.get("free_form_answer"):
            text = a["free_form_answer"].strip()
        elif a.get("extractive_spans"):
            text = " ".join(s.strip() for s in a["extractive_spans"] if s.strip())
        elif a.get("yes_no") is not None:
            text = "Yes" if a["yes_no"] else "No"
        else:
            continue
        text = clean(text).strip()
        if text and text not in out:
            out.append(text)
    return out


def text_evidence(qa: dict) -> list[str]:
    """Annotator-marked evidence paragraphs, union over annotators, cleaned."""
    out: list[str] = []
    for a in qa.get("answers", []):
        a = a.get("answer", {})
        if a.get("unanswerable"):
            continue
        for e in a.get("evidence") or []:
            if not e.strip() or e.startswith("FLOAT SELECTED"):
                continue
            e = clean(e).strip()
            if e and e not in out:
                out.append(e)
    return out


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
        out += ["## Abstract", "", clean(paper["abstract"].strip()), ""]
    for section in paper.get("full_text", []):
        name = (section.get("section_name") or "").replace(" ::: ", " — ").strip()
        paras = [p.strip() for p in section.get("paragraphs", []) if p and p.strip()]
        if not paras:
            continue
        if name:
            out += [f"## {name}", ""]
        for p in paras:
            out += [clean(p), ""]
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


def write_golden(
    path: Path, cases: list[dict], negatives: list[dict], split: str, titled: bool
) -> None:
    lines = [
        "# Golden set generated from QASPER (CC BY 4.0) by scripts/qasper_corpus.py.",
        f"# Split: {split}. Queries are "
        + ("prefixed with the paper's title." if titled else "title-free (as a user would ask)."),
        "#",
        "# Retrieval is scored on relevant_evidence (the annotators' evidence paragraphs);",
        "# relevant_papers gives paper-level recall alongside. reference_answers holds",
        "# every annotator's answer, for token F1. Negative cases should be declined.",
        "",
        "cases:",
    ]
    for c in cases:
        lines += [
            f"  - id: {c['id']}",
            f"    query: {yaml_str(c['query'])}",
            "    relevant_papers:",
            f"      - {yaml_str(c['paper'])}",
        ]
        if c.get("evidence"):
            lines.append("    relevant_evidence:")
            lines += [f"      - {yaml_str(e)}" for e in c["evidence"]]
        if c.get("references"):
            lines.append("    reference_answers:")
            lines += [f"      - {yaml_str(r)}" for r in c["references"]]
        lines += [f"    notes: {yaml_str(c['notes'])}", ""]
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


#: Off-corpus negatives: some plainly unrelated, some adjacent to NLP but not
#: about any paper, which is where a weak refusal guard fails.
OFF_CORPUS = [
    "What is the boiling point of water at the top of Mount Everest?",
    "Who won the 2018 FIFA World Cup?",
    "How do I reset a forgotten Windows password?",
    "What is the recommended daily intake of vitamin D for adults?",
    "How does a car's catalytic converter work?",
    "What year did the Berlin Wall fall?",
    "Which GPU should I buy for gaming at 1440p?",
    "How do I cook risotto without it going gluey?",
    # Everyday and general knowledge.
    "What is the capital of Australia?",
    "How many bones are in the adult human body?",
    "What causes the seasons on Earth?",
    "How long should I boil an egg for a soft yolk?",
    "What is the best way to remove a red wine stain from carpet?",
    "Who painted the ceiling of the Sistine Chapel?",
    "How do I change a flat tyre on a car?",
    "What is the difference between a crocodile and an alligator?",
    "When is the best time of year to visit Japan?",
    "How do I file an income tax return online?",
    "What are the rules of cricket's LBW decision?",
    "How many players are on a basketball team on court?",
    # Other sciences: adjacent in register, absent from an NLP corpus.
    "What is the half-life of carbon-14?",
    "How does CRISPR-Cas9 cut DNA?",
    "What is the mechanism of action of metformin?",
    "How do mRNA vaccines produce an immune response?",
    "What is the Chandrasekhar limit for white dwarfs?",
    "How does a lithium-ion battery degrade over charge cycles?",
    "What is the Reynolds number used for in fluid dynamics?",
    "How do plate tectonics cause earthquakes?",
    "What determines the yield strength of steel?",
    "How does photosynthesis convert light into chemical energy?",
    "What is the Black-Scholes formula used to price?",
    "How do central banks use interest rates to control inflation?",
    # Computing, but not NLP research.
    "How do I configure a static IP address on Ubuntu?",
    "What is the difference between TCP and UDP?",
    "How does garbage collection work in the Java virtual machine?",
    "How do I resolve a merge conflict in git?",
    "What is the time complexity of quicksort in the worst case?",
    "How do I centre a div horizontally with CSS?",
    "What is a race condition in multithreaded code?",
    "How does public-key encryption establish a shared secret?",
    # Recent AI, after the corpus was written (QASPER papers are 2016-2020).
    "What context window does GPT-4o support?",
    "How does Llama 3 differ from Llama 2 in training data size?",
    "What is the Model Context Protocol used for?",
    "How does speculative decoding speed up LLM inference?",
    "What is direct preference optimization and how does it replace RLHF?",
    "How does vLLM's PagedAttention manage the KV cache?",
    "What is the Mixtral mixture-of-experts architecture?",
    "How do diffusion models generate images from noise?",
    "What GPU memory does a 70B parameter model need in 4-bit?",
    "What benchmark scores did Gemini 1.5 report?",
]


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
    ap.add_argument("--title-prefix", action="store_true",
                    help='Prefix queries with the paper title (old behaviour; inflates recall)')
    ap.add_argument("--min-specific-words", type=int, default=3,
                    help="Title-free only: drop questions with fewer specific words")
    ap.add_argument("--keep-placeholders", action="store_true",
                    help="Keep QASPER's BIBREF/INLINEFORM/... tokens")
    args = ap.parse_args()
    global CLEAN
    CLEAN = not args.keep_placeholders
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
    dropped_generic = 0
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
            query = f'In the paper "{title}": {question}' if args.title_prefix else question
            qid = qa.get("question_id", stable_rank(question, 0))[:10]
            if unanswerable:
                if args.title_prefix and len(negatives) < args.max_negatives:
                    negatives.append({
                        "id": f"neg-{qid}",
                        "query": query,
                        "notes": f"Paper {pid} is in the corpus but does not answer this.",
                    })
                continue
            if answer and has_text_evidence(qa) and taken < args.questions_per_paper:
                if not args.title_prefix and specific_words(question) < args.min_specific_words:
                    dropped_generic += 1
                    continue
                cases.append({
                    "id": f"q-{qid}",
                    "query": query,
                    "paper": stem,
                    "evidence": text_evidence(qa),
                    "references": all_answers(qa),
                    "notes": f"Gold answer: {clean(answer)[:400]}",
                })
                taken += 1

    for i, q in enumerate(OFF_CORPUS, start=1):
        negatives.append({"id": f"off-corpus-{i}", "query": q, "notes": "Outside the corpus."})
    write_golden(profile.golden_path, cases, negatives, args.split, args.title_prefix)
    if dropped_generic:
        print(f"Dropped {dropped_generic} questions too generic to ask without the title.")
    registry.refresh_counts(profile)

    print(f"\nCorpus '{profile.name}':")
    print(f"  documents  {written} {'PDFs' if args.pdf else 'Markdown files'} in {profile.papers_dir}")
    print(f"  golden set {len(cases)} questions + {len(negatives)} negatives in {profile.golden_path}")
    print("\nNext:")
    print(f"  set VECTORA_CORPUS={profile.name} in .env, then")
    print("  vectora ingest\n  vectora index\n  vectora calibrate\n  vectora eval")


if __name__ == "__main__":
    main()
