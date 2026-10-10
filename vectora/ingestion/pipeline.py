"""
Ingestion orchestration.

Turns a folder of PDFs into ``chunks.jsonl``, incrementally, with a report
of what was kept and what was discarded and why. Also provides a **repair**
pass that re-applies the current cleaning and quality rules to an existing
``chunks.jsonl`` — needed here because the corpus was built by the old
pipeline and the source PDFs are no longer in the working tree.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from vectora.config import Settings
from vectora.embeddings import Encoder
from vectora.ingestion import clean
from vectora.ingestion import metadata as meta_mod
from vectora.ingestion.chunking import enforce_size, semantic_chunk
from vectora.ingestion.parse import parse_pdf
from vectora.ingestion.sections import split_into_sections
from vectora.models import Chunk, content_hash
from vectora.observability import get_logger

log = get_logger(__name__)


@dataclass
class IngestionReport:
    """What happened, in enough detail to act on."""

    papers_processed: int = 0
    papers_skipped: int = 0
    papers_failed: list[tuple[str, str]] = field(default_factory=list)
    chunks_emitted: int = 0
    chunks_dropped: int = 0
    drop_reasons: Counter[str] = field(default_factory=Counter)
    chunk_types: Counter[str] = field(default_factory=Counter)
    excluded_sections: Counter[str] = field(default_factory=Counter)
    metadata_gaps: Counter[str] = field(default_factory=Counter)
    """Fields that could not be extracted and fell back to the filename or to
    blank. A title counted here is not absent — it is the filename."""
    low_confidence_papers: list[tuple[str, float]] = field(default_factory=list)
    space_loss_papers: Counter[str] = field(default_factory=Counter)
    parse_errors: int = 0

    def summary(self) -> str:
        lines = [
            "",
            "=" * 62,
            "  INGESTION REPORT",
            "=" * 62,
            f"  papers processed   {self.papers_processed}",
            f"  papers skipped     {self.papers_skipped}  (unchanged)",
            f"  papers failed      {len(self.papers_failed)}",
            f"  chunks emitted     {self.chunks_emitted}",
            f"  chunks dropped     {self.chunks_dropped}",
        ]
        if self.chunk_types:
            lines.append(
                "  by type            "
                + ", ".join(f"{k} {v}" for k, v in self.chunk_types.most_common())
            )
        if self.drop_reasons:
            lines.append("")
            lines.append("  dropped because:")
            for reason, n in self.drop_reasons.most_common(12):
                lines.append(f"    {n:>6}  {reason}")
        if self.metadata_gaps:
            lines.append("")
            lines.append(
                "  metadata not extractable, fallback used (papers):"
            )
            for fld, n in self.metadata_gaps.most_common():
                lines.append(f"    {n:>6}  {fld}")
        if self.low_confidence_papers:
            lines.append("")
            lines.append("  low-confidence metadata — worth eyeballing:")
            for name, conf in sorted(self.low_confidence_papers, key=lambda t: t[1])[:12]:
                lines.append(f"    {conf:.2f}  {name}")
        if self.space_loss_papers:
            lines.append("")
            lines.append(
                "  PDF parsing lost the spaces between words in these papers."
            )
            lines.append(
                "  Cleaning cannot recover that — re-ingest them from the source PDFs:"
            )
            for name, n in self.space_loss_papers.most_common(15):
                lines.append(f"    {n:>6} chunks  {name}")
        if self.papers_failed:
            lines.append("")
            lines.append("  failures:")
            for name, err in self.papers_failed[:12]:
                lines.append(f"    {name}: {err[:90]}")
        lines += ["=" * 62, ""]
        return "\n".join(lines)


# ── manifest ──────────────────────────────────────────────────────────────


def file_fingerprint(path: Path) -> str:
    """Hash of size plus the head and tail of the file.

    Reading whole PDFs to detect changes is wasteful and a rewrite always
    changes size or one of the ends. Cheap and good enough for a manifest.
    """
    stat = path.stat()
    h = hashlib.sha256(str(stat.st_size).encode())
    with open(path, "rb") as f:
        h.update(f.read(65_536))
        if stat.st_size > 131_072:
            f.seek(-65_536, 2)
            h.update(f.read(65_536))
    return h.hexdigest()[:24]


def load_manifest(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text("utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("manifest unreadable, starting fresh", extra={"err": str(exc)})
        return {}


def save_manifest(manifest: dict[str, dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), "utf-8")
    tmp.replace(path)  # atomic, so an interrupted run cannot corrupt it


# ── chunk emission for one paper ──────────────────────────────────────────


def chunks_for_pdf(
    path: Path,
    settings: Settings,
    *,
    encode: Callable[[Sequence[str]], np.ndarray] | None,
    report: IngestionReport,
    seen_hashes: set[str],
) -> list[Chunk]:
    """Parse one PDF and return the chunks worth indexing."""
    paper_id = path.stem
    parsed = parse_pdf(
        path,
        image_dir=settings.images_dir,
        min_image_width=settings.image_min_width,
        min_image_height=settings.image_min_height,
        extract_images=settings.caption_images or settings.keep_uncaptioned_images,
    )
    report.parse_errors += len(parsed.errors)
    if not parsed.pages:
        raise RuntimeError(parsed.errors[0] if parsed.errors else "no pages extracted")

    info = meta_mod.extract(parsed.header_text, paper_id, parsed.info)
    for field_name in ("title", "authors", "year", "doi"):
        if not getattr(info, field_name):
            report.metadata_gaps[field_name] += 1
    if info.confidence < 0.5:
        report.low_confidence_papers.append((paper_id, info.confidence))

    common = {
        "paper_id": paper_id,
        "title": info.title,
        "authors": info.authors,
        "year": info.year,
        "doi": info.doi,
    }
    out: list[Chunk] = []

    def emit(content: str, *, section: str, page: int, ctype: str, image_path: str = "") -> None:
        q = clean.score_quality(
            content,
            min_tokens=settings.min_chunk_tokens,
            min_alpha_ratio=settings.min_alpha_ratio,
            chunk_type=ctype,
        )
        if not q.ok:
            report.chunks_dropped += 1
            for r in q.reasons or ("rejected",):
                report.drop_reasons[r] += 1
            if "spaces_lost_in_parsing" in q.reasons:
                report.space_loss_papers[paper_id] += 1
            return
        digest = content_hash(content)
        if digest in seen_hashes:
            report.chunks_dropped += 1
            report.drop_reasons["duplicate"] += 1
            return
        seen_hashes.add(digest)
        out.append(
            Chunk.build(
                content=content,
                section=section,
                page=page,
                chunk_type=ctype,  # type: ignore[arg-type]
                image_path=image_path,
                quality=q.score,
                **common,
            )
        )
        report.chunk_types[ctype] += 1

    for section in split_into_sections(parsed.pages):
        if clean.is_excluded_section(section.name):
            report.excluded_sections[section.name] += 1
            report.drop_reasons[f"excluded_section:{section.name}"] += (
                len(section.text) // 800 + 1
            )
            continue

        body = clean.strip_inline_noise(clean.clean_block(section.text))
        if body:
            body = clean.unglue(body)
            pieces = semantic_chunk(
                body,
                encode=encode,
                similarity_threshold=settings.semantic_similarity_threshold,
                min_tokens=settings.min_chunk_tokens,
                max_tokens=settings.max_chunk_tokens,
                overlap=settings.chunk_overlap_sentences,
            )
            for piece in pieces:
                emit(piece, section=section.name, page=section.page, ctype="text")

        for page_num, md in section.tables:
            # A table is prefixed with its context so a retrieved grid is
            # interpretable on its own.
            labelled = f"Table from {info.title or paper_id}, section {section.name}:\n{md}"
            for piece in enforce_size(
                [labelled],
                min_tokens=1,  # tables have their own quality rule
                max_tokens=settings.max_chunk_tokens,
                overlap=0,
            ):
                emit(piece, section=section.name, page=page_num, ctype="table")

        if settings.caption_images:
            for page_num, img_path in section.images:
                caption = caption_image(img_path, settings)
                if caption:
                    emit(
                        f"Figure from {info.title or paper_id} ({section.name}): {caption}",
                        section=section.name,
                        page=page_num,
                        ctype="image",
                        image_path=img_path,
                    )
        elif settings.keep_uncaptioned_images:
            for page_num, img_path in section.images:
                emit(
                    f"Figure from {info.title or paper_id}, section {section.name}, "
                    f"page {page_num}. No caption available.",
                    section=section.name,
                    page=page_num,
                    ctype="image",
                    image_path=img_path,
                )
        # Default: uncaptioned images are not emitted at all. A chunk whose
        # only content is "[Figure: p3_x12.png]" has no retrievable text and
        # the old corpus had 547 of them.

    return out


def caption_image(image_path: str, settings: Settings) -> str:
    """Caption a figure with a vision model. Returns "" on any failure.

    Optional and off by default: it costs one vision call per figure, and
    547 figures is real money for marginal retrieval gain on this corpus.
    """
    try:
        import base64

        import anthropic
    except ImportError:
        log.warning("caption_images is on but the anthropic package is missing")
        return ""

    try:
        raw = Path(image_path).read_bytes()
        suffix = Path(image_path).suffix.lstrip(".").lower()
        media = {
            "jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
            "gif": "image/gif", "webp": "image/webp",
        }.get(suffix)
        if not media:
            return ""
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=200,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media,
                                "data": base64.b64encode(raw).decode(),
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "This is a figure from a scientific paper. In 1-3 "
                                "sentences, state what it shows: the figure type, the "
                                "variables or concepts plotted, and any finding visible. "
                                "Write it as a caption, not as commentary."
                            ),
                        },
                    ],
                }
            ],
        )
        return msg.content[0].text.strip()
    except Exception as exc:
        log.warning("captioning failed", extra={"image": image_path, "err": str(exc)[:120]})
        return ""


# ── full run ──────────────────────────────────────────────────────────────


def iter_pdfs(papers_dir: Path) -> Iterator[Path]:
    yield from sorted(
        p for p in papers_dir.rglob("*.pdf") if p.is_file() and not p.name.startswith(".")
    )


def run_ingestion(
    settings: Settings,
    *,
    papers_dir: Path | None = None,
    output: Path | None = None,
    force: bool = False,
    semantic: bool = True,
    encoder: Encoder | None = None,
    limit: int | None = None,
) -> IngestionReport:
    """Ingest new or changed PDFs and rewrite ``chunks.jsonl``.

    Incremental by fingerprint. Chunks from unchanged papers are carried
    over from the existing JSONL rather than recomputed, so adding one paper
    to a corpus of 62 costs one paper's work.
    """
    papers_dir = papers_dir or settings.papers_dir
    output = output or settings.chunks_path
    report = IngestionReport()

    if not papers_dir.exists():
        raise FileNotFoundError(
            f"No papers directory at {papers_dir}. Put the PDFs there, or point "
            f"VECTORA_PAPERS_DIR at them."
        )

    pdfs = list(iter_pdfs(papers_dir))
    if limit:
        pdfs = pdfs[:limit]
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found under {papers_dir}")

    manifest = {} if force else load_manifest(settings.manifest_path)
    existing = list(read_chunks(output)) if output.exists() and not force else []
    by_paper: dict[str, list[Chunk]] = {}
    for chunk in existing:
        by_paper.setdefault(chunk.paper_id, []).append(chunk)

    encode = None
    if semantic:
        enc = encoder or Encoder(settings)
        encode = lambda texts: enc.encode(texts, progress=False)  # noqa: E731

    seen_hashes: set[str] = set()
    fresh: dict[str, list[Chunk]] = {}
    current_ids: set[str] = set()

    try:
        from tqdm import tqdm

        iterator: Iterable[Path] = tqdm(pdfs, desc="Ingesting", unit="pdf")
    except ImportError:
        iterator = pdfs

    for pdf in iterator:
        paper_id = pdf.stem
        current_ids.add(paper_id)
        try:
            fingerprint = file_fingerprint(pdf)
        except OSError as exc:
            report.papers_failed.append((paper_id, str(exc)))
            continue

        unchanged = (
            manifest.get(paper_id, {}).get("fingerprint") == fingerprint
            and paper_id in by_paper
        )
        if unchanged:
            fresh[paper_id] = by_paper[paper_id]
            for chunk in fresh[paper_id]:
                seen_hashes.add(chunk.chunk_id)
            report.papers_skipped += 1
            continue

        try:
            chunks = chunks_for_pdf(
                pdf, settings, encode=encode, report=report, seen_hashes=seen_hashes
            )
        except Exception as exc:
            log.error("ingestion failed", extra={"file": pdf.name, "err": str(exc)})
            report.papers_failed.append((paper_id, str(exc)))
            continue

        fresh[paper_id] = chunks
        manifest[paper_id] = {
            "fingerprint": fingerprint,
            "chunks": len(chunks),
            "path": str(pdf.relative_to(papers_dir)),
        }
        report.papers_processed += 1

    # Papers deleted from disk fall out of both the manifest and the output.
    for gone in set(manifest) - current_ids:
        manifest.pop(gone, None)
        log.info("paper removed from corpus", extra={"paper_id": gone})

    all_chunks = [c for pid in sorted(fresh) for c in fresh[pid]]
    report.chunks_emitted = len(all_chunks)
    write_chunks(all_chunks, output)
    save_manifest(manifest, settings.manifest_path)
    log.info("ingestion complete", extra={"chunks": len(all_chunks), "out": str(output)})
    return report


# ── repair an existing corpus ─────────────────────────────────────────────


def repair_chunks(
    settings: Settings,
    *,
    source: Path | None = None,
    output: Path | None = None,
) -> IngestionReport:
    """Re-apply the current cleaning, metadata and quality rules to an
    existing ``chunks.jsonl``, without the source PDFs.

    This exists because the shipped corpus was produced by the old pipeline
    and the PDFs are not in the working tree. It cannot recover information
    that was never extracted — a chunk whose text was never captured stays
    missing — but it does:

    * drop layout debris, reference lists and excluded sections,
    * re-derive the year (fixing the ``"20"`` bug) and clean up titles,
    * repair ligatures, hyphenation and glued words,
    * re-merge fragments below the token floor,
    * discard uncaptioned image placeholders,
    * score and record quality.

    Re-ingesting from the PDFs is still strictly better; run this when you
    do not have them.
    """
    source = source or settings.chunks_path
    output = output or source
    report = IngestionReport()

    if not source.exists():
        raise FileNotFoundError(f"No chunk file at {source}")

    raw = list(read_chunks(source, validate=False))
    log.info("repairing corpus", extra={"chunks_in": len(raw)})

    # Pass 1: per-paper metadata consensus. A field correct on one chunk is
    # correct for the paper, so the best value per paper is propagated.
    best: dict[str, dict[str, str]] = {}
    for d in raw:
        pid = str(d.get("paper_id") or "")
        if not pid:
            continue
        slot = best.setdefault(pid, {"title": "", "authors": "", "year": "", "doi": ""})

        year, _ = meta_mod.extract_year(
            " ".join(str(d.get(k, "")) for k in ("year", "content", "title")), filename=pid
        )
        if year and not slot["year"]:
            slot["year"] = year

        doi = str(d.get("doi") or "").strip()
        if doi and not slot["doi"]:
            cleaned_doi, _ = meta_mod.extract_doi(doi)
            slot["doi"] = cleaned_doi or doi

        title = clean.normalise_chars(str(d.get("title") or "")).strip()
        if title and meta_mod._title_score(title, 0) > 0:
            if not slot["title"] or len(title) > len(slot["title"]):
                slot["title"] = title

        authors = str(d.get("authors") or "").strip()
        if authors and len(authors) > len(slot["authors"]):
            slot["authors"] = meta_mod._tidy_authors(authors)

    for pid, slot in best.items():
        if not slot["title"]:
            slot["title"] = meta_mod.title_from_filename(pid)
            report.metadata_gaps["title"] += 1
        for fld in ("authors", "year", "doi"):
            if not slot[fld]:
                report.metadata_gaps[fld] += 1

    # Pass 2: regroup by (paper, section) so merging a runt keeps it with
    # text it actually belongs to.
    grouped: dict[tuple[str, str, int], list[str]] = {}
    order: list[tuple[str, str, int]] = []
    for d in raw:
        pid = str(d.get("paper_id") or "")
        ctype = str(d.get("chunk_type") or "text")
        section_raw = str(d.get("section") or "")
        content = str(d.get("content") or "")

        if ctype == "image" and not settings.keep_uncaptioned_images:
            # Old image chunks are placeholders like "[Figure: x.png — enable
            # CAPTION_IMAGES...]" and carry nothing retrievable.
            if "enable CAPTION_IMAGES" in content or len(content) < 120:
                report.chunks_dropped += 1
                report.drop_reasons["uncaptioned_image"] += 1
                continue

        from vectora.ingestion.sections import canonicalise

        canonical = canonicalise(section_raw)
        if canonical:
            section = canonical
        elif (
            3 <= len(section_raw) <= 60
            and not clean.is_noise_line(section_raw)
            and not clean.is_furniture_section(section_raw)
        ):
            section = section_raw
        else:
            # Better an unlabelled chunk than one labelled "GModel ARTICLE IN
            # PRESS", which is what the old corpus carried and what then got
            # shown to the user as the section of a citation.
            section = ""
        if clean.is_excluded_section(section):
            report.chunks_dropped += 1
            report.drop_reasons[f"excluded_section:{section}"] += 1
            report.excluded_sections[section] += 1
            continue

        cleaned = clean.clean_block(content)
        if ctype == "text":
            # strip_inline_noise before unglue: the furniture patterns expect
            # the original spacing, and unglue would break them up.
            cleaned = clean.unglue(clean.strip_inline_noise(cleaned))
        if not cleaned:
            report.chunks_dropped += 1
            report.drop_reasons["empty_after_cleaning"] += 1
            continue

        key = (pid, section, int(d.get("page") or 0) if ctype != "text" else 0)
        if ctype != "text":
            key = (pid, section, int(d.get("page") or 0))
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(f"{ctype}\u0000{d.get('page') or 0}\u0000{cleaned}")

    # Pass 3: merge, size-enforce and quality-gate per group.
    out: list[Chunk] = []
    seen: set[str] = set()
    for key in order:
        pid, section, _ = key
        slot = best.get(pid, {"title": pid, "authors": "", "year": "", "doi": ""})
        entries = [e.split("\u0000", 2) for e in grouped[key]]

        for ctype in ("text", "table", "image"):
            group = [(int(p), c) for t, p, c in entries if t == ctype]
            if not group:
                continue
            page = group[0][0]
            if ctype == "text":
                pieces = enforce_size(
                    [c for _, c in group],
                    min_tokens=settings.min_chunk_tokens,
                    max_tokens=settings.max_chunk_tokens,
                    overlap=0,
                )
            else:
                pieces = [c for _, c in group]

            for piece in pieces:
                q = clean.score_quality(
                    piece,
                    min_tokens=settings.min_chunk_tokens,
                    min_alpha_ratio=settings.min_alpha_ratio,
                    chunk_type=ctype,
                )
                if not q.ok:
                    report.chunks_dropped += 1
                    for r in q.reasons or ("rejected",):
                        report.drop_reasons[r] += 1
                    if "spaces_lost_in_parsing" in q.reasons:
                        report.space_loss_papers[pid] += 1
                    continue
                digest = content_hash(piece)
                if digest in seen:
                    report.chunks_dropped += 1
                    report.drop_reasons["duplicate"] += 1
                    continue
                seen.add(digest)
                out.append(
                    Chunk.build(
                        content=piece,
                        paper_id=pid,
                        title=slot["title"],
                        authors=slot["authors"],
                        year=slot["year"],
                        doi=slot["doi"],
                        section=section,
                        page=page,
                        chunk_type=ctype,  # type: ignore[arg-type]
                        quality=q.score,
                    )
                )
                report.chunk_types[ctype] += 1

    report.chunks_emitted = len(out)
    report.papers_processed = len({c.paper_id for c in out})
    write_chunks(out, output)
    log.info(
        "repair complete",
        extra={"chunks_in": len(raw), "chunks_out": len(out), "out": str(output)},
    )
    return report


# ── jsonl io ──────────────────────────────────────────────────────────────


def read_chunks(path: Path, *, validate: bool = True) -> Iterator[Chunk] | Iterator[dict]:
    """Stream chunks from JSONL. Malformed lines are logged and skipped."""
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                log.warning("bad jsonl line", extra={"line": lineno, "err": str(exc)})
                continue
            if not validate:
                yield data
                continue
            try:
                yield Chunk.model_validate(data)
            except Exception as exc:
                log.warning("invalid chunk", extra={"line": lineno, "err": str(exc)[:160]})


def write_chunks(chunks: Sequence[Chunk], path: Path) -> None:
    """Write JSONL atomically, so an interrupted run cannot truncate the
    corpus that is currently indexed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.payload(), ensure_ascii=False) + "\n")
    tmp.replace(path)
