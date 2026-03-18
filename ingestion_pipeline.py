"""
Research Paper RAG — Ingestion Pipeline
========================================
Converts a folder of PDFs into clean, metadata-rich chunks ready for embedding.

Pipeline:
  PDF → parse (text + tables + images) → section split → semantic chunking
      → noise reduction → deduplication → output JSONL

Usage:
  python ingestion_pipeline.py --input ./papers --output ./chunks.jsonl

Dependencies (install once):
  pip install unstructured[pdf] pdfplumber pymupdf sentence-transformers
              Pillow numpy tqdm hashlib
  # For vision-model image captioning (optional):
  pip install anthropic
"""

import os
import re
import json
import hashlib
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pdfplumber
import fitz  # PyMuPDF
from PIL import Image
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────

@dataclass
class Chunk:
    chunk_id: str           # SHA-256 of content (used for deduplication)
    paper_id: str           # stem of the source PDF filename
    title: str              # paper title (extracted or filename fallback)
    authors: str            # comma-joined author string
    year: str               # publication year (extracted or "")
    doi: str                # DOI if found in PDF text
    section: str            # section heading this chunk belongs to
    page: int               # 1-indexed page where the chunk starts
    chunk_type: str         # "text" | "table" | "image"
    content: str            # the actual text (or caption for images)
    image_path: str         # path to extracted image file (blank for text/table chunks)
    token_estimate: int     # rough token count (len / 4)


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

SECTION_HEADING_PATTERNS = [
    # Numbered: "1 Introduction", "2.3 Related Work"
    r"^(\d+\.?\d*\.?\d*)\s+([A-Z][^\n]{3,60})$",
    # ALL CAPS headings: "ABSTRACT", "CONCLUSION"
    r"^([A-Z][A-Z\s]{3,40})$",
    # Title-case short lines (≤8 words, no trailing punctuation)
    r"^([A-Z][a-zA-Z\s\-:]{3,60})$",
]

NOISE_PATTERNS = [
    r"^\s*\d+\s*$",                        # bare page numbers
    r"^(https?://\S+)$",                   # bare URLs
    r"^\s*©.*$",                           # copyright lines
    r"^\s*arXiv:\d{4}\.\d+.*$",           # arXiv identifiers
    r"^\s*(Received|Accepted|Published).*\d{4}.*$",  # submission dates
    r"^\s*\d+\s+[A-Z][a-z]+\s+et al\.",   # running headers like "12  Smith et al."
    r"^\s*$",                              # blank lines
]

DOI_PATTERN = re.compile(r"10\.\d{4,9}/[^\s]+")
YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")

# Semantic chunking: split where cosine similarity between adjacent sentences drops below this
SEMANTIC_SIMILARITY_THRESHOLD = 0.72

# Maximum tokens per chunk before forced sub-splitting
MAX_CHUNK_TOKENS = 512

# Minimum tokens — chunks shorter than this are merged with their neighbour
MIN_CHUNK_TOKENS = 40

# Image captions via Claude vision API (set False to skip)
CAPTION_IMAGES = False
IMAGE_MIN_WIDTH = 100   # pixels — ignore tiny decorative images
IMAGE_MIN_HEIGHT = 100


# ─────────────────────────────────────────────
# Step 1: PDF → raw text + tables + images
# ─────────────────────────────────────────────

def extract_pdf(pdf_path: Path) -> dict:
    """
    Returns:
      {
        "pages": [{"page_num": int, "text": str, "tables": [str], "image_paths": [str]}],
        "metadata": {"title": str, "authors": str, "year": str, "doi": str}
      }
    Tables are converted to markdown. Images are saved to ./extracted_images/.
    """
    result = {"pages": [], "metadata": {}}
    image_dir = pdf_path.parent / "extracted_images" / pdf_path.stem
    image_dir.mkdir(parents=True, exist_ok=True)

    # ── Extract with pdfplumber (text + tables) ──
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            raw_text = page.extract_text() or ""
            full_text += raw_text + "\n"

            # Extract tables as markdown
            tables_md = []
            for table in page.extract_tables():
                if table:
                    tables_md.append(_table_to_markdown(table))

            result["pages"].append({
                "page_num": page_num,
                "text": raw_text,
                "tables": tables_md,
                "image_paths": [],   # filled in below
            })

    # ── Extract images with PyMuPDF ──
    doc = fitz.open(pdf_path)
    for page_num, page in enumerate(doc, start=1):
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]
            ext = base_image["ext"]
            width = base_image.get("width", 0)
            height = base_image.get("height", 0)

            if width < IMAGE_MIN_WIDTH or height < IMAGE_MIN_HEIGHT:
                continue  # skip tiny decorative images

            img_filename = image_dir / f"page{page_num}_img{img_index}.{ext}"
            img_filename.write_bytes(img_bytes)
            result["pages"][page_num - 1]["image_paths"].append(str(img_filename))

    doc.close()

    # ── Extract metadata from first ~2 pages ──
    header_text = "\n".join(p["text"] for p in result["pages"][:3])
    result["metadata"] = _extract_metadata(header_text, pdf_path.stem)

    return result


def _table_to_markdown(table: list[list]) -> str:
    """Convert pdfplumber table (list of rows) to GitHub-flavored markdown."""
    if not table or not table[0]:
        return ""
    # Clean None cells
    rows = [[str(cell or "").strip() for cell in row] for row in table]
    header = rows[0]
    separator = ["---"] * len(header)
    body = rows[1:]
    md_rows = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in body:
        # Pad short rows
        while len(row) < len(header):
            row.append("")
        md_rows.append("| " + " | ".join(row) + " |")
    return "\n".join(md_rows)


def _extract_metadata(text: str, filename_stem: str) -> dict:
    doi = ""
    doi_match = DOI_PATTERN.search(text)
    if doi_match:
        doi = doi_match.group(0).rstrip(".,)")

    year = ""
    year_matches = YEAR_PATTERN.findall(text)
    if year_matches:
        year = year_matches[0]  # first plausible year in header

    # Very naive title extraction: first non-empty line in first 10 lines
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    title = lines[0] if lines else filename_stem

    # Authors: look for "Author1, Author2" patterns below the title
    authors = ""
    for line in lines[1:8]:
        if re.search(r"[A-Z][a-z]+,?\s+[A-Z]", line) and len(line) < 200:
            authors = line
            break

    return {"title": title, "authors": authors, "year": year, "doi": doi}


# ─────────────────────────────────────────────
# Step 2: Section detection
# ─────────────────────────────────────────────

def split_into_sections(pages: list[dict]) -> list[dict]:
    """
    Returns list of sections:
      [{"section": str, "page": int, "text": str, "tables": [str], "image_paths": [str]}]
    Each section = a heading + all content until the next heading.
    """
    sections = []
    current_section = "Abstract"
    current_page = 1
    current_lines = []
    current_tables = []
    current_images = []

    for page in pages:
        text = page["text"] or ""
        lines = text.split("\n")
        for line in lines:
            stripped = line.strip()
            if _is_section_heading(stripped):
                # Save current section
                if current_lines:
                    sections.append({
                        "section": current_section,
                        "page": current_page,
                        "text": "\n".join(current_lines),
                        "tables": current_tables,
                        "image_paths": current_images,
                    })
                current_section = stripped
                current_page = page["page_num"]
                current_lines = []
                current_tables = []
                current_images = []
            else:
                current_lines.append(line)

        # Accumulate tables and images at the page level
        current_tables.extend(page.get("tables", []))
        current_images.extend(page.get("image_paths", []))

    # Flush last section
    if current_lines:
        sections.append({
            "section": current_section,
            "page": current_page,
            "text": "\n".join(current_lines),
            "tables": current_tables,
            "image_paths": current_images,
        })

    return sections


def _is_section_heading(line: str) -> bool:
    if not line or len(line) > 80 or len(line) < 3:
        return False
    for pattern in SECTION_HEADING_PATTERNS:
        if re.match(pattern, line):
            return True
    return False


# ─────────────────────────────────────────────
# Step 3: Noise reduction
# ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Remove noise lines, normalize whitespace."""
    lines = text.split("\n")
    clean = []
    noise_re = [re.compile(p) for p in NOISE_PATTERNS]
    for line in lines:
        if any(r.match(line.strip()) for r in noise_re):
            continue
        # Normalize whitespace inside line
        line = re.sub(r"\s{2,}", " ", line).strip()
        if line:
            clean.append(line)
    return " ".join(clean)


# ─────────────────────────────────────────────
# Step 4: Semantic chunking
# ─────────────────────────────────────────────

_embedder: Optional[SentenceTransformer] = None

def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        log.info("Loading sentence-transformer model (first run only)...")
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def semantic_chunk(text: str, section: str) -> list[str]:
    """
    Split text into semantically coherent chunks.
    Algorithm:
      1. Split into sentences.
      2. Embed each sentence.
      3. Where cosine similarity between adjacent sentences drops below threshold,
         start a new chunk.
      4. If any chunk exceeds MAX_CHUNK_TOKENS, split it further by sentences.
      5. Merge chunks shorter than MIN_CHUNK_TOKENS into their neighbour.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []
    if len(sentences) == 1:
        return sentences

    embedder = _get_embedder()
    embeddings = embedder.encode(sentences, batch_size=64, show_progress_bar=False)

    # Compute cosine similarities between adjacent sentences
    sims = []
    for i in range(len(embeddings) - 1):
        a, b = embeddings[i], embeddings[i + 1]
        cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        sims.append(float(cos))

    # Build chunks: start new chunk where similarity drops
    chunks = []
    current = [sentences[0]]
    for i, sim in enumerate(sims):
        if sim < SEMANTIC_SIMILARITY_THRESHOLD:
            chunks.append(" ".join(current))
            current = [sentences[i + 1]]
        else:
            current.append(sentences[i + 1])
    if current:
        chunks.append(" ".join(current))

    # Enforce max/min token limits
    chunks = _enforce_size_limits(chunks)

    return [c for c in chunks if c.strip()]


def _split_sentences(text: str) -> list[str]:
    """Simple but robust sentence splitter (no NLTK dependency)."""
    # Split on sentence-ending punctuation followed by whitespace and a capital letter
    text = text.strip()
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [s.strip() for s in sentences if s.strip()]


def _enforce_size_limits(chunks: list[str]) -> list[str]:
    result = []
    for chunk in chunks:
        tokens = len(chunk) // 4
        if tokens > MAX_CHUNK_TOKENS:
            # Hard-split oversized chunks by sentences
            sentences = _split_sentences(chunk)
            sub, sub_tokens = [], 0
            for sent in sentences:
                st = len(sent) // 4
                if sub_tokens + st > MAX_CHUNK_TOKENS and sub:
                    result.append(" ".join(sub))
                    sub, sub_tokens = [sent], st
                else:
                    sub.append(sent)
                    sub_tokens += st
            if sub:
                result.append(" ".join(sub))
        else:
            result.append(chunk)

    # Merge short chunks into previous
    merged = []
    for chunk in result:
        if merged and len(chunk) // 4 < MIN_CHUNK_TOKENS:
            merged[-1] += " " + chunk
        else:
            merged.append(chunk)
    return merged


# ─────────────────────────────────────────────
# Step 5: Image captioning (optional)
# ─────────────────────────────────────────────

def caption_image(image_path: str) -> str:
    """
    Calls Claude claude-sonnet-4-20250514 vision to caption a figure.
    Only runs if CAPTION_IMAGES=True and anthropic package is installed.
    Falls back to a placeholder if API call fails.
    """
    if not CAPTION_IMAGES:
        return f"[Figure: {Path(image_path).name} — enable CAPTION_IMAGES for auto-caption]"
    try:
        import anthropic, base64
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ext = Path(image_path).suffix.lstrip(".").lower()
        media_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                     "png": "image/png", "gif": "image/gif", "webp": "image/webp"}
        media_type = media_map.get(ext, "image/png")

        client = anthropic.Anthropic()
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64",
                                                  "media_type": media_type, "data": b64}},
                    {"type": "text", "text":
                     "You are processing a figure from a scientific paper. "
                     "In 1-3 sentences, describe what this figure shows: its type "
                     "(chart, diagram, microscopy, etc.), what variables or concepts "
                     "it depicts, and any notable finding visible. Be concise."}
                ]
            }]
        )
        return msg.content[0].text.strip()
    except Exception as e:
        log.warning(f"Image captioning failed for {image_path}: {e}")
        return f"[Figure: {Path(image_path).name}]"


# ─────────────────────────────────────────────
# Step 6: Deduplication
# ─────────────────────────────────────────────

def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ─────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────

def process_pdf(pdf_path: Path, seen_hashes: set) -> list[Chunk]:
    """Full pipeline for a single PDF. Returns a list of Chunk objects."""
    log.info(f"Processing: {pdf_path.name}")
    paper_id = pdf_path.stem

    # Step 1: Extract
    extracted = extract_pdf(pdf_path)
    meta = extracted["metadata"]

    # Step 2: Section split
    sections = split_into_sections(extracted["pages"])
    log.info(f"  → {len(sections)} sections detected")

    chunks: list[Chunk] = []

    for sec in sections:
        # Step 3: Clean text
        cleaned = clean_text(sec["text"])

        # Step 4: Semantic chunking of section text
        if cleaned.strip():
            text_chunks = semantic_chunk(cleaned, sec["section"])
            for chunk_text in text_chunks:
                h = content_hash(chunk_text)
                if h in seen_hashes:
                    continue  # deduplicate
                seen_hashes.add(h)
                chunks.append(Chunk(
                    chunk_id=h,
                    paper_id=paper_id,
                    title=meta["title"],
                    authors=meta["authors"],
                    year=meta["year"],
                    doi=meta["doi"],
                    section=sec["section"],
                    page=sec["page"],
                    chunk_type="text",
                    content=chunk_text,
                    image_path="",
                    token_estimate=len(chunk_text) // 4,
                ))

        # Tables → each table is its own chunk
        for table_md in sec.get("tables", []):
            table_cleaned = table_md.strip()
            if not table_cleaned:
                continue
            h = content_hash(table_cleaned)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            chunks.append(Chunk(
                chunk_id=h,
                paper_id=paper_id,
                title=meta["title"],
                authors=meta["authors"],
                year=meta["year"],
                doi=meta["doi"],
                section=sec["section"],
                page=sec["page"],
                chunk_type="table",
                content=table_cleaned,
                image_path="",
                token_estimate=len(table_cleaned) // 4,
            ))

        # Images → caption and store
        for img_path in sec.get("image_paths", []):
            caption = caption_image(img_path)
            h = content_hash(img_path)  # hash by path, captions may vary
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            chunks.append(Chunk(
                chunk_id=h,
                paper_id=paper_id,
                title=meta["title"],
                authors=meta["authors"],
                year=meta["year"],
                doi=meta["doi"],
                section=sec["section"],
                page=sec["page"],
                chunk_type="image",
                content=caption,
                image_path=img_path,
                token_estimate=len(caption) // 4,
            ))

    log.info(f"  → {len(chunks)} chunks produced")
    return chunks


# ─────────────────────────────────────────────
# Manifest — tracks which PDFs have been ingested
# ─────────────────────────────────────────────

MANIFEST_PATH = Path("./ingestion_manifest.json")


def pdf_file_hash(pdf_path: Path) -> str:
    """SHA-256 of the raw PDF bytes — changes if the file is modified."""
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()[:24]


def load_manifest(manifest_path: Path) -> dict:
    """
    Returns dict: { pdf_stem: { "file_hash": str, "chunks": int, "ingested_at": str } }
    """
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_manifest(manifest: dict, manifest_path: Path):
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    log.info(f"Manifest saved -> {manifest_path}")


def get_new_pdfs(
    pdf_files: list,
    manifest: dict,
    force: bool = False,
) -> tuple:
    """
    Compare pdf_files against the manifest.
    Returns (new_or_changed, already_done).
      new_or_changed  -- PDFs not in manifest OR whose file hash changed
      already_done    -- PDFs already ingested with no changes
    force=True reprocesses everything regardless.
    """
    new_or_changed = []
    already_done   = []

    for pdf_path in pdf_files:
        stem = pdf_path.stem
        if force or stem not in manifest:
            new_or_changed.append(pdf_path)
        else:
            stored_hash  = manifest[stem].get("file_hash", "")
            current_hash = pdf_file_hash(pdf_path)
            if current_hash != stored_hash:
                log.info(f"  {pdf_path.name}: file changed -- will reprocess")
                new_or_changed.append(pdf_path)
            else:
                already_done.append(pdf_path)

    return new_or_changed, already_done


# ─────────────────────────────────────────────
# Incremental pipeline
# ─────────────────────────────────────────────

def run_pipeline(
    input_dir: Path,
    output_path: Path,
    manifest_path: Path = None,
    force: bool = False,
):
    """
    Incremental ingestion pipeline.

    On first run: processes all PDFs, writes chunks.jsonl, creates manifest.
    On subsequent runs: only processes new or changed PDFs, appends to chunks.jsonl.

    The manifest (ingestion_manifest.json) tracks:
      - which PDFs have been processed
      - their file hash (detects if a PDF was replaced/modified)
      - how many chunks were produced
      - when they were ingested

    Use --force to reprocess everything from scratch.
    """
    if manifest_path is None:
        manifest_path = MANIFEST_PATH

    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        log.error(f"No PDF files found in {input_dir}")
        return

    manifest = load_manifest(manifest_path)
    to_process, skipped = get_new_pdfs(pdf_files, manifest, force=force)

    log.info(f"Found {len(pdf_files)} PDF(s) total")
    log.info(f"  -> {len(skipped)} already ingested -- skipping")
    log.info(f"  -> {len(to_process)} new or changed -- processing")

    if not to_process:
        log.info("\n✓ Nothing to do. All papers already ingested.")
        log.info("   Add new PDFs to the papers/ folder and run again.")
        log.info("   Use --force to reprocess everything.")
        return

    # Load existing chunk hashes for cross-paper deduplication
    seen_hashes: set = set()
    if output_path.exists() and not force:
        log.info("Loading existing chunk hashes for deduplication...")
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    chunk = json.loads(line)
                    seen_hashes.add(chunk.get("chunk_id", ""))
                except json.JSONDecodeError:
                    pass
        log.info(f"  -> {len(seen_hashes)} existing chunks loaded")

    write_mode = "w" if force else "a"
    if force:
        seen_hashes = set()
        manifest    = {}
        log.info("--force: reprocessing all PDFs from scratch")

    import datetime
    total_new_chunks = 0
    now = datetime.datetime.utcnow().isoformat()

    with open(output_path, write_mode, encoding="utf-8") as out_f:
        for pdf_path in tqdm(to_process, desc="Processing PDFs"):
            try:
                file_hash = pdf_file_hash(pdf_path)
                chunks    = process_pdf(pdf_path, seen_hashes)

                for chunk in chunks:
                    out_f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

                total_new_chunks += len(chunks)

                manifest[pdf_path.stem] = {
                    "file_hash":   file_hash,
                    "filename":    pdf_path.name,
                    "chunks":      len(chunks),
                    "ingested_at": now,
                }

            except Exception as e:
                log.error(f"Failed to process {pdf_path.name}: {e}")

    save_manifest(manifest, manifest_path)

    log.info(f"\n✓ Done.")
    log.info(f"   New chunks written  : {total_new_chunks}")
    log.info(f"   Total papers tracked: {len(manifest)}")
    log.info(f"   Output              : {output_path}")
    log.info(f"   Manifest            : {manifest_path}")


def show_status(input_dir: Path, manifest_path: Path = None):
    """Print ingestion status for all PDFs in the folder."""
    if manifest_path is None:
        manifest_path = MANIFEST_PATH
    manifest  = load_manifest(manifest_path)
    pdf_files = sorted(input_dir.glob("*.pdf"))

    if not pdf_files:
        print("No PDFs found.")
        return

    print(f"\n{'PDF':<40} {'Status':<16} {'Chunks':>6}  Ingested at")
    print("-" * 80)
    for pdf_path in pdf_files:
        stem = pdf_path.stem
        if stem not in manifest:
            print(f"{pdf_path.name:<40} {'NOT INGESTED':<16} {'':>6}")
        else:
            entry       = manifest[stem]
            file_hash   = pdf_file_hash(pdf_path)
            stored_hash = entry.get("file_hash", "")
            status      = "✓ done (CHANGED)" if file_hash != stored_hash else "✓ done"
            ingested_at = entry.get("ingested_at", "")[:10]
            print(f"{pdf_path.name:<40} {status:<16} {entry['chunks']:>6}  {ingested_at}")
    print()


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Research Paper RAG -- Incremental Ingestion Pipeline",
    )
    parser.add_argument("--input",    default="./papers",                   help="Directory containing PDF files")
    parser.add_argument("--output",   default="./chunks.jsonl",              help="Output JSONL file path")
    parser.add_argument("--manifest", default="./ingestion_manifest.json",   help="Manifest file path")
    parser.add_argument("--force",    action="store_true",                   help="Reprocess all PDFs from scratch")
    parser.add_argument("--status",   action="store_true",                   help="Show ingestion status and exit")
    parser.add_argument("--caption-images", action="store_true",
                        help="Enable vision captioning for figures (requires ANTHROPIC_API_KEY)")
    args = parser.parse_args()

    if args.caption_images:
        CAPTION_IMAGES = True

    if args.status:
        show_status(Path(args.input), Path(args.manifest))
    else:
        run_pipeline(
            input_dir     = Path(args.input),
            output_path   = Path(args.output),
            manifest_path = Path(args.manifest),
            force         = args.force,
        )
