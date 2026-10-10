"""
PDF parsing: text, tables and images.

Kept separate from chunking so a parse failure on one page does not take out
the paper, and so the parser can be tested against a fixture PDF without
loading an embedding model.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vectora.observability import get_logger

log = get_logger(__name__)


@dataclass
class ParsedPDF:
    pages: list[dict[str, Any]] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)
    header_text: str = ""
    """Raw text of the first pages, for metadata extraction."""
    page_count: int = 0
    errors: list[str] = field(default_factory=list)


def table_to_markdown(table: list[list[Any]]) -> str:
    """Render a pdfplumber table as a markdown grid.

    Blank columns and fully blank rows are dropped — pdfplumber emits one
    column per ruling line, so a bordered table often comes back with empty
    columns that waste the chunk's token budget.
    """
    if not table:
        return ""

    rows = [[("" if c is None else str(c)).strip().replace("\n", " ") for c in row] for row in table]
    rows = [r for r in rows if any(cell for cell in r)]
    if len(rows) < 2:
        return ""

    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]

    keep = [i for i in range(width) if any(r[i] for r in rows)]
    if len(keep) < 2:
        return ""
    rows = [[r[i] for i in keep] for r in rows]

    header, *body = rows
    if not any(header):
        header = [f"col{i + 1}" for i in range(len(keep))]

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def parse_pdf(
    path: Path,
    *,
    image_dir: Path | None = None,
    min_image_width: int = 120,
    min_image_height: int = 120,
    extract_images: bool = True,
    header_pages: int = 2,
) -> ParsedPDF:
    """Extract text, tables and (optionally) images from one PDF.

    Per-page failures are collected rather than raised: a single
    unparseable page in a 30-page paper should cost that page, not the
    paper.
    """
    import pdfplumber

    result = ParsedPDF()

    try:
        pdf_ctx = pdfplumber.open(str(path))
    except Exception as exc:
        result.errors.append(f"open failed: {exc}")
        log.error("cannot open pdf", extra={"file": path.name, "err": str(exc)})
        return result

    with pdf_ctx as pdf:
        result.info = {k: v for k, v in (pdf.metadata or {}).items() if v}
        result.page_count = len(pdf.pages)

        for page_num, page in enumerate(pdf.pages, start=1):
            text, tables = "", []
            try:
                text = page.extract_text(x_tolerance=1.5, y_tolerance=2.5) or ""
            except Exception as exc:
                result.errors.append(f"page {page_num} text: {exc}")

            try:
                for raw in page.extract_tables() or []:
                    md = table_to_markdown(raw)
                    if md:
                        tables.append(md)
            except Exception as exc:
                result.errors.append(f"page {page_num} tables: {exc}")

            result.pages.append(
                {"page_num": page_num, "text": text, "tables": tables, "image_paths": []}
            )

    result.header_text = "\n".join(
        p["text"] for p in result.pages[:header_pages] if p.get("text")
    )

    if extract_images and image_dir is not None:
        _extract_images(path, result, image_dir, min_image_width, min_image_height)

    log.info(
        "parsed pdf",
        extra={
            "file": path.name,
            "pages": result.page_count,
            "tables": sum(len(p["tables"]) for p in result.pages),
            "images": sum(len(p["image_paths"]) for p in result.pages),
            "errors": len(result.errors),
        },
    )
    return result


def _extract_images(
    path: Path,
    result: ParsedPDF,
    image_dir: Path,
    min_w: int,
    min_h: int,
) -> None:
    """Save embedded raster images above the size floor.

    Deduplicated by xref within a document, because a logo repeated on every
    page would otherwise be written 30 times.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        result.errors.append("PyMuPDF not installed; skipped images")
        return

    out_dir = image_dir / path.stem
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        result.errors.append(f"image dir: {exc}")
        return

    seen: set[int] = set()
    doc = None
    try:
        doc = fitz.open(str(path))
        for page_index, page in enumerate(doc):
            page_num = page_index + 1
            if page_num > len(result.pages):
                break
            for img in page.get_images(full=True):
                xref = img[0]
                if xref in seen:
                    continue
                try:
                    base = doc.extract_image(xref)
                except Exception:
                    continue
                if base.get("width", 0) < min_w or base.get("height", 0) < min_h:
                    continue
                seen.add(xref)
                ext = base.get("ext", "png")
                target = out_dir / f"p{page_num:03d}_x{xref}.{ext}"
                try:
                    target.write_bytes(base["image"])
                except OSError as exc:
                    result.errors.append(f"image write: {exc}")
                    continue
                result.pages[page_num - 1]["image_paths"].append(str(target))
    except Exception as exc:
        result.errors.append(f"image extraction: {exc}")
    finally:
        if doc is not None:
            with contextlib.suppress(Exception):
                doc.close()
