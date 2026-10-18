"""
Document loading for formats other than PDF.

The pipeline was PDF-only, which quietly limited what a corpus could be.
Markdown, plain text and HTML are all perfectly good corpus material —
documentation, RFCs, wikis, scraped articles — and they parse more reliably
than PDFs do, because the text was never laid out for print in the first
place.

Everything here returns the same :class:`~vectora.ingestion.parse.ParsedPDF`
shape the PDF parser produces, so the section-splitting, chunking and
quality stages downstream do not branch on format. Pagination is the one
thing that has to be invented: a text file has no pages, so a synthetic
page boundary is inserted every ``LINES_PER_PAGE`` lines, which keeps
citations pointing at a locatable part of the document rather than at
"page 0".
"""

from __future__ import annotations

import html
import re
from pathlib import Path

from vectora.ingestion.parse import ParsedPDF, table_to_markdown
from vectora.observability import get_logger

log = get_logger(__name__)

#: Formats the pipeline can ingest. Anything else is skipped with a warning
#: rather than failing the run, so one stray file in a folder of documents
#: does not stop the ingest.
TEXT_SUFFIXES = frozenset({".txt", ".text", ".md", ".markdown", ".rst"})
HTML_SUFFIXES = frozenset({".html", ".htm", ".xhtml"})
PDF_SUFFIXES = frozenset({".pdf"})
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | HTML_SUFFIXES | PDF_SUFFIXES

#: Synthetic page size for formats without pages. ~45 lines is close to a
#: printed page, which keeps page numbers intuitive for a reader.
LINES_PER_PAGE = 45

_MD_FENCE = re.compile(r"^```|^~~~")
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")
_MD_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_MD_EMPHASIS = re.compile(r"(\*{1,3}|_{1,3})(?=\S)(.+?)(?<=\S)\1")
_MD_INLINE_CODE = re.compile(r"`([^`]+)`")
_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_RST_OVERLINE = re.compile(r"^[=\-~^\"'`#*+]{3,}\s*$")


def is_supported(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_SUFFIXES


def iter_documents(root: Path) -> list[Path]:
    """Every ingestible document under ``root``, sorted and deduplicated.

    Hidden files and anything in a dot-directory are skipped, so a stray
    ``.git`` or ``.ipynb_checkpoints`` does not become corpus material.
    """
    out: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        if is_supported(path):
            out.append(path)
    return out


def parse_document(
    path: Path,
    *,
    image_dir: Path | None = None,
    min_image_width: int = 120,
    min_image_height: int = 120,
    extract_images: bool = False,
    header_pages: int = 2,
) -> ParsedPDF:
    """Parse any supported document into the common page structure."""
    suffix = path.suffix.lower()
    if suffix in PDF_SUFFIXES:
        from vectora.ingestion.parse import parse_pdf

        return parse_pdf(
            path,
            image_dir=image_dir,
            min_image_width=min_image_width,
            min_image_height=min_image_height,
            extract_images=extract_images,
            header_pages=header_pages,
        )
    if suffix in HTML_SUFFIXES:
        return parse_html(path, header_pages=header_pages)
    if suffix in TEXT_SUFFIXES:
        return parse_text(path, header_pages=header_pages)
    raise ValueError(f"unsupported document type: {path.suffix} ({path.name})")


# ── plain text and markdown ───────────────────────────────────────────────


def parse_text(path: Path, *, header_pages: int = 2) -> ParsedPDF:
    """Read a text, markdown or reStructuredText file.

    Markdown structure is converted to the plain conventions the section
    detector already understands: ``## Methods`` becomes a bare ``Methods``
    line, and a pipe table is handed through as markdown because that is
    what the table path expects anyway.
    """
    result = ParsedPDF()
    try:
        raw = path.read_text("utf-8")
    except UnicodeDecodeError:
        try:
            raw = path.read_text("latin-1")
            result.errors.append("decoded as latin-1")
        except OSError as exc:
            result.errors.append(f"read failed: {exc}")
            return result
    except OSError as exc:
        result.errors.append(f"read failed: {exc}")
        return result

    front, body = _split_frontmatter(raw)
    if front:
        result.info.update(front)

    is_markdown = path.suffix.lower() in {".md", ".markdown"}
    is_rst = path.suffix.lower() == ".rst"
    lines, tables = _flatten_markup(body, markdown=is_markdown, rst=is_rst)

    for index, start in enumerate(range(0, max(len(lines), 1), LINES_PER_PAGE)):
        page_lines = lines[start : start + LINES_PER_PAGE]
        result.pages.append(
            {
                "page_num": index + 1,
                "text": "\n".join(page_lines),
                # Tables land on the first page: without layout information
                # there is no honest way to say which page one belongs to,
                # and claiming a precise page would be worse than not.
                "tables": tables if index == 0 else [],
                "image_paths": [],
            }
        )

    result.page_count = len(result.pages)
    result.header_text = "\n".join(
        p["text"] for p in result.pages[:header_pages] if p.get("text")
    )
    if not result.info.get("title"):
        inferred = _infer_title(lines, path.stem)
        if inferred:
            result.info["title"] = inferred

    log.info(
        "parsed text document",
        extra={
            "file": path.name,
            "lines": len(lines),
            "pages": result.page_count,
            "tables": len(tables),
        },
    )
    return result


def _split_frontmatter(raw: str) -> tuple[dict[str, str], str]:
    """Pull simple ``key: value`` YAML front matter off a markdown file.

    Deliberately not a YAML parser: front matter is metadata we are glad to
    have and not willing to add a dependency or a failure mode for.
    """
    match = _FRONTMATTER.match(raw)
    if not match:
        return {}, raw
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" not in line or line.lstrip().startswith("#"):
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip().strip("'\"")
        if key in {"title", "author", "authors", "date", "year", "doi", "description"}:
            meta[key if key != "authors" else "author"] = value
    return meta, raw[match.end() :]


def _flatten_markup(
    body: str, *, markdown: bool, rst: bool
) -> tuple[list[str], list[str]]:
    """Strip markup, returning content lines and any tables found."""
    lines: list[str] = []
    tables: list[str] = []
    in_fence = False

    raw_lines = body.splitlines()
    for line in raw_lines:
        if markdown and _MD_FENCE.match(line.strip()):
            in_fence = not in_fence
            continue
        if in_fence:
            # Code blocks are kept verbatim: an identifier or a config
            # snippet is often exactly what someone searches for.
            lines.append(line.rstrip())
            continue

        if markdown and _MD_TABLE_ROW.match(line):
            # Kept in the text flow rather than extracted. A markdown table
            # is already legible text, and diverting it into the structural
            # table path made it inherit whichever section happened to be
            # open at the end of the page — which silently lost a table
            # whose document ended in a reference list.
            lines.append(line.strip())
            continue

        if markdown:
            heading = _MD_HEADING.match(line.strip())
            if heading:
                lines.append(heading.group(2).strip())
                continue
            line = _strip_inline_markdown(line)

        if rst and _RST_OVERLINE.match(line) and lines:
            # An underline promotes the line above it to a heading, which it
            # already is once the underline is dropped.
            continue

        lines.append(line.rstrip())

    return lines, [t for t in tables if t]


def _strip_inline_markdown(line: str) -> str:
    line = _MD_IMAGE.sub("", line)
    line = _MD_LINK.sub(r"\1", line)
    line = _MD_INLINE_CODE.sub(r"\1", line)
    line = _MD_EMPHASIS.sub(r"\2", line)
    line = re.sub(r"^\s{0,3}>\s?", "", line)
    line = re.sub(r"^\s*[-*+]\s+", "", line)
    line = re.sub(r"^\s*\d+[.)]\s+", "", line)
    return line


def _infer_title(lines: list[str], fallback: str) -> str:
    for line in lines[:12]:
        stripped = line.strip()
        if 12 <= len(stripped) <= 200 and len(stripped.split()) >= 3:
            return stripped
    return fallback.replace("-", " ").replace("_", " ")


# ── HTML ──────────────────────────────────────────────────────────────────

#: Removed before anything else is read. Note that <head> is deliberately
#: NOT in this list: it holds <title> and the citation meta tags, which are
#: the most reliable metadata an HTML document has. Stripping it discards
#: exactly the information worth keeping.
_SCRIPT_STYLE = re.compile(
    r"<(script|style|noscript|svg|template)\b[^>]*>.*?</\1>",
    re.IGNORECASE | re.DOTALL,
)
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_TITLE_TAG = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_META_NAME = re.compile(
    r'<meta\s+[^>]*name=["\'](author|description|citation_title|citation_author|'
    r'citation_date|citation_doi)["\'][^>]*content=["\']([^"\']*)["\']',
    re.IGNORECASE,
)
_META_CONTENT_FIRST = re.compile(
    r'<meta\s+[^>]*content=["\']([^"\']*)["\'][^>]*name=["\'](author|description|'
    r'citation_title|citation_author|citation_date|citation_doi)["\']',
    re.IGNORECASE,
)
_HEADING_TAG = re.compile(
    r"<h([1-6])\b[^>]*>(.*?)</h\1>", re.IGNORECASE | re.DOTALL
)
_TABLE_TAG = re.compile(r"<table\b[^>]*>(.*?)</table>", re.IGNORECASE | re.DOTALL)
_TR = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
_TD = re.compile(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", re.IGNORECASE | re.DOTALL)
_BLOCK_END = re.compile(
    r"</(p|div|section|article|li|tr|h[1-6]|blockquote|pre|ul|ol|table)>",
    re.IGNORECASE,
)
_BR = re.compile(r"<br\s*/?>", re.IGNORECASE)
_ANY_TAG = re.compile(r"<[^>]+>")
_BLANKS = re.compile(r"\n{3,}")


def parse_html(path: Path, *, header_pages: int = 2) -> ParsedPDF:
    """Extract readable text, headings and tables from an HTML file.

    A regex reader rather than a parser dependency: the job is to recover
    paragraphs and headings from saved articles, not to build a DOM. Script,
    style and head content is removed first, so no JavaScript ends up in the
    corpus — which is the one failure mode that would really pollute
    retrieval.
    """
    result = ParsedPDF()
    try:
        raw = path.read_text("utf-8", errors="replace")
    except OSError as exc:
        result.errors.append(f"read failed: {exc}")
        return result

    raw = _COMMENT.sub(" ", raw)
    raw = _SCRIPT_STYLE.sub(" ", raw)

    for pattern in (_META_NAME, _META_CONTENT_FIRST):
        for a, b in pattern.findall(raw):
            key, value = (a, b) if pattern is _META_NAME else (b, a)
            key = key.lower()
            value = html.unescape(value).strip()
            if not value:
                continue
            if key in {"citation_title"}:
                result.info.setdefault("title", value)
            elif key in {"author", "citation_author"}:
                existing = result.info.get("author", "")
                result.info["author"] = (
                    f"{existing}, {value}" if existing and value not in existing else value
                )
            elif key == "citation_date":
                result.info.setdefault("creationDate", value)
            elif key == "citation_doi":
                result.info.setdefault("doi", value)
            elif key == "description":
                result.info.setdefault("description", value)

    title_match = _TITLE_TAG.search(raw)
    if title_match:
        result.info.setdefault(
            "title", _collapse(html.unescape(_ANY_TAG.sub(" ", title_match.group(1))))
        )

    tables = [md for md in (_html_table_to_markdown(m) for m in _TABLE_TAG.findall(raw)) if md]
    body = _TABLE_TAG.sub(" ", raw)

    # Headings become their own lines so the section detector can see them.
    body = _HEADING_TAG.sub(lambda m: f"\n\n{_collapse(_ANY_TAG.sub(' ', m.group(2)))}\n\n", body)
    body = _BR.sub("\n", body)
    body = _BLOCK_END.sub("\n\n", body)
    body = _ANY_TAG.sub(" ", body)
    body = html.unescape(body)
    body = "\n".join(_collapse(line) for line in body.splitlines())
    body = _BLANKS.sub("\n\n", body).strip()

    lines = body.splitlines()
    for index, start in enumerate(range(0, max(len(lines), 1), LINES_PER_PAGE)):
        result.pages.append(
            {
                "page_num": index + 1,
                "text": "\n".join(lines[start : start + LINES_PER_PAGE]),
                "tables": tables if index == 0 else [],
                "image_paths": [],
            }
        )

    result.page_count = len(result.pages)
    result.header_text = "\n".join(
        p["text"] for p in result.pages[:header_pages] if p.get("text")
    )
    log.info(
        "parsed html document",
        extra={"file": path.name, "pages": result.page_count, "tables": len(tables)},
    )
    return result


def _html_table_to_markdown(inner: str) -> str:
    rows: list[list[str]] = []
    for tr in _TR.findall(inner):
        cells = [
            _collapse(html.unescape(_ANY_TAG.sub(" ", td))) for td in _TD.findall(tr)
        ]
        if any(cells):
            rows.append(cells)
    return table_to_markdown(rows) if len(rows) >= 2 else ""


def _collapse(text: str) -> str:
    return re.sub(r"[ \t ]+", " ", text).strip()
