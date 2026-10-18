"""
Corpus acquisition.

The hardest part of standing up a RAG system on a new subject is not the
code, it is getting a few hundred documents onto disk. This turns that into
one command.

arXiv is the default source because its API is open, needs no key, and
returns both structured metadata and a PDF for every hit — which means the
title, authors, date and identifier are *known* rather than scraped out of
the first page, and the ingestion pipeline's metadata heuristics become a
fallback rather than the only option.

Two rules this module follows, because a fetcher that ignores them is a
fetcher that gets an IP banned:

* **Rate limiting.** arXiv asks for roughly one request every three
  seconds. The default is deliberately slower than the stated limit.
* **Resumability.** Every download is checked against what is already on
  disk, so an interrupted fetch continues instead of starting over, and
  re-running is cheap rather than rude.

The sidecar metadata written next to each PDF is the real payoff: it is read
back at ingestion time, so the citations for a fetched corpus are exact.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from vectora.observability import get_logger

log = get_logger(__name__)

ARXIV_API = "http://export.arxiv.org/api/query"
#: arXiv's guidance is one request per three seconds; we leave headroom.
ARXIV_MIN_INTERVAL = 3.5
ARXIV_PAGE_SIZE = 100
USER_AGENT = "vectora/3.0 (+https://github.com/SiddharthaRaja/Vectora)"

#: Sidecar filename next to each downloaded document.
SIDECAR_SUFFIX = ".meta.json"

_ATOM = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}
_WS = re.compile(r"\s+")
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")

#: Ready-made corpus definitions, so "a corpus about X" needs no API
#: knowledge. Each is an arXiv category query with a human description.
PRESETS: dict[str, dict[str, str]] = {
    "ml-systems": {
        "query": "cat:cs.LG OR cat:cs.DC OR cat:cs.PF",
        "description": "Machine learning systems, distributed training and performance",
        "document_kind": "machine learning papers",
        "terms": "throughput, latency, quantisation, kernel, scheduler, inference, "
        "batching, parallelism, memory bandwidth, FLOPs",
    },
    "nlp": {
        "query": "cat:cs.CL",
        "description": "Computational linguistics and natural language processing",
        "document_kind": "NLP papers",
        "terms": "tokenisation, embedding, attention, perplexity, fine-tuning, "
        "benchmark, corpus, alignment, decoder, pretraining",
    },
    "retrieval": {
        "query": "cat:cs.IR",
        "description": "Information retrieval, search and recommendation",
        "document_kind": "information retrieval papers",
        "terms": "BM25, dense retrieval, reranking, recall, nDCG, index, "
        "query expansion, relevance, embedding, hybrid search",
    },
    "rl": {
        "query": "cat:cs.AI AND abs:reinforcement learning",
        "description": "Reinforcement learning",
        "document_kind": "reinforcement learning papers",
        "terms": "policy, reward, value function, exploration, on-policy, "
        "off-policy, advantage, rollout, sample efficiency",
    },
    "robotics": {
        "query": "cat:cs.RO",
        "description": "Robotics, control and autonomous systems",
        "document_kind": "robotics papers",
        "terms": "controller, trajectory, kinematics, actuator, SLAM, "
        "manipulation, sim-to-real, sensor fusion",
    },
    "vision": {
        "query": "cat:cs.CV",
        "description": "Computer vision",
        "document_kind": "computer vision papers",
        "terms": "convolution, segmentation, detection, augmentation, "
        "backbone, mAP, diffusion, feature map",
    },
}


@dataclass
class FetchedDocument:
    """Metadata known from the source, not guessed from the document."""

    doc_id: str
    title: str
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    published: str = ""
    updated: str = ""
    categories: list[str] = field(default_factory=list)
    doi: str = ""
    pdf_url: str = ""
    source: str = "arxiv"
    filename: str = ""

    @property
    def year(self) -> str:
        match = re.search(r"(19|20)\d{2}", self.published or self.updated or "")
        return match.group(0) if match else ""

    @property
    def author_string(self) -> str:
        return ", ".join(self.authors[:14])

    def to_sidecar(self) -> dict[str, object]:
        data = asdict(self)
        data["year"] = self.year
        data["fetched_at"] = datetime.now(timezone.utc).isoformat()
        return data


@dataclass
class FetchReport:
    requested: int = 0
    found: int = 0
    downloaded: int = 0
    skipped_existing: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)
    bytes_written: int = 0

    def summary(self) -> str:
        lines = [
            "",
            "=" * 60,
            "  FETCH REPORT",
            "=" * 60,
            f"  matched upstream   {self.found}",
            f"  downloaded         {self.downloaded}",
            f"  already present    {self.skipped_existing}",
            f"  failed             {len(self.failed)}",
            f"  written            {self.bytes_written / 1e6:.1f} MB",
        ]
        if self.failed:
            lines.append("")
            lines.append("  failures:")
            for doc_id, err in self.failed[:10]:
                lines.append(f"    {doc_id}: {err[:70]}")
        lines += ["=" * 60, ""]
        return "\n".join(lines)


class _RateLimiter:
    """Sleeps so that calls are at least ``interval`` seconds apart."""

    def __init__(self, interval: float) -> None:
        self.interval = interval
        self._last = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last
        if self._last and elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self._last = time.monotonic()


def safe_filename(doc_id: str, title: str, *, suffix: str = ".pdf") -> str:
    """A filename that is stable, readable and legal on Windows.

    The id leads so the name is unique and sorts chronologically; a slug of
    the title follows so a human can tell the files apart in a folder.
    """
    ident = _UNSAFE.sub("-", doc_id).strip("-")[:24]
    slug = _UNSAFE.sub("-", _WS.sub(" ", title).strip()).strip("-")[:70].rstrip("-")
    stem = f"{ident}__{slug}" if slug else ident
    return stem[:120] + suffix


def _request(url: str, *, timeout: float) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def parse_arxiv_feed(xml_bytes: bytes) -> tuple[list[FetchedDocument], int]:
    """Parse an arXiv Atom response into documents and the total result count.

    Separated from the HTTP call so it can be tested against a recorded
    response without network access.
    """
    root = ET.fromstring(xml_bytes)

    total = 0
    total_el = root.find("opensearch:totalResults", _ATOM)
    if total_el is not None and (total_el.text or "").strip().isdigit():
        total = int(total_el.text.strip())

    docs: list[FetchedDocument] = []
    for entry in root.findall("atom:entry", _ATOM):
        raw_id = _text(entry.find("atom:id", _ATOM))
        if not raw_id:
            continue
        # "http://arxiv.org/abs/2401.01234v2" -> "2401.01234v2"
        doc_id = raw_id.rstrip("/").split("/abs/")[-1]

        pdf_url = ""
        for link in entry.findall("atom:link", _ATOM):
            if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                pdf_url = link.get("href", "")
                break
        if not pdf_url and doc_id:
            pdf_url = f"https://arxiv.org/pdf/{doc_id}"

        doc = FetchedDocument(
            doc_id=doc_id,
            title=_WS.sub(" ", _text(entry.find("atom:title", _ATOM))),
            authors=[
                _WS.sub(" ", _text(a.find("atom:name", _ATOM)))
                for a in entry.findall("atom:author", _ATOM)
                if _text(a.find("atom:name", _ATOM))
            ],
            abstract=_WS.sub(" ", _text(entry.find("atom:summary", _ATOM))),
            published=_text(entry.find("atom:published", _ATOM)),
            updated=_text(entry.find("atom:updated", _ATOM)),
            categories=[
                c.get("term", "")
                for c in entry.findall("atom:category", _ATOM)
                if c.get("term")
            ],
            doi=_text(entry.find("arxiv:doi", _ATOM)),
            pdf_url=pdf_url,
        )
        doc.filename = safe_filename(doc.doc_id, doc.title)
        docs.append(doc)

    return docs, total


def _text(el: ET.Element | None) -> str:
    return (el.text or "").strip() if el is not None else ""


def search_arxiv(
    query: str,
    *,
    limit: int = 50,
    sort: str = "relevance",
    timeout: float = 45.0,
    limiter: _RateLimiter | None = None,
) -> list[FetchedDocument]:
    """Query the arXiv API, paginating up to ``limit`` results."""
    limiter = limiter or _RateLimiter(ARXIV_MIN_INTERVAL)
    sort_by = {
        "relevance": "relevance",
        "recent": "submittedDate",
        "updated": "lastUpdatedDate",
    }.get(sort, "relevance")

    found: list[FetchedDocument] = []
    seen: set[str] = set()
    start = 0

    while len(found) < limit:
        page = min(ARXIV_PAGE_SIZE, limit - len(found))
        params = urllib.parse.urlencode(
            {
                "search_query": query,
                "start": start,
                "max_results": page,
                "sortBy": sort_by,
                "sortOrder": "descending",
            }
        )
        url = f"{ARXIV_API}?{params}"
        limiter.wait()
        log.info("querying arxiv", extra={"start": start, "want": page})
        try:
            body = _request(url, timeout=timeout)
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(
                f"arXiv query failed: {exc}. Check your network, or that "
                f"export.arxiv.org is reachable from here."
            ) from exc

        try:
            docs, total = parse_arxiv_feed(body)
        except ET.ParseError as exc:
            raise RuntimeError(f"arXiv returned unparseable XML: {exc}") from exc

        if not docs:
            break
        for doc in docs:
            base = doc.doc_id.split("v")[0]
            if base not in seen:
                seen.add(base)
                found.append(doc)
        start += len(docs)
        if total and start >= total:
            break

    log.info("arxiv search complete", extra={"results": len(found)})
    return found[:limit]


def download_documents(
    docs: list[FetchedDocument],
    target_dir: Path,
    *,
    timeout: float = 120.0,
    min_bytes: int = 20_000,
    limiter: _RateLimiter | None = None,
    progress: bool = True,
) -> FetchReport:
    """Download each document's PDF plus a metadata sidecar.

    Already-present files are skipped, so this is resumable and cheap to
    re-run. A suspiciously small response is treated as a failure rather
    than written: arXiv sometimes returns an HTML holding page instead of a
    PDF, and a corpus of error pages is worse than a short corpus.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    limiter = limiter or _RateLimiter(ARXIV_MIN_INTERVAL)
    report = FetchReport(requested=len(docs), found=len(docs))

    iterator = docs
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(docs, desc="Downloading", unit="doc")
        except ImportError:
            pass

    for doc in iterator:
        pdf_path = target_dir / doc.filename
        sidecar = pdf_path.with_suffix(pdf_path.suffix + SIDECAR_SUFFIX)

        if pdf_path.exists() and pdf_path.stat().st_size >= min_bytes:
            if not sidecar.exists():
                sidecar.write_text(
                    json.dumps(doc.to_sidecar(), indent=2), "utf-8"
                )
            report.skipped_existing += 1
            continue

        if not doc.pdf_url:
            report.failed.append((doc.doc_id, "no PDF link in the feed"))
            continue

        limiter.wait()
        try:
            body = _request(doc.pdf_url, timeout=timeout)
        except (urllib.error.URLError, TimeoutError) as exc:
            report.failed.append((doc.doc_id, str(exc)))
            continue

        if len(body) < min_bytes or not body[:5].startswith(b"%PDF"):
            report.failed.append(
                (doc.doc_id, f"not a PDF ({len(body)} bytes); likely a holding page")
            )
            continue

        tmp = pdf_path.with_suffix(pdf_path.suffix + ".part")
        tmp.write_bytes(body)
        tmp.replace(pdf_path)
        sidecar.write_text(json.dumps(doc.to_sidecar(), indent=2), "utf-8")
        report.downloaded += 1
        report.bytes_written += len(body)

    log.info(
        "download complete",
        extra={
            "downloaded": report.downloaded,
            "skipped": report.skipped_existing,
            "failed": len(report.failed),
        },
    )
    return report


def read_sidecar(document_path: Path) -> dict[str, object] | None:
    """Load the metadata written beside a fetched document, if present.

    This is what makes a fetched corpus cite accurately: the title and
    authors come from the source API, so the ingestion heuristics never have
    to guess them off the first page.
    """
    sidecar = document_path.with_suffix(document_path.suffix + SIDECAR_SUFFIX)
    if not sidecar.exists():
        return None
    try:
        data = json.loads(sidecar.read_text("utf-8"))
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError) as exc:
        log.warning(
            "unreadable sidecar", extra={"path": str(sidecar), "err": str(exc)[:100]}
        )
        return None


def fetch_arxiv_corpus(
    query: str,
    target_dir: Path,
    *,
    limit: int = 50,
    sort: str = "relevance",
    progress: bool = True,
) -> FetchReport:
    """Search arXiv and download the results into ``target_dir``."""
    limiter = _RateLimiter(ARXIV_MIN_INTERVAL)
    docs = search_arxiv(query, limit=limit, sort=sort, limiter=limiter)
    if not docs:
        return FetchReport(requested=limit, found=0)
    return download_documents(
        docs, target_dir, limiter=limiter, progress=progress
    )
