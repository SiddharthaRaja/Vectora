"""
Corpora as named, swappable objects.

Until now the corpus was implicit: one ``chunks.jsonl``, one Qdrant
collection, one BM25 index, one golden set, all named by settings that
happened to point at the vitamin D papers. That makes "try this on a
different corpus" a config edit plus a mental note about which index is
currently loaded, and it makes two corpora impossible to hold at once.

A corpus is now a named profile. Each one owns its derived state:

    corpora/<name>/
        corpus.json        the profile itself
        papers/            source documents
        chunks.jsonl       ingestion output
        bm25.pkl           lexical index
        golden.yaml        evaluation set for this corpus
        reports/           eval and ablation output

and its own Qdrant collection, ``vectora_<name>``. Nothing is shared, so
switching is one flag and rebuilding one corpus cannot disturb another.

The profile also carries what the pipeline cannot infer: what the documents
are, what a reader would ask of them, and the vocabulary the expansion
prompts should lean on. That is the difference between a system that works
on research papers and one that works on *this* set of documents, and it is
the only place domain knowledge is allowed to live.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vectora.observability import get_logger

log = get_logger(__name__)

CORPORA_DIRNAME = "corpora"
PROFILE_FILENAME = "corpus.json"

_SLUG = re.compile(r"[^a-z0-9]+")
_VALID_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,38}[a-z0-9]$")


def slugify(text: str) -> str:
    """A corpus name that is safe as a directory and a collection name."""
    slug = _SLUG.sub("-", text.strip().lower()).strip("-")
    return slug[:40].strip("-") or "corpus"


@dataclass
class CorpusProfile:
    """What a corpus is, and where its parts live."""

    name: str
    description: str = ""
    document_kind: str = "documents"
    """What one item is, in the words a user would use: "research papers",
    "RFCs", "internal runbooks". Used in prompts, so it should read
    naturally in a sentence."""
    domain_terms: list[str] = field(default_factory=list)
    """Vocabulary the query-rewriting prompt should prefer. Keeps expansion
    inside the corpus's register without hardcoding a field in the prompt
    source — which is what tied the old prompts to analytical chemistry."""
    source: str = ""
    """Where the documents came from, free text or a fetch spec."""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    documents: int = 0
    chunks: int = 0
    notes: str = ""

    # Not serialised: filled in by the registry so paths are available.
    root: Path | None = field(default=None, compare=False, repr=False)

    # ── validation ────────────────────────────────────────────────────────

    def __post_init__(self) -> None:
        if not _VALID_NAME.match(self.name):
            raise ValueError(
                f"corpus name {self.name!r} must be 2-40 characters of "
                "lowercase letters, digits, hyphen or underscore, starting "
                "and ending alphanumeric"
            )

    # ── paths ─────────────────────────────────────────────────────────────

    def _require_root(self) -> Path:
        if self.root is None:
            raise RuntimeError(
                f"corpus {self.name!r} has no root; load it through CorpusRegistry"
            )
        return self.root

    @property
    def papers_dir(self) -> Path:
        return self._require_root() / "papers"

    @property
    def chunks_path(self) -> Path:
        return self._require_root() / "chunks.jsonl"

    @property
    def bm25_path(self) -> Path:
        return self._require_root() / "bm25.pkl"

    @property
    def manifest_path(self) -> Path:
        return self._require_root() / "ingestion_manifest.json"

    @property
    def images_dir(self) -> Path:
        return self._require_root() / "extracted_images"

    @property
    def golden_path(self) -> Path:
        return self._require_root() / "golden.yaml"

    @property
    def reports_dir(self) -> Path:
        return self._require_root() / "reports"

    @property
    def profile_path(self) -> Path:
        return self._require_root() / PROFILE_FILENAME

    @property
    def collection(self) -> str:
        """Qdrant collection name. Derived, never configured separately —
        two corpora sharing a collection is a class of bug worth making
        unrepresentable."""
        return f"vectora_{self.name.replace('-', '_')}"

    # ── state ─────────────────────────────────────────────────────────────

    @property
    def has_documents(self) -> bool:
        return self.papers_dir.exists() and any(self.papers_dir.iterdir())

    @property
    def has_chunks(self) -> bool:
        return self.chunks_path.exists() and self.chunks_path.stat().st_size > 0

    @property
    def has_golden(self) -> bool:
        return self.golden_path.exists()

    def ensure_dirs(self) -> None:
        for path in (self.papers_dir, self.reports_dir, self.images_dir):
            path.mkdir(parents=True, exist_ok=True)

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()

    # ── serialisation ─────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        data = {k: v for k, v in asdict(self).items() if k != "root"}
        return data

    def save(self) -> None:
        path = self.profile_path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2), "utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, root: Path) -> CorpusProfile:
        data = json.loads((root / PROFILE_FILENAME).read_text("utf-8"))
        data.pop("root", None)
        known = {f for f in cls.__dataclass_fields__ if f != "root"}
        profile = cls(**{k: v for k, v in data.items() if k in known})
        profile.root = root
        return profile

    def summary(self) -> str:
        bits = [
            f"{self.name}",
            f"{self.documents} docs" if self.documents else "no docs",
            f"{self.chunks} chunks" if self.chunks else "not ingested",
            f"collection={self.collection}",
        ]
        return " · ".join(bits)


class CorpusRegistry:
    """Finds, creates and switches corpora under one directory."""

    def __init__(self, base: Path) -> None:
        self.base = base
        self.base.mkdir(parents=True, exist_ok=True)

    # ── discovery ─────────────────────────────────────────────────────────

    def list(self) -> list[CorpusProfile]:
        out: list[CorpusProfile] = []
        for child in sorted(self.base.iterdir()):
            if not child.is_dir() or not (child / PROFILE_FILENAME).exists():
                continue
            try:
                out.append(CorpusProfile.load(child))
            except Exception as exc:
                log.warning(
                    "unreadable corpus profile",
                    extra={"path": str(child), "err": str(exc)[:120]},
                )
        return out

    def names(self) -> list[str]:
        return [p.name for p in self.list()]

    def exists(self, name: str) -> bool:
        return (self.base / name / PROFILE_FILENAME).exists()

    def get(self, name: str) -> CorpusProfile:
        root = self.base / name
        if not (root / PROFILE_FILENAME).exists():
            available = ", ".join(self.names()) or "none yet"
            raise KeyError(
                f"no corpus named {name!r}. Available: {available}. "
                f"Create one with `vectora corpus create {name}`."
            )
        return CorpusProfile.load(root)

    # ── mutation ──────────────────────────────────────────────────────────

    def create(
        self,
        name: str,
        *,
        description: str = "",
        document_kind: str = "documents",
        domain_terms: list[str] | None = None,
        source: str = "",
        exist_ok: bool = False,
    ) -> CorpusProfile:
        name = slugify(name)
        if self.exists(name):
            if not exist_ok:
                raise FileExistsError(f"corpus {name!r} already exists")
            return self.get(name)

        profile = CorpusProfile(
            name=name,
            description=description,
            document_kind=document_kind,
            domain_terms=list(domain_terms or []),
            source=source,
        )
        profile.root = self.base / name
        profile.ensure_dirs()
        profile.save()
        log.info("corpus created", extra={"corpus": name, "root": str(profile.root)})
        return profile

    def delete(self, name: str, *, keep_documents: bool = True) -> None:
        """Remove a corpus's derived state, and optionally its documents.

        Documents are kept by default: re-fetching a corpus can be slow or
        impossible, and deleting an index is a routine operation while
        deleting source material is not.
        """
        profile = self.get(name)
        root = profile._require_root()
        if keep_documents:
            for target in (
                profile.chunks_path,
                profile.bm25_path,
                profile.manifest_path,
            ):
                target.unlink(missing_ok=True)
            shutil.rmtree(profile.images_dir, ignore_errors=True)
            shutil.rmtree(profile.reports_dir, ignore_errors=True)
            log.info("corpus derived state cleared", extra={"corpus": name})
        else:
            shutil.rmtree(root, ignore_errors=True)
            log.info("corpus deleted", extra={"corpus": name})

    def refresh_counts(self, profile: CorpusProfile) -> CorpusProfile:
        """Recount documents and chunks from disk."""
        if profile.papers_dir.exists():
            profile.documents = sum(
                1
                for p in profile.papers_dir.rglob("*")
                if p.is_file() and not p.name.startswith(".")
            )
        if profile.has_chunks:
            with open(profile.chunks_path, encoding="utf-8") as f:
                profile.chunks = sum(1 for line in f if line.strip())
        else:
            profile.chunks = 0
        profile.touch()
        profile.save()
        return profile


#: The corpus used when none is named. Kept as a constant rather than a
#: literal so the fallback is greppable.
DEFAULT_CORPUS = "default"


def migrate_legacy_layout(
    base: Path,
    *,
    chunks_path: Path,
    papers_dir: Path,
    collection_hint: str = "",
    name: str = DEFAULT_CORPUS,
) -> CorpusProfile | None:
    """Adopt a pre-profile layout as a corpus, without moving the files.

    The repository ships a top-level ``chunks.jsonl``. Rather than require a
    manual move, the first run wraps it in a profile and symlinks or copies
    it in. Returns the profile, or ``None`` if there was nothing to adopt.
    """
    registry = CorpusRegistry(base)
    if registry.exists(name):
        return registry.get(name)
    if not chunks_path.exists():
        return None

    profile = registry.create(
        name,
        description="Adopted from the pre-v3 single-corpus layout.",
        document_kind="research papers",
        source=str(chunks_path),
    )
    target = profile.chunks_path
    if not target.exists():
        try:
            target.symlink_to(chunks_path.resolve())
            log.info("linked legacy chunks into corpus", extra={"corpus": name})
        except (OSError, NotImplementedError):
            # Windows without developer mode cannot symlink; copy instead.
            shutil.copy2(chunks_path, target)
            log.info("copied legacy chunks into corpus", extra={"corpus": name})

    if papers_dir.exists() and papers_dir.is_dir():
        for pdf in papers_dir.glob("*"):
            if pdf.is_file():
                link = profile.papers_dir / pdf.name
                if not link.exists():
                    try:
                        link.symlink_to(pdf.resolve())
                    except (OSError, NotImplementedError):
                        shutil.copy2(pdf, link)

    if collection_hint:
        profile.notes = (
            f"The pre-v3 collection was {collection_hint!r}; this corpus uses "
            f"{profile.collection!r}. Re-index to populate it."
        )
    registry.refresh_counts(profile)
    return profile


@dataclass
class CorpusContext:
    """A resolved corpus: the profile, its concrete paths, its collection.

    Components take this rather than reading paths off ``Settings``, so
    there is exactly one place that decides where a corpus's files live and
    no component can disagree with another about which index is active.
    """

    profile: CorpusProfile
    chunks: Path
    manifest: Path
    papers: Path
    images: Path
    bm25: Path
    golden: Path
    reports: Path
    collection: str

    @classmethod
    def build(cls, settings: Any, name: str | None = None) -> CorpusContext:
        profile = settings.resolve_corpus(name)
        paths = settings.paths_for(profile)
        return cls(
            profile=profile,
            chunks=paths["chunks"],
            manifest=paths["manifest"],
            papers=paths["papers"],
            images=paths["images"],
            bm25=paths["bm25"],
            golden=paths["golden"],
            reports=paths["reports"],
            collection=settings.collection_for(profile),
        )

    @property
    def name(self) -> str:
        return self.profile.name

    def describe(self) -> str:
        return (
            f"corpus={self.name} collection={self.collection} "
            f"chunks={self.chunks} docs={self.profile.documents}"
        )
