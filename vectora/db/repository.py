"""
Conversation persistence.

SQLite, with the settings that make it safe under a threaded server: WAL so
readers do not block the writer, a busy timeout so a concurrent write waits
rather than raising ``database is locked``, and foreign keys actually
enforced (they are off by default in SQLite, which is why the old schema's
``REFERENCES`` clause did nothing).

Connections are per-thread. ``sqlite3`` connections are not safe to share
across threads, and the previous code kept one on Flask's ``g`` — correct
per request, but it also meant every request paid connection setup and the
pragmas were never applied.

Schema changes go through ``_MIGRATIONS``: the table is created at version 1
and each later entry is applied in order, so an existing ``chats.db`` keeps
its history instead of being replaced.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vectora.models import Citation, Message, Session, utcnow
from vectora.observability import get_logger

log = get_logger(__name__)

_MIGRATIONS: tuple[str, ...] = (
    # v1 — base schema
    """
    CREATE TABLE IF NOT EXISTS sessions (
        id         TEXT PRIMARY KEY,
        title      TEXT NOT NULL DEFAULT 'New chat',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS messages (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        role       TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
        content    TEXT NOT NULL,
        citations  TEXT,
        created_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_messages_session
        ON messages(session_id, id);
    CREATE INDEX IF NOT EXISTS idx_sessions_updated
        ON sessions(updated_at DESC);
    """,
    # v2 — per-message metadata (mode, query type, grounding, usage)
    """
    ALTER TABLE messages ADD COLUMN metadata TEXT;
    """,
)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _parse_dt(value: Any) -> datetime:
    """Parse a stored timestamp, tolerating the old naive format.

    Rows written by the previous implementation used
    ``datetime.utcnow().isoformat()`` — no timezone — so they are read as
    UTC rather than being rejected.
    """
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return utcnow()


class ChatRepository:
    """Sessions and messages. One instance per process."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._local = threading.local()
        self._write_lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate()

    # ── connection handling ───────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=10.0, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=10000")
            self._local.conn = conn
        return conn

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        """Serialised write transaction.

        SQLite allows one writer; taking the lock in-process turns a
        would-be ``database is locked`` into a short wait.
        """
        conn = self._connect()
        with self._write_lock:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def _migrate(self) -> None:
        conn = self._connect()
        current = conn.execute("PRAGMA user_version").fetchone()[0]
        if current >= len(_MIGRATIONS):
            return
        for version in range(current, len(_MIGRATIONS)):
            log.info("applying migration", extra={"version": version + 1})
            try:
                conn.executescript(_MIGRATIONS[version])
            except sqlite3.OperationalError as exc:
                # A column added by hand, or a partially applied migration.
                if "duplicate column" not in str(exc).lower():
                    raise
                log.info("migration already present", extra={"version": version + 1})
            conn.execute(f"PRAGMA user_version = {version + 1}")
        log.info("database ready", extra={"path": str(self.path), "version": len(_MIGRATIONS)})

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    # ── sessions ──────────────────────────────────────────────────────────

    def create_session(self, title: str = "New chat") -> Session:
        session = Session(id=str(uuid.uuid4()), title=title)
        with self._write() as conn:
            conn.execute(
                "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?,?,?,?)",
                (session.id, session.title, _iso(session.created_at), _iso(session.updated_at)),
            )
        return session

    def get_session(self, session_id: str) -> Session | None:
        row = (
            self._connect()
            .execute(
                "SELECT s.id, s.title, s.created_at, s.updated_at, "
                "(SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS n "
                "FROM sessions s WHERE s.id = ?",
                (session_id,),
            )
            .fetchone()
        )
        return _row_to_session(row) if row else None

    def list_sessions(self, limit: int = 200, offset: int = 0) -> list[Session]:
        rows = (
            self._connect()
            .execute(
                "SELECT s.id, s.title, s.created_at, s.updated_at, "
                "(SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS n "
                "FROM sessions s ORDER BY s.updated_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
            .fetchall()
        )
        return [_row_to_session(r) for r in rows]

    def rename_session(self, session_id: str, title: str) -> bool:
        with self._write() as conn:
            cur = conn.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                (title[:200], _iso(utcnow()), session_id),
            )
        return (cur.rowcount or 0) > 0

    def delete_session(self, session_id: str) -> bool:
        with self._write() as conn:
            # Explicit, rather than relying on the cascade, so the behaviour
            # is identical on a database created before foreign keys were on.
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            cur = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        return (cur.rowcount or 0) > 0

    def touch_session(self, session_id: str) -> None:
        with self._write() as conn:
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (_iso(utcnow()), session_id),
            )

    # ── messages ──────────────────────────────────────────────────────────

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        citations: Sequence[Citation] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Message:
        created = utcnow()
        payload = (
            json.dumps([c.model_dump() for c in citations], ensure_ascii=False)
            if citations
            else None
        )
        meta = json.dumps(metadata, ensure_ascii=False, default=str) if metadata else None
        with self._write() as conn:
            cur = conn.execute(
                "INSERT INTO messages (session_id, role, content, citations, metadata, created_at) "
                "VALUES (?,?,?,?,?,?)",
                (session_id, role, content, payload, meta, _iso(created)),
            )
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (_iso(created), session_id),
            )
        return Message(
            id=cur.lastrowid,
            session_id=session_id,
            role=role,  # type: ignore[arg-type]
            content=content,
            citations=list(citations or []),
            metadata=metadata or {},
            created_at=created,
        )

    def add_exchange(
        self,
        session_id: str,
        question: str,
        answer: str,
        *,
        citations: Sequence[Citation] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store both turns in one transaction.

        Previously these were two statements that could be interrupted
        between them, leaving a question with no answer in the history —
        which then got sent to the model on the next turn.
        """
        now = _iso(utcnow())
        payload = (
            json.dumps([c.model_dump() for c in citations], ensure_ascii=False)
            if citations
            else None
        )
        meta = json.dumps(metadata, ensure_ascii=False, default=str) if metadata else None
        with self._write() as conn:
            conn.execute(
                "INSERT INTO messages (session_id, role, content, citations, metadata, created_at) "
                "VALUES (?,?,?,?,?,?)",
                (session_id, "user", question, None, None, now),
            )
            conn.execute(
                "INSERT INTO messages (session_id, role, content, citations, metadata, created_at) "
                "VALUES (?,?,?,?,?,?)",
                (session_id, "assistant", answer, payload, meta, now),
            )
            row = conn.execute(
                "SELECT title FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            title = row["title"] if row else "New chat"
            if title == "New chat":
                title = question[:60] + ("…" if len(question) > 60 else "")
                conn.execute(
                    "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                    (title, now, session_id),
                )
            else:
                conn.execute(
                    "UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id)
                )

    def get_messages(self, session_id: str, limit: int | None = None) -> list[Message]:
        sql = (
            "SELECT id, session_id, role, content, citations, metadata, created_at "
            "FROM messages WHERE session_id = ? ORDER BY id ASC"
        )
        params: tuple[Any, ...] = (session_id,)
        if limit:
            sql += " LIMIT ?"
            params = (session_id, limit)
        rows = self._connect().execute(sql, params).fetchall()
        return [_row_to_message(r) for r in rows]

    def get_history(self, session_id: str, turns: int) -> list[dict[str, str]]:
        """The last ``turns`` messages as plain role/content dicts."""
        rows = (
            self._connect()
            .execute(
                "SELECT role, content FROM ("
                "  SELECT id, role, content FROM messages WHERE session_id = ? "
                "  ORDER BY id DESC LIMIT ?"
                ") ORDER BY id ASC",
                (session_id, turns),
            )
            .fetchall()
        )
        return [{"role": r["role"], "content": r["content"]} for r in rows]

    # ── housekeeping ──────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        conn = self._connect()
        sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        messages = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        return {
            "path": str(self.path),
            "sessions": sessions,
            "messages": messages,
            "size_bytes": self.path.stat().st_size if self.path.exists() else 0,
        }

    def prune_empty_sessions(self) -> int:
        """Delete sessions with no messages.

        The UI creates a session as soon as 'New chat' is clicked, so an
        abandoned click leaves a row forever.
        """
        with self._write() as conn:
            cur = conn.execute(
                "DELETE FROM sessions WHERE id NOT IN (SELECT DISTINCT session_id FROM messages)"
            )
        removed = cur.rowcount or 0
        if removed:
            log.info("pruned empty sessions", extra={"count": removed})
        return removed


def _row_to_session(row: sqlite3.Row) -> Session:
    return Session(
        id=row["id"],
        title=row["title"],
        created_at=_parse_dt(row["created_at"]),
        updated_at=_parse_dt(row["updated_at"]),
        message_count=row["n"] if "n" in row.keys() else 0,
    )


def _row_to_message(row: sqlite3.Row) -> Message:
    citations: list[Citation] = []
    raw = row["citations"]
    if raw:
        try:
            for item in json.loads(raw):
                try:
                    citations.append(Citation.model_validate(item))
                except Exception:
                    # Rows written by the old format lack `index`/`chunk_id`.
                    citations.append(
                        Citation(
                            index=0,
                            chunk_id="",
                            paper_id=str(item.get("paper_id", "")),
                            title=str(item.get("title", "")),
                            authors=str(item.get("authors", "")),
                            year=str(item.get("year", "")),
                            doi=str(item.get("doi", "")),
                            section=str(item.get("section", "")),
                            page=int(item.get("page") or 0),
                        )
                    )
        except json.JSONDecodeError:
            log.warning("unreadable citations column", extra={"message_id": row["id"]})

    metadata: dict[str, Any] = {}
    meta_raw = row["metadata"] if "metadata" in row.keys() else None
    if meta_raw:
        try:
            metadata = json.loads(meta_raw)
        except json.JSONDecodeError:
            pass

    return Message(
        id=row["id"],
        session_id=row["session_id"],
        role=row["role"],
        content=row["content"],
        citations=citations,
        metadata=metadata,
        created_at=_parse_dt(row["created_at"]),
    )
