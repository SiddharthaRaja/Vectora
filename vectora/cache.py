"""
A small SQLite-backed key/value cache with TTL.

Used for two things that are expensive and deterministic enough to reuse:
query embeddings, and utility LLM responses (routing, rewriting, HyDE). It
deliberately does not cache answers — those depend on conversation history
and the user expects them to be fresh.

SQLite rather than an in-memory dict because the cache should survive a
restart, and rather than Redis because one process and one file is the right
amount of infrastructure for this.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from vectora.observability import get_logger

log = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache (
    key        TEXT PRIMARY KEY,
    value      BLOB NOT NULL,
    expires_at REAL NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache(expires_at);
"""


def make_key(namespace: str, *parts: Any) -> str:
    """Stable cache key from arbitrary JSON-able parts."""
    blob = json.dumps(parts, sort_keys=True, default=str, ensure_ascii=False)
    return f"{namespace}:{hashlib.sha256(blob.encode()).hexdigest()[:32]}"


class Cache:
    """Thread-safe, process-local. Misses are always silent."""

    def __init__(self, path: Path, ttl_seconds: int, enabled: bool = True) -> None:
        self.path = path
        self.ttl = ttl_seconds
        self.enabled = enabled
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self.hits = 0
        self.misses = 0

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(
                self.path, check_same_thread=False, timeout=5.0
            )
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
        return self._conn

    def get(self, key: str) -> Any | None:
        if not self.enabled:
            return None
        try:
            with self._lock:
                row = (
                    self._connect()
                    .execute(
                        "SELECT value, expires_at FROM cache WHERE key = ?", (key,)
                    )
                    .fetchone()
                )
            if row is None:
                self.misses += 1
                return None
            value, expires_at = row
            if expires_at < time.time():
                self.delete(key)
                self.misses += 1
                return None
            self.hits += 1
            return pickle.loads(value)
        except Exception as exc:  # cache must never break a request
            log.debug("cache get failed", extra={"err": str(exc)})
            return None

    def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        if not self.enabled:
            return
        ttl = self.ttl if ttl_seconds is None else ttl_seconds
        now = time.time()
        try:
            blob = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            with self._lock:
                conn = self._connect()
                conn.execute(
                    "INSERT OR REPLACE INTO cache (key, value, expires_at, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (key, blob, now + ttl, now),
                )
                conn.commit()
        except Exception as exc:
            log.debug("cache set failed", extra={"err": str(exc)})

    def delete(self, key: str) -> None:
        try:
            with self._lock:
                conn = self._connect()
                conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                conn.commit()
        except Exception:
            pass

    def purge_expired(self) -> int:
        try:
            with self._lock:
                conn = self._connect()
                cur = conn.execute(
                    "DELETE FROM cache WHERE expires_at < ?", (time.time(),)
                )
                conn.commit()
                return cur.rowcount or 0
        except Exception:
            return 0

    def clear(self) -> None:
        try:
            with self._lock:
                conn = self._connect()
                conn.execute("DELETE FROM cache")
                conn.commit()
        except Exception:
            pass

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 3),
        }

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
