"""Disk cache for extraction results.

Free-tier quotas are small and unforgiving (we measured 20 requests/day on one
model), so re-reading a window we have already read is not merely wasteful — it
is the difference between the demo running and not running.

The cache key is content-addressed over everything that could change the answer:
the window text, the model, and a prompt version. Edit the prompt and the version
bumps, so stale results never masquerade as fresh ones.

This is also the mechanism behind incremental ingest: adding a seventh document
re-reads only that document's windows, because the other six are already keyed
and hit.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path

DEFAULT_PATH = Path("data/extraction_cache.db")

# Bump when the extraction prompt or RawFact schema changes in a way that would
# alter results. Cheaper and less error-prone than remembering to clear the cache.
PROMPT_VERSION = "v1"


class ExtractionCache:
    def __init__(self, path: str | Path = DEFAULT_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS extractions ("
            " key TEXT PRIMARY KEY,"
            " doc_id TEXT, page INTEGER, model TEXT,"
            " payload TEXT NOT NULL,"
            " created_at REAL DEFAULT (strftime('%s','now'))"
            ")"
        )
        self._conn.commit()

    @staticmethod
    def make_key(window_text: str, model: str) -> str:
        digest = hashlib.sha256()
        digest.update(PROMPT_VERSION.encode())
        digest.update(b"\x00")
        digest.update(model.encode())
        digest.update(b"\x00")
        digest.update(window_text.encode("utf-8"))
        return digest.hexdigest()

    def get(self, key: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM extractions WHERE key = ?", (key,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, payload: dict, *, doc_id: str = "", page: int = 0, model: str = ""):
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO extractions (key, doc_id, page, model, payload)"
                " VALUES (?,?,?,?,?)",
                (key, doc_id, page, model, json.dumps(payload, ensure_ascii=False)),
            )
            self._conn.commit()

    def stats(self) -> dict:
        with self._lock:
            total = self._conn.execute("SELECT COUNT(*) FROM extractions").fetchone()[0]
            docs = self._conn.execute(
                "SELECT COUNT(DISTINCT doc_id) FROM extractions"
            ).fetchone()[0]
        return {"windows_cached": total, "documents": docs}

    def close(self):
        with self._lock:
            self._conn.close()


_default: ExtractionCache | None = None


def get_cache(path: str | Path | None = None) -> ExtractionCache:
    global _default
    if path is not None:
        return ExtractionCache(path)
    if _default is None:
        _default = ExtractionCache()
    return _default
