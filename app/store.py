"""SQLite storage for documents, facts, and relations.

Deliberately relational rather than a graph database. The assignment says
plainly that "a graph database or visualization alone is not the solution", and
nothing here needs graph traversal: relations are pairwise judgements, which is
exactly a table with two foreign keys. The payoff is that the whole system runs
from `git clone` with no service to install.

Facts are keyed by a content hash, so re-ingesting the same document updates
rather than duplicates. That, plus the extraction cache, is what makes
incremental ingest work: adding a seventh document leaves the first six alone.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from app.schema import Evidence, Fact, Quantity, Relation

DEFAULT_DB = Path("data/veritas.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id      TEXT PRIMARY KEY,
    filename    TEXT NOT NULL,
    page_count  INTEGER,
    char_count  INTEGER,
    title       TEXT,
    ingested_at REAL DEFAULT (strftime('%s','now')),
    stats       TEXT
);

CREATE TABLE IF NOT EXISTS facts (
    fact_id     TEXT PRIMARY KEY,
    doc_id      TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    subject     TEXT, metric TEXT,
    subject_key TEXT, metric_key TEXT,
    fact_type   TEXT,
    value_text  TEXT,
    canonical_value REAL, canonical_unit TEXT,
    context     TEXT,
    valid_from  TEXT, valid_to TEXT,
    confidence  REAL,
    notes       TEXT,
    payload     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_facts_doc    ON facts(doc_id);
CREATE INDEX IF NOT EXISTS idx_facts_metric ON facts(metric_key);
CREATE INDEX IF NOT EXISTS idx_facts_unit   ON facts(canonical_unit);

CREATE TABLE IF NOT EXISTS relations (
    left_id   TEXT NOT NULL,
    right_id  TEXT NOT NULL,
    verdict   TEXT NOT NULL,
    differing_keys TEXT,
    unstated_keys  TEXT,
    reasoning TEXT,
    method    TEXT,
    confidence REAL,
    PRIMARY KEY (left_id, right_id)
);

CREATE INDEX IF NOT EXISTS idx_rel_verdict ON relations(verdict);
CREATE INDEX IF NOT EXISTS idx_rel_left    ON relations(left_id);
CREATE INDEX IF NOT EXISTS idx_rel_right   ON relations(right_id);
"""


class Store:
    def __init__(self, path: str | Path = DEFAULT_DB):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    # -- documents ---------------------------------------------------------

    def upsert_document(self, doc_id, filename, page_count, char_count, title=None, stats=None):
        with self._lock:
            self._conn.execute(
                "INSERT INTO documents (doc_id, filename, page_count, char_count, title, stats)"
                " VALUES (?,?,?,?,?,?)"
                " ON CONFLICT(doc_id) DO UPDATE SET filename=excluded.filename,"
                " page_count=excluded.page_count, char_count=excluded.char_count,"
                " title=excluded.title, stats=excluded.stats,"
                " ingested_at=strftime('%s','now')",
                (doc_id, filename, page_count, char_count, title,
                 json.dumps(stats or {})),
            )
            self._conn.commit()

    def documents(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT d.*, (SELECT COUNT(*) FROM facts f WHERE f.doc_id=d.doc_id) AS fact_count"
                " FROM documents d ORDER BY ingested_at DESC"
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["stats"] = json.loads(d.get("stats") or "{}")
            out.append(d)
        return out

    def has_document(self, doc_id: str) -> bool:
        with self._lock:
            return self._conn.execute(
                "SELECT 1 FROM documents WHERE doc_id=?", (doc_id,)
            ).fetchone() is not None

    def delete_document(self, doc_id: str):
        with self._lock:
            ids = [r[0] for r in self._conn.execute(
                "SELECT fact_id FROM facts WHERE doc_id=?", (doc_id,))]
            self._conn.executemany(
                "DELETE FROM relations WHERE left_id=? OR right_id=?",
                [(i, i) for i in ids],
            )
            self._conn.execute("DELETE FROM facts WHERE doc_id=?", (doc_id,))
            self._conn.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))
            self._conn.commit()

    # -- facts -------------------------------------------------------------

    def save_facts(self, facts: list[Fact]):
        rows = []
        for f in facts:
            q = f.quantity
            rows.append((
                f.fact_id, f.doc_id, f.subject, f.metric, f.subject_key, f.metric_key,
                f.fact_type, f.value_text,
                q.canonical_value if q else None,
                q.canonical_unit if q else None,
                json.dumps(f.context, ensure_ascii=False),
                f.valid_from, f.valid_to, f.confidence, f.notes,
                f.model_dump_json(),
            ))
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO facts (fact_id, doc_id, subject, metric,"
                " subject_key, metric_key, fact_type, value_text, canonical_value,"
                " canonical_unit, context, valid_from, valid_to, confidence, notes,"
                " payload) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                rows,
            )
            self._conn.commit()

    def facts(
        self,
        *,
        doc_id: str | None = None,
        search: str | None = None,
        min_confidence: float = 0.0,
        limit: int = 200,
        offset: int = 0,
    ) -> list[Fact]:
        sql = "SELECT payload FROM facts WHERE confidence >= ?"
        args: list = [min_confidence]
        if doc_id:
            sql += " AND doc_id = ?"
            args.append(doc_id)
        if search:
            sql += " AND (subject LIKE ? OR metric LIKE ? OR context LIKE ?)"
            args += [f"%{search}%"] * 3
        sql += " ORDER BY confidence DESC LIMIT ? OFFSET ?"
        args += [limit, offset]
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return [Fact.model_validate_json(r[0]) for r in rows]

    def all_facts(self) -> list[Fact]:
        with self._lock:
            rows = self._conn.execute("SELECT payload FROM facts").fetchall()
        return [Fact.model_validate_json(r[0]) for r in rows]

    def fact(self, fact_id: str) -> Fact | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM facts WHERE fact_id=?", (fact_id,)
            ).fetchone()
        return Fact.model_validate_json(row[0]) if row else None

    def count_facts(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]

    # -- relations ---------------------------------------------------------

    def save_relations(self, relations: list[Relation]):
        rows = [
            (r.left_id, r.right_id, r.verdict,
             json.dumps(r.differing_keys), json.dumps(r.unstated_keys),
             r.reasoning, r.method, r.confidence)
            for r in relations
        ]
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO relations (left_id, right_id, verdict,"
                " differing_keys, unstated_keys, reasoning, method, confidence)"
                " VALUES (?,?,?,?,?,?,?,?)",
                rows,
            )
            self._conn.commit()

    def relations(
        self,
        *,
        verdict: str | None = None,
        fact_id: str | None = None,
        cross_document: bool = False,
        min_confidence: float = 0.0,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Relation]:
        sql = (
            "SELECT r.* FROM relations r"
            " JOIN facts l ON l.fact_id = r.left_id"
            " JOIN facts g ON g.fact_id = r.right_id"
            " WHERE r.confidence >= ?"
        )
        args: list = [min_confidence]
        if verdict:
            sql += " AND r.verdict = ?"
            args.append(verdict.upper())
        if fact_id:
            sql += " AND (r.left_id = ? OR r.right_id = ?)"
            args += [fact_id, fact_id]
        if cross_document:
            sql += " AND l.doc_id != g.doc_id"
        sql += " ORDER BY r.confidence DESC LIMIT ? OFFSET ?"
        args += [limit, offset]
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return [
            Relation(
                left_id=r["left_id"], right_id=r["right_id"], verdict=r["verdict"],
                differing_keys=json.loads(r["differing_keys"] or "[]"),
                unstated_keys=json.loads(r["unstated_keys"] or "[]"),
                reasoning=r["reasoning"], method=r["method"], confidence=r["confidence"],
            )
            for r in rows
        ]

    def verdict_counts(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT verdict, COUNT(*) c FROM relations GROUP BY verdict"
            ).fetchall()
        return {r["verdict"]: r["c"] for r in rows}

    def close(self):
        with self._lock:
            self._conn.close()


_default: Store | None = None


def get_store(path: str | Path | None = None) -> Store:
    global _default
    if path is not None:
        return Store(path)
    if _default is None:
        _default = Store()
    return _default
