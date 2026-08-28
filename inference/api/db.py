"""SQLite storage for batches and per-call results.

Milestone 6. No DB existed before this — SQLite (stdlib-adjacent, zero
extra infra) fits the single-box deployment; Celery/Redis would be
unnecessary complexity at this scale (batch processing is already
sequential and slow by design — see docs/cost_analysis.md).
"""

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "autoace.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS batches (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,        -- pending | processing | done | failed
    manifest_name TEXT NOT NULL,
    total_calls INTEGER NOT NULL,
    completed_calls INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS calls (
    id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL REFERENCES batches(id),
    filename TEXT NOT NULL,
    status TEXT NOT NULL,        -- pending | processing | done | failed
    result_json TEXT,
    error TEXT
);
"""


@contextmanager
def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(SCHEMA)


@dataclass
class CallRow:
    id: str
    batch_id: str
    filename: str
    status: str
    result: dict | None
    error: str | None


@dataclass
class BatchRow:
    id: str
    created_at: str
    status: str
    manifest_name: str
    total_calls: int
    completed_calls: int


def create_batch(batch_id: str, created_at: str, manifest_name: str, filenames: list[str]) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO batches (id, created_at, status, manifest_name, total_calls, completed_calls) "
            "VALUES (?, ?, 'pending', ?, ?, 0)",
            (batch_id, created_at, manifest_name, len(filenames)),
        )
        conn.executemany(
            "INSERT INTO calls (id, batch_id, filename, status) VALUES (?, ?, ?, 'pending')",
            [(f"{batch_id}:{name}", batch_id, name) for name in filenames],
        )


def set_batch_status(batch_id: str, status: str) -> None:
    with _connect() as conn:
        conn.execute("UPDATE batches SET status = ? WHERE id = ?", (status, batch_id))


def set_call_result(call_id: str, status: str, result: dict | None = None, error: str | None = None) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE calls SET status = ?, result_json = ?, error = ? WHERE id = ?",
            (status, json.dumps(result) if result is not None else None, error, call_id),
        )
        if status in ("done", "failed"):
            batch_id = conn.execute("SELECT batch_id FROM calls WHERE id = ?", (call_id,)).fetchone()["batch_id"]
            conn.execute(
                "UPDATE batches SET completed_calls = completed_calls + 1 WHERE id = ?",
                (batch_id,),
            )


def get_batch(batch_id: str) -> BatchRow | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM batches WHERE id = ?", (batch_id,)).fetchone()
        return BatchRow(**dict(row)) if row else None


def list_batches() -> list[BatchRow]:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM batches ORDER BY created_at DESC").fetchall()
        return [BatchRow(**dict(r)) for r in rows]


def delete_batch(batch_id: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM calls WHERE batch_id = ?", (batch_id,))
        conn.execute("DELETE FROM batches WHERE id = ?", (batch_id,))


def list_calls(batch_id: str) -> list[CallRow]:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM calls WHERE batch_id = ? ORDER BY filename", (batch_id,)).fetchall()
        return [
            CallRow(
                id=r["id"], batch_id=r["batch_id"], filename=r["filename"], status=r["status"],
                result=json.loads(r["result_json"]) if r["result_json"] else None,
                error=r["error"],
            )
            for r in rows
        ]
