"""
jobs.py — SQLite-backed job store.

Tables:
  jobs        — one row per task, full lifecycle
  job_events  — append-only log of everything that happened
  approvals   — who approved what and when
"""

import json
import sqlite3
import logging
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional
import os

logger = logging.getLogger(__name__)

DB_PATH = Path(os.environ.get("MEMORY_DB", Path.home() / "agent" / "memory.db"))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Job states ────────────────────────────────────────────────────────────────

class State:
    NEW                    = "new"
    PLANNED                = "planned"
    AWAITING_PLAN_APPROVAL = "awaiting_plan_approval"
    RUNNING                = "running"
    AWAITING_DIFF_APPROVAL = "awaiting_diff_approval"
    COMMITTED              = "committed"
    FAILED                 = "failed"
    CANCELLED              = "cancelled"


@dataclass
class Job:
    id:          int
    user_id:     int
    repo:        Optional[str]
    repo_path:   Optional[str]
    branch:      Optional[str]
    task:        str
    status:      str
    model:       str
    plan:        Optional[str]
    result:      Optional[str]
    commit_hash: Optional[str]
    created_at:  str
    updated_at:  str


# ── Schema ────────────────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL,
    repo        TEXT,
    repo_path   TEXT,
    branch      TEXT,
    task        TEXT    NOT NULL,
    status      TEXT    NOT NULL DEFAULT 'new',
    model       TEXT,
    plan        TEXT,
    result      TEXT,
    commit_hash TEXT,
    created_at  TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS job_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id     INTEGER NOT NULL,
    type       TEXT    NOT NULL,
    message    TEXT,
    created_at TEXT    NOT NULL,
    FOREIGN KEY (job_id) REFERENCES jobs(id)
);

CREATE TABLE IF NOT EXISTS approvals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id      INTEGER NOT NULL,
    stage       TEXT    NOT NULL,
    approved_by INTEGER NOT NULL,
    approved_at TEXT    NOT NULL,
    FOREIGN KEY (job_id) REFERENCES jobs(id)
);

CREATE INDEX IF NOT EXISTS idx_jobs_user    ON jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_jobs_status  ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_events_job   ON job_events(job_id);
"""


@contextmanager
def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.commit()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_job(row) -> Job:
    return Job(
        id=row["id"], user_id=row["user_id"],
        repo=row["repo"], repo_path=row["repo_path"],
        branch=row["branch"], task=row["task"],
        status=row["status"], model=row["model"],
        plan=row["plan"], result=row["result"],
        commit_hash=row["commit_hash"],
        created_at=row["created_at"], updated_at=row["updated_at"],
    )


# ── Write ─────────────────────────────────────────────────────────────────────

def create_job(user_id: int, task: str, model: str,
               repo: str = None, repo_path: str = None) -> Job:
    now = _now()
    with _db() as conn:
        cur = conn.execute(
            """INSERT INTO jobs (user_id, repo, repo_path, task, status, model, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (user_id, repo, repo_path, task, State.NEW, model, now, now),
        )
        job_id = cur.lastrowid
    log_event(job_id, "created", f"task: {task[:120]}")
    return get_job(job_id)


def update_job(job_id: int, **kwargs):
    kwargs["updated_at"] = _now()
    cols  = ", ".join(f"{k} = ?" for k in kwargs)
    vals  = list(kwargs.values()) + [job_id]
    with _db() as conn:
        conn.execute(f"UPDATE jobs SET {cols} WHERE id = ?", vals)


def log_event(job_id: int, event_type: str, message: str = None):
    with _db() as conn:
        conn.execute(
            "INSERT INTO job_events (job_id, type, message, created_at) VALUES (?,?,?,?)",
            (job_id, event_type, message, _now()),
        )


def record_approval(job_id: int, stage: str, user_id: int):
    with _db() as conn:
        conn.execute(
            "INSERT INTO approvals (job_id, stage, approved_by, approved_at) VALUES (?,?,?,?)",
            (job_id, stage, user_id, _now()),
        )


# ── Read ──────────────────────────────────────────────────────────────────────

def get_job(job_id: int) -> Optional[Job]:
    with _db() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return _row_to_job(row) if row else None


def get_active_job(user_id: int) -> Optional[Job]:
    """Return the most recent non-terminal job for a user."""
    terminal = (State.COMMITTED, State.FAILED, State.CANCELLED)
    placeholders = ",".join("?" * len(terminal))
    with _db() as conn:
        row = conn.execute(
            f"SELECT * FROM jobs WHERE user_id = ? AND status NOT IN ({placeholders}) "
            f"ORDER BY id DESC LIMIT 1",
            (user_id, *terminal),
        ).fetchone()
    return _row_to_job(row) if row else None


def get_recent_jobs(user_id: int, n: int = 5) -> list[Job]:
    with _db() as conn:
        rows = conn.execute(
            "SELECT * FROM jobs WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, n),
        ).fetchall()
    return [_row_to_job(r) for r in rows]


def get_interrupted_jobs() -> dict[int, list[Job]]:
    """Return all non-terminal jobs grouped by user_id (survives restarts)."""
    terminal = (State.COMMITTED, State.FAILED, State.CANCELLED)
    placeholders = ",".join("?" * len(terminal))
    with _db() as conn:
        rows = conn.execute(
            f"SELECT * FROM jobs WHERE status NOT IN ({placeholders}) ORDER BY id",
            terminal,
        ).fetchall()
    result: dict[int, list[Job]] = {}
    for row in rows:
        job = _row_to_job(row)
        result.setdefault(job.user_id, []).append(job)
    return result


def get_job_events(job_id: int) -> list[dict]:
    with _db() as conn:
        rows = conn.execute(
            "SELECT type, message, created_at FROM job_events WHERE job_id = ? ORDER BY id",
            (job_id,),
        ).fetchall()
    return [dict(r) for r in rows]
