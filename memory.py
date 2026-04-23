"""
Memory module — SQLite backend.

Tables:
  lessons  (id, repo, lesson, created_at, hit_count)
  history  (id, repo, task, outcome, model, created_at)
"""

import os
import re
import json
import sqlite3
import logging
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager
from openai import OpenAI

logger = logging.getLogger(__name__)

DB_PATH      = Path(os.environ.get("MEMORY_DB", Path.home() / "agent" / "memory.db"))
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MAX_HISTORY  = int(os.environ.get("MAX_HISTORY", 100))

DB_PATH.parent.mkdir(parents=True, exist_ok=True)


# ── Schema ────────────────────────────────────────────────────────────────────

def _init_db(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS lessons (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            repo       TEXT,                        -- NULL = global
            lesson     TEXT NOT NULL,
            created_at TEXT NOT NULL,
            hit_count  INTEGER DEFAULT 0            -- times this was injected into a prompt
        );

        CREATE TABLE IF NOT EXISTS history (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            repo       TEXT,
            task       TEXT NOT NULL,
            outcome    TEXT NOT NULL,
            model      TEXT,
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_lessons_repo ON lessons(repo);
        CREATE INDEX IF NOT EXISTS idx_history_repo ON history(repo);
    """)
    conn.commit()


@contextmanager
def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    _init_db(conn)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ── Write ─────────────────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower().lstrip("•-* "))


def save_lessons(lessons: list[str], repo: str = None):
    """Insert lessons, skipping duplicates (fuzzy: ignores whitespace/case)."""
    if not lessons:
        return
    now = datetime.now(timezone.utc).isoformat()
    saved = 0
    with _db() as conn:
        # Load existing for this scope to dedup
        existing = {
            _normalise(r["lesson"])
            for r in conn.execute(
                "SELECT lesson FROM lessons WHERE repo IS ?", (repo,)
            )
        }
        for lesson in lessons:
            norm = _normalise(lesson)
            if not norm or norm in existing:
                continue
            conn.execute(
                "INSERT INTO lessons (repo, lesson, created_at) VALUES (?, ?, ?)",
                (repo, lesson.strip(), now),
            )
            existing.add(norm)
            saved += 1
    if saved:
        scope = repo or "global"
        logger.info(f"Memory: saved {saved} lesson(s) [{scope}]")


def log_task(task: str, outcome: str, repo: str = None, model: str = None):
    now = datetime.now(timezone.utc).isoformat()
    with _db() as conn:
        conn.execute(
            "INSERT INTO history (repo, task, outcome, model, created_at) VALUES (?,?,?,?,?)",
            (repo, task[:500], outcome[:1000], model, now),
        )
        # Trim to MAX_HISTORY rows
        conn.execute(
            "DELETE FROM history WHERE id NOT IN "
            "(SELECT id FROM history ORDER BY id DESC LIMIT ?)",
            (MAX_HISTORY,),
        )


# ── Read ──────────────────────────────────────────────────────────────────────

def get_lessons(repo: str = None, limit: int = 30) -> list[str]:
    """
    Return the most relevant lessons for a task.
    Priority: repo-specific first, then global.
    Bumps hit_count so we can surface frequently-useful lessons.
    """
    with _db() as conn:
        rows = conn.execute(
            """
            SELECT id, lesson FROM lessons
            WHERE repo IS ? OR repo IS NULL
            ORDER BY (repo IS NOT NULL) DESC, hit_count DESC, id DESC
            LIMIT ?
            """,
            (repo, limit),
        ).fetchall()

        ids = [r["id"] for r in rows]
        if ids:
            conn.execute(
                f"UPDATE lessons SET hit_count = hit_count + 1 WHERE id IN ({','.join('?'*len(ids))})",
                ids,
            )

    return [r["lesson"] for r in rows]


def get_history(repo: str = None, n: int = 5) -> list[dict]:
    with _db() as conn:
        rows = conn.execute(
            """
            SELECT repo, task, outcome, model, created_at FROM history
            WHERE (? IS NULL OR repo = ?)
            ORDER BY id DESC LIMIT ?
            """,
            (repo, repo, n),
        ).fetchall()
    return [dict(r) for r in reversed(rows)]


def build_memory_block(repo: str = None) -> str:
    """Assemble memory string for injection into the system prompt."""
    sections = []

    lessons = get_lessons(repo)
    if lessons:
        bullet_list = "\n".join(f"- {l}" for l in lessons)
        scope = f"for `{repo}`" if repo else "(global + repo)"
        sections.append(f"### Lessons learned {scope}\n{bullet_list}")

    history = get_history(repo, n=5)
    if history:
        lines = []
        for h in history:
            ts    = h["created_at"][:16].replace("T", " ")
            label = f"[{ts}]"
            if h["repo"]: label += f" ({h['repo']})"
            lines.append(f"{label} {h['task'][:100]}\n  → {h['outcome'][:150]}")
        sections.append("### Recent task history\n" + "\n\n".join(lines))

    if not sections:
        return ""
    return "## Memory\n\n" + "\n\n".join(sections)


# ── Stats (for /memory command) ───────────────────────────────────────────────

def stats() -> dict:
    with _db() as conn:
        total_lessons = conn.execute("SELECT COUNT(*) FROM lessons").fetchone()[0]
        global_lessons = conn.execute("SELECT COUNT(*) FROM lessons WHERE repo IS NULL").fetchone()[0]
        repos = conn.execute(
            "SELECT repo, COUNT(*) as n FROM lessons WHERE repo IS NOT NULL GROUP BY repo ORDER BY n DESC"
        ).fetchall()
        total_tasks = conn.execute("SELECT COUNT(*) FROM history").fetchone()[0]
    return {
        "total_lessons": total_lessons,
        "global_lessons": global_lessons,
        "repos": [{"repo": r["repo"], "lessons": r["n"]} for r in repos],
        "total_tasks": total_tasks,
    }


def clear(repo: str = None, scope: str = "repo"):
    """
    scope='repo'    → delete lessons for that repo
    scope='global'  → delete global lessons
    scope='all'     → delete everything
    """
    with _db() as conn:
        if scope == "all":
            conn.execute("DELETE FROM lessons")
            conn.execute("DELETE FROM history")
        elif scope == "global":
            conn.execute("DELETE FROM lessons WHERE repo IS NULL")
        elif repo:
            conn.execute("DELETE FROM lessons WHERE repo = ?", (repo,))


# ── User preferences (stored as lessons with special repo scope) ───────────────

_USER_SCOPE = "__user__"


def save_user_preferences(preferences: list[str]):
    """Save facts/preferences learned about the user from conversation."""
    save_lessons(preferences, repo=_USER_SCOPE)


def get_user_preferences() -> list[str]:
    """Return stored facts and preferences about the user."""
    with _db() as conn:
        rows = conn.execute(
            "SELECT lesson FROM lessons WHERE repo = ? ORDER BY hit_count DESC, id DESC LIMIT 30",
            (_USER_SCOPE,),
        ).fetchall()
    return [r["lesson"] for r in rows]


# ── Reflection ────────────────────────────────────────────────────────────────

REFLECT_PROMPT = """\
You are reviewing a coding agent's completed task. Extract concise, actionable lessons.

Focus on:
- Mistakes made and how they were fixed (so they're not repeated)
- Repo-specific quirks: test commands, build tools, config, conventions
- Patterns that worked well
- Things to always/never do

Skip obvious or trivial observations.

Respond ONLY with valid JSON (no markdown, no extra text):
{
  "repo_name": "<detected repo name, or null>",
  "global_lessons": ["lesson 1", "lesson 2"],
  "repo_lessons": ["lesson 1", "lesson 2"]
}

If there are no useful lessons, use empty arrays.
"""

CONVO_REFLECT_PROMPT = """\
You are reviewing a conversation between a user and their AI assistant.
Extract facts or preferences about the user worth remembering for future conversations.

Focus on:
- How the user likes to work or communicate
- Their technical background, tools, or stack preferences
- Goals, projects, or context they've shared
- Things they explicitly said they like or dislike

Skip anything obvious, generic, or already implied by the context.
Return an empty array if nothing is worth remembering.

Respond ONLY with valid JSON:
{"preferences": ["fact 1", "fact 2"]}
"""


async def reflect_and_save(task: str, result: str, repo: str = None, model: str = None):
    log_task(task, result[:800], repo=repo, model=model)

    if not GROQ_API_KEY:
        return

    client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=512,
            messages=[
                {"role": "system", "content": REFLECT_PROMPT},
                {"role": "user",   "content": f"TASK:\n{task}\n\nRESULT:\n{result[:3000]}"},
            ],
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        data = json.loads(raw)

        detected_repo = data.get("repo_name") or repo

        save_lessons(data.get("global_lessons", []), repo=None)
        if detected_repo:
            save_lessons(data.get("repo_lessons", []), repo=detected_repo)

    except Exception as e:
        logger.warning(f"Reflection failed: {e}")


async def reflect_on_conversation(user_msg: str, assistant_reply: str):
    """Extract user preferences from a conversational exchange and save them."""
    if not GROQ_API_KEY:
        return

    client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=256,
            messages=[
                {"role": "system", "content": CONVO_REFLECT_PROMPT},
                {"role": "user",   "content": f"USER: {user_msg}\nASSISTANT: {assistant_reply}"},
            ],
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$",          "", raw)
        data = json.loads(raw)
        save_user_preferences(data.get("preferences", []))
    except Exception as e:
        logger.warning(f"Conversation reflection failed: {e}")
