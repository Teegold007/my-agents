"""
conversation.py — Per-user conversation history, job queue, and background task registry.
No LLM or Telegram imports — safe to import from anywhere.
"""

import asyncio

# ── Conversation history ──────────────────────────────────────────────────────

_conversations: dict[int, list[dict]] = {}
_MAX_HISTORY = 20


def convo_add(user_id: int, role: str, content: str) -> None:
    hist = _conversations.setdefault(user_id, [])
    hist.append({"role": role, "content": content})
    if len(hist) > _MAX_HISTORY:
        _conversations[user_id] = hist[-_MAX_HISTORY:]


def convo_get(user_id: int) -> list[dict]:
    return list(_conversations.get(user_id, []))


def convo_clear(user_id: int) -> None:
    _conversations.pop(user_id, None)


# ── Per-user job queue ────────────────────────────────────────────────────────

_queues: dict[int, asyncio.Queue] = {}


def get_queue(user_id: int) -> asyncio.Queue:
    if user_id not in _queues:
        _queues[user_id] = asyncio.Queue()
    return _queues[user_id]


# ── Background task registry ──────────────────────────────────────────────────
# Prevents fire-and-forget asyncio tasks from being garbage-collected mid-run.

_bg_tasks: set[asyncio.Task] = set()


def bg(coro) -> asyncio.Task:
    t = asyncio.create_task(coro)
    _bg_tasks.add(t)
    t.add_done_callback(_bg_tasks.discard)
    return t
