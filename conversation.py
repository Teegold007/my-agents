"""
conversation.py — Per-user conversation history, job queue, and background task registry.
No LLM or Telegram imports — safe to import from anywhere.
"""

import asyncio
import logging

logger = logging.getLogger(__name__)

# ── Conversation history ──────────────────────────────────────────────────────

_conversations: dict[int, list[dict]] = {}
_MAX_HISTORY   = 40    # keep up to 40 turns in memory
_COMPACT_AFTER = 30    # trigger compaction when we exceed this


def convo_add(user_id: int, role: str, content: str) -> None:
    hist = _conversations.setdefault(user_id, [])
    hist.append({"role": role, "content": content})
    if len(hist) > _MAX_HISTORY:
        # Compact: summarise the oldest half, keep the recent half verbatim
        bg(_compact_history(user_id))


def convo_get(user_id: int) -> list[dict]:
    return list(_conversations.get(user_id, []))


async def _compact_history(user_id: int) -> None:
    """
    Summarise the oldest messages into a single assistant turn so the context
    window doesn't grow unboundedly. Inspired by code_puppy's _compaction.py.
    """
    hist = _conversations.get(user_id, [])
    if len(hist) <= _COMPACT_AFTER:
        return

    # Split: summarise everything except the most recent _COMPACT_AFTER turns
    cutoff    = len(hist) - _COMPACT_AFTER
    to_squash = hist[:cutoff]
    keep      = hist[cutoff:]

    text = "\n".join(
        f"{m['role'].upper()}: {str(m['content'])[:300]}"
        for m in to_squash
    )

    summary = await _call_summary_llm(text)
    if summary:
        compacted = [{"role": "assistant", "content": f"[Earlier conversation summary]\n{summary}"}]
        _conversations[user_id] = compacted + keep
        logger.info(f"Compacted {cutoff} messages for user {user_id}")


async def _call_summary_llm(text: str) -> str:
    """Call Groq (free, fast) to summarise old conversation turns."""
    try:
        from config import GROQ_API_KEY
        from openai import AsyncOpenAI
        if not GROQ_API_KEY:
            return ""
        client = AsyncOpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
        resp   = await client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=300,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Summarise this conversation history in 3-5 bullet points. "
                        "Focus on tasks completed, key decisions, and files changed. "
                        "Be concise."
                    ),
                },
                {"role": "user", "content": text[:4000]},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"History compaction failed: {e}")
        return ""


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
