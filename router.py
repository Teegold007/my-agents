"""
router.py — Incoming message router.

Single Responsibility: inspect each plain-text message and dispatch to the
right handler — approve/push/revert actions, clone shortcut, plan feedback,
queue management, or new task.
"""

import re
import json
import logging

from telegram import Update
from telegram.ext import ContextTypes

from config import DEFAULT_MODEL, MODELS, esc, send_html
from conversation import get_queue, convo_get, convo_add
from executor import safe_run
from jobs import State, get_job, get_active_job, update_job, log_event, record_approval, get_job_events
from pipeline import execute_job, commit_job, revert_job, start_task, refine_job
from planner import generate_plan, format_plan_html
from commands import cmd_clone

logger = logging.getLogger(__name__)

# ── Conversational dispatcher ─────────────────────────────────────────────────

_DISPATCH_SYSTEM = """You are a routing assistant embedded in a Telegram bot.
Classify the user's message as either a coding task or a conversation.

OUTPUT RULES — strictly one of two formats:
A) If it is a coding task (implement, fix, refactor, add, change, create, build):
   Output the single word:  TASK
   Nothing else. No summary, no explanation, no punctuation.

B) If it is conversational (greeting, thanks, confirmation like "yes"/"ok"/"sure",
   question about code concepts, asking what was done, general chat):
   Output a helpful reply in plain text.
   NEVER include the word TASK anywhere in a conversational reply.

IMPORTANT:
- "yes", "ok", "sure", "go ahead" are ALWAYS conversational confirmations — never TASK.
- A message describing a feature but not asking you to build it yet = conversational.
- Only output TASK when the user is directly requesting you to do the work right now."""


# Regex heuristic for obvious non-task messages — used when all LLMs fail.
_CONVO_RE = re.compile(
    r"^(hi|hello|hey|thanks?|thank you|cheers|great|nice|awesome|good job|"
    r"well done|perfect|ok|okay|cool|got it|sounds good|"
    r"what did you (do|change|fix)|explain|why did|how does|what is|what's|"
    r"who are you|can you help|help me understand)\b",
    re.IGNORECASE,
)


def _build_dispatch_system() -> str:
    from memory import get_user_preferences
    prefs = get_user_preferences()
    if prefs:
        prefs_block = "\n".join(f"- {p}" for p in prefs)
        return _DISPATCH_SYSTEM + f"\n\n## What you know about this user:\n{prefs_block}"
    return _DISPATCH_SYSTEM


async def _try_anthropic(system: str, messages: list) -> str | None:
    try:
        from llm import get_anthropic_client
        client = get_anthropic_client()
        resp = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            system=system,
            messages=messages,
        )
        return resp.content[0].text.strip()
    except Exception as e:
        logger.warning(f"Anthropic dispatcher failed: {e}")
        return None


async def _try_groq(system: str, messages: list) -> str | None:
    try:
        from config import GROQ_API_KEY, groq_available, groq_mark_rate_limited, parse_groq_retry_after
        from openai import AsyncOpenAI, RateLimitError
        if not GROQ_API_KEY or not groq_available():
            return None
        client = AsyncOpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
        groq_messages = [{"role": "system", "content": system}] + messages
        resp = await client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=512,
            messages=groq_messages,
        )
        return resp.choices[0].message.content.strip()
    except RateLimitError as e:
        from config import groq_mark_rate_limited, parse_groq_retry_after
        groq_mark_rate_limited(parse_groq_retry_after(str(e)))
        logger.warning(f"Groq rate-limited in dispatcher")
        return None
    except Exception as e:
        logger.warning(f"Groq dispatcher failed: {e}")
        return None


async def _dispatch(text: str, user_id: int) -> tuple[bool, str]:
    """
    Ask a fast LLM whether the message is a coding task or conversation.
    Returns (is_task, chat_reply).
    Fallback chain: Anthropic haiku → Groq llama → regex heuristic → treat as task.
    """
    history  = convo_get(user_id)
    messages = history + [{"role": "user", "content": text}]
    system   = _build_dispatch_system()

    for attempt in (_try_anthropic, _try_groq):
        reply = await attempt(system, messages)
        if reply is not None:
            if reply.upper() == "TASK":
                return True, ""
            return False, reply

    # Last resort: regex heuristic
    if _CONVO_RE.match(text.strip()):
        logger.info("Dispatcher: regex fallback → conversational")
        return False, "I'm here! What would you like help with?"

    logger.warning("All dispatchers failed — treating as task")
    return True, ""


async def _chat_about_diff(question: str, job, user_id: int) -> str:
    """
    Answer a question about a job's diff using the stored diff as context.
    Falls back to a generic reply if all LLMs fail.
    """
    from config import job_log_dir
    from jobs import get_job_events

    # Build context: diff content + job summary
    diff_path = job_log_dir(job.id) / "diff.txt"
    diff_text = diff_path.read_text()[:4000] if diff_path.exists() else "(diff not available)"

    events    = get_job_events(job.id)
    event_log = "\n".join(
        f"[{e['type']}] {(e['message'] or '')[:120]}" for e in events[-10:]
    )

    system = (
        "You are a senior engineer helping a developer review code changes before committing.\n"
        "Answer their questions clearly and concisely based on the diff and job context below.\n"
        "If they ask about a specific file or line, quote it from the diff.\n"
        "Do NOT start making code changes — only answer questions about what was done.\n\n"
        f"Job #{job.id} — Task: {job.task[:300]}\n\n"
        f"Recent events:\n{event_log}\n\n"
        f"Diff:\n{diff_text}"
    )

    history  = convo_get(user_id)
    messages = history + [{"role": "user", "content": question}]

    # Try Anthropic haiku first, then Groq
    for attempt in (_try_anthropic, _try_groq):
        reply = await attempt(system, messages)
        if reply is not None:
            return reply

    return "I couldn't load an LLM right now. Check the diff at runs/{job.id}/diff.txt directly."


# ── Message router ────────────────────────────────────────────────────────────

async def handle_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    from config import auth
    if not auth(update):
        return

    text    = update.message.text.strip()
    user_id = update.effective_user.id

    # ── status [job_id] — works even while a job is running ──────────────────
    m = re.match(r"^status(?:\s+(\d+))?$", text, re.IGNORECASE)
    if m:
        job_id = int(m.group(1)) if m.group(1) else None
        job    = get_job(job_id) if job_id else get_active_job(user_id)
        if not job or (job_id and job.user_id != user_id):
            await update.message.reply_html("No active job found.")
            return
        events = get_job_events(job.id)
        recent = "\n".join(
            f"• [{e['created_at'][11:19]}] {e['type']}: {(e['message'] or '')[:80]}"
            for e in events[-6:]
        )
        await update.message.reply_html(
            f"<b>Job #{job.id}</b> — <code>{esc(job.status)}</code>\n"
            f"Model: {esc(job.model)}  Branch: <code>{esc(job.branch or '—')}</code>\n\n"
            f"<b>Recent events:</b>\n{esc(recent)}"
        )
        return

    # ── cancel <job_id> — kill a stuck running job ────────────────────────────
    m = re.match(r"^cancel\s+(\d+)$", text, re.IGNORECASE)
    if m:
        job_id = int(m.group(1))
        job    = get_job(job_id)
        if not job or job.user_id != user_id:
            await update.message.reply_html("❌ Job not found.")
            return
        if job.status in (State.COMMITTED, State.CANCELLED, State.FAILED):
            await update.message.reply_html(
                f"Job #{job_id} is already <code>{esc(job.status)}</code>."
            )
            return
        update_job(job_id, status=State.CANCELLED)
        log_event(job_id, "cancelled_by_user")
        await update.message.reply_html(f"🛑 Job #{job_id} cancelled.")
        return

    # ── approve <job_id> ──────────────────────────────────────────────────────
    m = re.match(r"^approve\s+(\d+)$", text, re.IGNORECASE)
    if m:
        job_id = int(m.group(1))
        job    = get_job(job_id)

        if not job or job.user_id != user_id:
            await update.message.reply_html("❌ Job not found.")
            return

        if job.status == State.AWAITING_PLAN_APPROVAL:
            record_approval(job_id, "plan", user_id)
            update_job(job_id, status=State.NEW)
            await update.message.reply_html(f"▶️ Approved plan. Executing job #{job_id}…")
            try:
                await execute_job(update, get_job(job_id))
            except Exception as e:
                logger.exception(f"execute_job failed for job #{job_id}")
                update_job(job_id, status=State.FAILED)
                await update.message.reply_html(
                    f"❌ Job #{job_id} crashed: <code>{esc(str(e))}</code>"
                )

        elif job.status == State.AWAITING_DIFF_APPROVAL:
            await commit_job(update, job)

        else:
            await update.message.reply_html(
                f"Job #{job_id} is in state <code>{esc(job.status)}</code> — nothing to approve."
            )
        return

    # ── push <job_id> ─────────────────────────────────────────────────────────
    m = re.match(r"^push\s+(\d+)$", text, re.IGNORECASE)
    if m:
        job_id = int(m.group(1))
        job    = get_job(job_id)

        if not job or job.user_id != user_id:
            await update.message.reply_html("❌ Job not found.")
            return

        if job.status != State.COMMITTED or not job.branch:
            await update.message.reply_html(
                f"Job #{job_id} is in state <code>{esc(job.status)}</code> — nothing to push."
            )
            return

        rc, out = safe_run(["git", "push", "-u", "origin", job.branch], cwd=job.repo_path)
        if rc == 0:
            log_event(job_id, "pushed", job.branch)
            await update.message.reply_html(
                f"🚀 Pushed <code>{esc(job.branch)}</code> to origin."
            )
        else:
            await update.message.reply_html(f"❌ Push failed:\n<pre>{esc(out[:500])}</pre>")
        return

    # ── revert <job_id> ───────────────────────────────────────────────────────
    m = re.match(r"^revert\s+(\d+)$", text, re.IGNORECASE)
    if m:
        job_id = int(m.group(1))
        job    = get_job(job_id)

        if not job or job.user_id != user_id:
            await update.message.reply_html("❌ Job not found.")
            return

        await revert_job(update, job)
        return

    # ── Clone shortcut — bypass the LLM ──────────────────────────────────────
    if re.search(r"\bclone\b", text, re.IGNORECASE):
        m_url = re.search(
            r"(https?://github\.com/[\w\-]+/[\w\-\.]+(?:\.git)?"
            r"|[\w\-]+/[\w\-\.]+\.git"
            r"|[\w\-]+/[\w\-]+)",
            text,
        )
        if m_url:
            ctx.args = [m_url.group(1)]
            await cmd_clone(update, ctx)
            return

    # ── Diff review chat / refinement ────────────────────────────────────────
    model_key = ctx.user_data.get("model", DEFAULT_MODEL)
    active    = get_active_job(user_id)

    if active and active.status == State.AWAITING_DIFF_APPROVAL:
        is_task, chat_reply = await _dispatch(text, user_id)
        if not is_task:
            # Conversational — answer with full diff context
            reply = await _chat_about_diff(text, active, user_id)
            convo_add(user_id, "user",      text)
            convo_add(user_id, "assistant", reply)
            await update.message.reply_html(send_html(esc(reply)))
        else:
            # Refinement instruction — make more code changes
            await update.message.reply_html(f"🔧 Applying changes to job #{active.id}…")
            try:
                await refine_job(update, active, text)
            except Exception as e:
                logger.exception("Refinement error")
                await update.message.reply_html(f"❌ Refinement failed: <code>{esc(str(e))}</code>")
        return

    # ── Plan feedback — revise instead of queuing a new job ──────────────────
    if active and active.status == State.AWAITING_PLAN_APPROVAL:
        cfg          = MODELS[active.model]
        previous_plan = json.loads(active.plan) if active.plan else None
        await update.message.reply_html("🔄 Revising plan…")
        try:
            plan = await generate_plan(
                active.task, active.repo or "?",
                cfg["id"], cfg["provider"],
                feedback=text,
                previous_plan=previous_plan,
            )
        except Exception as e:
            await update.message.reply_html(f"❌ Could not revise plan: <code>{esc(str(e))}</code>")
            return
        plan_json = json.dumps(plan)
        update_job(active.id, plan=plan_json)
        log_event(active.id, "plan_revised", text[:120])
        await update.message.reply_html(send_html(format_plan_html(plan, active.id)))
        return

    # ── Queue if a job is already running ────────────────────────────────────
    if active:
        q = get_queue(user_id)
        await q.put((text, model_key))
        await update.message.reply_html(
            f"📥 Queued (position {q.qsize()}) — job #{active.id} is still running."
        )
        return

    # ── Dispatch: conversation or coding task? ────────────────────────────────
    is_task, chat_reply = await _dispatch(text, user_id)

    if not is_task:
        # Store exchange in history so future turns have context
        convo_add(user_id, "user",      text)
        convo_add(user_id, "assistant", chat_reply)
        await update.message.reply_html(send_html(esc(chat_reply)))
        # Learn from this exchange in the background
        from memory import reflect_on_conversation
        from conversation import bg
        bg(reflect_on_conversation(text, chat_reply))
        return

    # ── Start a new coding task ───────────────────────────────────────────────
    try:
        await start_task(update, ctx, text, model_key)
    except Exception as e:
        logger.exception("Task error")
        await update.message.reply_html(f"❌ Error: <code>{esc(str(e))}</code>")
