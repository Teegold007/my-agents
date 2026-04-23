"""
commands.py — Telegram slash-command handlers.

Single Responsibility: one function per bot command, no business logic beyond
delegating to pipeline/memory/executor.
"""

import os
import sys
import re
from pathlib import Path

from telegram import Update
from telegram.ext import ContextTypes

from config import (
    WORKSPACE, AGENT_DIR, MODELS, DEFAULT_MODEL, ALLOWED_USER_IDS,
    auth, esc, send_html,
)
from conversation import convo_clear, get_queue
from pipeline import execute_job, revert_job
from executor import safe_run
from jobs import State, get_job, get_recent_jobs, update_job, log_event
from memory import stats, clear, get_lessons, get_history


async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not auth(update): return
    model = ctx.user_data.get("model", DEFAULT_MODEL)
    await update.message.reply_html(
        "<b>👋 Coding Agent ready!</b>\n\n"
        f"Model: <code>{esc(model)}</code>\n\n"
        "<b>Commands:</b>\n"
        "• /model — list &amp; switch models\n"
        "• /clone &lt;url&gt; — clone a repo\n"
        "• /repos — list repos\n"
        "• /status &lt;repo&gt; — git log &amp; status\n"
        "• /jobs — recent jobs\n"
        "• /cancel &lt;job_id&gt; — cancel a job\n"
        "• /memory — lessons learned\n"
        "• /forget &lt;repo|global&gt; — clear memory\n"
        "• /new — reset conversation context\n"
        "• /update — pull latest agent code &amp; restart\n\n"
        "<b>Approvals &amp; actions:</b>\n"
        "• <code>approve &lt;job_id&gt;</code> — approve plan or commit\n"
        "• <code>push &lt;job_id&gt;</code> — push branch to origin\n"
        "• <code>revert &lt;job_id&gt;</code> — undo changes\n\n"
        "Just send any task in plain English!"
    )


async def cmd_model(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not auth(update): return
    args = ctx.args

    if not args:
        current = ctx.user_data.get("model", DEFAULT_MODEL)
        lines   = [f"<b>Current:</b> <code>{esc(current)}</code>\n\n<b>Available:</b>"]
        for k, v in MODELS.items():
            mark = "👉 " if k == current else "    "
            lines.append(f"{mark}<code>{k}</code> — {esc(v['label'])}")
        lines.append("    <code>auto</code> — smart routing")
        lines.append("\nSwitch: <code>/model &lt;name&gt;</code>")
        await update.message.reply_html("\n".join(lines))
        return

    key = args[0].lower()
    if key != "auto" and key not in MODELS:
        await update.message.reply_html(f"❌ Unknown model <code>{esc(key)}</code>")
        return
    ctx.user_data["model"] = key
    label = "auto-routing" if key == "auto" else MODELS[key]["label"]
    await update.message.reply_html(f"✅ Switched to <code>{esc(key)}</code> — {esc(label)}")


async def cmd_jobs(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not auth(update): return
    jobs = get_recent_jobs(update.effective_user.id, n=5)

    if not jobs:
        await update.message.reply_html("No jobs yet.")
        return

    STATUS_ICONS = {
        State.NEW: "🆕", State.PLANNED: "📋",
        State.AWAITING_PLAN_APPROVAL: "⏳", State.RUNNING: "⚙️",
        State.AWAITING_DIFF_APPROVAL: "👀", State.COMMITTED: "✅",
        State.FAILED: "❌", State.CANCELLED: "🚫",
    }
    lines = ["<b>Recent jobs:</b>"]
    for j in jobs:
        icon = STATUS_ICONS.get(j.status, "❓")
        ts   = j.created_at[:16].replace("T", " ")
        repo = f" [{esc(j.repo)}]" if j.repo else ""
        lines.append(f"{icon} <code>#{j.id}</code>{repo} <i>{esc(j.task[:60])}</i>")
        lines.append(f"    {esc(j.status)} · {ts}")
    await update.message.reply_html("\n".join(lines))


async def cmd_cancel(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not auth(update): return
    if not ctx.args:
        await update.message.reply_html("Usage: <code>/cancel &lt;job_id&gt;</code>")
        return

    job_id = int(ctx.args[0])
    job    = get_job(job_id)

    if not job or job.user_id != update.effective_user.id:
        await update.message.reply_html("❌ Job not found.")
        return
    if job.status in (State.COMMITTED, State.CANCELLED):
        await update.message.reply_html(f"Job #{job_id} is already {job.status}.")
        return

    await revert_job(update, job)

    q = get_queue(update.effective_user.id)
    while not q.empty():
        try: q.get_nowait()
        except Exception: break

    await update.message.reply_html(f"🛑 Job #{job_id} cancelled.")


async def cmd_clone(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not auth(update): return
    if not ctx.args:
        await update.message.reply_html(
            "Usage: <code>/clone &lt;github_url&gt;</code>\n"
            "Example: <code>/clone https://github.com/you/myapp</code>"
        )
        return

    url = ctx.args[0].strip()

    if re.match(r"^[\w\-]+/[\w\-\.]+$", url):
        url = f"git@github.com:{url}.git"
    elif url.startswith("https://github.com/"):
        path = url.replace("https://github.com/", "").removesuffix(".git")
        url  = f"git@github.com:{path}.git"

    if not re.match(r"^git@github\.com:[\w\-]+/[\w\-\.]+\.git$", url):
        await update.message.reply_html("❌ URL doesn't look like a valid GitHub repo.")
        return

    repo_name = url.split("/")[-1].removesuffix(".git")
    dest      = str(Path(WORKSPACE) / repo_name)

    if Path(dest).exists():
        await update.message.reply_html(
            f"📁 <code>{esc(repo_name)}</code> already exists.\n"
            f"Use <code>/status {esc(repo_name)}</code> to check it."
        )
        return

    msg = await update.message.reply_html(f"⏳ Cloning <code>{esc(repo_name)}</code>…")
    rc, out = safe_run(["git", "clone", url, dest], cwd=WORKSPACE, timeout=120)

    if Path(dest).exists():
        await msg.edit_text(
            f"✅ Cloned <code>{esc(repo_name)}</code> — now tell me what to do with it!",
            parse_mode="HTML",
        )
    else:
        await msg.edit_text(
            f"❌ Clone failed:\n<pre>{esc(out[:500])}</pre>",
            parse_mode="HTML",
        )


async def cmd_repos(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not auth(update): return
    try:
        repos = [r for r in os.listdir(WORKSPACE) if (Path(WORKSPACE) / r).is_dir()]
    except Exception:
        repos = []
    items = "\n".join(f"• <code>{esc(r)}</code>" for r in sorted(repos)) or "No repos found."
    await update.message.reply_html(f"<b>📁 Repos:</b>\n{items}")


async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not auth(update): return
    if not ctx.args:
        await update.message.reply_html("Usage: <code>/status &lt;repo&gt;</code>")
        return
    repo      = ctx.args[0]
    repo_path = str(Path(WORKSPACE) / repo)
    if not Path(repo_path).exists():
        await update.message.reply_html(f"❌ Repo <code>{esc(repo)}</code> not found.")
        return
    _, log = safe_run(["git", "log", "--oneline", "-5"], cwd=repo_path)
    _, st  = safe_run(["git", "status", "--short"],      cwd=repo_path)
    await update.message.reply_html(
        f"<b>{esc(repo)}</b>\n<pre>{esc(log)}\n---\n{esc(st)}</pre>"
    )


async def cmd_memory(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not auth(update): return
    s     = stats()
    lines = [
        "<b>🧠 Agent memory</b>\n",
        f"Total lessons: <b>{s['total_lessons']}</b> ({s['global_lessons']} global)",
        f"Tasks completed: <b>{s['total_tasks']}</b>",
    ]
    if s["repos"]:
        lines.append("Repos: " + ", ".join(
            f"<code>{esc(r['repo'])}</code> ({r['lessons']})" for r in s["repos"]
        ))
    lessons = get_lessons(repo=None, limit=5)
    if lessons:
        lines.append("\n<b>Top global lessons:</b>")
        for lesson in lessons:
            lines.append(f"• {esc(lesson)}")
    history = get_history(n=3)
    if history:
        lines.append("\n<b>Recent tasks:</b>")
        for h in history:
            ts = h["created_at"][:16].replace("T", " ")
            lines.append(f"<code>{ts}</code> {esc(h['task'][:80])}")
    await update.message.reply_html("\n".join(lines))


async def cmd_new(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not auth(update): return
    convo_clear(update.effective_user.id)
    await update.message.reply_html("🆕 Conversation reset. Starting fresh!")


async def cmd_forget(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not auth(update): return
    if not ctx.args:
        await update.message.reply_html(
            "Usage: <code>/forget global</code> or <code>/forget &lt;reponame&gt;</code>"
        )
        return
    target = ctx.args[0].lower()
    if target == "global":
        clear(scope="global")
        await update.message.reply_html("🗑️ Cleared global lessons.")
    elif target == "all":
        clear(scope="all")
        await update.message.reply_html("🗑️ Cleared all memory.")
    else:
        clear(repo=target, scope="repo")
        await update.message.reply_html(f"🗑️ Cleared lessons for <code>{esc(target)}</code>.")


async def cmd_update(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not auth(update): return
    msg = await update.message.reply_html("⏳ Pulling latest code…")
    rc, out = safe_run(["git", "pull"], cwd=AGENT_DIR)
    await msg.edit_text(f"<pre>{esc(out)}</pre>", parse_mode="HTML")

    if "Already up to date" in out:
        await update.message.reply_html("✅ Already on latest version.")
        return

    pip = str(Path(AGENT_DIR) / "venv" / "bin" / "pip")
    safe_run([pip, "install", "-r", str(Path(AGENT_DIR) / "requirements.txt"), "-q"],
             cwd=AGENT_DIR)
    await update.message.reply_html("🔄 Restarting… I'll message you when I'm back.")
    os.execv(sys.executable, [sys.executable] + sys.argv)
