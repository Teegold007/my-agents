"""
agent.py — Entry point. Wires handlers, manages startup notification, runs polling.

Single Responsibility: application bootstrap only — no business logic here.
"""

import logging

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler, filters,
)

from config import TELEGRAM_TOKEN, ALLOWED_USER_IDS, WORKSPACE, esc
from jobs import State, get_interrupted_jobs, update_job, log_event
from commands import (
    cmd_start, cmd_model, cmd_jobs, cmd_cancel,
    cmd_clone, cmd_repos, cmd_status,
    cmd_memory, cmd_forget, cmd_new, cmd_update,
)
from router import handle_message

load_dotenv()
logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


async def on_startup(app: Application) -> None:
    """Notify all allowed users that the agent is online; surface interrupted jobs."""
    interrupted = get_interrupted_jobs()

    # Jobs that were mid-run when the process died cannot be resumed — mark them failed.
    for jobs in interrupted.values():
        for job in jobs:
            if job.status == State.RUNNING:
                update_job(job.id, status=State.FAILED, result="Interrupted by restart.")
                log_event(job.id, "interrupted", "agent restarted while job was running")

    STATUS_LABELS = {
        State.RUNNING:                "⚙️ was running (interrupted — please retry)",
        State.PLANNED:                "📋 planned",
        State.AWAITING_PLAN_APPROVAL: "⏳ awaiting plan approval — reply <b>approve {id}</b>",
        State.AWAITING_DIFF_APPROVAL: "👀 awaiting diff approval — reply <b>approve {id}</b>",
        State.NEW:                    "🆕 queued",
    }

    for uid in ALLOWED_USER_IDS:
        lines   = ["✅ <b>Agent is online and ready.</b>"]
        pending = interrupted.get(uid, [])

        if pending:
            lines.append("")
            lines.append("<b>Pending jobs from before restart:</b>")
            for job in pending:
                label = STATUS_LABELS.get(job.status, job.status).format(id=job.id)
                lines.append(f"• Job <code>#{job.id}</code> {label}")
                lines.append(f"  <i>{esc(job.task[:80])}</i>")

        try:
            await app.bot.send_message(uid, "\n".join(lines), parse_mode="HTML")
        except Exception:
            pass


def main() -> None:
    app = Application.builder().token(TELEGRAM_TOKEN).post_init(on_startup).build()

    app.add_handler(CommandHandler("start",  cmd_start))
    app.add_handler(CommandHandler("model",  cmd_model))
    app.add_handler(CommandHandler("jobs",   cmd_jobs))
    app.add_handler(CommandHandler("cancel", cmd_cancel))
    app.add_handler(CommandHandler("clone",  cmd_clone))
    app.add_handler(CommandHandler("repos",  cmd_repos))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("memory", cmd_memory))
    app.add_handler(CommandHandler("forget", cmd_forget))
    app.add_handler(CommandHandler("new",    cmd_new))
    app.add_handler(CommandHandler("update", cmd_update))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Agent started.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
