"""
pipeline.py — Job execution pipeline.

Single Responsibility: orchestrating the lifecycle of a job from creation
through branch management, agent execution, diff review, and commit.
"""

import json
import logging
from pathlib import Path

from telegram import Update
from telegram.ext import ContextTypes

from config import (
    MODELS, DEFAULT_MODEL, WORKSPACE,
    detect_complexity, detect_repo, make_branch, job_log_dir,
    esc, send_html,
)
from conversation import convo_add, bg
from llm import run_agent
from planner import generate_plan, format_plan_html
from executor import safe_run
from jobs import (
    Job, State,
    create_job, update_job, log_event, record_approval, get_job,
)
from memory import reflect_and_save

logger = logging.getLogger(__name__)


async def start_task(
    update: Update,
    ctx: ContextTypes.DEFAULT_TYPE,
    task: str,
    model_key: str,
) -> None:
    """Entry point: create job, optionally plan, then execute."""
    user_id         = update.effective_user.id
    repo, repo_path = detect_repo(task)

    if model_key == "auto":
        complexity = detect_complexity(task)
        model_key  = "sonnet" if complexity == "complex" else "haiku"
        await update.message.reply_html(
            f"🎯 Auto-selected <b>{model_key}</b> ({complexity} task)"
        )

    if model_key not in MODELS:
        await update.message.reply_html(f"❌ Unknown model <code>{esc(model_key)}</code>")
        return

    job = create_job(user_id, task, model_key, repo=repo, repo_path=repo_path)
    cfg = MODELS[model_key]

    repo_label = f" · <code>{esc(repo)}</code>" if repo else ""
    await update.message.reply_html(
        f"🤖 <b>{esc(cfg['label'])}</b>{repo_label} · Job <code>#{job.id}</code>"
    )

    if detect_complexity(task) == "complex":
        update_job(job.id, status=State.PLANNED)
        await update.message.reply_html("📋 Planning…")
        plan      = await generate_plan(task, repo or "?", cfg["id"], cfg["provider"])
        plan_json = json.dumps(plan)
        update_job(job.id, plan=plan_json, status=State.AWAITING_PLAN_APPROVAL)
        log_event(job.id, "plan_ready", plan_json[:200])
        await update.message.reply_html(send_html(format_plan_html(plan, job.id)))
        return

    await execute_job(update, job)


async def execute_job(update: Update, job: Job) -> None:
    """Create branch, run agent, show diff, await approval."""
    repo_path = job.repo_path
    is_git    = repo_path and (Path(repo_path) / ".git").exists()

    if is_git:
        branch = make_branch(job.task)
        update_job(job.id, branch=branch)
        safe_run(["git", "checkout", "-b", branch], cwd=repo_path)
        log_event(job.id, "branch_created", branch)
        await update.message.reply_html(f"🌿 Branch: <code>{esc(branch)}</code>")
        job = get_job(job.id)

    update_job(job.id, status=State.RUNNING)
    log_event(job.id, "execution_started")

    async def status_cb(msg: str) -> None:
        try:
            await update.message.reply_html(send_html(msg))
        except Exception:
            pass

    thinking = await update.message.reply_html("⏳ Working…")

    try:
        result, tools_used = await run_agent(job, status_cb)
    except Exception as e:
        update_job(job.id, status=State.FAILED, result=str(e))
        log_event(job.id, "error", str(e))
        await thinking.delete()
        await update.message.reply_html(f"❌ Job #{job.id} failed: <code>{esc(str(e))}</code>")
        return

    await thinking.delete()
    update_job(job.id, result=result[:2000])
    log_event(job.id, "execution_done", result[:200])

    convo_add(job.user_id, "user",      job.task)
    convo_add(job.user_id, "assistant", result)

    await update.message.reply_html(send_html(esc(result)))

    if is_git:
        _, diff_stat = safe_run(["git", "diff", "--stat", "HEAD"], cwd=repo_path)
        _, diff_full = safe_run(["git", "diff", "HEAD"],           cwd=repo_path)
        _, untracked = safe_run(["git", "status", "--short"],      cwd=repo_path)

        if diff_stat.strip() or untracked.strip():
            (job_log_dir(job.id) / "diff.txt").write_text(diff_full)
            preview = diff_stat[:1500] if diff_stat.strip() else untracked[:1500]
            update_job(job.id, status=State.AWAITING_DIFF_APPROVAL)
            log_event(job.id, "awaiting_diff_approval")
            await update.message.reply_html(
                f"<b>📋 Diff — Job #{job.id}</b>\n"
                f"<pre>{esc(preview)}</pre>\n\n"
                f"Full diff saved to <code>runs/{job.id}/diff.txt</code>\n\n"
                f"Reply <b>approve {job.id}</b> to commit, or <b>revert {job.id}</b> to undo."
            )
            return

        if tools_used:
            await update.message.reply_html(
                "⚠️ The agent ran commands but made no file changes. "
                "Try being more specific about what to change, or check if the repo path is correct."
            )
        else:
            await update.message.reply_html(
                "⚠️ The agent gave a response without running any commands — "
                "this is likely a hallucination. Please rephrase your task."
            )

    finish_job(job.id)


async def commit_job(update: Update, job: Job) -> None:
    """Commit approved changes and notify the user."""
    import re
    commit_msg = re.sub(r"[^\w\s\-]", "", job.task)[:72]
    cwd        = job.repo_path

    safe_run(["git", "add", "-A"], cwd=cwd)
    safe_run(
        ["git", "commit", "-m", f"agent: {commit_msg}"],
        cwd=cwd,
        log_path=str(job_log_dir(job.id) / "commit.log"),
    )
    _, hash_out = safe_run(["git", "rev-parse", "--short", "HEAD"], cwd=cwd)
    commit_hash = hash_out.strip()

    update_job(job.id, status=State.COMMITTED, commit_hash=commit_hash)
    log_event(job.id, "committed", commit_hash)
    record_approval(job.id, "commit", update.effective_user.id)

    bg(reflect_and_save(job.task, job.result or "", repo=job.repo, model=job.model))

    await update.message.reply_html(
        f"✅ <b>Job #{job.id} committed</b>\n"
        f"Branch: <code>{esc(job.branch or 'unknown')}</code>\n"
        f"Commit: <code>{esc(commit_hash)}</code>\n\n"
        f"Reply <b>push {job.id}</b> to push the branch, or merge manually."
    )


async def revert_job(update: Update, job: Job) -> None:
    """Revert all changes and delete the branch."""
    cwd = job.repo_path
    safe_run(["git", "checkout", "--", "."], cwd=cwd)
    safe_run(["git", "clean", "-fd"],        cwd=cwd)
    safe_run(["git", "checkout", "main"],    cwd=cwd)
    if job.branch:
        safe_run(["git", "branch", "-D", job.branch], cwd=cwd)

    update_job(job.id, status=State.CANCELLED)
    log_event(job.id, "reverted")
    await update.message.reply_html(f"↩️ Job #{job.id} reverted. Branch deleted.")


def finish_job(job_id: int) -> None:
    """Mark a no-diff job as complete and trigger memory reflection."""
    job = get_job(job_id)
    update_job(job_id, status=State.COMMITTED)
    log_event(job_id, "finished")
    bg(reflect_and_save(job.task, job.result or "", repo=job.repo, model=job.model))
