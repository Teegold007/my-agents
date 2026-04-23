"""
pipeline.py — Job execution pipeline.

Single Responsibility: orchestrating the lifecycle of a job from creation
through branch management, agent execution, diff review, and commit.
"""

import json
import logging
import re
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
from aider_runner import aider_available, run_aider
from planner import generate_plan, format_plan_html
from executor import safe_run
from jobs import (
    Job, State,
    create_job, update_job, log_event, record_approval, get_job,
)
from memory import reflect_and_save

logger = logging.getLogger(__name__)


def _is_git_task(job: Job) -> bool:
    """
    Return True when the task is primarily a git workflow operation
    (fetch, checkout, cherry-pick, merge, rebase, diff inspection, etc.).
    Aider is a code editor and handles these poorly — the bash-tool agent loop
    should be used instead.
    """
    # Check structured plan steps first (most reliable signal)
    if job.plan:
        try:
            steps = json.loads(job.plan).get("steps", [])
            git_steps = sum(
                1 for s in steps
                if re.match(r"^\s*(run:\s*)?git\s", s, re.IGNORECASE)
            )
            if steps and git_steps / len(steps) >= 0.5:
                return True
        except Exception:
            pass

    # Fall back to keyword scan of the task text
    task_lower = job.task.lower()
    git_keywords = (
        "cherry-pick", "cherry pick", "git fetch", "git checkout",
        "git diff", "git merge", "git rebase", "git log",
        "remote branch", "origin/", "audit-log", "merge branch",
        "pick changes from", "apply changes from",
    )
    return any(kw in task_lower for kw in git_keywords)


def _inject_plan_steps(job: Job) -> Job:
    """
    Return a shallow copy of job with the task replaced by an execution-ready
    message that includes the approved plan steps as explicit bash instructions.
    Without this the agent just narrates what it would do instead of running commands.
    """
    import copy
    if not job.plan:
        return job

    try:
        plan  = json.loads(job.plan)
        steps = plan.get("steps", [])
        if not steps:
            return job

        numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
        enriched = (
            f"Task: {job.task}\n\n"
            f"Execute these steps IN ORDER using the bash tool. "
            f"Run each as a separate command — do NOT describe, actually run them.\n\n"
            f"IMPORTANT CONSTRAINTS:\n"
            f"- Shell operators are FORBIDDEN: no $var, no $(cmd), no pipes |, no semicolons ;, no loops\n"
            f"- If a step says 'for each file', first run the discovery command, read its output, "
            f"then issue ONE separate bash tool call per file\n\n"
            f"{numbered}\n\n"
            f"After all steps complete, summarise what changed."
        )
        exec_job      = copy.copy(job)
        exec_job.task = enriched
        return exec_job
    except Exception:
        return job


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

    # When a plan was approved, inject the steps as explicit instructions so the
    # agent executes them rather than narrating what it would do.
    exec_job = _inject_plan_steps(job)

    # Git/shell tasks require tool calls. If the selected model doesn't support
    # tools it will just narrate and nothing will execute — force a capable model.
    if _is_git_task(exec_job) and not MODELS.get(exec_job.model, {}).get("supports_tools"):
        capable = next(
            (k for k in ("deepseek", "sonnet", "haiku") if MODELS.get(k, {}).get("supports_tools")),
            None,
        )
        if capable:
            await status_cb(
                f"🔀 <i>{esc(MODELS[exec_job.model]['label'])} can't run commands — "
                f"switching to {esc(MODELS[capable]['label'])} for this git task.</i>"
            )
            update_job(job.id, model=capable)
            exec_job.model = capable

    if aider_available() and not _is_git_task(exec_job):
        thinking = await update.message.reply_html("⏳ Working (aider)…")
        _runner = run_aider
    else:
        label   = "⏳ Working (git)…" if _is_git_task(exec_job) else "⏳ Working…"
        thinking = await update.message.reply_html(label)
        _runner = run_agent

    try:
        result, tools_used = await _runner(exec_job, status_cb)
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
            update_job(job.id, status=State.AWAITING_DIFF_APPROVAL)
            log_event(job.id, "awaiting_diff_approval")

            review = await _generate_diff_review(job.task, diff_stat, diff_full)

            stat_preview = diff_stat[:1000] if diff_stat.strip() else untracked[:1000]
            parts = [f"<b>📋 Review — Job #{job.id}</b>"]

            if review.get("summary"):
                parts.append(f"\n<b>What changed:</b>\n{esc(review['summary'])}")
            if review.get("notes"):
                parts.append(f"\n<b>⚠️ Notes:</b>\n{esc(review['notes'])}")
            if review.get("commit_subject"):
                parts.append(f"\n<b>Proposed commit:</b>\n<code>{esc(review['commit_subject'])}</code>")

            parts.append(f"\n<b>Files:</b>\n<pre>{esc(stat_preview)}</pre>")
            parts.append(
                f"\nReply <b>approve {job.id}</b> to commit, "
                f"<b>revert {job.id}</b> to discard, "
                f"or tell me what to change."
            )

            await update.message.reply_html(send_html("\n".join(parts)))
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


async def refine_job(update: Update, job: Job, instruction: str) -> None:
    """Apply further instructions to an already-executed job (on its branch)."""
    repo_path = job.repo_path

    # Temporarily put the job back in RUNNING so get_active_job blocks new tasks
    update_job(job.id, status=State.RUNNING, task=f"{job.task}\n\nRefinement: {instruction}")
    log_event(job.id, "refinement_started", instruction[:200])

    async def status_cb(msg: str) -> None:
        try:
            await update.message.reply_html(send_html(msg))
        except Exception:
            pass

    # Swap the job task for the instruction so the agent only sees what to change
    refined_job        = get_job(job.id)
    original_task      = refined_job.task
    refined_job.task   = instruction

    if aider_available() and not _is_git_task(refined_job):
        thinking = await update.message.reply_html("⏳ Refining (aider)…")
        _runner = run_aider
    else:
        thinking = await update.message.reply_html("⏳ Refining…")
        _runner = run_agent

    try:
        result, _ = await _runner(refined_job, status_cb)
    except Exception as e:
        update_job(job.id, status=State.AWAITING_DIFF_APPROVAL, task=original_task, result=str(e))
        log_event(job.id, "refinement_error", str(e))
        await thinking.delete()
        await update.message.reply_html(f"❌ Refinement failed: <code>{esc(str(e))}</code>")
        return

    await thinking.delete()
    update_job(job.id, result=result[:2000])
    log_event(job.id, "refinement_done", result[:200])
    convo_add(job.user_id, "user",      instruction)
    convo_add(job.user_id, "assistant", result)
    await update.message.reply_html(send_html(esc(result)))

    # Show updated diff
    _, diff_stat = safe_run(["git", "diff", "--stat", "HEAD"], cwd=repo_path)
    _, diff_full = safe_run(["git", "diff", "HEAD"],           cwd=repo_path)
    _, untracked = safe_run(["git", "status", "--short"],      cwd=repo_path)

    if diff_stat.strip() or untracked.strip():
        (job_log_dir(job.id) / "diff.txt").write_text(diff_full)
        update_job(job.id, status=State.AWAITING_DIFF_APPROVAL)
        log_event(job.id, "awaiting_diff_approval")

        review = await _generate_diff_review(job.task, diff_stat, diff_full)
        stat_preview = diff_stat[:1000] if diff_stat.strip() else untracked[:1000]
        parts = [f"<b>📋 Updated Review — Job #{job.id}</b>"]
        if review.get("summary"):
            parts.append(f"\n<b>What changed:</b>\n{esc(review['summary'])}")
        if review.get("notes"):
            parts.append(f"\n<b>⚠️ Notes:</b>\n{esc(review['notes'])}")
        if review.get("commit_subject"):
            parts.append(f"\n<b>Proposed commit:</b>\n<code>{esc(review['commit_subject'])}</code>")
        parts.append(f"\n<b>Files:</b>\n<pre>{esc(stat_preview)}</pre>")
        parts.append(
            f"\nReply <b>approve {job.id}</b> to commit, "
            f"<b>revert {job.id}</b> to discard, "
            f"or keep giving instructions."
        )
        await update.message.reply_html(send_html("\n".join(parts)))
    else:
        update_job(job.id, status=State.AWAITING_DIFF_APPROVAL)
        await update.message.reply_html(
            "⚠️ No file changes detected after refinement. Try a more specific instruction."
        )


_DIFF_REVIEW_PROMPT = """\
You are a senior engineer reviewing a code change before it is committed.
Given the original task and the git diff, produce:
1. A clear explanation of WHAT was changed and WHY (2-4 sentences, plain English, no code blocks)
2. Any risks, assumptions, or follow-up items the reviewer should know
3. A conventional-commit formatted commit message (subject ≤72 chars, blank line, body)

Respond ONLY with valid JSON, no markdown fences:
{
  "summary": "plain-English explanation of what changed and why",
  "notes": "risks, assumptions, or follow-up items (empty string if none)",
  "commit_subject": "type(scope): short description",
  "commit_body": "longer explanation of why these changes were made"
}"""


async def _generate_diff_review(task: str, diff_stat: str, diff_full: str) -> dict:
    """Call an LLM to produce a human summary + proper commit message. Never raises."""
    import re as _re, json as _json
    prompt = (
        f"Task: {task}\n\n"
        f"Files changed:\n{diff_stat[:800]}\n\n"
        f"Diff (truncated):\n{diff_full[:3000]}"
    )
    providers = [
        ("groq",       "llama-3.3-70b-versatile",         "openai"),
        ("anthropic",  "claude-haiku-4-5-20251001",        "anthropic"),
        ("openrouter", "deepseek/deepseek-chat",            "openai"),
    ]
    for provider, model_id, api_type in providers:
        try:
            if api_type == "anthropic":
                from llm import get_anthropic_client
                client = get_anthropic_client()
                resp   = await client.messages.create(
                    model=model_id, max_tokens=512,
                    system=_DIFF_REVIEW_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = resp.content[0].text.strip()
            else:
                from llm import get_openai_client
                client = get_openai_client(provider)
                resp   = await client.chat.completions.create(
                    model=model_id, max_tokens=512,
                    messages=[
                        {"role": "system", "content": _DIFF_REVIEW_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                )
                raw = resp.choices[0].message.content.strip()
            raw = _re.sub(r"^```(?:json)?\s*", "", raw)
            raw = _re.sub(r"\s*```$", "", raw)
            return _json.loads(raw)
        except Exception as e:
            logger.warning(f"Diff review via {provider}/{model_id} failed: {e}")

    return {}   # all providers failed — caller uses fallback


async def commit_job(update: Update, job: Job) -> None:
    """Commit approved changes and notify the user."""
    cwd       = job.repo_path

    # Generate a proper commit message from the stored diff
    diff_path = job_log_dir(job.id) / "diff.txt"
    diff_full = diff_path.read_text() if diff_path.exists() else ""
    _, diff_stat = safe_run(["git", "diff", "--stat", "HEAD"], cwd=cwd)

    review = await _generate_diff_review(job.task, diff_stat, diff_full)

    if review.get("commit_subject"):
        subject = review["commit_subject"][:72]
        body    = review.get("commit_body", "")
        commit_msg = f"{subject}\n\n{body}".strip() if body else subject
    else:
        # Fallback: clean slug from task
        commit_msg = re.sub(r"[^\w\s\-]", "", job.task)[:72]

    safe_run(["git", "add", "-A"], cwd=cwd)
    safe_run(
        ["git", "commit", "-m", commit_msg],
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
        f"Commit: <code>{esc(commit_hash)}</code>\n"
        f"Message: <i>{esc(commit_msg.splitlines()[0])}</i>\n\n"
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
