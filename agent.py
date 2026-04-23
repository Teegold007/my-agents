"""
agent.py — Telegram coding agent v2.

Improvements over v1:
- No shell=True anywhere
- SQLite job store (jobs.py) — survives restarts
- Job-ID based approvals — no stale state confusion
- Structured JSON plans from the LLM
- HTML formatting for Telegram — no MarkdownV2 escaping issues
- Command allowlist via executor.py
- Logs saved to disk per job
- Repo config file support (.agent.json in repo root)
"""

import os
import re
import sys
import json
import logging
import asyncio
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv

import anthropic
from openai import AsyncOpenAI, BadRequestError, RateLimitError, APIConnectionError
from telegram import Update
from telegram.ext import (
    Application, MessageHandler, CommandHandler,
    filters, ContextTypes,
)

from executor import safe_run, parse_argv, ExecutionError, validate_path
from jobs import (
    Job, State,
    create_job, update_job, log_event, record_approval,
    get_job, get_active_job, get_recent_jobs, get_job_events,
    get_interrupted_jobs,
)
from memory import build_memory_block, reflect_and_save, stats, clear, get_lessons, get_history

load_dotenv()
logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

TELEGRAM_TOKEN     = os.environ["TELEGRAM_TOKEN"]
ALLOWED_USER_IDS   = set(int(x) for x in os.environ["ALLOWED_USER_IDS"].split(","))
WORKSPACE          = os.environ.get("WORKSPACE", "/home/agent3/repos")
AGENT_DIR          = os.environ.get("AGENT_DIR", "/home/agent3/agent")
LOGS_DIR           = Path(os.environ.get("LOGS_DIR", "/home/agent3/runs"))
ANTHROPIC_API_KEY  = os.getenv("ANTHROPIC_API_KEY", "")
GROQ_API_KEY       = os.getenv("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
DEFAULT_MODEL      = os.getenv("DEFAULT_MODEL", "auto")

LOGS_DIR.mkdir(parents=True, exist_ok=True)
try:
    Path(WORKSPACE).mkdir(parents=True, exist_ok=True)
except PermissionError:
    logging.warning(f"Cannot create WORKSPACE at {WORKSPACE!r} — check the path and permissions.")

# ── Models ────────────────────────────────────────────────────────────────────

MODELS = {
    "haiku":    {"provider": "anthropic",  "id": "claude-haiku-4-5-20251001",         "label": "Claude Haiku 4.5 (fast)",   "supports_tools": True},
    "sonnet":   {"provider": "anthropic",  "id": "claude-sonnet-4-6",                 "label": "Claude Sonnet 4.6",         "supports_tools": True},
    "claude":   {"provider": "anthropic",  "id": "claude-opus-4-5",                   "label": "Claude Opus 4.5",           "supports_tools": True},
    "llama":    {"provider": "groq",       "id": "llama-3.3-70b-versatile",           "label": "Llama 3.3 70B (free)",      "supports_tools": False},
    "deepseek": {"provider": "openrouter", "id": "deepseek/deepseek-chat",            "label": "DeepSeek V3",               "supports_tools": True},
    "qwen":     {"provider": "openrouter", "id": "qwen/qwen-2.5-coder-32b-instruct", "label": "Qwen 2.5 Coder",            "supports_tools": True},
    "r1":       {"provider": "openrouter", "id": "deepseek/deepseek-r1",             "label": "DeepSeek R1 (reasoning)",   "supports_tools": False},
}

FALLBACK_MODELS = {
    "haiku":    "sonnet",
    "llama":    "haiku",
    "qwen":     "haiku",
    "deepseek": "sonnet",
    "r1":       "sonnet",
}

COMPLEX_KEYWORDS = {
    "refactor", "architecture", "redesign", "security", "optimize",
    "performance", "migrate", "authentication", "database", "schema",
    "review", "audit", "rewrite", "overhaul", "implement", "build",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def detect_complexity(msg: str) -> str:
    if len(msg.split()) > 50: return "complex"
    if any(kw in msg.lower() for kw in COMPLEX_KEYWORDS): return "complex"
    return "simple"


def detect_repo(msg: str) -> tuple[str, str] | tuple[None, None]:
    """Returns (repo_name, repo_path) or (None, None)."""
    try:
        repos = os.listdir(WORKSPACE)
    except Exception:
        return None, None
    for repo in repos:
        if repo.lower() in msg.lower():
            return repo, str(Path(WORKSPACE) / repo)
    return None, None


def make_branch(task: str) -> str:
    slug = re.sub(r"[^\w\s]", "", task.lower())
    slug = re.sub(r"\s+", "-", slug.strip())[:40].rstrip("-")
    ts   = datetime.now(timezone.utc).strftime("%m%d-%H%M")
    return f"agent/{ts}-{slug}"


def job_log_dir(job_id: int) -> Path:
    d = LOGS_DIR / str(job_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_repo_config(repo_path: str) -> dict:
    """Load .agent.json from repo root if present."""
    cfg_path = Path(repo_path) / ".agent.json"
    if cfg_path.exists():
        try:
            return json.loads(cfg_path.read_text())
        except Exception:
            pass
    return {}


def esc(text: str) -> str:
    """Escape text for Telegram HTML."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def send_html(text: str) -> str:
    """Truncate a message to Telegram's 4096 char limit."""
    return text[:4090]

# ── API clients ───────────────────────────────────────────────────────────────

def get_anthropic_client():
    return anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)


def get_openai_client(provider: str) -> AsyncOpenAI:
    if provider == "groq":
        return AsyncOpenAI(api_key=GROQ_API_KEY,       base_url="https://api.groq.com/openai/v1")
    if provider == "openrouter":
        return AsyncOpenAI(api_key=OPENROUTER_API_KEY,  base_url="https://openrouter.ai/api/v1")
    raise ValueError(f"Unknown provider: {provider}")

# ── Conversation history ──────────────────────────────────────────────────────

_conversations: dict[int, list[dict]] = {}
_MAX_HISTORY   = 20   # message turns to keep per user

def convo_add(user_id: int, role: str, content: str) -> None:
    hist = _conversations.setdefault(user_id, [])
    hist.append({"role": role, "content": content})
    if len(hist) > _MAX_HISTORY:
        _conversations[user_id] = hist[-_MAX_HISTORY:]

def convo_get(user_id: int) -> list[dict]:
    return list(_conversations.get(user_id, []))

def convo_clear(user_id: int) -> None:
    _conversations.pop(user_id, None)

# ── Background task registry (prevents GC of fire-and-forget tasks) ──────────

_bg_tasks: set[asyncio.Task] = set()

def bg(coro) -> asyncio.Task:
    t = asyncio.create_task(coro)
    _bg_tasks.add(t)
    t.add_done_callback(_bg_tasks.discard)
    return t

# ── Tool definitions ──────────────────────────────────────────────────────────

BASH_TOOL_DESC = (
    "Run a single shell command on the server. "
    "Only allowlisted binaries are permitted (git, python, node, npm, pytest, ls, cat, find, grep, etc). "
    "No shell operators like &&, |, ;, or redirects — run one command at a time. "
    "CWD defaults to the repo directory."
)

ANTHROPIC_TOOLS = [{
    "name": "bash",
    "description": BASH_TOOL_DESC,
    "input_schema": {
        "type": "object",
        "properties": {
            "command":     {"type": "string", "description": "Single command to run, e.g. 'git status'"},
            "description": {"type": "string", "description": "One-line summary of what this does."},
        },
        "required": ["command", "description"],
    },
}]

OPENAI_TOOLS = [{
    "type": "function",
    "function": {
        "name": "bash",
        "description": BASH_TOOL_DESC,
        "parameters": {
            "type": "object",
            "properties": {
                "command":     {"type": "string"},
                "description": {"type": "string"},
            },
            "required": ["command", "description"],
        },
    },
}]

# ── Safe bash wrapper for the agent loop ─────────────────────────────────────

def agent_run(command_str: str, cwd: str, job_id: int, step: int) -> str:
    """Parse and safely execute one LLM-generated command."""
    log_file = str(job_log_dir(job_id) / f"step_{step:03d}.log")
    try:
        argv = parse_argv(command_str)
        rc, output = safe_run(argv, cwd=cwd, log_path=log_file)
        log_event(job_id, "bash", f"[{rc}] {command_str[:100]}")
        return output
    except ExecutionError as e:
        log_event(job_id, "blocked", str(e))
        return f"[blocked] {e}"

# ── System prompt ─────────────────────────────────────────────────────────────

def build_system_prompt(job: Job) -> str:
    memory = build_memory_block(job.repo)
    cfg    = load_repo_config(job.repo_path) if job.repo_path else {}

    test_cmd  = cfg.get("test_cmd",  "")
    lint_cmd  = cfg.get("lint_cmd",  "")
    protected = cfg.get("protected_paths", [])

    extras = []
    if test_cmd:  extras.append(f"Test command: {test_cmd}")
    if lint_cmd:  extras.append(f"Lint command: {lint_cmd}")
    if protected: extras.append(f"NEVER modify: {', '.join(protected)}")

    repo_note   = f"\nRepo: {job.repo} at {job.repo_path}" if job.repo else ""
    branch_note = f"\nBranch: {job.branch}" if job.branch else ""
    extras_note = ("\n" + "\n".join(extras)) if extras else ""

    base = f"""You are an expert coding agent on a Linux server.{repo_note}{branch_note}{extras_note}

Rules:
- Run ONE command at a time — no && or | chaining
- Only use allowed binaries: git, python, node, npm, pytest, ls, cat, find, grep, cp, mv, mkdir, etc.
- Read files before editing them
- Check memory/lessons before starting — avoid known mistakes
- DO NOT run git commit — the orchestrator handles that after user approval
- After making changes, summarise what you did

End your response with:
📝 Changes made
📁 Files changed
⚠️  Notes"""

    if memory:
        return base + "\n\n" + memory
    return base

# ── Planning ──────────────────────────────────────────────────────────────────

PLAN_SYSTEM = """You are a coding agent producing a structured plan.
Analyse the task and return ONLY valid JSON — no markdown, no explanation.
Format:
{
  "summary": "one line description",
  "steps": ["step 1", "step 2", ...],
  "files_to_read": ["path/to/file"],
  "files_to_change": ["path/to/file"],
  "test_commands": ["npm test"],
  "risks": ["may break X"]
}
Keep steps concrete and under 10 items."""


async def generate_plan(task: str, repo: str, model_id: str, provider: str) -> dict:
    prompt = f"Repo: {repo or 'unknown'}\nTask: {task}"
    try:
        if provider == "anthropic":
            client = get_anthropic_client()
            resp   = client.messages.create(
                model=model_id, max_tokens=1024,
                system=PLAN_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
        else:
            client = get_openai_client(provider)
            resp   = client.chat.completions.create(
                model=model_id, max_tokens=1024,
                messages=[
                    {"role": "system", "content": PLAN_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
            )
            raw = resp.choices[0].message.content.strip()

        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw)

    except Exception as e:
        return {"summary": task, "steps": ["Complete the task"], "error": str(e)}


def format_plan_html(plan: dict, job_id: int) -> str:
    lines = [
        f"<b>📋 Plan — Job #{job_id}</b>",
        f"<i>{esc(plan.get('summary', ''))}</i>",
        "",
    ]
    for i, step in enumerate(plan.get("steps", []), 1):
        lines.append(f"{i}. {esc(step)}")

    if plan.get("files_to_change"):
        lines.append("")
        lines.append("<b>Files to change:</b>")
        for f in plan["files_to_change"]:
            lines.append(f"  • <code>{esc(f)}</code>")

    if plan.get("risks"):
        lines.append("")
        lines.append("<b>⚠️ Risks:</b>")
        for r in plan["risks"]:
            lines.append(f"  • {esc(r)}")

    lines.append("")
    lines.append(f"Reply <b>approve {job_id}</b> to proceed, or tell me what to change.")
    return "\n".join(lines)

# ── Agentic loops ─────────────────────────────────────────────────────────────

async def loop_anthropic(job: Job, status_cb) -> tuple[str, bool]:
    client     = get_anthropic_client()
    history    = convo_get(job.user_id)
    messages   = history + [{"role": "user", "content": job.task}]
    cwd        = job.repo_path or WORKSPACE
    tools_used = False

    for step in range(30):
        resp = await client.messages.create(
            model=MODELS[job.model]["id"],
            max_tokens=8096,
            system=build_system_prompt(job),
            tools=ANTHROPIC_TOOLS,
            messages=messages,
        )
        tool_uses   = [b for b in resp.content if b.type == "tool_use"]
        text_blocks = [b for b in resp.content if b.type == "text"]

        if not tool_uses:
            reply = "\n".join(b.text for b in text_blocks).strip() or "Done."
            return reply, tools_used

        tools_used = True
        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for tu in tool_uses:
            desc   = tu.input.get("description", tu.input["command"][:80])
            await status_cb(f"⚙️ <i>{esc(desc)}</i>")
            output = await asyncio.to_thread(agent_run, tu.input["command"], cwd, job.id, step)
            results.append({"type": "tool_result", "tool_use_id": tu.id, "content": output})
        messages.append({"role": "user", "content": results})

    return "⚠️ Hit 30-step limit.", tools_used


async def loop_openai_compat(job: Job, status_cb) -> tuple[str, bool]:
    cfg           = MODELS[job.model]
    client        = get_openai_client(cfg["provider"])
    cwd           = job.repo_path or WORKSPACE
    tools_enabled = cfg.get("supports_tools", True)
    tools_used    = False
    history       = convo_get(job.user_id)
    messages      = [{"role": "system", "content": build_system_prompt(job)}] \
                  + history \
                  + [{"role": "user", "content": job.task}]

    for step in range(30):
        kwargs: dict = {"model": cfg["id"], "max_tokens": 8096, "messages": messages}
        if tools_enabled:
            kwargs["tools"]       = OPENAI_TOOLS
            kwargs["tool_choice"] = "auto"

        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(**kwargs)
                break
            except RateLimitError:
                if attempt < 2:
                    wait = 5 * (2 ** attempt)
                    logger.warning(f"Rate-limited by {cfg['provider']}, retrying in {wait}s…")
                    await asyncio.sleep(wait)
                    continue
                raise
            except APIConnectionError:
                if attempt < 2:
                    await asyncio.sleep(3)
                    continue
                raise
            except BadRequestError as e:
                if tools_enabled and "tool" in str(e).lower():
                    logger.warning(f"Malformed tool call from {cfg['id']}, disabling tools: {e}")
                    tools_enabled = False
                    kwargs.pop("tools",       None)
                    kwargs.pop("tool_choice", None)
                    continue
                raise

        msg = resp.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            return (msg.content or "Done.").strip(), tools_used

        tools_used = True
        for tc in msg.tool_calls:
            args   = json.loads(tc.function.arguments)
            desc   = args.get("description", args["command"][:80])
            await status_cb(f"⚙️ <i>{esc(desc)}</i>")
            output = await asyncio.to_thread(agent_run, args["command"], cwd, job.id, step)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})

    return "⚠️ Hit 30-step limit.", tools_used


async def run_agent(job: Job, status_cb) -> tuple[str, bool]:
    async def _run(model_key: str) -> tuple[str, bool]:
        provider = MODELS[model_key]["provider"]
        if provider == "anthropic":
            return await loop_anthropic(job, status_cb)
        return await loop_openai_compat(job, status_cb)

    try:
        return await _run(job.model)
    except Exception as e:
        fallback = FALLBACK_MODELS.get(job.model)
        if not fallback:
            raise
        logger.warning(f"Model {job.model} failed ({e}), falling back to {fallback}")
        await status_cb(
            f"⚠️ <i>{esc(MODELS[job.model]['label'])} failed — retrying with "
            f"{esc(MODELS[fallback]['label'])}</i>"
        )
        update_job(job.id, model=fallback)
        log_event(job.id, "model_fallback", f"{job.model} → {fallback}: {str(e)[:120]}")
        job.model = fallback
        return await _run(fallback)

# ── Task pipeline ─────────────────────────────────────────────────────────────

async def start_task(
    update: Update, ctx: ContextTypes.DEFAULT_TYPE,
    task: str, model_key: str,
):
    """Entry point — create job, optionally plan, then execute."""
    user_id   = update.effective_user.id
    repo, repo_path = detect_repo(task)

    # Resolve model
    if model_key == "auto":
        complexity = detect_complexity(task)
        model_key  = "sonnet" if complexity == "complex" else "haiku"
        await update.message.reply_html(
            f"🎯 Auto-selected <b>{model_key}</b> ({complexity} task)"
        )

    if model_key not in MODELS:
        await update.message.reply_html(f"❌ Unknown model <code>{esc(model_key)}</code>")
        return

    # Create job in DB
    job = create_job(user_id, task, model_key, repo=repo, repo_path=repo_path)
    cfg = MODELS[model_key]

    repo_label = f" · <code>{esc(repo)}</code>" if repo else ""
    await update.message.reply_html(
        f"🤖 <b>{esc(cfg['label'])}</b>{repo_label} · Job <code>#{job.id}</code>"
    )

    # Complex tasks get a plan first
    if detect_complexity(task) == "complex":
        update_job(job.id, status=State.PLANNED)
        await update.message.reply_html("📋 Planning…")
        plan = await generate_plan(task, repo or "?", cfg["id"], cfg["provider"])
        plan_json = json.dumps(plan)
        update_job(job.id, plan=plan_json, status=State.AWAITING_PLAN_APPROVAL)
        log_event(job.id, "plan_ready", plan_json[:200])
        await update.message.reply_html(
            send_html(format_plan_html(plan, job.id))
        )
        return  # wait for approve <job_id>

    await execute_job(update, job)


async def execute_job(update: Update, job: Job):
    """Create branch, run agent, show diff."""
    repo_path = job.repo_path
    is_git    = repo_path and (Path(repo_path) / ".git").exists()

    # Create branch
    if is_git:
        branch = make_branch(job.task)
        update_job(job.id, branch=branch)
        rc, out = safe_run(["git", "checkout", "-b", branch], cwd=repo_path)
        log_event(job.id, "branch_created", branch)
        await update.message.reply_html(f"🌿 Branch: <code>{esc(branch)}</code>")
        # Reload job with branch
        job = get_job(job.id)

    update_job(job.id, status=State.RUNNING)
    log_event(job.id, "execution_started")

    async def status_cb(msg: str):
        try: await update.message.reply_html(send_html(msg))
        except Exception: pass

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

    # Update conversation history so follow-ups have context
    convo_add(job.user_id, "user",      job.task)
    convo_add(job.user_id, "assistant", result)

    # Show result summary
    await update.message.reply_html(send_html(esc(result)))

    # Show diff and wait for commit approval
    if is_git:
        _, diff_stat = safe_run(["git", "diff", "--stat", "HEAD"], cwd=repo_path)
        _, diff_full = safe_run(["git", "diff", "HEAD"],           cwd=repo_path)
        _, untracked = safe_run(["git", "status", "--short"],      cwd=repo_path)

        has_changes = diff_stat.strip() or untracked.strip()

        if has_changes:
            diff_file = job_log_dir(job.id) / "diff.txt"
            diff_file.write_text(diff_full)

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

        # Agent ran but produced no file changes
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

    await finish_job(job.id)


async def commit_job(update: Update, job: Job):
    """Commit changes after user approval."""
    commit_msg = re.sub(r"[^\w\s\-]", "", job.task)[:72]
    cwd        = job.repo_path

    rc, out = safe_run(["git", "add", "-A"], cwd=cwd)
    rc, out = safe_run(
        ["git", "commit", "-m", f"agent: {commit_msg}"],
        cwd=cwd,
        log_path=str(job_log_dir(job.id) / "commit.log"),
    )
    rc, hash_out = safe_run(["git", "rev-parse", "--short", "HEAD"], cwd=cwd)
    commit_hash  = hash_out.strip()

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


async def revert_job(update: Update, job: Job):
    """Revert all changes and delete branch."""
    cwd = job.repo_path
    safe_run(["git", "checkout", "--", "."], cwd=cwd)
    safe_run(["git", "clean", "-fd"],        cwd=cwd)
    safe_run(["git", "checkout", "main"],    cwd=cwd)

    if job.branch:
        safe_run(["git", "branch", "-D", job.branch], cwd=cwd)

    update_job(job.id, status=State.CANCELLED)
    log_event(job.id, "reverted")

    await update.message.reply_html(
        f"↩️ Job #{job.id} reverted. Branch deleted."
    )


def finish_job(job_id: int) -> None:
    job = get_job(job_id)
    update_job(job_id, status=State.COMMITTED)
    log_event(job_id, "finished")
    bg(reflect_and_save(job.task, job.result or "", repo=job.repo, model=job.model))

# ── Queue ─────────────────────────────────────────────────────────────────────

_queues: dict[int, asyncio.Queue] = {}

def get_queue(user_id: int) -> asyncio.Queue:
    if user_id not in _queues:
        _queues[user_id] = asyncio.Queue()
    return _queues[user_id]

# ── Auth ──────────────────────────────────────────────────────────────────────

def auth(update: Update) -> bool:
    return update.effective_user.id in ALLOWED_USER_IDS

# ── Message router ────────────────────────────────────────────────────────────

async def handle_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not auth(update): return

    text    = update.message.text.strip()
    user_id = update.effective_user.id

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
            await execute_job(update, get_job(job_id))

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

    # ── Clone shortcut — no LLM needed ───────────────────────────────────────
    if re.search(r"\bclone\b", text, re.IGNORECASE):
        m_url = re.search(
            r"(https?://github\.com/[\w\-]+/[\w\-\.]+(?:\.git)?|[\w\-]+/[\w\-\.]+\.git|[\w\-]+/[\w\-]+)",
            text,
        )
        if m_url:
            ctx.args = [m_url.group(1)]
            await cmd_clone(update, ctx)
            return

    # ── New task ──────────────────────────────────────────────────────────────
    model_key = ctx.user_data.get("model", DEFAULT_MODEL)

    # Queue if something is already active
    active = get_active_job(user_id)
    if active:
        q = get_queue(user_id)
        await q.put((text, model_key))
        await update.message.reply_html(
            f"📥 Queued (position {q.qsize()}) — job #{active.id} is still running."
        )
        return

    try:
        await start_task(update, ctx, text, model_key)
    except Exception as e:
        logger.exception("Task error")
        await update.message.reply_html(f"❌ Error: <code>{esc(str(e))}</code>")

# ── Commands ──────────────────────────────────────────────────────────────────

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not auth(update): return
    model = ctx.user_data.get("model", DEFAULT_MODEL)
    await update.message.reply_html(
        "<b>👋 Coding Agent v2 ready!</b>\n\n"
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


async def cmd_model(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
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


async def cmd_jobs(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not auth(update): return
    jobs  = get_recent_jobs(update.effective_user.id, n=5)

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


async def cmd_cancel(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
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

    # Clear queue
    q = get_queue(update.effective_user.id)
    while not q.empty():
        try: q.get_nowait()
        except Exception: break

    await update.message.reply_html(f"🛑 Job #{job_id} cancelled.")


async def cmd_clone(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not auth(update): return
    if not ctx.args:
        await update.message.reply_html(
            "Usage: <code>/clone &lt;github_url&gt;</code>\n"
            "Example: <code>/clone https://github.com/you/myapp</code>"
        )
        return

    url = ctx.args[0].strip()

    # Convert shorthand / HTTPS to SSH
    if re.match(r"^[\w\-]+/[\w\-\.]+$", url):
        url = f"git@github.com:{url}.git"
    elif url.startswith("https://github.com/"):
        path = url.replace("https://github.com/", "").removesuffix(".git")
        url  = f"git@github.com:{path}.git"

    # Validate URL looks like a GitHub SSH URL
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


async def cmd_repos(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not auth(update): return
    try:
        repos = os.listdir(WORKSPACE)
        repos = [r for r in repos if (Path(WORKSPACE) / r).is_dir()]
    except Exception:
        repos = []
    if repos:
        items = "\n".join(f"• <code>{esc(r)}</code>" for r in sorted(repos))
    else:
        items = "No repos found."
    await update.message.reply_html(f"<b>📁 Repos:</b>\n{items}")


async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not auth(update): return
    if not ctx.args:
        await update.message.reply_html("Usage: <code>/status &lt;repo&gt;</code>")
        return
    repo      = ctx.args[0]
    repo_path = str(Path(WORKSPACE) / repo)

    if not Path(repo_path).exists():
        await update.message.reply_html(f"❌ Repo <code>{esc(repo)}</code> not found.")
        return

    rc, log = safe_run(["git", "log", "--oneline", "-5"],  cwd=repo_path)
    rc, st  = safe_run(["git", "status", "--short"],       cwd=repo_path)
    await update.message.reply_html(
        f"<b>{esc(repo)}</b>\n<pre>{esc(log)}\n---\n{esc(st)}</pre>"
    )


async def cmd_memory(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
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
        for l in lessons: lines.append(f"• {esc(l)}")
    history = get_history(n=3)
    if history:
        lines.append("\n<b>Recent tasks:</b>")
        for h in history:
            ts = h["created_at"][:16].replace("T", " ")
            lines.append(f"<code>{ts}</code> {esc(h['task'][:80])}")
    await update.message.reply_html("\n".join(lines))


async def cmd_new(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not auth(update): return
    convo_clear(update.effective_user.id)
    await update.message.reply_html("🆕 Conversation reset. Starting fresh!")


async def cmd_forget(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
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


async def cmd_update(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not auth(update): return
    msg = await update.message.reply_html("⏳ Pulling latest code…")
    rc, out = safe_run(["git", "pull"], cwd=AGENT_DIR)
    await msg.edit_text(f"<pre>{esc(out)}</pre>", parse_mode="HTML")

    if "Already up to date" in out:
        await update.message.reply_html("✅ Already on latest version.")
        return

    # Install any new dependencies
    pip = str(Path(AGENT_DIR) / "venv" / "bin" / "pip")
    safe_run([pip, "install", "-r", str(Path(AGENT_DIR) / "requirements.txt"), "-q"],
             cwd=AGENT_DIR)

    await update.message.reply_html("🔄 Restarting… I'll message you when I'm back.")

    # Replace the current process with a fresh copy — no sudo/systemctl needed.
    # systemd will restart the service if it's configured with Restart=always.
    os.execv(sys.executable, [sys.executable] + sys.argv)

# ── Startup notification ──────────────────────────────────────────────────────

async def on_startup(app: Application) -> None:
    interrupted = get_interrupted_jobs()

    # Mark any jobs that were mid-run as failed — they won't resume
    for jobs in interrupted.values():
        for job in jobs:
            if job.status == State.RUNNING:
                update_job(job.id, status=State.FAILED, result="Interrupted by restart.")
                log_event(job.id, "interrupted", "agent restarted while job was running")

    STATUS_LABELS = {
        State.RUNNING:                "⚙️ was running (interrupted)",
        State.PLANNED:                "📋 planned",
        State.AWAITING_PLAN_APPROVAL: "⏳ awaiting plan approval — reply <b>approve {id}</b>",
        State.AWAITING_DIFF_APPROVAL: "👀 awaiting diff approval — reply <b>approve {id}</b>",
        State.NEW:                    "🆕 queued",
    }

    for uid in ALLOWED_USER_IDS:
        lines = ["✅ <b>Agent is online and ready.</b>"]
        pending = interrupted.get(uid, [])
        if pending:
            lines.append("")
            lines.append("<b>Pending jobs from before restart:</b>")
            for job in pending:
                label = STATUS_LABELS.get(job.status, job.status)
                label = label.format(id=job.id)
                lines.append(f"• Job <code>#{job.id}</code> {label}")
                lines.append(f"  <i>{esc(job.task[:80])}</i>")
        try:
            await app.bot.send_message(uid, "\n".join(lines), parse_mode="HTML")
        except Exception:
            pass

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
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
    logger.info("Agent v2 started.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
