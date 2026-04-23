"""
llm.py — LLM API clients, tool definitions, agentic loops, and model fallback.

Single Responsibility: everything that talks to an LLM lives here.
"""

import json
import logging
import asyncio

import anthropic
from openai import AsyncOpenAI, BadRequestError, RateLimitError, APIConnectionError

from config import (
    ANTHROPIC_API_KEY, GROQ_API_KEY, OPENROUTER_API_KEY,
    MODELS, FALLBACK_MODELS, WORKSPACE,
    esc, job_log_dir, load_repo_config,
)
from conversation import convo_get, bg
from executor import safe_run, parse_argv, ExecutionError
from jobs import Job, update_job, log_event
from memory import build_memory_block

logger = logging.getLogger(__name__)

# ── API clients ───────────────────────────────────────────────────────────────

def get_anthropic_client() -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)


def get_openai_client(provider: str) -> AsyncOpenAI:
    if provider == "groq":
        return AsyncOpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
    if provider == "openrouter":
        return AsyncOpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
    raise ValueError(f"Unknown provider: {provider}")

# ── Tool definitions ──────────────────────────────────────────────────────────

_BASH_DESC = (
    "Run a single shell command on the server. "
    "Only allowlisted binaries are permitted (git, python, node, npm, pytest, ls, cat, find, grep, etc). "
    "No shell operators like &&, |, ;, or redirects — run one command at a time. "
    "CWD defaults to the repo directory."
)

ANTHROPIC_TOOLS = [{
    "name": "bash",
    "description": _BASH_DESC,
    "input_schema": {
        "type": "object",
        "properties": {
            "command":     {"type": "string", "description": "Single command to run."},
            "description": {"type": "string", "description": "One-line summary of what this does."},
        },
        "required": ["command", "description"],
    },
}]

OPENAI_TOOLS = [{
    "type": "function",
    "function": {
        "name": "bash",
        "description": _BASH_DESC,
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

    extras = []
    if cfg.get("test_cmd"):  extras.append(f"Test command: {cfg['test_cmd']}")
    if cfg.get("lint_cmd"):  extras.append(f"Lint command: {cfg['lint_cmd']}")
    if cfg.get("protected_paths"):
        extras.append(f"NEVER modify: {', '.join(cfg['protected_paths'])}")

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

    return base + "\n\n" + memory if memory else base

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
    messages      = (
        [{"role": "system", "content": build_system_prompt(job)}]
        + convo_get(job.user_id)
        + [{"role": "user", "content": job.task}]
    )

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
    """Run the appropriate loop with automatic model fallback on failure."""
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
