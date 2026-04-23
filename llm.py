"""
llm.py — LLM API clients, tool definitions, agentic loops, and model fallback.

Single Responsibility: everything that talks to an LLM lives here.
"""

import json
import logging
import asyncio

import anthropic
from openai import AsyncOpenAI, BadRequestError, NotFoundError, RateLimitError, APIConnectionError

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

# ── Skill blocks (distilled from superpowers.dev) ────────────────────────────

_SKILL_DEBUGGING = """
## Skill: Systematic Debugging
ALWAYS find the root cause before proposing any fix. Symptom-fixes are failure.

Phase 1 — Root cause investigation (MANDATORY before touching any code):
  1. Read the full error message and stack trace — it often contains the answer.
  2. Reproduce reliably. If you can't reproduce it, gather more data first.
  3. Check recent changes: git diff, new deps, config edits.
  4. In multi-layer systems, add diagnostic logging at each boundary to find
     WHICH layer is broken before writing a fix.
  5. Trace data flow backward — find where the bad value originates, not just
     where it surfaces.

Phase 2 — Pattern analysis: find a working counterexample, compare line by line.

Phase 3 — Hypothesis: state ONE clear theory, make the smallest possible change
to test it, verify before adding anything else.

Phase 4 — Implementation:
  - Write a failing test that reproduces the bug FIRST.
  - Apply ONE fix targeting the root cause.
  - If 3+ fixes have failed, STOP — the architecture may be wrong. Say so.

Red flags — return to Phase 1 if you notice:
  "quick fix for now" · "just try changing X" · "probably X, let me fix it" ·
  proposing solutions before tracing data flow · each fix revealing a new problem."""

_SKILL_TDD = """
## Skill: Test-Driven Development
Iron law: NO production code without a failing test first.

Red → Green → Refactor:
  RED:    Write one minimal test describing the desired behaviour.
          Run it. Confirm it fails for the RIGHT reason (feature missing, not syntax error).
  GREEN:  Write the simplest code that makes the test pass. Nothing more.
          Run the test suite. All tests must be green.
  REFACTOR: Clean up duplication and names while keeping tests green.
  Repeat for the next behaviour.

Rules:
  - If you wrote code before the test: delete it, start over.
  - If the test passes immediately: you're testing existing behaviour, fix the test.
  - Never fix a bug without first writing a test that reproduces it.
  - "I'll add tests after" = not TDD. The value is in watching the test fail first."""

_SKILL_DEFENSE_IN_DEPTH = """
## Skill: Defense-in-Depth Validation
When you fix a bug caused by bad data, validate at EVERY layer, not just one.

Layer 1 — Entry point: reject obviously invalid input at the API boundary.
Layer 2 — Business logic: ensure data makes sense for this specific operation.
Layer 3 — Environment guards: in tests, refuse dangerous operations outside safe paths
           (e.g. refuse git init outside a tmp directory).
Layer 4 — Debug instrumentation: log context + stack trace before the dangerous operation.

Goal: make the bug structurally impossible, not just "fixed at one place"."""

_SKILL_SENIOR_ENGINEER = """
## Skill: Senior Engineer Mindset

### Before writing a single line of code
- Understand the FULL context: who owns this code, what depends on it, what could break.
- Read the surrounding code, not just the file you're about to edit.
- Check git log on the affected files — understand WHY it was written the way it was.
- Ask: is this the right place to make this change, or is there a deeper design issue?

### Architecture and design
- Prefer changing data structures over adding control flow. Complexity in logic is harder to reason about than complexity in data.
- If a function needs more than ~3 parameters, the abstraction is probably wrong.
- Duplication is far cheaper than the wrong abstraction. Don't unify things that merely look similar.
- When adding a feature, look for the minimal surface area change. New code is new liability.
- If you have to add a comment to explain what the code does, consider renaming instead.

### Code review mindset (apply to your own output)
- Would a new engineer understand this in 6 months without asking questions?
- Is every branch reachable? Is every error handled or deliberately ignored?
- Are there race conditions, missing locks, or TOCTOU issues?
- Does this introduce a new external dependency? Is that justified?
- Could this silently fail in production in a way that's hard to detect?

### When the task feels harder than it should
- Stop. The difficulty is a signal, not a challenge to push through.
- Hard-to-test code is poorly designed code. Redesign before writing tests.
- If you're patching a patch, you're in the wrong layer. Go up one level of abstraction.
- State one clear invariant the system should maintain. If you can't, you don't understand it yet.

### Communication (in your summary)
- State what you changed AND what you deliberately did not change and why.
- Flag any assumptions you made that the user should verify.
- If you took a shortcut, say so explicitly — don't bury it.
- If you see a bigger problem adjacent to the task, name it without gold-plating the fix."""

_SKILL_GIT = """
## Skill: Git Operations

### Inspecting a remote branch
Always fetch before referencing a remote branch:
  git fetch origin <branch>
Then reference it as origin/<branch> (no checkout needed).

### Copying specific files from another branch (selective cherry-pick)
DO NOT use git cherry-pick when the goal is to take only certain files.
Cherry-pick applies whole commits and will bring unwanted changes.
Instead, checkout individual files directly from the remote ref:
  git fetch origin <branch>
  git checkout origin/<branch> -- path/to/file1 path/to/file2
This stages exactly those files from the other branch, nothing else.

### Viewing what changed on another branch vs current HEAD
  git fetch origin <branch>
  git diff HEAD origin/<branch>                     -- all changes
  git diff HEAD origin/<branch> -- path/to/file     -- one file
  git diff HEAD origin/<branch> --name-only         -- just filenames

### Excluding a file from a cherry-pick commit
If you cherry-picked a commit and need to drop one file:
  git cherry-pick <hash>
  git checkout HEAD -- path/to/unwanted-file        -- restore to HEAD version
  (The orchestrator will commit after user approval — do NOT run git commit.)

### Safe sequence for "apply branch changes, exclude file X"
  1. git fetch origin <branch>
  2. git diff HEAD origin/<branch> --name-only       -- confirm which files changed
  3. For each desired file: git checkout origin/<branch> -- <file>
  4. Verify with: git diff --stat HEAD
  (Skip files you were told to exclude — never check them out.)"""

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
- DO NOT run git commit or git push — the orchestrator handles that after user approval
- After making changes, summarise what you did

End your response with:
📝 Changes made
📁 Files changed
⚠️  Notes

{_SKILL_SENIOR_ENGINEER}

{_SKILL_DEBUGGING}

{_SKILL_TDD}

{_SKILL_DEFENSE_IN_DEPTH}

{_SKILL_GIT}"""

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


def _is_credit_error(e: Exception) -> bool:
    lower = str(e).lower()
    return "balance is too low" in lower or "credit balance" in lower


def _is_tool_error(e: Exception) -> bool:
    """Return True for any error caused by the model not supporting tool/function calls."""
    msg = str(e).lower()
    return (
        isinstance(e, NotFoundError) and "tool use" in msg
    ) or (
        isinstance(e, BadRequestError) and "tool" in msg
    )


def _best_tool_capable_fallback(current_key: str) -> str | None:
    """Return the best available model that supports tools, excluding the current one."""
    # Preference order: deepseek → sonnet → haiku
    for candidate in ("deepseek", "sonnet", "haiku"):
        if candidate != current_key and MODELS.get(candidate, {}).get("supports_tools"):
            return candidate
    return None


async def run_agent(job: Job, status_cb) -> tuple[str, bool]:
    """Run the appropriate loop with automatic model fallback on failure.

    Fallback priority:
    1. Tool-unsupported error (404/400 tool) → best tool-capable model
    2. Anthropic credit exhaustion           → DeepSeek V3
    3. Any other failure                     → FALLBACK_MODELS chain
    """
    CREDIT_FALLBACK = "deepseek"  # DeepSeek V3 via OpenRouter — supports tools, cheap

    async def _run(model_key: str) -> tuple[str, bool]:
        provider = MODELS[model_key]["provider"]
        if provider == "anthropic":
            return await loop_anthropic(job, status_cb)
        return await loop_openai_compat(job, status_cb)

    async def _switch(from_key: str, to_key: str, reason: str) -> tuple[str, bool]:
        logger.warning(f"Model {from_key} failed ({reason}), falling back to {to_key}")
        await status_cb(
            f"⚠️ <i>{esc(MODELS[from_key]['label'])} unavailable — retrying with "
            f"{esc(MODELS[to_key]['label'])}</i>"
        )
        update_job(job.id, model=to_key)
        log_event(job.id, "model_fallback", f"{from_key} → {to_key}: {reason[:120]}")
        job.model = to_key
        return await _run(to_key)

    try:
        return await _run(job.model)
    except Exception as e:
        err_str = str(e)

        # Model doesn't support tool use → escalate to best tool-capable alternative
        if _is_tool_error(e):
            fallback = _best_tool_capable_fallback(job.model)
            if fallback:
                await status_cb(
                    f"🔧 <i>{esc(MODELS[job.model]['label'])} doesn't support tool use "
                    f"— switching to {esc(MODELS[fallback]['label'])}.</i>"
                )
                return await _switch(job.model, fallback, "tool use not supported")

        # Anthropic credit exhaustion → DeepSeek
        if _is_credit_error(e) and MODELS[job.model]["provider"] == "anthropic":
            await status_cb("💳 <i>Anthropic credits exhausted — switching to DeepSeek V3.</i>")
            return await _switch(job.model, CREDIT_FALLBACK, "credit balance too low")

        # Any other failure → follow FALLBACK_MODELS chain
        fallback = FALLBACK_MODELS.get(job.model)
        if not fallback:
            raise
        return await _switch(job.model, fallback, err_str)
