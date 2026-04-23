"""
aider_runner.py — Aider-based code editing backend.

Replaces the raw bash agentic loop with aider-chat for accurate multi-file
editing, proper repo-map context, and fewer hallucinations.

Install: pip install aider-chat
"""

import asyncio
import logging
import os
import shutil

from config import (
    ANTHROPIC_API_KEY, GROQ_API_KEY, OPENROUTER_API_KEY,
    MODELS, FALLBACK_MODELS, esc,
)
from executor import _AUGMENTED_PATH
from jobs import Job, log_event, update_job

logger = logging.getLogger(__name__)

# Map our model keys → litellm model strings that aider accepts.
AIDER_MODEL_MAP: dict[str, str] = {
    "haiku":    "anthropic/claude-haiku-4-5-20251001",
    "sonnet":   "anthropic/claude-sonnet-4-6",
    "claude":   "anthropic/claude-opus-4-5",
    "deepseek": "openrouter/deepseek/deepseek-chat",
    "qwen":     "openrouter/qwen/qwen-2.5-coder-32b-instruct",
    "llama":    "groq/llama-3.3-70b-versatile",
    "r1":       "openrouter/deepseek/deepseek-r1",
}

# Ordered preference for non-Anthropic models when credits run out.
# First untried model in this list is used.
_CREDIT_FALLBACK_ORDER = ("deepseek", "llama", "qwen")

# Lines that are pure decoration — skip to avoid spamming the user.
_NOISE_PREFIXES = ("─", "━", "Aider v", "Model:", "Git repo:", "Repo-map:")


def aider_available() -> bool:
    """Return True if the aider binary is on the (augmented) PATH."""
    return shutil.which("aider", path=_AUGMENTED_PATH) is not None


def _is_noise(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    return any(s.startswith(p) for p in _NOISE_PREFIXES)


def _is_credit_error(output: str) -> bool:
    # The message wraps across lines ("Your credit\nbalance is too low"),
    # so match on the part that always appears on its own line.
    lower = output.lower()
    return "balance is too low" in lower or "credit balance" in lower


def _extract_error(output: str) -> str:
    """Pull the most relevant error line out of aider's output."""
    for line in reversed(output.splitlines()):
        s = line.strip()
        if any(kw in s for kw in ("Error", "error", "Exception", "timed out")):
            # Trim litellm prefix noise for readability
            s = s.replace("litellm.APIError: APIError: ", "")
            s = s.replace("litellm.BadRequestError: ", "")
            return s[:200]
    return "unknown error"


def _is_provider_error(output: str) -> bool:
    """Return True for transient or fatal provider-side failures."""
    lower = output.lower()
    return any(phrase in lower for phrase in (
        "internal server error",
        "error_type': 'server'",
        "bad gateway",
        "service unavailable",
        "no endpoints found",
        "apiconnectionerror",
        "apierror",
    ))


_AIDER_TIMEOUT = 180   # seconds before we kill aider and try the next model
# How long output can be idle before we assume aider is stuck retrying.
_IDLE_TIMEOUT   = 30


async def _run_aider_once(model_str: str, job: Job, status_cb) -> tuple[str, int, str | None]:
    """Spawn aider for a single model attempt. Returns (output, returncode, killed_by_reason).

    Kills the process if:
    - total wall time exceeds _AIDER_TIMEOUT, or
    - no new output for _IDLE_TIMEOUT seconds (stuck in litellm retry loop).
    """
    cwd = job.repo_path or os.getcwd()

    env = {
        **os.environ,
        "PATH":                _AUGMENTED_PATH,
        "ANTHROPIC_API_KEY":   ANTHROPIC_API_KEY,
        "OPENROUTER_API_KEY":  OPENROUTER_API_KEY,
        "GROQ_API_KEY":        GROQ_API_KEY,
        "GIT_TERMINAL_PROMPT": "0",
    }

    cmd = [
        "aider",
        "--message",              job.task,
        "--model",                model_str,
        "--no-auto-commits",
        "--yes-always",
        "--no-pretty",
        "--no-fancy-input",
        "--no-gitignore",         # don't touch .gitignore
        "--chat-history-file",    f"/tmp/aider-chat-{job.id}.md",
        "--input-history-file",   f"/tmp/aider-input-{job.id}.history",
    ]

    log_event(job.id, "aider_start", f"model={model_str}")
    logger.info(f"[job {job.id}] aider: {model_str} in {cwd}")

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=cwd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except FileNotFoundError:
        raise RuntimeError("aider not found — run: pip install aider-chat")

    lines:     list[str] = []
    batch:     list[str] = []
    killed_by: str | None = None

    # Consecutive error lines before we give up and kill the process.
    # Aider reprints the analysis on every retry, so we look for the
    # error line specifically repeating rather than any output stopping.
    _MAX_ERROR_REPEATS = 2
    error_count = 0

    async def flush() -> None:
        if batch:
            await status_cb("⚙️ <code>" + esc("\n".join(batch)) + "</code>")
            batch.clear()

    assert proc.stdout
    try:
        async with asyncio.timeout(_AIDER_TIMEOUT):
            while True:
                try:
                    raw = await asyncio.wait_for(proc.stdout.readline(), timeout=_IDLE_TIMEOUT)
                except asyncio.TimeoutError:
                    killed_by = "idle timeout"
                    proc.kill()
                    break

                if not raw:
                    break

                line = raw.decode(errors="replace").rstrip()
                lines.append(line)

                # Kill as soon as the provider error line appears enough times —
                # aider will just keep retrying otherwise.
                if _is_credit_error(line):
                    error_count += 1
                    if error_count >= _MAX_ERROR_REPEATS:
                        killed_by = "credit exhausted"
                        proc.kill()
                        break
                elif _is_provider_error(line):
                    error_count += 1
                    if error_count >= _MAX_ERROR_REPEATS:
                        killed_by = "provider error"
                        proc.kill()
                        break

                if _is_noise(line):
                    continue
                batch.append(line)
                if len(batch) >= 6:
                    await flush()
    except asyncio.TimeoutError:
        killed_by = "wall-clock timeout"
        proc.kill()
        logger.warning(f"[job {job.id}] aider timeout — killed {model_str}")

    await flush()
    # Drain remaining stdout so the pipe doesn't block
    if proc.stdout and not proc.stdout.at_eof():
        try:
            await asyncio.wait_for(proc.stdout.read(), timeout=2)
        except asyncio.TimeoutError:
            pass
    await proc.wait()

    if killed_by:
        lines.append(f"[aider killed: {killed_by}]")

    rc = proc.returncode if not killed_by else 1
    return "\n".join(lines), rc, killed_by


async def run_aider(job: Job, status_cb) -> tuple[str, bool]:
    """
    Run aider non-interactively with automatic fallback on credit/model errors.

    Returns (full_output, tools_used=True).
    Raises RuntimeError only when all fallbacks are exhausted.
    """
    model_key = job.model
    tried: set[str] = set()

    while True:
        if model_key in tried:
            raise RuntimeError(f"All fallbacks exhausted. Last model attempted: {model_key}")
        tried.add(model_key)

        model_str = AIDER_MODEL_MAP.get(model_key, "anthropic/claude-sonnet-4-6")
        output, rc, killed_by = await _run_aider_once(model_str, job, status_cb)
        log_event(job.id, "aider_done", f"rc={rc} killed_by={killed_by} model={model_key}")

        # Check for errors in output regardless of exit code, as aider's exit
        # code is not always reliable for API failures.
        if _is_credit_error(output):
            killed_by = "credit exhausted"
        elif _is_provider_error(output):
            killed_by = killed_by or "provider error"

        # Success condition: exit code is 0 and no errors were detected.
        if rc == 0 and not killed_by:
            return output, True

        err_msg = _extract_error(output)
        await status_cb(f"❌ <i>{esc(err_msg)}</i>")
        logger.warning(f"[job {job.id}] aider failed ({model_key}): {err_msg}")

        # Credit exhaustion — pick the first untried non-Anthropic model
        if killed_by == "credit exhausted":
            fallback = next(
                (m for m in _CREDIT_FALLBACK_ORDER if m not in tried),
                None,
            )
            if fallback:
                await status_cb(
                    f"💳 <i>Anthropic credits exhausted — switching to "
                    f"{esc(MODELS[fallback]['label'])}.</i>"
                )
                log_event(job.id, "model_fallback", f"{model_key} → {fallback}: credit exhausted")
                update_job(job.id, model=fallback)
                job.model = fallback
                model_key  = fallback
                continue
            raise RuntimeError("Anthropic credits exhausted and all non-Anthropic fallbacks already tried.")

        # Provider error or timeout — follow fallback chain
        fallback = FALLBACK_MODELS.get(model_key) or (
            CREDIT_FALLBACK if model_key != CREDIT_FALLBACK else None
        )
        if fallback and fallback not in tried:
            await status_cb(
                f"🔄 <i>{esc(model_key)} failed ({killed_by or 'error'}) — switching to {esc(MODELS[fallback]['label'])}…</i>"
            )
            log_event(job.id, "model_fallback", f"{model_key} → {fallback}: {err_msg[:80]}")
            update_job(job.id, model=fallback)
            job.model = fallback
            model_key = fallback
            continue

        raise RuntimeError(f"aider exited {rc}: {err_msg}")
