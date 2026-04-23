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

# Fallback when Anthropic credits run out — must support tools via aider
CREDIT_FALLBACK = "deepseek"

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
    return "credit balance is too low" in output.lower()


async def _run_aider_once(model_str: str, job: Job, status_cb) -> tuple[str, int]:
    """Spawn aider for a single model attempt. Returns (output, returncode)."""
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
        "--message",        job.task,
        "--model",          model_str,
        "--no-auto-commits",
        "--yes-always",
        "--no-pretty",
        "--no-fancy-input",
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

    lines: list[str] = []
    batch: list[str] = []

    async def flush() -> None:
        if batch:
            await status_cb("⚙️ <code>" + esc("\n".join(batch)) + "</code>")
            batch.clear()

    assert proc.stdout
    async for raw in proc.stdout:
        line = raw.decode(errors="replace").rstrip()
        lines.append(line)
        if _is_noise(line):
            continue
        batch.append(line)
        if len(batch) >= 6:
            await flush()

    await flush()
    await proc.wait()

    return "\n".join(lines), proc.returncode


async def run_aider(job: Job, status_cb) -> tuple[str, bool]:
    """
    Run aider non-interactively with automatic fallback on credit/model errors.

    Returns (full_output, tools_used=True).
    Raises RuntimeError only when all fallbacks are exhausted.
    """
    model_key = job.model
    tried: set[str] = set()

    while True:
        model_str = AIDER_MODEL_MAP.get(model_key, "anthropic/claude-sonnet-4-6")

        if model_key in tried:
            raise RuntimeError(f"All fallbacks exhausted. Last model: {model_key}")
        tried.add(model_key)

        output, rc = await _run_aider_once(model_str, job, status_cb)
        log_event(job.id, "aider_done", f"rc={rc} model={model_key}")

        if rc == 0:
            return output, True

        # Detect credit exhaustion → switch to non-Anthropic fallback
        if _is_credit_error(output) and MODELS.get(model_key, {}).get("provider") == "anthropic":
            fallback = CREDIT_FALLBACK
            await status_cb(
                f"💳 <i>Anthropic credits exhausted — switching to "
                f"{esc(MODELS[fallback]['label'])}.</i>"
            )
            log_event(job.id, "model_fallback", f"{model_key} → {fallback}: credit exhausted")
            update_job(job.id, model=fallback)
            job.model = fallback
            model_key  = fallback
            continue

        # Any other non-zero exit → try FALLBACK_MODELS chain
        fallback = FALLBACK_MODELS.get(model_key)
        if fallback and fallback not in tried:
            await status_cb(
                f"⚠️ <i>{esc(MODELS[model_key]['label'])} failed — retrying with "
                f"{esc(MODELS[fallback]['label'])}.</i>"
            )
            log_event(job.id, "model_fallback", f"{model_key} → {fallback}: rc={rc}")
            update_job(job.id, model=fallback)
            job.model = fallback
            model_key  = fallback
            continue

        raise RuntimeError(f"aider exited {rc}:\n{output[-600:]}")
