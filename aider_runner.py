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
    MODELS, esc,
)
from executor import _AUGMENTED_PATH
from jobs import Job, log_event

logger = logging.getLogger(__name__)

# Map our model keys → litellm model strings that aider accepts.
# Anthropic models work bare; OpenRouter and Groq need the provider prefix.
AIDER_MODEL_MAP: dict[str, str] = {
    "haiku":    "anthropic/claude-haiku-4-5-20251001",
    "sonnet":   "anthropic/claude-sonnet-4-6",
    "claude":   "anthropic/claude-opus-4-5",
    "deepseek": "openrouter/deepseek/deepseek-chat",
    "qwen":     "openrouter/qwen/qwen-2.5-coder-32b-instruct",
    "llama":    "groq/llama-3.3-70b-versatile",
    "r1":       "openrouter/deepseek/deepseek-r1",
}

# Lines that are pure decoration — skip them to avoid spamming the user.
_NOISE_PREFIXES = ("─", "━", "Aider v", "Model:", "Git repo:", "Repo-map:")


def aider_available() -> bool:
    """Return True if the aider binary is on the (augmented) PATH."""
    return shutil.which("aider", path=_AUGMENTED_PATH) is not None


def _is_noise(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    return any(s.startswith(p) for p in _NOISE_PREFIXES)


async def run_aider(job: Job, status_cb) -> tuple[str, bool]:
    """
    Run aider non-interactively for job.task in job.repo_path.

    Streams meaningful output back via status_cb in small batches.
    Returns (full_output, tools_used=True).
    Raises RuntimeError on non-zero exit.
    """
    model_str = AIDER_MODEL_MAP.get(job.model, "anthropic/claude-sonnet-4-6")
    cwd       = job.repo_path or os.getcwd()

    env = {
        **os.environ,
        "PATH":               _AUGMENTED_PATH,
        "ANTHROPIC_API_KEY":  ANTHROPIC_API_KEY,
        "OPENROUTER_API_KEY": OPENROUTER_API_KEY,
        "GROQ_API_KEY":       GROQ_API_KEY,
        "GIT_TERMINAL_PROMPT": "0",
    }

    cmd = [
        "aider",
        "--message",        job.task,
        "--model",          model_str,
        "--no-auto-commits",   # we handle git commit after user approval
        "--yes-always",        # no interactive prompts
        "--no-pretty",         # plain text output
        "--no-fancy-input",    # no readline / colour escape codes
    ]

    log_event(job.id, "aider_start", f"model={model_str}")
    logger.info(f"[job {job.id}] aider start: {model_str} in {cwd}")

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=cwd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "aider not found — install it with: pip install aider-chat"
        )

    lines:  list[str] = []
    batch:  list[str] = []

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

    output = "\n".join(lines)
    log_event(job.id, "aider_done", f"rc={proc.returncode}")

    if proc.returncode != 0:
        raise RuntimeError(
            f"aider exited {proc.returncode}:\n{output[-600:]}"
        )

    return output, True   # tools_used=True — aider always edits files directly
