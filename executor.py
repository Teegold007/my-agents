"""
executor.py — safe command runner, no shell=True, allowlisted commands only.
"""

import os
import re
import subprocess
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Allowlist ─────────────────────────────────────────────────────────────────
# Only these top-level binaries may be executed.

ALLOWED_BINARIES = {
    "git", "python", "python3", "pip", "pip3",
    "node", "npm", "npx", "yarn", "pnpm",
    "cargo", "rustc",
    "go",
    "mvn", "gradle",
    "pytest", "jest", "vitest",
    "make",
    "ls", "cat", "find", "grep", "echo", "head", "tail",
    "mkdir", "cp", "mv", "rm", "touch", "chmod",
    "curl", "wget",
    "which", "env", "printenv",
    "wc", "sort", "uniq", "diff",
}

# Characters that indicate shell injection attempts
SHELL_INJECTION_CHARS = re.compile(r"[;&|`$<>\\]|\$\(|\beval\b|\bexec\b")


class ExecutionError(Exception):
    pass


def validate_path(path: str, workspace: str) -> str:
    """Ensure a path stays within the workspace. Raises on traversal."""
    resolved  = str(Path(workspace, path).resolve())
    workspace = str(Path(workspace).resolve())
    if not resolved.startswith(workspace):
        raise ExecutionError(f"Path traversal blocked: {path}")
    return resolved


def safe_run(
    argv: list[str],
    cwd: str,
    timeout: int = 120,
    log_path: str = None,
) -> tuple[int, str]:
    """
    Run a command safely:
    - No shell=True
    - Binary must be in ALLOWED_BINARIES
    - No shell injection characters in args
    - Timeout enforced
    - Returns (exit_code, combined_output)
    """
    if not argv:
        raise ExecutionError("Empty command")

    binary = Path(argv[0]).name
    if binary not in ALLOWED_BINARIES:
        raise ExecutionError(f"Binary not allowed: {binary}")

    for arg in argv[1:]:
        if SHELL_INJECTION_CHARS.search(str(arg)):
            raise ExecutionError(f"Suspicious argument blocked: {arg!r}")

    logger.info(f"safe_run: {' '.join(argv)} (cwd={cwd})")

    try:
        result = subprocess.run(
            argv,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
    except subprocess.TimeoutExpired:
        return -1, f"[timeout after {timeout}s]"
    except FileNotFoundError:
        return -1, f"[binary not found: {argv[0]}]"
    except Exception as e:
        return -1, f"[error: {e}]"

    parts = []
    if result.stdout.strip(): parts.append(result.stdout.strip())
    if result.stderr.strip(): parts.append(result.stderr.strip())
    output = "\n".join(parts) or f"[exit {result.returncode}]"

    # Truncate for Telegram but save full to disk
    if log_path:
        try:
            Path(log_path).write_text(output)
        except Exception:
            pass

    if len(output) > 6000:
        output = output[:2800] + "\n\n...[truncated — full log on server]...\n\n" + output[-2800:]

    return result.returncode, output


def parse_argv(command_str: str) -> list[str]:
    """
    Parse a command string from the LLM into an argv list.
    Handles quoted arguments. Raises if injection chars found.
    """
    if SHELL_INJECTION_CHARS.search(command_str):
        raise ExecutionError(f"Shell injection blocked in: {command_str!r}")

    import shlex
    try:
        argv = shlex.split(command_str)
    except ValueError as e:
        raise ExecutionError(f"Could not parse command: {e}")

    return argv
