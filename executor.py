"""
executor.py — safe command runner, no shell=True, allowlisted commands only.
"""

import os
import re
import shutil
import subprocess
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ── PATH augmentation ─────────────────────────────────────────────────────────
# System services often run with a stripped PATH. Combine the process PATH with
# common install locations so binaries like git are always findable.

_EXTRA_PATHS = [
    "/usr/bin", "/usr/local/bin", "/bin",
    "/usr/local/git/bin",          # macOS Xcode git
    "/opt/homebrew/bin",           # Homebrew (Apple Silicon)
    "/opt/homebrew/sbin",
    "/home/linuxbrew/.linuxbrew/bin",  # Homebrew (Linux)
    "/snap/bin",                   # Ubuntu snap packages
]

# Include the venv that the agent itself runs inside, so tools installed there
# (e.g. aider) are always findable regardless of how the service was launched.
_VENV_BIN = str(Path(__file__).parent / "venv" / "bin")
if Path(_VENV_BIN).is_dir():
    _EXTRA_PATHS.insert(0, _VENV_BIN)

_AUGMENTED_PATH = ":".join(filter(None, [os.environ.get("PATH", "")] + _EXTRA_PATHS))

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


# ── Native file tools (no shell, no injection risk) ───────────────────────────

_MAX_READ_BYTES  = 200_000   # ~5k lines
_MAX_WRITE_BYTES = 500_000


def tool_read_file(path: str, cwd: str) -> str:
    """Read a file and return its content. Path is relative to cwd."""
    target = Path(cwd, path).resolve()
    cwd_r  = Path(cwd).resolve()
    if not str(target).startswith(str(cwd_r)):
        raise ExecutionError(f"Path traversal blocked: {path}")
    if not target.exists():
        raise ExecutionError(f"File not found: {path}")
    if target.stat().st_size > _MAX_READ_BYTES:
        raise ExecutionError(f"File too large to read in one call (>{_MAX_READ_BYTES} bytes): {path}")
    return target.read_text(errors="replace")


def tool_write_file(path: str, content: str, cwd: str) -> str:
    """Write (overwrite) a file. Creates parent directories as needed."""
    target = Path(cwd, path).resolve()
    cwd_r  = Path(cwd).resolve()
    if not str(target).startswith(str(cwd_r)):
        raise ExecutionError(f"Path traversal blocked: {path}")
    if len(content.encode()) > _MAX_WRITE_BYTES:
        raise ExecutionError(f"Content too large to write (>{_MAX_WRITE_BYTES} bytes)")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return f"Written {len(content)} chars to {path}"


def tool_replace_in_file(path: str, old_str: str, new_str: str, cwd: str) -> str:
    """
    Replace the FIRST occurrence of old_str in the file with new_str.
    Falls back to normalised-whitespace matching when exact match fails.
    Returns a short diff summary on success.
    """
    target = Path(cwd, path).resolve()
    cwd_r  = Path(cwd).resolve()
    if not str(target).startswith(str(cwd_r)):
        raise ExecutionError(f"Path traversal blocked: {path}")
    if not target.exists():
        raise ExecutionError(f"File not found: {path}")

    original = target.read_text(errors="replace")

    # Exact match
    if old_str in original:
        updated = original.replace(old_str, new_str, 1)
        target.write_text(updated)
        added   = len(new_str.splitlines())
        removed = len(old_str.splitlines())
        return f"Replaced in {path}: -{removed} lines +{added} lines"

    # Normalised-whitespace fallback (handles minor indent/spacing diffs)
    norm_orig = re.sub(r"[ \t]+", " ", original)
    norm_old  = re.sub(r"[ \t]+", " ", old_str)
    if norm_old in norm_orig:
        idx     = norm_orig.index(norm_old)
        updated = original[:idx] + new_str + original[idx + len(norm_old):]
        target.write_text(updated)
        return f"Replaced (normalised) in {path}"

    raise ExecutionError(
        f"old_str not found in {path}. "
        f"Verify the exact text exists in the file first with read_file."
    )


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

    # Locate the binary and resolve any symlinks to confirm the target exists.
    # shutil.which can return a broken symlink (e.g. /usr/bin/git -> /snap/bin/git
    # when snap isn't mounted), so we verify the real path before using it.
    found = shutil.which(binary, path=_AUGMENTED_PATH)
    if found is None:
        return -1, (
            f"[{binary} not found in PATH — install it with: sudo apt install {binary}]"
        )
    real = os.path.realpath(found)
    if not os.path.isfile(real):
        return -1, (
            f"[{binary} found at {found} but resolves to a missing target ({real}). "
            f"Try: sudo apt install {binary}]"
        )
    argv = [real] + list(argv[1:])

    logger.info(f"safe_run: {' '.join(argv)} (cwd={cwd})")

    try:
        result = subprocess.run(
            argv,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PATH": _AUGMENTED_PATH, "GIT_TERMINAL_PROMPT": "0"},
        )
    except subprocess.TimeoutExpired:
        return -1, f"[timeout after {timeout}s]"
    except FileNotFoundError as e:
        if cwd and not os.path.isdir(cwd):
            return -1, f"[working directory does not exist: {cwd}]"
        return -1, f"[{argv[0]} not found — is it installed?]"
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
