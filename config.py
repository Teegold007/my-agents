"""
config.py — Environment config, model registry, and shared utilities.
No telegram or LLM imports — safe to import from anywhere.
"""

import os
import re
import json
import logging
import time
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Environment ───────────────────────────────────────────────────────────────

TELEGRAM_TOKEN     = os.environ["TELEGRAM_TOKEN"]
ALLOWED_USER_IDS   = {int(x) for x in os.environ["ALLOWED_USER_IDS"].split(",")}
WORKSPACE          = os.environ.get("WORKSPACE", "/home/agent3/repos")
AGENT_DIR          = os.environ.get("AGENT_DIR", "/home/agent3/agent")
LOGS_DIR           = Path(os.environ.get("LOGS_DIR", "/home/agent3/runs"))
ANTHROPIC_API_KEY  = os.getenv("ANTHROPIC_API_KEY", "")
GROQ_API_KEY       = os.getenv("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
DEFAULT_MODEL      = os.getenv("DEFAULT_MODEL", "deepseek")

LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ── Groq rate-limit circuit breaker ──────────────────────────────────────────
# When Groq's daily token limit is hit, all callers check this before attempting
# a call. The backoff resets after _GROQ_BACKOFF_SECONDS.

_GROQ_BACKOFF_UNTIL: float = 0.0
_GROQ_BACKOFF_SECONDS = 3600  # 1 hour default; overridden by error message


def groq_available() -> bool:
    """Return False while Groq is in its rate-limit backoff window."""
    return time.time() >= _GROQ_BACKOFF_UNTIL


def groq_mark_rate_limited(retry_after_seconds: float = _GROQ_BACKOFF_SECONDS) -> None:
    """Call this when a Groq RateLimitError is received."""
    global _GROQ_BACKOFF_UNTIL
    _GROQ_BACKOFF_UNTIL = time.time() + retry_after_seconds
    logger.warning(
        f"Groq rate-limited — disabling for {retry_after_seconds:.0f}s "
        f"(until {datetime.fromtimestamp(_GROQ_BACKOFF_UNTIL).strftime('%H:%M:%S')})"
    )


def parse_groq_retry_after(error_message: str) -> float:
    """Parse '1h44m30s' style retry-after from Groq error messages."""
    m = re.search(r"try again in\s+((?:\d+h)?(?:\d+m)?(?:\d+(?:\.\d+)?s)?)", error_message, re.IGNORECASE)
    if not m:
        return _GROQ_BACKOFF_SECONDS
    text = m.group(1)
    seconds = 0.0
    for value, unit in re.findall(r"(\d+(?:\.\d+)?)([hms])", text):
        if unit == "h":   seconds += float(value) * 3600
        elif unit == "m": seconds += float(value) * 60
        else:             seconds += float(value)
    return seconds if seconds > 0 else _GROQ_BACKOFF_SECONDS
try:
    Path(WORKSPACE).mkdir(parents=True, exist_ok=True)
except PermissionError:
    logger.warning(f"Cannot create WORKSPACE at {WORKSPACE!r} — check the path and permissions.")

# ── Model registry ────────────────────────────────────────────────────────────

MODELS: dict[str, dict] = {
    "haiku":    {"provider": "anthropic",  "id": "claude-haiku-4-5-20251001",         "label": "Claude Haiku 4.5 (fast)",  "supports_tools": True},
    "sonnet":   {"provider": "anthropic",  "id": "claude-sonnet-4-6",                 "label": "Claude Sonnet 4.6",        "supports_tools": True},
    "claude":   {"provider": "anthropic",  "id": "claude-opus-4-5",                   "label": "Claude Opus 4.5",          "supports_tools": True},
    "llama":    {"provider": "groq",       "id": "llama-3.3-70b-versatile",           "label": "Llama 3.3 70B (free)",     "supports_tools": False},
    "deepseek": {"provider": "openrouter", "id": "deepseek/deepseek-chat",            "label": "DeepSeek V3",              "supports_tools": True},
    "qwen":     {"provider": "openrouter", "id": "qwen/qwen-2.5-coder-32b-instruct", "label": "Qwen 2.5 Coder",           "supports_tools": False},
    "r1":       {"provider": "openrouter", "id": "deepseek/deepseek-r1",             "label": "DeepSeek R1 (reasoning)",  "supports_tools": False},
}

FALLBACK_MODELS: dict[str, str] = {
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

# ── Utility functions ─────────────────────────────────────────────────────────

def detect_complexity(msg: str) -> str:
    if len(msg.split()) > 50:
        return "complex"
    if any(kw in msg.lower() for kw in COMPLEX_KEYWORDS):
        return "complex"
    return "simple"


def detect_repo(msg: str) -> tuple[str, str] | tuple[None, None]:
    """Return (repo_name, repo_path) if a known repo name appears in msg."""
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
    """Load .agent.json and AGENTS.md / AGENT.md from repo root if present."""
    cfg: dict = {}

    json_path = Path(repo_path) / ".agent.json"
    if json_path.exists():
        try:
            cfg = json.loads(json_path.read_text())
        except Exception:
            pass

    # Load AGENTS.md or AGENT.md (code_puppy convention) as free-form rules
    for name in ("AGENTS.md", "AGENT.md", ".agent.md"):
        md_path = Path(repo_path) / name
        if md_path.exists():
            try:
                cfg["agents_md"] = md_path.read_text()[:4000]
            except Exception:
                pass
            break

    return cfg


def auth(update) -> bool:
    """Return True if the Telegram user is in the allowed set."""
    return update.effective_user.id in ALLOWED_USER_IDS


def esc(text: str) -> str:
    """Escape text for Telegram HTML."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def send_html(text: str) -> str:
    """Truncate to Telegram's 4096-char message limit."""
    return text[:4090]
