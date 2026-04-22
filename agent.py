import os
import json
import re
import subprocess
import logging
import asyncio
from dotenv import load_dotenv

import anthropic
from openai import OpenAI
from telegram import Update
from telegram.ext import (
    Application, MessageHandler, CommandHandler,
    filters, ContextTypes
)
from memory import build_memory_block, reflect_and_save, stats, clear, get_lessons, get_history

load_dotenv()
logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Env ───────────────────────────────────────────────────────────────────────

TELEGRAM_TOKEN     = os.environ["TELEGRAM_TOKEN"]
ALLOWED_USER_IDS   = set(int(x) for x in os.environ["ALLOWED_USER_IDS"].split(","))
WORKSPACE          = os.environ.get("WORKSPACE", "/home/agent/repos")
ANTHROPIC_API_KEY  = os.getenv("ANTHROPIC_API_KEY", "")
GROQ_API_KEY       = os.getenv("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OLLAMA_BASE_URL    = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL       = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
DEFAULT_MODEL      = os.getenv("DEFAULT_MODEL", "auto")

# ── Model registry ────────────────────────────────────────────────────────────

MODELS = {
    "claude":   {"provider": "anthropic",  "id": "claude-opus-4-5",                   "label": "Claude Opus 4.5        (~$0.075/1k)"},
    "sonnet":   {"provider": "anthropic",  "id": "claude-sonnet-4-6",                 "label": "Claude Sonnet 4.6      (~$0.015/1k)"},
    "llama":    {"provider": "groq",       "id": "llama-3.3-70b-versatile",           "label": "Llama 3.3 70B on Groq  (FREE)"},
    "deepseek": {"provider": "openrouter", "id": "deepseek/deepseek-chat",            "label": "DeepSeek V3            (~$0.001/1k)"},
    "qwen":     {"provider": "openrouter", "id": "qwen/qwen-2.5-coder-32b-instruct", "label": "Qwen 2.5 Coder         (~$0.002/1k)"},
    "local":    {"provider": "ollama",     "id": OLLAMA_MODEL,                        "label": f"Local Ollama ({OLLAMA_MODEL})"},
}

COMPLEX_KEYWORDS = {
    "refactor", "architecture", "redesign", "security", "optimize", "performance",
    "migrate", "authentication", "database", "schema", "review", "audit",
    "rewrite", "overhaul",
}

def detect_complexity(msg: str) -> str:
    if len(msg.split()) > 60: return "complex"
    if any(kw in msg.lower() for kw in COMPLEX_KEYWORDS): return "complex"
    return "simple"

def detect_repo(msg: str) -> str | None:
    try:
        repos = os.listdir(WORKSPACE)
    except Exception:
        return None
    for repo in repos:
        if repo.lower() in msg.lower():
            return repo
    return None

# ── Clients ───────────────────────────────────────────────────────────────────

def get_anthropic_client():
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

def get_openai_client(provider: str) -> OpenAI:
    if provider == "groq":
        return OpenAI(api_key=GROQ_API_KEY,       base_url="https://api.groq.com/openai/v1")
    if provider == "openrouter":
        return OpenAI(api_key=OPENROUTER_API_KEY,  base_url="https://openrouter.ai/api/v1")
    if provider == "ollama":
        return OpenAI(api_key="ollama",             base_url=OLLAMA_BASE_URL)
    raise ValueError(f"Unknown provider: {provider}")

# ── Tool definitions ──────────────────────────────────────────────────────────

ANTHROPIC_TOOLS = [{
    "name": "bash",
    "description": "Run a bash command on the server. CWD is WORKSPACE. Output truncated at 6000 chars.",
    "input_schema": {
        "type": "object",
        "properties": {
            "command":     {"type": "string", "description": "The bash command to run."},
            "description": {"type": "string", "description": "One-line summary of what this does."},
        },
        "required": ["command", "description"],
    },
}]

OPENAI_TOOLS = [{
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a bash command on the server. CWD is WORKSPACE. Output truncated at 6000 chars.",
        "parameters": {
            "type": "object",
            "properties": {
                "command":     {"type": "string", "description": "The bash command to run."},
                "description": {"type": "string", "description": "One-line summary of what this does."},
            },
            "required": ["command", "description"],
        },
    },
}]

# ── Bash execution ────────────────────────────────────────────────────────────

def run_bash(cmd: str, timeout: int = 120) -> str:
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=timeout, cwd=WORKSPACE,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
        parts = []
        if r.stdout.strip(): parts.append(r.stdout.strip())
        if r.stderr.strip(): parts.append(f"[stderr]\n{r.stderr.strip()}")
        if not parts: parts.append(f"[exit {r.returncode}]")
        out = "\n".join(parts)
        if len(out) > 6000:
            out = out[:2800] + "\n\n...[truncated]...\n\n" + out[-2800:]
        return out
    except subprocess.TimeoutExpired:
        return f"[timeout after {timeout}s]"
    except Exception as e:
        return f"[error] {e}"

# ── System prompt ─────────────────────────────────────────────────────────────

def build_system_prompt(repo: str = None) -> str:
    memory = build_memory_block(repo)
    base = f"""You are an expert coding agent on a Linux server.
Repos are at: {WORKSPACE}

Workflow:
1. Explore the repo structure first (ls, cat key files like package.json / pyproject.toml)
2. Check the memory/lessons below carefully — avoid repeating known mistakes
3. Make focused, minimal changes to solve the task
4. Run existing tests or build commands if they exist
5. Commit everything: git add -A && git commit -m "clear message"
6. Report back concisely

Finish every task with:
✅ What was done
📁 Files changed
🔀 Git commit hash
⚠️  Any decisions or issues worth noting"""

    if memory:
        return base + "\n\n" + memory
    return base

# ── Agentic loops ─────────────────────────────────────────────────────────────

async def loop_anthropic(message: str, model_id: str, repo: str, status_cb) -> str:
    client   = get_anthropic_client()
    messages = [{"role": "user", "content": message}]

    for _ in range(30):
        resp = client.messages.create(
            model=model_id, max_tokens=8096,
            system=build_system_prompt(repo),
            tools=ANTHROPIC_TOOLS, messages=messages,
        )
        tool_uses   = [b for b in resp.content if b.type == "tool_use"]
        text_blocks = [b for b in resp.content if b.type == "text"]

        if not tool_uses:
            return "\n".join(b.text for b in text_blocks).strip() or "Done."

        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for tu in tool_uses:
            desc = tu.input.get("description", tu.input["command"][:80])
            await status_cb(f"⚙️ _{desc}_")
            output = run_bash(tu.input["command"])
            logger.info(f"bash: {tu.input['command'][:80]}")
            results.append({"type": "tool_result", "tool_use_id": tu.id, "content": output})
        messages.append({"role": "user", "content": results})

    return "⚠️ Hit 30-step limit."


async def loop_openai_compat(message: str, provider: str, model_id: str, repo: str, status_cb) -> str:
    client   = get_openai_client(provider)
    messages = [
        {"role": "system", "content": build_system_prompt(repo)},
        {"role": "user",   "content": message},
    ]

    for _ in range(30):
        resp = client.chat.completions.create(
            model=model_id, max_tokens=8096,
            tools=OPENAI_TOOLS, tool_choice="auto",
            messages=messages,
        )
        msg = resp.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            return (msg.content or "Done.").strip()

        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            desc = args.get("description", args["command"][:80])
            await status_cb(f"⚙️ _{desc}_")
            output = run_bash(args["command"])
            logger.info(f"bash: {args['command'][:80]}")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})

    return "⚠️ Hit 30-step limit."


async def run_agent(message: str, model_key: str, status_cb) -> str:
    repo = detect_repo(message)

    if model_key == "auto":
        complexity = detect_complexity(message)
        model_key  = "sonnet" if complexity == "complex" else "llama"
        await status_cb(f"🎯 Auto-selected *{model_key}* ({complexity} task)")

    cfg = MODELS.get(model_key)
    if not cfg:
        return f"❌ Unknown model: `{model_key}`"

    repo_label = f" · repo: `{repo}`" if repo else ""
    await status_cb(f"🤖 *{cfg['label'].split('(')[0].strip()}*{repo_label}")

    provider = cfg["provider"]
    model_id = cfg["id"]

    if provider == "anthropic":
        result = await loop_anthropic(message, model_id, repo, status_cb)
    else:
        result = await loop_openai_compat(message, provider, model_id, repo, status_cb)

    # Reflect and save lessons in background — free, uses Groq
    asyncio.create_task(reflect_and_save(message, result, repo=repo, model=model_key))

    return result

# ── Auth ──────────────────────────────────────────────────────────────────────

def auth(update: Update) -> bool:
    return update.effective_user.id in ALLOWED_USER_IDS

# ── Telegram command handlers ─────────────────────────────────────────────────

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not auth(update): return
    model = ctx.user_data.get("model", DEFAULT_MODEL)
    await update.message.reply_text(
        "👋 *Coding Agent ready!*\n\n"
        f"Current model: `{model}`\n\n"
        "*Commands:*\n"
        "• /model — list & switch models\n"
        "• /memory — view lessons learned\n"
        "• /forget `<repo|global>` — clear memory\n"
        "• /repos — list repos in workspace\n"
        "• /status `<repo>` — git log & status\n\n"
        "*Example tasks:*\n"
        "• _Add a /health endpoint to the Express app in backend_\n"
        "• _Fix the failing tests in myapp_\n"
        "• _Refactor the auth module to use JWT_",
        parse_mode="Markdown",
    )


async def cmd_model(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not auth(update): return
    args = ctx.args

    if not args:
        current = ctx.user_data.get("model", DEFAULT_MODEL)
        lines = [f"*Current model:* `{current}`\n", "*Available:*"]
        for k, v in MODELS.items():
            mark = "👉 " if k == current else "    "
            lines.append(f"{mark}`{k}` — {v['label']}")
        lines.append("    `auto` — smart routing (simple→llama, complex→sonnet)")
        lines.append("\nSwitch with `/model <name>`")
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
        return

    key = args[0].lower()
    if key != "auto" and key not in MODELS:
        await update.message.reply_text(f"❌ Unknown model `{key}`. Use /model to list options.", parse_mode="Markdown")
        return
    ctx.user_data["model"] = key
    label = "auto-routing" if key == "auto" else MODELS[key]["label"].split("(")[0].strip()
    await update.message.reply_text(f"✅ Switched to `{key}` — {label}", parse_mode="Markdown")


async def cmd_memory(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not auth(update): return
    s = stats()

    lines = [f"🧠 *Agent memory*\n"]
    lines.append(f"Total lessons: *{s['total_lessons']}* ({s['global_lessons']} global)")
    lines.append(f"Tasks completed: *{s['total_tasks']}*")

    if s["repos"]:
        repo_lines = ", ".join(f"`{r['repo']}` ({r['lessons']})" for r in s["repos"])
        lines.append(f"Repo-specific: {repo_lines}")

    # Show top 5 global lessons
    global_lessons = get_lessons(repo=None, limit=5)
    if global_lessons:
        lines.append("\n*Top global lessons:*")
        for l in global_lessons:
            lines.append(f"• {l}")

    # Show last 3 tasks
    history = get_history(n=3)
    if history:
        lines.append("\n*Recent tasks:*")
        for h in history:
            ts = h["created_at"][:16].replace("T", " ")
            lines.append(f"`{ts}` {h['task'][:80]}")

    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def cmd_forget(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not auth(update): return
    if not ctx.args:
        await update.message.reply_text(
            "Usage:\n`/forget global` — clear global lessons\n`/forget <reponame>` — clear repo lessons",
            parse_mode="Markdown",
        )
        return
    target = ctx.args[0].lower()
    if target == "global":
        clear(scope="global")
        await update.message.reply_text("🗑️ Cleared global lessons.", parse_mode="Markdown")
    elif target == "all":
        clear(scope="all")
        await update.message.reply_text("🗑️ Cleared all memory.", parse_mode="Markdown")
    else:
        clear(repo=target, scope="repo")
        await update.message.reply_text(f"🗑️ Cleared lessons for `{target}`.", parse_mode="Markdown")


async def cmd_repos(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not auth(update): return
    out = run_bash(f"ls -1 {WORKSPACE}")
    await update.message.reply_text(f"📁 *Repos:*\n```\n{out}\n```", parse_mode="Markdown")


async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not auth(update): return
    if not ctx.args:
        await update.message.reply_text("Usage: `/status <repo>`", parse_mode="Markdown")
        return
    repo = ctx.args[0]
    out = run_bash(f"cd {WORKSPACE}/{repo} && git log --oneline -5 && echo '---' && git status -s")
    await update.message.reply_text(f"```\n{out}\n```", parse_mode="Markdown")


async def handle_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not auth(update):
        logger.warning(f"Unauthorized attempt from user ID {update.effective_user.id}")
        return

    text      = update.message.text
    model_key = ctx.user_data.get("model", DEFAULT_MODEL)
    thinking  = await update.message.reply_text("🤖 On it…")

    async def status_cb(msg: str):
        try:
            await update.message.reply_text(msg, parse_mode="Markdown")
        except Exception:
            pass

    try:
        result = await run_agent(text, model_key, status_cb)
        await thinking.delete()
        await update.message.reply_text(result, parse_mode="Markdown")
    except Exception as e:
        logger.exception("Agent error")
        await thinking.delete()
        await update.message.reply_text(f"❌ Error: `{e}`", parse_mode="Markdown")


# ── ADD THIS FUNCTION ─────────────────────────────────────────────────────────

async def cmd_clone(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not auth(update): return

    if not ctx.args:
        await update.message.reply_text(
            "Usage: `/clone <github_url>`\n\nExample:\n`/clone https://github.com/you/myapp`",
            parse_mode="Markdown",
        )
        return

    url = ctx.args[0].strip()

    # Accept both https and ssh URLs, also bare "user/repo" shorthand
    if re.match(r"^[\w\-]+/[\w\-\.]+$", url):
        url = f"git@github.com:{url}.git"
    elif url.startswith("https://github.com/"):
        # Convert https to ssh so pushing works
        path = url.replace("https://github.com/", "").removesuffix(".git")
        url  = f"git@github.com:{path}.git"

    repo_name = url.split("/")[-1].removesuffix(".git")
    dest      = f"{WORKSPACE}/{repo_name}"

    # Don't clone if already exists
    if os.path.exists(dest):
        await update.message.reply_text(
            f"📁 `{repo_name}` already exists in workspace.\nUse `/status {repo_name}` to check it.",
            parse_mode="Markdown",
        )
        return

    msg = await update.message.reply_text(f"⏳ Cloning `{repo_name}`…", parse_mode="Markdown")
    output = run_bash(f"git clone {url} {dest}")

    if os.path.exists(dest):
        await msg.edit_text(
            f"✅ Cloned `{repo_name}` successfully.\n\nNow just tell me what to do with it!",
            parse_mode="Markdown",
        )
    else:
        await msg.edit_text(
            f"❌ Clone failed for `{repo_name}`:\n```\n{output[:600]}\n```",
            parse_mode="Markdown",
        )


# ── ADD THIS TO main() ────────────────────────────────────────────────────────
# app.add_handler(CommandHandler("clone", cmd_clone))


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start",  cmd_start))
    app.add_handler(CommandHandler("model",  cmd_model))
    app.add_handler(CommandHandler("memory", cmd_memory))
    app.add_handler(CommandHandler("forget", cmd_forget))
    app.add_handler(CommandHandler("repos",  cmd_repos))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("clone",  cmd_clone))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    logger.info("Agent started and polling for messages.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
