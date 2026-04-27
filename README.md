# my-agents

A personal AI coding agent that runs on a €5 VPS and talks through Telegram.

Built from scratch to understand how agents actually work — model routing,
diff review, conversation management, and SQLite-backed memory.

## Stack
python-telegram-bot · Anthropic API · OpenRouter · Groq · Aider · SQLite

## Key files
- agent.py — entry point and Telegram handler
- llm.py — model routing and fallback chain
- conversation.py — history management and compaction
- memory.py — SQLite-backed lessons and task history
- planner.py / executor.py — plan/approve/commit loop

## Setup
[brief env var / install steps]
