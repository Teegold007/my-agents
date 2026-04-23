"""
planner.py — Plan generation and formatting.

Single Responsibility: turning a task description into a structured plan
and presenting it to the user.
"""

import re
import json

from config import esc
from llm import get_anthropic_client, get_openai_client

PLAN_SYSTEM = """You are a senior software engineer producing a detailed implementation plan.
You will be given a bug report or feature request and a repo name.
Return ONLY valid JSON — no markdown fences, no prose outside the JSON.

Rules:
- steps must be CONCRETE actions (e.g. "Read src/attachments/upload.js to find the filename extraction logic", not "Investigate the code")
- Identify the exact files likely involved based on the task description
- steps should cover: read/understand → locate root cause → implement fix → verify
- Minimum 4 steps, maximum 12
- If feedback is provided, revise the plan to address it

JSON format:
{
  "summary": "one-line description of the fix/feature",
  "root_cause": "hypothesis about what is causing the bug",
  "steps": ["step 1", "step 2", ...],
  "files_to_read": ["path/to/file"],
  "files_to_change": ["path/to/file"],
  "test_commands": ["npm test"],
  "risks": ["may break X"]
}"""


import logging
logger = logging.getLogger(__name__)

# Ordered preference for planning — most capable first, free fallback last.
_PLAN_MODEL_CHAIN = [
    ("anthropic", "claude-haiku-4-5-20251001"),
    ("openrouter", "deepseek/deepseek-chat"),
    ("groq",       "llama-3.3-70b-versatile"),
]


async def _call_plan_model(provider: str, model_id: str, prompt: str) -> dict:
    if provider == "anthropic":
        client = get_anthropic_client()
        resp   = await client.messages.create(
            model=model_id, max_tokens=2048,
            system=PLAN_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
    else:
        client = get_openai_client(provider)
        resp   = await client.chat.completions.create(
            model=model_id, max_tokens=2048,
            messages=[
                {"role": "system", "content": PLAN_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
        )
        raw = resp.choices[0].message.content.strip()

    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$",          "", raw)
    return json.loads(raw)


async def generate_plan(task: str, repo: str, model_id: str, provider: str,
                        feedback: str = "", previous_plan: dict = None) -> dict:
    prompt = f"Repo: {repo or 'unknown'}\nTask: {task}"
    if previous_plan and feedback:
        prev_steps = "\n".join(f"- {s}" for s in previous_plan.get("steps", []))
        prompt += (
            f"\n\nPrevious plan:\n{prev_steps}"
            f"\n\nUser feedback — revise the plan to address this:\n{feedback}"
        )
    elif feedback:
        prompt += f"\n\nFeedback on previous plan — revise to address this:\n{feedback}"

    # Try the requested model first, then fall back down the chain.
    chain = [(provider, model_id)] + [
        (p, m) for p, m in _PLAN_MODEL_CHAIN if m != model_id
    ]
    last_error = None
    for p, m in chain:
        try:
            return await _call_plan_model(p, m, prompt)
        except Exception as e:
            logger.warning(f"Plan model {m} failed: {e}")
            last_error = e

    raise RuntimeError(f"All plan models failed. Last error: {last_error}")


def format_plan_html(plan: dict, job_id: int) -> str:
    lines = [
        f"<b>📋 Plan — Job #{job_id}</b>",
        f"<i>{esc(plan.get('summary', ''))}</i>",
    ]
    if plan.get("root_cause"):
        lines += ["", f"<b>Root cause:</b> {esc(plan['root_cause'])}"]

    lines.append("")
    for i, step in enumerate(plan.get("steps", []), 1):
        lines.append(f"{i}. {esc(step)}")

    if plan.get("files_to_change"):
        lines.append("")
        lines.append("<b>Files to change:</b>")
        for f in plan["files_to_change"]:
            lines.append(f"  • <code>{esc(f)}</code>")

    if plan.get("risks"):
        lines.append("")
        lines.append("<b>⚠️ Risks:</b>")
        for r in plan["risks"]:
            lines.append(f"  • {esc(r)}")

    lines.append("")
    lines.append(
        f"Reply <b>approve {job_id}</b> to proceed, "
        f"<b>revert {job_id}</b> to cancel, "
        f"or just tell me what to change and I'll revise the plan."
    )
    return "\n".join(lines)
