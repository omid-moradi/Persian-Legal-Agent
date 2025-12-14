from __future__ import annotations

from typing import Any, Dict

from openai import OpenAI
from langsmith.wrappers import wrap_openai
from langsmith import traceable

from legal_multi_agent.state.schemas import MASharedState
from legal_multi_agent.utils.toon import extract_toon_answer

import os


OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL_ID = "qwen/qwen3-235b-a22b-2507"

_raw_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)
client = wrap_openai(_raw_client)


@traceable(name="reasoner_agent")
def reasoner_agent(state: MASharedState) -> MASharedState:
    """ایجنت استدلال‌گر که TOON پاسخ را می‌سازد (MCQ چهارگزینه‌ای)."""
    q = state["question"]
    options = state["options_text"]
    ctx = state.get("context", "")

    critic = state.get("critic_toon")
    critic_hint = ""
    if critic and critic.get("needs_revision", False):
        critic_hint = (
            f"Issue: {critic.get('issue','')}\n"
            f"Action: {critic.get('action','')}\n"
        )

    system_msg = (
        "You are an Iranian legal exam QA assistant.\n"
        "Rules:\n"
        "1) Use ONLY the provided SOURCES.\n"
        "2) Answer in Persian.\n"
        "3) Choose exactly one option number (1-4).\n"
        "4) Provide a short legal explanation (1-2 Persian sentences).\n"
        "5) Output MUST be exactly one TOON table with one row.\n\n"
        "Output format (exact):\n"
        "results{explanation,answer,confidence}:\n"
        "<explanation>,<1-4>,<1-5>\n"
        "Do not output anything else."
    )

    user_msg = f"""SOURCES:
{ctx}

QUESTION (Persian):
{q}

OPTIONS:
{options}

CRITIC FEEDBACK (if any):
{critic_hint}

INSTRUCTIONS:
- If there is no CRITIC FEEDBACK, answer normally.
- If CRITIC FEEDBACK exists and needs_revision=true, you MUST correct the explanation
  according to the critic's issue/action. The new explanation MUST be different from
  the previous one and MUST fix the reported problem.
"""

    resp = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )

    draft_raw = resp.choices[0].message.content
    draft_toon: Dict[str, Any] = extract_toon_answer(draft_raw)

    rc = int(state.get("revision_count", 0))
    # اگر قبلاً draft_toon داشتیم یا critic فیدبک داده، یعنی در حلقه اصلاح هستیم
    if state.get("draft_toon") or critic:
        rc += 1

    return {
        "draft_raw": draft_raw,
        "draft_toon": draft_toon,
        "revision_count": rc,
    }
