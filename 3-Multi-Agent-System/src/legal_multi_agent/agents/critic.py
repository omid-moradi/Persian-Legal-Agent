from __future__ import annotations

from typing import Any, Dict

from openai import OpenAI
from langsmith.wrappers import wrap_openai
from langsmith import traceable

from legal_multi_agent.state.schemas import MASharedState
from legal_multi_agent.utils.toon import extract_toon_critic

import os


OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL_ID = "qwen/qwen3-235b-a22b-2507"

_raw_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)
client = wrap_openai(_raw_client)


@traceable(name="critic_agent")
def critic_agent(state: MASharedState) -> MASharedState:
    """ایجنت منتقد/قاضی که TOON-CRITIC را برمی‌گرداند."""
    q = state["question"]
    options = state["options_text"]
    ctx = state.get("context", "")
    draft_raw = state.get("draft_raw", "")

    system_msg = (
        "Role: exam QA critic for Iranian law.\n"
        "You see SOURCES (legal texts + cases), the question, options, and the assistant's TOON answer.\n"
        "Your job is to check for CLEAR ERRORS, not to reject every unstated doctrinal detail.\n"
        "You MUST treat the assistant's legal doctrine as acceptable UNLESS it directly contradicts the SOURCES.\n"
        "Only set needs_revision=true if:\n"
        "- The chosen option clearly conflicts with the SOURCES, OR\n"
        "- The explanation clearly contradicts or misquotes the SOURCES, OR\n"
        "- The TOON format is invalid.\n"
        "If the explanation adds reasonable doctrinal concepts (e.g., distinguishing 'condition of act' vs 'condition of result')"
        " that do not conflict with SOURCES, you MUST NOT mark them as unsupported.\n"
        "Return ONLY a TOON table:\n"
        "results{needs_revision,issue,action}:\n"
        "<true/false>,<short issue>,<short action>\n"
        "Constraints:\n"
        "- issue: one short English phrase (no comma).\n"
        "- action: one short English phrase (no comma), max 10 words.\n"
        "Do not output anything else."
    )

    user_msg = f"""SOURCES:
{ctx}

QUESTION:
{q}

OPTIONS:
{options}

ASSISTANT OUTPUT:
{draft_raw}

Check:
- Valid TOON format (one row)
- answer is 1-4
- explanation is grounded in SOURCES
"""

    resp = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
    )

    critic_raw = resp.choices[0].message.content
    critic_toon: Dict[str, Any] | None = extract_toon_critic(critic_raw)

    # اگر نتوانستیم TOON-CRITIC را parse کنیم، fail-safe:
    if critic_toon is None:
        critic_toon = {
            "needs_revision": True,
            "issue": "Critic output could not be parsed as TOON.",
            "action": "Return a valid TOON-CRITIC table exactly as specified.",
        }

    return {
        "critic_raw": critic_raw,
        "critic_toon": critic_toon,
    }
