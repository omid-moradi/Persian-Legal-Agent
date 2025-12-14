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
        "\n"
        "Your job is to detect ONLY CLEAR, OBJECTIVE ERRORS, not to nitpick wording or unstated doctrines.\n"
        "\n"
        "DEFINITIONS:\n"
        "- A CLEAR CONFLICT exists ONLY if the chosen option or explanation directly contradicts the explicit text of the SOURCES "
        "(e.g., chooses option 2 while the statute explicitly says option 3).\n"
        "- Mere differences in phrasing, emphasis, or level of detail DO NOT count as conflict if they are compatible with the SOURCES.\n"
        "- If the SOURCES are silent or ambiguous on a doctrinal nuance, you MUST treat the assistant's doctrine as acceptable.\n"
        "- Disagreement with a possible fiqhi/doctrinal view is NOT enough to mark an error unless the view clearly contradicts the statute or binding case.\n"
        "\n"
        "CRITERIA FOR needs_revision=true:\n"
        "Set needs_revision=true ONLY if at least one of these holds:\n"
        "1) The CHOSEN OPTION NUMBER clearly conflicts with the SOURCES.\n"
        "   (Example: statute says the correct share is one-eighth, but the assistant chooses one-fourth.)\n"
        "2) The EXPLANATION contains a specific legal claim that clearly contradicts or misquotes the SOURCES.\n"
        "   (Example: the statute explicitly includes degree 6, but the assistant says 'degree 6 does not exist in this law'.)\n"
        "3) The TOON format is invalid (missing fields, wrong answer range, or not a single row).\n"
        "\n"
        "IMPORTANT NEGATIVE RULES (when NOT to flag):\n"
        "- Do NOT set needs_revision=true just because the wording does not exactly match the statute if the meaning is consistent.\n"
        "- Do NOT set needs_revision=true when the assistant reasonably extends the SOURCES with compatible doctrinal explanation.\n"
        "- Do NOT claim 'contradiction' unless you can point to a specific article/vote in SOURCES that says the opposite.\n"
        "\n"
        "OUTPUT FORMAT (STRICT):\n"
        "You MUST return ONLY one TOON table in this exact format:\n"
        "results{needs_revision,issue,action}:\n"
        "<true/false>,<short issue>,<short action>\n"
        "\n"
        "Constraints:\n"
        "- needs_revision: exactly 'true' or 'false' (lowercase).\n"
        "- issue: one short English phrase (no comma), describing the main problem, "
        "or 'no clear error' if there is none.\n"
        "- action: one short English phrase (no comma), max 10 words, suggesting how to fix or 'keep answer as is' if no revision is needed.\n"
        "Do not output anything else."
    )

    user_msg = f"""SOURCES:
{ctx}

QUESTION:
{q}

OPTIONS:
{options}

ASSISTANT OUTPUT (TOON):
{draft_raw}

Check step by step:
1) Is the TOON format valid (one row, fields explanation,answer,confidence)?
2) Is the chosen answer (1-4) clearly inconsistent with any specific article or binding case in SOURCES?
3) Does the explanation contain any statement that explicitly contradicts or misquotes a statute or case in SOURCES?

If you do NOT find a clear, source-based conflict in steps 2 or 3, and the format is valid, you MUST set needs_revision=false.
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
            "issue": "Critic output could not be parsed as TOON",
            "action": "Return a valid TOON-CRITIC table exactly as specified",
        }

    return {
        "critic_raw": critic_raw,
        "critic_toon": critic_toon,
    }
