from __future__ import annotations

from typing import Any, Dict

from openai import OpenAI
from langsmith.wrappers import wrap_openai
from langsmith import traceable

from legal_multi_agent.state.schemas import MASharedState
from legal_multi_agent.utils.toon import extract_toon_critic

import os
import re
from dotenv import load_dotenv

load_dotenv()

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

    # 🔧 اضافه: اگر verifier وجود دارد، به critic اطلاع بده
    verifier = state.get("verifier_output")
    verifier_hint = ""
    if verifier and verifier.get("scores"):
        supp_opts = [
            s["option_number"] 
            for s in verifier["scores"] 
            if str(s.get("support_level", "")).upper() == "SUPPORTED"
        ]
        rec = verifier.get("recommended_answer")
        verifier_hint = (
            f"\n📊 VERIFIER INFO (advisory only):\n"
            f"- Supported options: {supp_opts}\n"
            f"- Recommended: {rec} (confidence {verifier.get('confidence')})\n"
            f"⚠️ You may override verifier ONLY with explicit citation from SOURCES.\n"
        )

    # 🔧 system_msg و user_msg باید خارج از بلوک if باشند
    system_msg = (
        "Role: strict exam QA auditor for Iranian law.\n\n"
        
        "────────────────────────\n"
        "CONTEXT\n"
        "────────────────────────\n"
        "You are given:\n"
        "- SOURCES (statutes + binding judicial decisions)\n"
        "- The question\n"
        "- The answer options\n"
        "- The assistant's TOON answer\n"
        "- (Optional) Option verifier analysis\n\n"
        
        "Your task is to detect ONLY clear, objective, source-based errors.\n"
        "You must NOT judge style, completeness, or doctrinal preferences.\n\n"
        
        "────────────────────────\n"
        "CORE PRINCIPLE\n"
        "────────────────────────\n"
        "Flag an error ONLY when the assistant's answer is in DIRECT and EXPLICIT\n"
        "conflict with the plain text of the SOURCES.\n\n"
        "If you cannot demonstrate such a conflict by citing a specific article\n"
        "or binding ruling, then NO error exists.\n\n"
        
        "🔍 EVIDENCE REQUIREMENT:\n"
        "- You MUST cite a specific source location: [منبع N] + article/principle number.\n"
        "- If you cannot provide a precise citation, you MUST set needs_revision=false.\n"
        "- Verifier suggestions are ADVISORY; you must independently verify against SOURCES.\n\n"
        
        "────────────────────────\n"
        "WHAT COUNTS AS A CLEAR ERROR\n"
        "────────────────────────\n"
        "A clear error exists ONLY if at least one of the following is true:\n\n"
        
        "1) OPTION CONFLICT (with citation)\n"
        "   The selected option number contradicts an explicit rule in the SOURCES.\n"
        "   Example:\n"
        "   - [منبع 2] ماده 1234 explicitly states 'one-eighth'.\n"
        "   - Assistant selects 'one-fourth'.\n\n"
        
        "2) EXPLANATION CONFLICT (with citation)\n"
        "   The explanation contains a concrete legal claim that clearly contradicts\n"
        "   or misquotes the SOURCES.\n"
        "   Example:\n"
        "   - [منبع 1] ماده 567 explicitly includes degree 3.\n"
        "   - Assistant claims 'degree 3 is not covered'.\n\n"
        
        "3) FORMAT ERROR\n"
        "   The TOON answer is invalid:\n"
        "   - Missing required fields.\n"
        "   - More than one row.\n"
        "   - Invalid value range.\n\n"
        
        "────────────────────────\n"
        "WHAT DOES NOT COUNT AS AN ERROR\n"
        "────────────────────────\n"
        "Do NOT flag an error in the following cases:\n\n"
        "- Differences in wording or level of detail when meaning remains compatible.\n"
        "- Reasonable doctrinal or fiqhi explanations when SOURCES are silent or ambiguous.\n"
        "- Interpretive choices that are not expressly prohibited by SOURCES.\n"
        "- Disagreement with a scholarly view, absent explicit statutory contradiction.\n"
        "- Verifier recommendation without explicit contradiction in SOURCES.\n\n"
        
        "If SOURCES do not clearly say the opposite, the assistant is considered correct.\n\n"
        
        "────────────────────────\n"
        "OUTPUT FORMAT (STRICT)\n"
        "────────────────────────\n"
        "You MUST return EXACTLY ONE TOON table and nothing else:\n\n"
        "results{needs_revision,issue,action,confidence}:\n"
        "<true/false>,<short English issue no comma>,"
        "<short English action max 10 words>,<1-5>\n\n"
        
        "────────────────────────\n"
        "OUTPUT CONSTRAINTS\n"
        "────────────────────────\n"
        "- needs_revision: exactly 'true' or 'false' (lowercase).\n"
        "- issue:\n"
        "  - One short English phrase.\n"
        "  - No commas.\n"
        "  - Use 'no clear error' if applicable.\n"
        "- action:\n"
        "  - One short English phrase.\n"
        "  - No commas.\n"
        "  - Max 10 words.\n"
        "  - Use 'keep answer as is' if no revision is needed.\n"
        "- confidence:\n"
        "  - 1-5 (1=very uncertain, 5=very certain).\n"
        "  - Be conservative: if any doubt, use 1-3.\n\n"
        
        "Do not add any explanation or extra text.\n"
    )

    user_msg = f"""SOURCES:
{ctx}

QUESTION:
{q}

OPTIONS:
{options}

ASSISTANT OUTPUT (TOON):
{draft_raw}

{verifier_hint}

STEP-BY-STEP CHECK:
1) Is the TOON format valid (one row, fields explanation,answer,confidence)?
2) Can you cite a SPECIFIC article/principle (with [منبع N] and number) that contradicts the chosen answer?
3) Can you cite a SPECIFIC article/principle that contradicts a claim in the explanation?
4) If the verifier suggests a different answer, can you cite an explicit source contradiction?

If you CANNOT provide precise citations for steps 2–4, and the format is valid,
you MUST set needs_revision=false.
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
            "confidence": 1,
        }

    # 🔧 Post-check gate: اگر true است اما citation معتبر ندارد، به false برگردان
    def _has_valid_citation(txt: str) -> bool:
        """چک می‌کند که آیا critic ارجاع معتبر داده ([منبع N] + شماره ماده)"""
        return bool(re.search(r"\[منبع\s*\d+\]", txt))

    if bool(critic_toon.get("needs_revision")):
        if not _has_valid_citation(critic_raw):
            print("   ⚠️ [Critic] needs_revision=true but no valid citation → forcing false")
            critic_toon = {
                "needs_revision": False,
                "issue": "no clear error",
                "action": "keep answer as is",
                "confidence": 3,
            }

    return {
        "critic_raw": critic_raw,
        "critic_toon": critic_toon,
    }
