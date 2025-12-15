from __future__ import annotations
from typing import Any, Dict

from openai import OpenAI
from langsmith.wrappers import wrap_openai
from langsmith import traceable
from dotenv import load_dotenv

from legal_multi_agent.state.schemas import MASharedState
from legal_multi_agent.utils.toon import extract_toon_answer

import os

load_dotenv()

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL_ID = "qwen/qwen3-235b-a22b-2507"

_raw_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)
client = wrap_openai(_raw_client)


@traceable(name="reasoner_agent")
def reasoner_agent(state: MASharedState) -> MASharedState:
    """
    ایجنت استدلال‌گر که TOON پاسخ را می‌سازد.
    """
    q = state["question"]
    options = state["options_text"]
    ctx = state.get("context", "")
    
    use_verifier = state.get("use_option_verifier", False)
    
    # 🔧 مدیریت verifier
    if use_verifier:
        tool_results = state.get("tool_results", {})
        
        # مرحله 2: اگر verifier اجرا شده، نتیجه را ذخیره کن
        if "option_verifier_tool" in tool_results and not state.get("verifier_output"):
            verifier_output = tool_results["option_verifier_tool"]
            return {
                "verifier_output": verifier_output,
            }
        
        # مرحله 1: اگر context داریم و verifier هنوز اجرا نشده
        if ctx and "option_verifier_tool" not in tool_results:
            messages = state.get("messages", [])
            
            # چک کن که قبلاً tool call نکردیم
            already_called = False
            if messages:
                for msg in messages:
                    if isinstance(msg, dict) and msg.get("tool_calls"):
                        for tc in msg["tool_calls"]:
                            if tc.get("name") == "option_verifier_tool":
                                already_called = True
                                break
            
            if not already_called:
                # ساخت tool call request
                new_message = {
                    "role": "assistant",
                    "content": "تحلیل گزینه‌ها با منابع...",
                    "tool_calls": [
                        {
                            "id": "verifier_001",
                            "name": "option_verifier_tool",
                            "arguments": {
                                "question": q,
                                "options_text": options,
                                "sources": ctx,
                            }
                        }
                    ]
                }
                
                messages_copy = messages.copy()
                messages_copy.append(new_message)
                
                return {
                    "messages": messages_copy,
                }
            
            # اگر tool call کردیم اما هنوز نتیجه نداریم، منتظر می‌مانیم
            return {}
    
    # 🔧 ساخت hints برای critic و verifier
    critic = state.get("critic_toon")
    critic_hint = ""
    if critic and critic.get("needs_revision", False):
        critic_hint = (
            f"Issue: {critic.get('issue','')}\n"
            f"Action: {critic.get('action','')}\n"
        )

    verifier_output = state.get("verifier_output")
    verifier_hint = ""
    if verifier_output and verifier_output.get("scores"):
        verifier_hint = "\n📊 نتایج تحلیل گزینه‌ها (ADVISORY - مستقلاً بررسی کنید):\n"
        for score in verifier_output["scores"]:
            verifier_hint += (
                f"گزینه {score['option_number']}: {score['support_level']} - "
                f"{score['reasoning'][:100]}...\n"
            )
        verifier_hint += (
            f"\n⚠️ گزینه پیشنهادی verifier: {verifier_output.get('recommended_answer')}\n"
            f"   ⚠️ این فقط یک نظر مشورتی است - شما باید مستقلاً منابع را بررسی کنید!\n"
        )

    # 🔧 تعریف system_msg و user_msg (باید خارج از همه if ها باشد)
    system_msg = (
        "You are an Iranian legal exam QA assistant.\n\n"
        
        "────────────────────────\n"
        "YOUR TASK\n"
        "────────────────────────\n"
        "Answer a multiple-choice legal question based ONLY on the provided SOURCES.\n\n"
        
        "────────────────────────\n"
        "CORE RULES\n"
        "────────────────────────\n"
        "1) Use ONLY the provided SOURCES - do not rely on general knowledge\n"
        "2) Answer entirely in Persian (explanation must be Persian)\n"
        "3) Choose exactly ONE option number (1, 2, 3, or 4)\n"
        "4) Provide a concise legal explanation (1-3 Persian sentences)\n"
        "5) Cite specific article/principle numbers from SOURCES in your explanation\n"
        "6) Output MUST be exactly ONE TOON table with ONE row\n\n"
        
        "────────────────────────\n"
        "HANDLING VERIFIER ANALYSIS (if provided)\n"
        "────────────────────────\n"
        "- Verifier analysis is ADVISORY, not binding\n"
        "- You MUST independently verify all claims against SOURCES\n"
        "- If verifier contradicts explicit article text in SOURCES, TRUST THE SOURCES\n"
        "- Pay special attention to:\n"
        "  • Numeric ranges (e.g., 'درجه چهار و بالاتر' = degrees ≥4, excludes 1-3)\n"
        "  • Legal terminology precision\n"
        "  • Exact article numbers and wording\n"
        "- Use verifier as a HINT to focus your attention, then verify yourself\n\n"
        
        "────────────────────────\n"
        "HANDLING CRITIC FEEDBACK (if provided)\n"
        "────────────────────────\n"
        "- If needs_revision=true, you MUST address the issue raised\n"
        "- Re-read the relevant SOURCES carefully\n"
        "- Correct any misquotes, wrong article numbers, or logical errors\n"
        "- Ensure your explanation is fully grounded in SOURCES\n\n"
        
        "────────────────────────\n"
        "OUTPUT FORMAT (STRICT)\n"
        "────────────────────────\n"
        "You MUST output EXACTLY this format:\n\n"
        
        "results{explanation,answer,confidence}:\n"
        "<explanation in Persian>,<1-4>,<1-5>\n\n"
        
        "Constraints:\n"
        "- explanation: 1-3 Persian sentences with article citations (no commas inside sentences - use semicolons)\n"
        "- answer: exactly one number: 1, 2, 3, or 4\n"
        "- confidence: 1-5 scale\n"
        "  • 5 = very certain (explicit article support)\n"
        "  • 4 = confident (clear source support)\n"
        "  • 3 = moderate (reasonable interpretation)\n"
        "  • 2 = uncertain (ambiguous sources)\n"
        "  • 1 = very uncertain (insufficient sources)\n\n"
        
        "Do not output anything else - no extra text, no explanations outside the TOON table.\n"
    )

    user_msg = f"""SOURCES:
{ctx}

QUESTION (Persian):
{q}

OPTIONS:
{options}

{verifier_hint}

{critic_hint}

────────────────────────
STEP-BY-STEP INSTRUCTIONS
────────────────────────
1. READ SOURCES carefully and identify relevant articles/principles
2. EVALUATE each option against the explicit text of SOURCES
3. If VERIFIER ANALYSIS exists:
   - Note its recommendations as hints
   - BUT independently verify each claim against SOURCES
   - If verifier says "SUPPORTED" but you cannot find explicit support in SOURCES, mark it as uncertain
4. If CRITIC FEEDBACK exists with needs_revision=true:
   - Address the specific issue mentioned
   - Re-verify article numbers and legal claims
   - Correct any errors in your previous explanation
5. SELECT the option with the STRONGEST explicit support in SOURCES
6. WRITE explanation in Persian with:
   - Specific article/principle citations
   - Clear reasoning
   - Use semicolons instead of commas in explanation text
7. ASSIGN confidence based on clarity of source support
8. OUTPUT exactly one TOON table as specified above

Remember: When in doubt between verifier recommendation and explicit SOURCES text, ALWAYS trust the SOURCES.
"""

    # 🔧 فراخوانی LLM
    resp = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.1,
    )

    draft_raw = resp.choices[0].message.content
    draft_toon: Dict[str, Any] = extract_toon_answer(draft_raw)

    # 🔧 افزایش revision_count فقط در revision
    rc = int(state.get("revision_count", 0))
    if critic and critic.get("needs_revision"):
        rc += 1

    return {
        "draft_raw": draft_raw,
        "draft_toon": draft_toon,
        "revision_count": rc,
        "critic_toon": None,  # پاک کردن critic قدیمی
    }
