from __future__ import annotations
from typing import Any, Dict
from uuid import uuid4

from openai import OpenAI
from langsmith.wrappers import wrap_openai
from langsmith import traceable
from dotenv import load_dotenv

from legal_multi_agent.state.schemas import MASharedState
from legal_multi_agent.utils.toon import extract_toon_answer
from legal_multi_agent.utils.logger import log_debug, log_info

import os

load_dotenv()

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL_ID = os.environ["MODEL"]

_raw_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)
client = wrap_openai(_raw_client)


def _has_pending_verifier_call(messages: list) -> bool:
    """
    بررسی اینکه آیا یک tool_call برای option_verifier_tool وجود دارد
    که هنوز پاسخ نگرفته است.
    """
    if not messages:
        return False
    
    # پیدا کردن آخرین tool_call برای verifier
    last_verifier_call_id = None
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                if tc.get("name") == "option_verifier_tool":
                    last_verifier_call_id = tc.get("id")
                    break
            if last_verifier_call_id:
                break
    
    if not last_verifier_call_id:
        return False
    
    # بررسی اینکه آیا پاسخ برای این call_id وجود دارد
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "tool":
            if msg.get("tool_call_id") == last_verifier_call_id:
                return False  # پاسخ پیدا شد، pending نیست
    
    return True  # tool_call وجود دارد ولی پاسخ نداریم


@traceable(name="reasoner_agent")
def reasoner_agent(state: MASharedState) -> MASharedState:
    """
    ایجنت استدلال‌گر که TOON پاسخ را می‌سازد.
    
    وظایف:
    1. اگر use_option_verifier=True: درخواست تحلیل گزینه‌ها از verifier
    2. دریافت نتیجه verifier و ذخیره در verifier_output
    3. استدلال نهایی بر اساس SOURCES + verifier hints (اختیاری) + critic feedback
    """
    log_debug("\n🔵 ═══ REASONER START ═══")
    
    q = state["question"]
    options = state["options_text"]
    ctx = state.get("context", "")
    
    log_debug(f"   📋 Question: {q[:60]}...")
    log_debug(f"   📄 Context length: {len(ctx)} chars")
    
    use_verifier = state.get("use_option_verifier", False)
    tool_results = state.get("tool_results") or {}
    messages = state.get("messages") or []
    
    log_debug(f"   🔧 use_verifier: {use_verifier}")
    log_debug(f"   🛠️  tool_results keys: {list(tool_results.keys())}")
    log_debug(f"   💬 messages count: {len(messages)}")
    
    # ═══════════════════════════════════════════════════════════
    # مرحله 1: مدیریت verifier (اگر فعال باشد)
    # ═══════════════════════════════════════════════════════════
    if use_verifier and ctx:
        log_debug("   ✓ Entering verifier logic (use_verifier=True and ctx exists)")
        
        verifier_output_in_state = state.get("verifier_output")
        log_debug(f"   📊 verifier_output in state: {bool(verifier_output_in_state)}")
        
        # ⭐ اگر verifier_output قبلاً در state ست شده → ادامه به استدلال
        if verifier_output_in_state:
            log_debug("   ✅ Verifier output already in state → proceeding to reasoning")
            # همه چیز آماده است - به استدلال می‌رویم
        
        # اگر verifier در tool_results هست ولی در state نیست → ذخیره کن
        elif "option_verifier_tool" in tool_results:
            log_info("🔵 Reasoner: Saving verifier output")
            log_debug("   💾 Verifier found in tool_results → saving to state")
            verifier_output = tool_results["option_verifier_tool"]
            log_debug(f"   ↩️  RETURNING: verifier_output")
            return {
                "verifier_output": verifier_output,
            }
        
        # اگر verifier هنوز اجرا نشده → tool_call بساز یا منتظر بمان
        else:
            log_debug("   ⚠️  Verifier not ready → checking pending calls")
            has_pending = _has_pending_verifier_call(messages)
            log_debug(f"   🔍 has_pending_verifier_call: {has_pending}")
            
            if has_pending:
                log_debug("   ⏳ Pending verifier call exists → waiting")
                log_debug("   ↩️  RETURNING: {} (empty dict)")
                return {}
            
            log_info("🔵 Reasoner: Requesting verifier tool")
            log_debug("   🆕 No pending call → creating new verifier tool_call")
            call_id = f"verifier_{uuid4().hex[:8]}"
            
            new_message = {
                "role": "assistant",
                "content": "🔍 در حال تحلیل گزینه‌ها بر اساس منابع...",
                "tool_calls": [
                    {
                        "id": call_id,
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
            
            log_debug(f"   📤 Created tool_call with id: {call_id}")
            log_debug(f"   ↩️  RETURNING: messages (with new tool_call)")
            return {
                "messages": messages_copy,
            }
    
    # ═══════════════════════════════════════════════════════════
    # مرحله 2: ساخت hints برای LLM
    # ═══════════════════════════════════════════════════════════
    log_debug("   ✅ All prerequisites met → proceeding to reasoning phase")
    
    # 2.1) Critic hint (اگر revision است)
    critic = state.get("critic_toon")
    critic_hint = ""
    if critic and isinstance(critic, dict) and critic.get("needs_revision", False):
        critic_hint = (
            f"⚠️ CRITIC FEEDBACK (you MUST address this):\n"
            f"Issue: {critic.get('issue', 'unknown')}\n"
            f"Action: {critic.get('action', 'revise')}\n"
        )
        log_debug(f"   📝 Critic hint: {len(critic_hint)} chars")
    else:
        log_debug(f"   ℹ️  No critic feedback")

    # 2.2) Verifier hint (اگر استفاده شده)
    verifier_output = state.get("verifier_output")
    verifier_hint = ""
    if verifier_output and isinstance(verifier_output, dict) and verifier_output.get("scores"):
        verifier_hint = "\n📊 VERIFIER ANALYSIS (ADVISORY - verify independently):\n"
        verifier_hint += "─" * 60 + "\n"
        
        for score in verifier_output["scores"]:
            verifier_hint += (
                f"• گزینه {score.get('option_number', '?')}: "
                f"{score.get('support_level', 'UNKNOWN')}\n"
                f"  └─ {score.get('reasoning', '')[:150]}...\n"
            )
        
        recommended = verifier_output.get("recommended_answer", "?")
        confidence = verifier_output.get("confidence", "?")
        verifier_hint += (
            f"\n💡 Recommended: گزینه {recommended} (confidence: {confidence}/5)\n"
            f"⚠️  IMPORTANT: This is advisory only - YOU must verify against SOURCES!\n"
            f"─" * 60 + "\n"
        )
        log_debug(f"   📊 Verifier hint: {len(verifier_hint)} chars")
    else:
        log_debug(f"   ℹ️  No verifier output")

    # ═══════════════════════════════════════════════════════════
    # مرحله 3: ساخت prompt و فراخوانی LLM
    # ═══════════════════════════════════════════════════════════
    
    log_debug("   🤖 Preparing LLM call...")
    
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

    try:
        log_debug(f"   📡 Calling OpenAI API...")
        log_debug(f"      Model: {MODEL_ID}")
        log_debug(f"      System msg: {len(system_msg)} chars")
        log_debug(f"      User msg: {len(user_msg)} chars")
        
        resp = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.1,
        )
        
        draft_raw = resp.choices[0].message.content
        log_debug(f"   ✅ LLM response received: {len(draft_raw)} chars")
        log_debug(f"   📄 First 200 chars: {draft_raw[:200]}")
        
        draft_toon: Dict[str, Any] = extract_toon_answer(draft_raw)
        log_debug(f"   🎯 Extracted TOON: {draft_toon}")
        
        if not draft_toon:
            log_debug(f"   ⚠️  WARNING: extract_toon_answer returned None/empty!")
        else:
            # نتیجه مهم (INFO level)
            log_info(f"🔵 Reasoner: Draft generated → answer={draft_toon['answer']}, confidence={draft_toon['confidence']}")
        
    except Exception as e:
        log_debug(f"   ❌ LLM CALL FAILED!")
        log_debug(f"   ❌ Exception: {e.__class__.__name__}: {str(e)}")
        log_debug(f"   ↩️  RETURNING: empty dict due to error")
        return {}

    # ═══════════════════════════════════════════════════════════
    # مرحله 4: افزایش revision_count (فقط در حالت revision)
    # ═══════════════════════════════════════════════════════════
    rc = int(state.get("revision_count", 0) or 0)
    if critic and isinstance(critic, dict) and critic.get("needs_revision", False):
        rc += 1
        log_debug(f"   🔄 Revision mode: incrementing revision_count to {rc}")

    log_debug(f"   ✅ REASONER COMPLETE")
    log_debug(f"   ↩️  RETURNING: draft_raw, draft_toon, revision_count={rc}")
    log_debug("🔵 ═══ REASONER END ═══\n")
    
    return {
        "draft_raw": draft_raw,
        "draft_toon": draft_toon,
        "revision_count": rc,
        "critic_toon": None,  # پاک کردن critic قدیمی برای دور بعدی
    }
