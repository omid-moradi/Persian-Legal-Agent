from __future__ import annotations
import json

from typing import Any, Dict, List, Optional
from uuid import uuid4
import os
import re

from openai import OpenAI
from langsmith.wrappers import wrap_openai
from langsmith import traceable
from dotenv import load_dotenv

from legal_multi_agent.state.schemas import MASharedState
from legal_multi_agent.utils.toon import extract_toon_answer
from legal_multi_agent.utils.logger import log_debug, log_info

load_dotenv()

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL_ID = os.environ["MODEL"]

_raw_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)
client = wrap_openai(_raw_client)


# ═══════════════════════════════════════════════════════════
# Helper: بررسی tool_call معلق برای verifier
# ═══════════════════════════════════════════════════════════
def _has_pending_verifier_call(messages: List[Dict[str, Any]]) -> bool:
    if not messages:
        return False

    # ── پیدا کردن آخرین tool_call مربوط به option_verifier_tool ──
    last_verifier_call_id = None
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                # ✅ فرمت صحیح OpenAI: tc["function"]["name"]
                func = tc.get("function", {})
                if func.get("name") == "option_verifier_tool":
                    last_verifier_call_id = tc.get("id")
            # ✅ FIX: خروج از حلقه خارجی فقط بعد از بررسی همه tc های یک message
            if last_verifier_call_id:
                break

    # اگر هیچ verifier call ای وجود ندارد → pending نیست
    if not last_verifier_call_id:
        return False

    # ── بررسی وجود tool response برای آن call_id ──────────────────
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "tool":
            if msg.get("tool_call_id") == last_verifier_call_id:
                return False  # جواب رسیده → pending نیست

    return True  # جواب نرسیده → هنوز pending است


# ═══════════════════════════════════════════════════════════
# Helper: بازیابی feedback منتقد
# ═══════════════════════════════════════════════════════════
def _get_latest_critic_feedback(
    state: MASharedState,
    messages: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    ابتدا از state["critic_toon"] می‌خواند.
    اگر supervisor آن را پاک کرده باشد، از آخرین پیام critic در messages بازیابی می‌کند.
    """
    critic = state.get("critic_toon")
    if isinstance(critic, dict) and critic.get("needs_revision", False):
        return critic

    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        if msg.get("name") != "critic":
            continue

        metadata = msg.get("metadata") or {}
        if metadata.get("agent") == "critic" and metadata.get("needs_revision", False):
            return {
                "needs_revision": True,
                "issue": metadata.get("issue", "بازبینی لازم است"),
                "action": metadata.get("action", "پاسخ باید بر اساس ایراد منتقد بازنویسی شود"),
            }

    return None


# ═══════════════════════════════════════════════════════════
# Helper: ساخت hint برای verifier
# ═══════════════════════════════════════════════════════════
def _build_verifier_hint(verifier_output: Optional[Dict[str, Any]]) -> str:
    if not verifier_output or not isinstance(verifier_output, dict):
        return ""

    scores = verifier_output.get("scores")
    if not scores:
        return ""

    lines = []
    lines.append("📊 تحلیل option verifier (صرفاً جنبه کمکی دارد و الزام‌آور نیست):")
    lines.append("شما باید همه ادعاها را مستقلاً با منابع کنترل کنید و اگر verifier با نص منبع تعارض داشت، فقط به منبع تکیه کنید.")

    for score in scores:
        option_number = score.get("option_number", "?")
        support_level = score.get("support_level", "UNKNOWN")
        reasoning = str(score.get("reasoning", "")).strip()
        reasoning = reasoning.replace("\n", " ")
        if len(reasoning) > 220:
            reasoning = reasoning[:220] + "..."
        lines.append(f"- گزینه {option_number}: {support_level} | توضیح verifier: {reasoning}")

    recommended = verifier_output.get("recommended_answer")
    if recommended not in (None, "", "?"):
        lines.append(f"- جمع‌بندی verifier: گزینه پیشنهادی {recommended} است؛ اما این فقط یک سرنخ است و ملاک نهایی نیست.")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
# Helper: ساخت hint برای critic
# ═══════════════════════════════════════════════════════════
def _build_critic_hint(
    critic_feedback: Optional[Dict[str, Any]],
    previous_draft_raw: str,
) -> str:
    if not critic_feedback or not critic_feedback.get("needs_revision", False):
        return ""

    issue = str(critic_feedback.get("issue", "")).strip()
    action = str(critic_feedback.get("action", "")).strip()

    parts = [
        "⚠️ بازبینی اجباری بر اساس نظر critic:",
        f"- ایراد اعلام‌شده: {issue}",
        f"- دستور اصلاح: {action}",
    ]

    if previous_draft_raw:
        parts.append("- پاسخ قبلی شما در ادامه آمده است و باید فقط در همان نقاط مورد ایراد اصلاح شود:")
        parts.append(previous_draft_raw)

    parts.append("در بازنویسی جدید، حتماً ایراد فوق را مستقیماً برطرف کن و پاسخ نهایی را دوباره در همان قالب ۵ گام + TOON ارائه بده.")
    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════
# Helper: پارس TOON جدید بدون confidence
# ═══════════════════════════════════════════════════════════
def _parse_reasoner_toon(raw_text: str) -> Optional[Dict[str, Any]]:
    if not raw_text or not isinstance(raw_text, str):
        return None

    # ── مرحله ۱: از extract_toon_answer کمک بگیر ─────────
    parsed = extract_toon_answer(raw_text)
    if isinstance(parsed, dict):
        answer = parsed.get("answer")
        explanation = parsed.get("explanation")
        if answer is not None and explanation:
            return {
                "explanation": str(explanation).strip(),
                "answer": str(answer).strip(),
            }

    # ── مرحله ۲: regex مقاوم — فاصله و بدون فاصله ───────
    # پشتیبانی از: results{explanation,answer}: و results{explanation, answer}:
    m = re.search(
        r"results\s*\{\s*explanation\s*,\s*answer(?:\s*,\s*confidence)?\s*\}\s*:\s*\n(.+)",
        raw_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        row = m.group(1).strip().splitlines()[0].strip()
        # آخرین بخش بعد از آخرین کاما = answer
        last_comma = row.rfind(",")
        if last_comma != -1:
            explanation = row[:last_comma].strip()
            answer = row[last_comma + 1:].strip()
            # پاکسازی کاراکترهای اضافه
            answer = re.sub(r"[^\d]", "", answer)
            if answer in {"1", "2", "3", "4"} and explanation:
                return {"explanation": explanation, "answer": answer}

    # ── مرحله ۳: fallback — جستجوی مستقیم گزینه در گام ۵ ─
    m2 = re.search(r"گزینه\s*([\d۱-۴])\s*(?:صحیح|درست|انتخاب)", raw_text)
    if not m2:
        # اعداد فارسی را هم چک کن
        m2 = re.search(r"گام\s*۵[^\n]*\n.*?گزینه\s*([\d۱-۴])", raw_text, flags=re.DOTALL)
    if m2:
        raw_answer = m2.group(1).strip()
        # تبدیل اعداد فارسی به لاتین
        fa_to_en = str.maketrans("۱۲۳۴", "1234")
        answer = raw_answer.translate(fa_to_en)
        step5_match = re.search(
            r"گام\s*۵[^\n:：]*[:：]\s*(.+?)(?:\nresults\{|\Z)",
            raw_text,
            flags=re.DOTALL,
        )
        explanation = ""
        if step5_match:
            explanation = " ".join(step5_match.group(1).strip().split())
        if explanation and answer in {"1", "2", "3", "4"}:
            return {"explanation": explanation, "answer": answer}

    # ── مرحله ۴: آخرین fallback — هر جایی در متن که گزینه ذکر شده ─
    last_match = None
    for m3 in re.finditer(r"گزینه\s*([\d۱-۴])", raw_text):
        last_match = m3
    if last_match:
        fa_to_en = str.maketrans("۱۲۳۴", "1234")
        answer = last_match.group(1).translate(fa_to_en)
        if answer in {"1", "2", "3", "4"}:
            return {
                "explanation": "استخراج از متن با روش fallback انجام شد.",
                "answer": answer,
            }

    return None


# ═══════════════════════════════════════════════════════════
# System Prompt — نسخه اصلاح‌شده reasoner
# ═══════════════════════════════════════════════════════════
SYSTEM_MSG = """# Iranian Legal MCQ Reasoning System (RAG-Augmented)

You are a **senior Iranian lawyer and legal examiner** with deep expertise in Iranian statutory law.
Your reasoning must reflect how a practicing lawyer thinks: precise, article-grounded, and conclusive.

---
## Task
- Language: **Persian (فارسی)** for all reasoning and output.
- You will receive relevant legal articles and precedents retrieved from a legal database (RAG context).
- **Prioritize the RAG context** when it directly addresses the question; use your background knowledge only if the context is genuinely insufficient.
- Analyze the question as a legal professional and select the **single correct option (1–4)**.
- You MUST follow the **exact 5-step reasoning structure** below.
- **Critical E2P rule**: In steps 1–4, do NOT reveal which option is correct. Evaluate all options neutrally. Only in Step 5 may you name the correct answer for the first time.
- Do NOT write short answers. Each step must be fully developed and legally precise.

---
## 5-Step Legal Reasoning Structure (MANDATORY)

**گام ۱ — تحلیل موضوع (Issue Spotting)**
مسئله حقوقی دقیق مطرح‌شده در سؤال را شناسایی کن.
روشن کن سؤال مربوط به کدام شاخه حقوق است و نزاع یا رابطه حقوقی اصلی چیست.

**گام ۲ — احصاء قواعد حاکم (Rule Identification)**
مواد قانونی، اصول، یا آرای لازم‌الاتباع مرتبط را به‌طور دقیق و با نام قانون ذکر کن.
الزاماً با این قالب بنویس:
«ماده X قانون Y» یا «اصل X قانون اساسی».
اگر در RAG context ماده‌ای مستقیماً مرتبط وجود دارد، حتماً باید صریحاً در همین گام ذکر شود.

**گام ۳ — تطبیق قاعده با گزینه‌ها (Neutral Application)**
برای هر چهار گزینه، جداگانه بیان کن:
الف) آن گزینه چه ادعایی دارد؛
ب) مواد و قواعد شناسایی‌شده درباره آن ادعا چه می‌گویند.
در این گام مطلقاً از واژه‌هایی مانند «صحیح»، «غلط»، «پاسخ درست» یا هر نشانه‌ای که جواب را لو بدهد استفاده نکن.

**گام ۴ — تمییز گزینه‌ها (Critical Discrimination)**
صرفاً بر اساس تحلیل گام ۳، عیب یا انحراف حقوقی هر گزینه‌ای را که از نص قانونی فاصله دارد روشن کن.
هنوز نباید پاسخ نهایی را اعلام کنی.
اگر یک گزینه فاقد عیب است، فقط توضیح بده که با قاعده حاکم تعارضی نشان داده نشده است؛ بدون اعلام اینکه همان پاسخ نهایی است.

**گام ۵ — نتیجه‌گیری قضایی (Legal Conclusion)**
این نخستین و تنها جایی است که مجاز به اعلام پاسخ درست هستی.
در یک جمله دقیق و روشن اعلام کن کدام گزینه صحیح است و دلیل کنترل‌کننده آن چیست.

---
## Revision Handling
اگر بازخورد critic داده شده باشد:
- باید همان ایراد مشخص را مستقیماً رفع کنی.
- شماره مواد، شروط، استثناها، حدود، مراتب، مهلت‌ها و اصطلاحات فنی را دوباره با منبع تطبیق بده.
- اگر verifier نظری داده، آن فقط جنبه کمکی دارد و هیچ اعتباری بالاتر از نص منبع ندارد.

---
## Mandatory Output Format
Your response MUST contain exactly two parts in this order:

Part 1 — 5-Step Legal Reasoning:
گام ۱ — تحلیل موضوع: ...
گام ۲ — احصاء قواعد حاکم: ...
گام ۳ — تطبیق قاعده با گزینه‌ها: ...
گام ۴ — تمییز گزینه‌ها: ...
گام ۵ — نتیجه‌گیری قضایی: ...

Part 2 — TOON (immediately after Part 1):
results{explanation,answer}:
خلاصه حقوقی یک‌خطی با ذکر ماده قانونی,X

---
## Strict Rules
- All 5 steps are mandatory.
- Article number AND law name must appear in گام ۲.
- All 4 options must be analyzed in گام ۳.
- In steps 1–4 do NOT reveal the correct option.
- The correct answer must appear for the first time only in گام ۵.
- The TOON header must be exactly: results{explanation,answer}:
- The TOON row must contain explanation on a single line, then a comma, then answer digit (1–4).
- No confidence score anywhere.
- No code block, no JSON, no markdown table, no extra text after the TOON row.
"""


# ═══════════════════════════════════════════════════════════
# Reasoner Agent
# ═══════════════════════════════════════════════════════════
@traceable(name="reasoner_agent")
def reasoner_agent(state: MASharedState) -> MASharedState:
    """
    ایجنت استدلال‌گر:
    1) در صورت فعال بودن verifier، اجرای آن را درخواست می‌کند یا نتیجه‌اش را می‌خواند.
    2) پاسخ نهایی را با ساختار ۵ گام + TOON تولید می‌کند.
    3) کل خروجی خام reasoner را در messages ذخیره می‌کند تا trace کامل بماند.
    """
    log_debug("\n🔵 ═══ REASONER START ═══")

    q = state["question"]
    options = state["options_text"]
    ctx = state.get("context", "")
    tool_results = state.get("tool_results") or {}
    messages = list(state.get("messages") or [])
    use_verifier = state.get("use_option_verifier", False)

    log_debug(f" 📋 Question: {q[:80]}...")
    log_debug(f" 📄 Context length: {len(ctx)} chars")
    log_debug(f" 🔧 use_verifier: {use_verifier}")
    log_debug(f" 🛠️ tool_results keys: {list(tool_results.keys())}")
    log_debug(f" 💬 messages count: {len(messages)}")

    # ── مرحله ۱: مدیریت verifier ──────────────────────────────────────
    if use_verifier and ctx:
        log_debug(" ✓ Entering verifier logic")

        verifier_output_in_state = state.get("verifier_output")
        if verifier_output_in_state:
            log_debug(" ✅ verifier_output already available in state")
        elif "option_verifier_tool" in tool_results:
            log_info("🔵 Reasoner: Saving verifier output from tool_results")
            return {
                "verifier_output": tool_results["option_verifier_tool"],
            }
        else:
            has_pending = _has_pending_verifier_call(messages)
            log_debug(f" 🔍 has_pending_verifier_call: {has_pending}")

            if has_pending:
                log_debug(" ⏳ Pending verifier call exists → waiting")
                log_debug("🔵 ═══ REASONER END ═══\n")
                return {}

            call_id = f"verifier_{uuid4().hex[:8]}"
            verifier_request_message = {
                "role": "assistant",
                "name": "reasoner",
                "content": "🔍 در حال تحلیل گزینه‌ها بر اساس منابع...",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "option_verifier_tool",
                            "arguments": json.dumps(
                                {
                                    "question": q,
                                    "options_text": options,
                                    "sources": ctx,
                                },
                                ensure_ascii=False,
                            ),
                        },
                    }
                ],
                "metadata": {
                    "agent": "reasoner",
                    "phase": "request_option_verifier",
                    "tool_name": "option_verifier_tool",
                },
            }

            messages.append(verifier_request_message)
            log_info("🔵 Reasoner: Requesting verifier tool")
            log_debug(f" 📤 Created verifier tool_call with id={call_id}")
            log_debug("🔵 ═══ REASONER END ═══\n")
            return {
                "messages": messages,
            }

    # ── مرحله ۲: ساخت hintها ──────────────────────────────────────────
    verifier_output = state.get("verifier_output")
    verifier_hint = _build_verifier_hint(verifier_output)

    critic_feedback = _get_latest_critic_feedback(state, messages)
    previous_draft_raw = str(state.get("draft_raw", "") or "")
    critic_hint = _build_critic_hint(critic_feedback, previous_draft_raw)

    log_debug(f" 📊 verifier_hint length: {len(verifier_hint)}")
    log_debug(f" 📝 critic_hint length: {len(critic_hint)}")

    # ── مرحله ۳: ساخت user prompt ─────────────────────────────────────
    user_parts = [
        "📚 قوانین و آراء مرتبط (RAG Context):",
        "از مواد و آراء زیر که از پایگاه داده قانونی بازیابی شده‌اند در استدلال خود استفاده کن.",
        "اگر ماده‌ای مستقیماً مرتبط با سؤال باشد، در گام ۲ به آن استناد کن.",
        "",
        ctx if ctx else "هیچ متن بازیابی‌شده‌ای ارائه نشده است.",
        "",
        "---",
        "",
        f"سؤال:\n{q}",
        "",
        f"گزینه‌ها:\n{options}",
    ]

    if verifier_hint:
        user_parts.extend([
            "",
            "---",
            "",
            verifier_hint,
        ])

    if critic_hint:
        user_parts.extend([
            "",
            "---",
            "",
            critic_hint,
        ])

    user_parts.extend([
        "",
        "---",
        "",
        "FOLLOW ALL 5 STEPS, then output the TOON.",
        "یادآوری مهم: در گام‌های ۱ تا ۴ به‌هیچ‌وجه پاسخ صحیح را افشا نکن و فقط در گام ۵ برای نخستین بار گزینه صحیح را اعلام کن.",
    ])

    user_msg = "\n".join(user_parts)

    # ── مرحله ۴: فراخوانی مدل ─────────────────────────────────────────
    try:
        log_debug(" 🤖 Calling LLM for legal reasoning...")
        log_debug(f" Model: {MODEL_ID}")
        log_debug(f" System msg length: {len(SYSTEM_MSG)}")
        log_debug(f" User msg length: {len(user_msg)}")

        resp = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=10240
        )

        draft_raw = (resp.choices[0].message.content or "").strip()
        log_debug(f" ✅ LLM response received: {len(draft_raw)} chars")

    except Exception as e:
        log_debug(" ❌ LLM CALL FAILED")
        log_debug(f" ❌ Exception: {e.__class__.__name__}: {str(e)}")
        log_debug("🔵 ═══ REASONER END ═══\n")
        return {}

    # ── مرحله ۵: Parse TOON ────────────────────────────────────────────
    draft_toon = _parse_reasoner_toon(draft_raw)

    if not draft_toon:
        log_debug(" ⚠️ Failed to parse TOON output")
        draft_toon = {
            "explanation": "استخراج خودکار ردیف TOON از پاسخ مدل با شکست مواجه شد و نیاز به بازبینی قالب خروجی وجود دارد.",
            "answer": "",
        }

    answer = str(draft_toon.get("answer", "")).strip()
    explanation = str(draft_toon.get("explanation", "")).strip()

    # ── مرحله ۶: ذخیره کامل پیام reasoner در messages ─────────────────
    reasoner_message = {
        "role": "assistant",
        "name": "reasoner",
        "content": draft_raw,
        "metadata": {
            "agent": "reasoner",
            "phase": "final_reasoning",
            "answer": answer,
            "used_verifier": bool(verifier_output),
            "is_revision": bool(critic_feedback and critic_feedback.get("needs_revision", False)),
            "revision_issue": (critic_feedback or {}).get("issue"),
        },
    }
    messages.append(reasoner_message)

    if answer in {"1", "2", "3", "4"}:
        log_info(f"🔵 Reasoner: Draft generated → answer={answer}")
    else:
        log_info("🔵 Reasoner: Draft generated but answer parsing is incomplete")

    log_debug(" ✅ REASONER COMPLETE")
    log_debug(f" 🧾 Parsed draft_toon: {draft_toon}")
    log_debug("🔵 ═══ REASONER END ═══\n")

    return {
        "draft_raw": draft_raw,
        "draft_toon": {
            "explanation": explanation,
            "answer": answer,
        },
        "messages": messages,
    }