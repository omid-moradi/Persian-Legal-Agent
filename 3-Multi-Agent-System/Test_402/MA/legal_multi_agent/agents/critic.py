from __future__ import annotations

from typing import Any, Dict, List, Optional
import os
import re
from dotenv import load_dotenv

from openai import OpenAI
from langsmith.wrappers import wrap_openai
from langsmith import traceable

from legal_multi_agent.state.schemas import MASharedState
from legal_multi_agent.utils.toon import extract_toon_critic
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
# Helper: بررسی وجود استناد معتبر — نسخه اصلاح‌شده
# ═══════════════════════════════════════════════════════════
def _has_valid_citation(txt: str) -> bool:
    """
    وجود استناد معتبر را بررسی می‌کند.
    
    تغییرات:
    - فرمت [N] و [۱] علاوه بر [منبع N] پذیرفته می‌شود
    - بررسی «اصل» حذف شد (خیلی عمومی بود و false positive می‌داد)
    - حداقل یک فرمت از فرمت‌های زیر کافی است
    """
    # فرمت استاندارد [منبع N]
    if re.search(r"\[منبع\s*[\d۰-۹]+\]", txt):
        return True
    # فرمت مختصر [N] — مدل گاهی این را استفاده می‌کند
    if re.search(r"\[[\d۰-۹]+\]", txt):
        return True
    # ذکر ماده با شماره
    if re.search(r"ماده\s*[\d۰-۹]+", txt):
        return True
    return False


# ═══════════════════════════════════════════════════════════
# Helper: ساخت verifier hint
# ═══════════════════════════════════════════════════════════
def _build_verifier_hint(verifier: Optional[Dict[str, Any]]) -> str:
    if not verifier or not isinstance(verifier, dict):
        return ""

    scores = verifier.get("scores")
    if not scores:
        return ""

    lines = [
        "📊 اطلاعات verifier (صرفاً جنبه کمکی دارد — الزام‌آور نیست):",
        "این اطلاعات فقط برای راهنمایی است. critic باید ادعاها را مستقلاً با SOURCES تطبیق دهد.",
    ]

    supp_opts = [
        str(s.get("option_number", "?"))
        for s in scores
        if str(s.get("support_level", "")).upper() == "SUPPORTED"
    ]
    rec = verifier.get("recommended_answer")

    if supp_opts:
        lines.append(f"- گزینه‌های دارای پشتوانه منبعی (طبق verifier): {', '.join(supp_opts)}")
    if rec:
        lines.append(f"- گزینه پیشنهادی verifier: {rec}")
    lines.append(
        "⚠️ critic فقط در صورتی مجاز به درخواست revision است که خطا را با ماده/[منبع N] مستند کرده باشد."
    )

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
# System Prompt — نسخه اصلاح‌شده
# ═══════════════════════════════════════════════════════════
SYSTEM_MSG = """# Iranian Legal QA Auditor — 5-Step Review

You are a **senior Iranian legal auditor** (بازرس ارشد حقوقی) reviewing a reasoner AI's answer to a bar-exam multiple-choice question.

Your mission:
Detect errors that are **directly and explicitly contradicted** by the provided SOURCES (statutes, codes, or binding judicial decisions).

You are NOT allowed to request revision based on:
- Stylistic or wording differences that do not change legal meaning
- Reasonable doctrinal interpretation when SOURCES are silent or ambiguous
- Interpretive choices not expressly prohibited by SOURCES
- Verifier suggestions that lack direct SOURCES contradiction

**When SOURCES are silent or ambiguous → NO ERROR.**
**When you find a clear contradiction with SOURCES → ERROR FOUND (with full citation).**

---
## Mandatory 5-Step Review Structure

**گام ۱ — شناسایی مسئله حقوقی**
هسته اصلی سؤال را در یک جمله دقیق فارسی بیان کن.
شاخه حقوقی (مدنی، کیفری، تجاری، آیین دادرسی، ...) را مشخص کن.

---
**گام ۲ — استخراج قاعده حاکم از SOURCES**
ماده یا اصل حاکم بر موضوع را از SOURCES به فرمت زیر نقل کن:
[منبع N] ماده/اصل X — متن یا مضمون دقیق ماده

اگر هیچ ماده‌ای مستقیماً در SOURCES حاکم نیست:
«هیچ ماده حاکم مستقیمی در SOURCES یافت نشد.»
← در این صورت در گام ۴ حتماً NO ERROR صادر کن.

---
**گام ۳ — مقایسه پاسخ reasoner با قاعده حاکم**
پاسخ reasoner را در این موارد بررسی کن:
الف) آیا شماره گزینه انتخابی با متن ماده سازگار است؟
ب) آیا شماره مواد استنادشده درست است؟
ج) آیا اصطلاحات فنی حقوقی به‌درستی به‌کار رفته‌اند؟
د) آیا اعداد، مهلت‌ها، حدود، نسبت‌ها با نص ماده منطبق است؟
ه) آیا شروط و استثناهای قاعده به‌درستی بیان شده‌اند؟

---
**گام ۴ — حکم (Verdict)**
دقیقاً یکی از دو گزینه زیر را بنویس:

اگر پاسخ با SOURCES سازگار است:
«NO ERROR: [توضیح فارسی]»

اگر خطای صریح و قابل مستندسازی یافتی:
«ERROR FOUND: [توضیح فارسی کامل شامل: شماره منبع + ماده + تعارض صریح]»

قوانین صدور ERROR FOUND:
۱. باید [منبع N] + شماره ماده + تعارض صریح ذکر شود.
۲. باید نشان دهی reasoner دقیقاً چه گفته که با نص ماده تعارض دارد.
۳. اگر هر دو شرط بالا را نمی‌توانی محقق کنی → NO ERROR بنویس.

---
**گام ۵ — دستور اصلاح**
فقط در صورت ERROR FOUND:
- کدام گزینه بازنگری شود و چرا؟
- کدام ماده مجدداً خوانده شود؟
- چه اصلاحی در استدلال لازم است؟

در صورت NO ERROR:
«نیازی به اصلاح نیست.»

---
## Output Format (STRICT — این بخش بسیار مهم است)

**پس از تکمیل همه ۵ گام، دقیقاً یک خط TOON بنویس:**

results{needs_revision,issue,action}:true,<مشکل فارسی>,<دستور اصلاح فارسی>

یا:

results{needs_revision,issue,action}:false,خطای صریحی یافت نشد,پاسخ قابل قبول است و نیازی به تغییر ندارد

قوانین TOON:
- needs_revision: دقیقاً 'true' یا 'false' — بدون فاصله — بدون ترجمه
- issue: عبارت فارسی کوتاه از مشکل
- action: دستور اصلاح فارسی
- **از کاما داخل issue یا action استفاده نکن — به‌جای کاما از نقطه‌ویرگول استفاده کن**
- **هیچ متنی بعد از خط TOON نباشد**
- **خط TOON باید آخرین خط پاسخ تو باشد**
"""


# ═══════════════════════════════════════════════════════════
# Critic Agent — نسخه اصلاح‌شده
# ═══════════════════════════════════════════════════════════
@traceable(name="critic_agent")
def critic_agent(state: MASharedState) -> MASharedState:
    """
    ایجنت منتقد — نسخه اصلاح‌شده.

    تغییرات کلیدی:
    1. draft_raw با fallback به draft_toon.explanation خوانده می‌شود
    2. parse fail → log بهتر + retry یک بار
    3. citation check با فرمت‌های بیشتر
    4. prompt واضح‌تر برای خط TOON
    5. اگر parse دوباره هم شکست خورد → fallback NO ERROR با log صریح
    """
    log_debug("\n🔍 ═══ CRITIC START ═══")

    q = state["question"]
    options = state["options_text"]
    ctx = state.get("context", "")
    messages: List[Dict[str, Any]] = list(state.get("messages") or [])

    # ── draft_raw با fallback ────────────────────────────────────────────
    draft_raw = state.get("draft_raw", "")
    if not draft_raw:
        # fallback: از draft_toon بساز
        draft_toon = state.get("draft_toon") or {}
        explanation = draft_toon.get("explanation", "")
        answer = draft_toon.get("answer", "")
        if explanation or answer:
            draft_raw = f"گزینه انتخابی: {answer}\n\nاستدلال:\n{explanation}"
            log_debug(f" ⚠️ draft_raw خالی بود — از draft_toon ساخته شد ({len(draft_raw)} chars)")
        else:
            log_debug(" ⚠️ draft_raw و draft_toon هر دو خالی هستند")

    log_debug(f" 📋 Question: {q[:80]}...")
    log_debug(f" 📄 Draft length: {len(draft_raw)} chars")

    # ── verifier hint ────────────────────────────────────────────────────
    verifier = state.get("verifier_output")
    verifier_hint = _build_verifier_hint(verifier)

    # ── User prompt ──────────────────────────────────────────────────────
    user_parts = [
        "SOURCES:",
        ctx if ctx else "هیچ متن بازیابی‌شده‌ای ارائه نشده است.",
        "",
        "---",
        "",
        f"سؤال:\n{q}",
        "",
        f"گزینه‌ها:\n{options}",
        "",
        "---",
        "",
        f"پاسخ کامل reasoner:\n{draft_raw}",
    ]

    if verifier_hint:
        user_parts.extend(["", "---", "", verifier_hint])

    user_parts.extend([
        "",
        "---",
        "",
        "اکنون بررسی ۵ گامه‌ات را انجام بده.",
        "⚠️ یادآوری مهم: آخرین خط پاسخت باید دقیقاً به این شکل باشد:",
        "results{needs_revision,issue,action}:true/false,<issue>,<action>",
        "هیچ متنی بعد از این خط ننویس.",
    ])

    user_msg = "\n".join(user_parts)

    # ── فراخوانی مدل ────────────────────────────────────────────────────
    log_debug(f" 📡 Calling LLM | model={MODEL_ID} | user_msg={len(user_msg)} chars")

    resp = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
        max_tokens=4096,
    )

    critic_raw: str = (resp.choices[0].message.content or "").strip()
    log_debug(f" ✅ LLM response: {len(critic_raw)} chars")

    # ── Parse TOON — با retry ────────────────────────────────────────────
    critic_toon: Optional[Dict[str, Any]] = extract_toon_critic(critic_raw)

    if critic_toon is None:
        log_debug(" ⚠️ Parse شکست خورد — retry با prompt ساده‌تر")
        log_info("🔍 Critic: TOON parse failed — retrying with simplified prompt")

        # retry: فقط از critic بخواه خط TOON را بنویسد
        retry_prompt = (
            f"متن زیر پاسخ یک بازرس حقوقی است:\n\n{critic_raw}\n\n"
            "بر اساس این پاسخ، دقیقاً یک خط TOON بنویس:\n"
            "اگر خطا یافت شد:\n"
            "results{needs_revision,issue,action}:true,<خلاصه مشکل>,<دستور اصلاح>\n"
            "اگر خطایی نبود:\n"
            "results{needs_revision,issue,action}:false,خطای صریحی یافت نشد,پاسخ قابل قبول است و نیازی به تغییر ندارد\n"
            "فقط همین یک خط را بنویس و چیز دیگری ننویس."
        )

        retry_resp = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": retry_prompt}],
            temperature=0.0,
            max_tokens=256,
        )
        retry_raw = (retry_resp.choices[0].message.content or "").strip()
        log_debug(f" 🔄 Retry response: {retry_raw[:200]}")

        critic_toon = extract_toon_critic(retry_raw)
        if critic_toon is not None:
            # خط TOON را به critic_raw اضافه کن
            critic_raw = critic_raw + "\n\n[TOON از retry]\n" + retry_raw
            log_info("🔍 Critic: TOON parse succeeded after retry")
        else:
            log_debug(" ❌ Retry هم شکست خورد → fail-safe NO ERROR")
            log_info("🔍 Critic: TOON parse failed after retry → NO ERROR fallback")
            critic_toon = {
                "needs_revision": False,
                "issue": "خطای پارس TOON critic — به‌صورت محافظه‌کارانه NO ERROR اعمال شد",
                "action": "پاسخ قابل قبول است و نیازی به تغییر ندارد",
            }

    # ── Post-check: citation validation ─────────────────────────────────
    needs_revision = bool(critic_toon.get("needs_revision"))

    if needs_revision:
        log_debug(" 🔍 Checking citation validity...")
        if not _has_valid_citation(critic_raw):
            log_debug(" ⚠️ needs_revision=true اما citation معتبر نیست → NO ERROR")
            log_info("🔍 Critic: Revision rejected — no valid citation found")
            critic_toon = {
                "needs_revision": False,
                "issue": "درخواست revision بدون استناد معتبر رد شد",
                "action": "پاسخ قابل قبول است و نیازی به تغییر ندارد",
            }
            needs_revision = False
        else:
            log_debug(" ✅ Citation valid — revision approved")
            log_info(f"🔍 Critic: Revision approved — issue: {critic_toon.get('issue', '')[:80]}")
    else:
        log_info("🔍 Critic: No revision needed")

    # ── ذخیره در messages ────────────────────────────────────────────────
    messages.append({
        "role": "assistant",
        "name": "critic",
        "content": critic_raw,
        "metadata": {
            "agent": "critic",
            "needs_revision": needs_revision,
            "issue": critic_toon.get("issue"),
            "action": critic_toon.get("action"),
        },
    })

    log_debug(f" 📊 Final: needs_revision={needs_revision} | issue={critic_toon.get('issue','')[:60]}")
    log_debug("🔍 ═══ CRITIC END ═══\n")

    return {
        "critic_raw": critic_raw,
        "critic_toon": critic_toon,
        "messages": messages,
    }