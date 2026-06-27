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
# Helper: بررسی وجود استناد معتبر در متن critic
# ═══════════════════════════════════════════════════════════
def _has_valid_citation(txt: str) -> bool:
    """
    وجود [منبع N] در متن critic را بررسی می‌کند.
    بدون این استناد هیچ revision معتبر نیست.
    """
    return bool(re.search(r"\[منبع\s*\d+\]", txt))


# ═══════════════════════════════════════════════════════════
# Helper: ساخت verifier hint بدون confidence
# ═══════════════════════════════════════════════════════════
def _build_verifier_hint(verifier: Optional[Dict[str, Any]]) -> str:
    """
    اطلاعات verifier را به‌صورت کمکی و بدون confidence به critic می‌دهد.
    """
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
        lines.append(
            f"- گزینه پیشنهادی verifier: {rec} "
            f"(این پیشنهاد بدون citation صریح از SOURCES ارزشی برای تأیید یا رد ندارد)"
        )
    lines.append(
        "⚠️ critic فقط در صورتی مجاز به درخواست revision است که خطا را با [منبع N] مستند کرده باشد."
    )

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
# System Prompt — نسخه اصلاح‌شده critic
# ═══════════════════════════════════════════════════════════
SYSTEM_MSG = """# Iranian Legal QA Auditor — 5-Step Review

You are a **senior Iranian legal auditor** (بازرس ارشد حقوقی) reviewing a reasoner AI's answer to a bar-exam multiple-choice question.

Your one and only mission:
Detect errors that are **directly and explicitly contradicted** by the provided SOURCES (statutes, codes, or binding judicial decisions).

You are NOT allowed to request revision based on:
- Stylistic or wording differences that do not change legal meaning
- Reasonable doctrinal interpretation when SOURCES are silent or ambiguous
- Interpretive choices not expressly prohibited by SOURCES
- Scholarly disagreement without explicit statutory contradiction
- Verifier suggestions that lack direct SOURCES contradiction

**When in doubt → NO ERROR.**

---
## Mandatory 5-Step Review Structure

You MUST complete all five steps fully before writing the TOON. Do NOT skip any step. Each step must be written in Persian.

---
**گام ۱ — شناسایی مسئله حقوقی**
در یک جمله دقیق فارسی بگو: هسته اصلی این سؤال چه قاعده یا موضوع حقوقی است؟
شاخه حقوقی (مدنی، کیفری، تجاری، آیین دادرسی، ...) و ماهیت رابطه یا اختلاف را مشخص کن.

---
**گام ۲ — استخراج قاعده حاکم از SOURCES**
ماده یا اصل حاکم بر موضوع را از SOURCES با فرمت زیر نقل یا بازنویسی دقیق کن:
[منبع N] ماده/اصل X — متن یا مضمون دقیق ماده
اگر هیچ ماده‌ای مستقیماً در SOURCES حاکم نیست، صریحاً بنویس:
«هیچ ماده حاکم مستقیمی در SOURCES یافت نشد.»
در این صورت در گام ۴ باید NO ERROR صادر کنی.

---
**گام ۳ — مقایسه پاسخ reasoner با قاعده حاکم**
پاسخ reasoner را در این نقاط بررسی کن:
الف) آیا شماره گزینه انتخابی با متن ماده سازگار است؟
ب) آیا شماره مواد استناد‌شده در توضیح درست است؟
ج) آیا اصطلاحات فنی حقوقی به‌درستی به‌کار رفته‌اند؟
د) آیا اعداد و ارقام (مهلت‌ها، حدود، درجات، نسبت‌ها، شروط کمّی) با نص ماده منطبق است؟
ه) آیا شروط استثناها، یا محدوده اعمال قاعده به‌درستی بیان شده‌اند؟
برای هر مورد، متن گفته‌شده توسط reasoner را با متن SOURCES کنار هم بگذار.

---
**گام ۴ — حکم (Verdict)**
دقیقاً یکی از دو گزینه زیر را بنویس:

الف) اگر پاسخ با SOURCES سازگار است (حتی اگر ناقص باشد):
«NO ERROR: [توضیح فارسی چرا اشکال صریحی یافت نشد]»

ب) اگر خطای صریح قابل مستندسازی یافتی:
«ERROR FOUND: [توضیح فارسی کامل]»

قوانین صدور ERROR FOUND:
- حتماً باید [منبع N] + شماره ماده + متن دقیق ماده ذکر شود.
- حتماً باید نشان دهی reasoner دقیقاً چه گفته که با نص ماده تعارض دارد.
- اگر نمی‌توانی هر دو شرط بالا را با هم محقق کنی → باید NO ERROR بنویسی.

---
**گام ۵ — دستور اصلاح (فقط در صورت ERROR FOUND)**
اگر در گام ۴ ERROR FOUND نوشتی، دستور اصلاح واضح و چند مرحله‌ای بنویس:
- کدام گزینه باید بازنگری شود و چرا؟
- کدام ماده باید مجدداً خوانده شود؟
- دقیقاً چه اصلاحی در استدلال یا گزینه انتخابی لازم است؟
- reasoner باید پاسخ را از کدام نقطه بازنویسی کند؟

اگر در گام ۴ NO ERROR نوشتی، فقط بنویس: «نیازی به اصلاح نیست.»

---
## Output Format (STRICT)

بعد از تکمیل همه ۵ گام، دقیقاً یک TOON به این شکل بنویس:

results{needs_revision,issue,action}:
<true/false>,<توضیح فارسی مسئله>,<دستور اصلاح فارسی چند مرحله‌ای>

محدودیت‌ها:
- needs_revision: دقیقاً 'true' یا 'false' (حروف کوچک انگلیسی).
- issue: یک عبارت فارسی واضح از مشکل شناسایی‌شده؛ اگر خطایی نیست بنویس: «خطای صریحی یافت نشد».
- action: دستور اصلاح کامل فارسی با جزئیات کافی برای اقدام؛ اگر اصلاحی لازم نیست بنویس: «پاسخ قابل قبول است و نیازی به تغییر ندارد».
- هیچ کاما داخل issue یا action استفاده نکن — به‌جای کاما از نقطه‌ویرگول استفاده کن.
- هیچ متنی بعد از ردیف TOON نباید باشد.
"""


# ═══════════════════════════════════════════════════════════
# Critic Agent
# ═══════════════════════════════════════════════════════════
@traceable(name="critic_agent")
def critic_agent(state: MASharedState) -> MASharedState:
    """
    ایجنت منتقد که پاسخ reasoner را در ۵ گام بررسی می‌کند.

    تغییرات نسبت به نسخه قبل:
    - confidence کامل حذف شد از verifier_hint
    - issue در TOON حالا فارسی و مفصل است
    - action در TOON حالا دستور چندمرحله‌ای واضح است
    - ساختار ۵ گام prompt عمق بیشتری دارد
    - گام‌های ۳ و ۴ اعداد، مهلت‌ها، درجات و شروط را صریحاً بررسی می‌کنند
    """
    log_debug("\n🔍 ═══ CRITIC START ═══")

    q = state["question"]
    options = state["options_text"]
    ctx = state.get("context", "")
    draft_raw = state.get("draft_raw", "")
    messages: List[Dict[str, Any]] = list(state.get("messages") or [])

    log_debug(f" 📋 Question: {q[:80]}...")
    log_debug(f" 📄 Draft length: {len(draft_raw)} chars")

    # ── verifier hint بدون confidence ──────────────────────────────────
    verifier = state.get("verifier_output")
    verifier_hint = _build_verifier_hint(verifier)
    log_debug(f" 📊 verifier_hint length: {len(verifier_hint)}")

    # ── User prompt ─────────────────────────────────────────────────────
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
        f"پاسخ کامل reasoner (شامل استدلال ۵ گام و TOON):\n{draft_raw}",
    ]

    if verifier_hint:
        user_parts.extend([
            "",
            "---",
            "",
            verifier_hint,
        ])

    user_parts.extend([
        "",
        "---",
        "",
        "اکنون بررسی ۵ گامه‌ات را انجام بده:",
        "",
        "گام ۱ — شناسایی مسئله حقوقی:",
        "گام ۲ — استخراج قاعده حاکم از SOURCES:",
        "گام ۳ — مقایسه پاسخ reasoner با قاعده حاکم:",
        "گام ۴ — حکم (Verdict):",
        "گام ۵ — دستور اصلاح:",
        "",
        "سپس TOON را بنویس.",
    ])

    user_msg = "\n".join(user_parts)

    # ── فراخوانی مدل ────────────────────────────────────────────────────
    log_debug(" 📡 Calling LLM for critic review...")
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
        max_tokens=4096
    )

    critic_raw: str = (resp.choices[0].message.content or "").strip()
    log_debug(f" ✅ LLM response: {len(critic_raw)} chars")

    # ── Parse TOON ──────────────────────────────────────────────────────
    critic_toon: Optional[Dict[str, Any]] = extract_toon_critic(critic_raw)

    if critic_toon is None:
        log_debug(" ⚠️ Failed to parse critic TOON → fail-safe NO ERROR")
        # fail-safe: parse شکست خورد → فرض بر عدم خطا تا loop ایجاد نشود
        critic_toon = {
            "needs_revision": False,
            "issue": "خطای پارس TOON critic — به‌صورت محافظه‌کارانه NO ERROR اعمال شد",
            "action": "پاسخ قابل قبول است و نیازی به تغییر ندارد",
        }
        log_info("🔍 Critic: TOON parse failed → defaulting to NO ERROR (safe fallback)")

    # ── Post-check: بدون citation معتبر revision رد می‌شود ─────────────
    needs_revision = bool(critic_toon.get("needs_revision"))

    if needs_revision:
        log_debug(" 🔍 Checking for valid citation in critic output...")
        if not _has_valid_citation(critic_raw):
            log_debug(" ⚠️ needs_revision=true but no valid [منبع N] citation → forcing NO ERROR")
            log_info("🔍 Critic: Revision request rejected — no [منبع N] citation found")
            critic_toon = {
                "needs_revision": False,
                "issue": "خطای صریحی یافت نشد",
                "action": "پاسخ قابل قبول است و نیازی به تغییر ندارد",
            }
            needs_revision = False
        else:
            log_debug(" ✅ Valid [منبع N] citation found — revision approved")
            log_info(f"🔍 Critic: Revision requested — issue: {critic_toon.get('issue', 'نامشخص')}")
    else:
        log_info("🔍 Critic: Draft approved — no revision needed")

    # ── ذخیره کامل پیام critic در messages ─────────────────────────────
    # کل استدلال ۵ گامه و تصمیم نهایی critic برای trace و CSV ذخیره می‌شود
    critic_message: Dict[str, Any] = {
        "role": "assistant",
        "name": "critic",
        "content": critic_raw,
        "metadata": {
            "agent": "critic",
            "needs_revision": needs_revision,
            "issue": critic_toon.get("issue"),
            "action": critic_toon.get("action"),
        },
    }
    messages.append(critic_message)

    log_debug(f" 📊 Final critic decision: needs_revision={needs_revision}")
    log_debug(f" 📝 issue: {critic_toon.get('issue')}")
    log_debug("🔍 ═══ CRITIC END ═══\n")

    return {
        "critic_raw": critic_raw,
        "critic_toon": critic_toon,
        "messages": messages,
    }