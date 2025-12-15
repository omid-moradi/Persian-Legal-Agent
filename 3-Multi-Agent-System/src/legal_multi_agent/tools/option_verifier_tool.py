from __future__ import annotations
import os
from typing import Dict, Any
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from openai import OpenAI
from langsmith.wrappers import wrap_openai
from langsmith import traceable

from legal_multi_agent.utils.toon import extract_toon_verifier
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL_ID = "google/gemini-2.5-flash"

_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)
client = wrap_openai(_client)


class OptionVerifierInput(BaseModel):
    """ورودی ابزار تایید گزینه‌ها."""
    question: str = Field(description="متن کامل سوال به فارسی")
    options_text: str = Field(
        description="گزینه‌های سوال به فرمت:\n1) گزینه اول\n2) گزینه دوم\n..."
    )
    sources: str = Field(
        description="متن اسناد و مواد قانونی (SOURCES) که باید با گزینه‌ها تطبیق داده شود."
    )


@traceable(name="call_verifier_llm")
def _call_verifier_llm(question: str, options_text: str, sources: str) -> str:
    """فراخوانی LLM برای تحلیل و امتیازدهی گزینه‌ها (option verification)."""
    system_msg = (
        """
شما یک **بازرس دقیق گزینه‌های حقوقی برای آزمون‌های چندگزینه‌ای حقوق ایران** هستید.

────────────────────────
نقش شما
────────────────────────
وظیفه شما:
- هر گزینه را فقط بر اساس منابع ارائه‌شده ارزیابی کنید.
- سطح حمایت هر گزینه را مشخص کنید: SUPPORTED, NOT_SUPPORTED, UNCLEAR.
- دقیقا یک گزینه را به عنوان پاسخ بهتر و پشتیبانی‌شده توصیه کنید.
- از حدس زدن یا ارجاع به منابع خارج از داده‌های ارائه‌شده خودداری کنید.

────────────────────────
قواعد حیاتی
────────────────────────
1) گزینه را فقط در صورتی SUPPORTED علامت بزنید که منابع به صراحت آن را پشتیبانی کنند.
2) اگر منابع سکوت کرده‌اند، مبهم یا غیرمستقیم هستند، گزینه را UNCLEAR بزنید.
3) گزینه را فقط زمانی NOT_SUPPORTED علامت بزنید که منابع به وضوح با آن مخالفت داشته باشند.
4) اگر دو یا چند گزینه SUPPORTED بودند:
   - گزینه‌ای را انتخاب کنید که به عنوان مرجع نهایی یا اصلی ذکر شده است.
   - اگر مرجع نهایی مشخص نیست، گزینه‌ای که متن صریح‌تر و دقیق‌تری دارد را انتخاب کنید.

────────────────────────
سطوح حمایت
────────────────────────
- SUPPORTED: منابع صریحاً گزینه را پشتیبانی می‌کنند.
- NOT_SUPPORTED: منابع به وضوح با گزینه مخالفت دارند.
- UNCLEAR: منابع سکوت، مبهم یا ارتباط ضعیف دارند.

────────────────────────
فرمت خروجی (سختگیرانه)
────────────────────────
شما باید دقیقا دو جدول TOON تولید کنید و هیچ چیز دیگری اضافه نکنید:

TOON اول (امتیاز گزینه‌ها):
results{option,support_level,reasoning}:
1,<SUPPORTED/NOT_SUPPORTED/UNCLEAR>,<دلیل به فارسی با ارجاع به ماده>
2,<SUPPORTED/NOT_SUPPORTED/UNCLEAR>,<دلیل به فارسی با ارجاع به ماده>
3,<SUPPORTED/NOT_SUPPORTED/UNCLEAR>,<دلیل به فارسی با ارجاع به ماده>
4,<SUPPORTED/NOT_SUPPORTED/UNCLEAR>,<دلیل به فارسی با ارجاع به ماده>

TOON دوم (توصیه نهایی):
results{recommended_answer,confidence}:
<1-4>,<1-5>

────────────────────────
محدودیت‌های خروجی
────────────────────────
- دلیل‌ها باید به فارسی باشند و شامل ارجاع دقیق به ماده قانون باشند.
- دلیل‌ها باید به طور مختصر توضیح دهند چرا گزینه پشتیبانی می‌شود یا نمی‌شود.
- اگر چند گزینه SUPPORTED هستند، گزینه‌ای انتخاب شود که مستقیم‌ترین و واضح‌ترین نام را دارد و به عنوان مرجع اصلی یا نهایی ذکر شده باشد.
- confidence: 5=خیلی مطمئن، 3-4=نسبتاً مطمئن، 1-2=اطمینان کم.
- هیچ متن دیگری خارج از دو جدول TOON اضافه نکنید.
- در متن دلیل‌ها از کاما استفاده نکنید؛ از نقطه‌ویرگول یا خط فاصله استفاده کنید.

"""
        )

    user_msg = f"""SOURCES:
{sources}

QUESTION (Persian):
{question}

OPTIONS:
{options_text}

وظیفه (TASK):
1. منابع (SOURCES) را با دقت مطالعه کرده و همه مواد و احکام مرتبط با سوال را شناسایی کنید.
2. برای هر گزینه (1-4) مشخص کنید که منابع:
   - به طور صریح از آن حمایت می‌کنند → SUPPORTED 
     (فقط یک گزینه می‌تواند SUPPORTED باشد؛ بقیه باید NOT_SUPPORTED یا UNCLEAR باشند)
   - به وضوح با آن مخالفت می‌کنند → NOT_SUPPORTED
   - سکوت یا ابهام دارند → UNCLEAR
3. برای هر گزینه، یک دلیل کوتاه به فارسی بنویسید شامل:
   - شماره دقیق مواد
   - عبارات کلیدی از منابع
   - توضیح روشن که چرا گزینه SUPPORTED / NOT_SUPPORTED / UNCLEAR است
4. سپس دقیقا یک گزینه را به عنوان بهترین پاسخ توصیه کنید و سطح اطمینان (1-5) را مشخص کنید. مطمئن شوید که گزینه SUPPORTED و گزینه پیشنهادی هماهنگ باشند.
5. دقیقا دو جدول TOON را طبق فرمت مشخص شده در خروجی ارائه دهید.
"""

    resp = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,  # deterministic
    )

    return resp.choices[0].message.content.strip()


@tool(args_schema=OptionVerifierInput)
def option_verifier_tool(
    question: str,
    options_text: str,
    sources: str,
) -> str:
    """
    امتیازدهی گزینه‌های MCQ بر اساس SOURCES با خروجی TOON.
    
    این ابزار:
    - هر گزینه (1-4) را جداگانه بررسی می‌کند.
    - تعیین می‌کند آیا SOURCES آن گزینه را تایید می‌کنند یا خیر.
    - بهترین گزینه را بر اساس حمایت منابع انتخاب می‌کند.
    
    خروجی: دو جدول TOON
    1) امتیاز هر گزینه (option, support_level, reasoning)
    2) گزینه پیشنهادی (recommended_answer, confidence)
    """
    try:
        raw_output = _call_verifier_llm(question, options_text, sources)

        # Parse TOON
        result = extract_toon_verifier(raw_output, verbose=False)

        if not result or not result["scores"]:
            return f"خطا: خروجی مدل TOON معتبر نبود.\n\nخروجی خام:\n{raw_output}"

        # ساخت خلاصه
        scores = result["scores"]
        recommended = result.get("recommended_answer")
        confidence = result.get("confidence")

        summary_lines = ["✓ نتیجه تحلیل گزینه‌ها:"]
        for sc in scores:
            reasoning_preview = (
                sc["reasoning"][:80] + "..." if len(sc["reasoning"]) > 80 else sc["reasoning"]
            )
            summary_lines.append(
                f"  گزینه {sc['option_number']}: {sc['support_level']} — {reasoning_preview}"
            )
        summary_lines.append(
            f"\n✓ گزینه پیشنهادی: {recommended} (اطمینان: {confidence}/5)"
        )
        summary_lines.append(
            "⚠️ توجه: این یک نظر مشورتی است - لطفاً مستقلاً منابع را بررسی کنید."
        )

        summary = "\n".join(summary_lines)

        return f"{summary}\n\n{'='*60}\nTOON خام:\n{'='*60}\n\n{raw_output}"

    except Exception as e:
        return f"خطا در اجرای option_verifier: {str(e)}"


@traceable(name="verify_options_direct")
def verify_options_direct(
    question: str,
    options_text: str,
    sources: str,
) -> Dict[str, Any]:
    """
    نسخه مستقیم برای استفاده در nodes (بدون tool wrapper).
    خروجی: dict با کلیدهای scores, recommended_answer, confidence
    """
    try:
        raw_output = _call_verifier_llm(question, options_text, sources)
        result = extract_toon_verifier(raw_output, verbose=False)

        if not result:
            result = {
                "scores": [],
                "recommended_answer": None,
                "confidence": None,
                "error": "TOON parse failed",
                "raw": raw_output[:500],
            }

        return result

    except Exception as e:
        return {
            "scores": [],
            "recommended_answer": None,
            "confidence": None,
            "error": str(e),
        }
