from __future__ import annotations
import os
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from openai import OpenAI
from langsmith.wrappers import wrap_openai
from langsmith import traceable

from legal_multi_agent.utils.toon import extract_toon_verifier


OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL_ID = "qwen/qwen3-235b-a22b-2507"

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


# تابع کمکی برای فراخوانی مدل (با traceable)
@traceable(name="call_verifier_llm")
def _call_verifier_llm(question: str, options_text: str, sources: str) -> str:
    """فراخوانی LLM برای option verification."""
    system_msg = (
        "You are a precise legal option verifier for Iranian law MCQ exams.\n"
        "Your ONLY job is to check EACH option (1-4) against the provided SOURCES.\n"
        "For each option, determine:\n"
        "- SUPPORTED: Sources clearly support this option (cite article/ruling).\n"
        "- NOT_SUPPORTED: Sources clearly contradict or do not support this option.\n"
        "- UNCLEAR: Sources are insufficient or ambiguous for this option.\n\n"
        "Output MUST be EXACTLY two TOON tables:\n\n"
        "First TOON (option scores):\n"
        "results{option,support_level,reasoning}:\n"
        "1,<SUPPORTED/NOT_SUPPORTED/UNCLEAR>,<reasoning in Persian with article cite>\n"
        "2,<SUPPORTED/NOT_SUPPORTED/UNCLEAR>,<reasoning in Persian with article cite>\n"
        "3,<SUPPORTED/NOT_SUPPORTED/UNCLEAR>,<reasoning in Persian with article cite>\n"
        "4,<SUPPORTED/NOT_SUPPORTED/UNCLEAR>,<reasoning in Persian with article cite>\n\n"
        "Second TOON (recommendation):\n"
        "results{recommended_answer,confidence}:\n"
        "<1-4>,<1-5>\n\n"
        "Rules:\n"
        "- reasoning MUST be in Persian and cite specific article numbers.\n"
        "- If multiple options are SUPPORTED, choose the one with strongest evidence.\n"
        "- confidence: 1=very uncertain, 5=very certain.\n"
        "- Do NOT add any text outside these two TOON tables.\n"
        "- Do NOT use commas inside reasoning text (use semicolons or dashes instead)."
    )

    user_msg = f"""SOURCES:
{sources}

QUESTION (Persian):
{question}

OPTIONS:
{options_text}

Task: Output exactly two TOON tables as specified in system instructions.
"""

    resp = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
    )

    return resp.choices[0].message.content.strip()


# Tool اصلی (بدون traceable مستقیم)
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
            reasoning_preview = sc['reasoning'][:80] + "..." if len(sc['reasoning']) > 80 else sc['reasoning']
            summary_lines.append(
                f"  گزینه {sc['option_number']}: {sc['support_level']} — {reasoning_preview}"
            )
        summary_lines.append(f"\n✓ گزینه پیشنهادی: {recommended} (اطمینان: {confidence}/5)")

        summary = "\n".join(summary_lines)

        return f"{summary}\n\n{'='*60}\nTOON خام:\n{'='*60}\n\n{raw_output}"

    except Exception as e:
        return f"خطا در اجرای option_verifier: {str(e)}"


# نسخه helper برای استفاده مستقیم در node
@traceable(name="verify_options_direct")
def verify_options_direct(
    question: str,
    options_text: str,
    sources: str,
) -> Dict[str, Any]:
    """
    نسخه مستقیم برای استفاده در nodes (بدون tool wrapper).
    خروجی: dict با کلیدهای scores و recommended_answer
    """
    try:
        raw_output = _call_verifier_llm(question, options_text, sources)
        result = extract_toon_verifier(raw_output, verbose=False)

        if not result:
            result = {
                "scores": [],
                "recommended_answer": None,
                "error": "TOON parse failed",
                "raw": raw_output[:500],
            }

        return result

    except Exception as e:
        return {
            "scores": [],
            "recommended_answer": None,
            "error": str(e),
        }
