from __future__ import annotations
from typing import TypedDict, List, Dict, Any

class MASharedState(TypedDict, total=False):
    # ورودی اصلی
    question_number: int
    category: str
    domain: str
    question: str
    options_text: str

    # پیام‌ها برای ایجنت‌ها (بعداً برای ToolNode هم استفاده می‌شود)
    messages: List[Dict[str, Any]]

    # خروجی RAG
    context: str
    rag_results: List[Dict[str, Any]]
    context_preview: str
    docs_meta: List[Dict[str, Any]]

    # خروجی Reasoner
    draft_raw: str
    draft_toon: Dict[str, Any]

    # خروجی Critic
    critic_raw: str
    critic_toon: Dict[str, Any]

    # خروجی نهایی
    final_raw: str
    final_toon: Dict[str, Any]

    # حلقه بازبینی
    revision_count: int
    max_revisions: int
    next: str
