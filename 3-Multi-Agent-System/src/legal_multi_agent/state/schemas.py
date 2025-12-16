from __future__ import annotations
from typing import TypedDict, List, Dict, Any, Optional, Annotated
from operator import add


def merge_messages(left: List[Dict[str, Any]], right: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Reducer برای messages: لیست راست را به چپ append می‌کند.
    """
    if not right:
        return left
    if not left:
        return right
    return left + right


class MASharedState(TypedDict, total=False):
    # ورودی اصلی
    question_number: int
    question: str
    options_text: str

    # ⭐ پیام‌ها برای ایجنت‌ها - با reducer برای append
    messages: Annotated[List[Dict[str, Any]], merge_messages]
    
    # نتایج tools (dict با key = tool name)
    tool_results: Dict[str, Any]
    
    # خروجی option_verifier_tool (اگر استفاده شود)
    verifier_output: Optional[Dict[str, Any]]

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
    
    # فلگ برای کنترل استفاده از tools
    use_option_verifier: bool
    use_retriever_tool: bool
