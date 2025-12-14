from __future__ import annotations
from typing import Any, Dict, List

from langsmith import traceable

from legal_multi_agent.state.schemas import MASharedState
from legal_multi_agent.rag.pipeline import legal_rag_retrieve, format_results_for_llm


def domain_top_k(domain: str) -> int:
    """تنظیم تعداد اسناد براساس دامین."""
    if domain in ("criminal", "criminal_procedure"):
        return 6
    if domain == "constitutional":
        return 6
    return 5


@traceable(name="researcher_agent")
def researcher_agent(state: MASharedState) -> MASharedState:
    """
    ایجنت پژوهشگر: اجرای RAG و ساخت context + متادیتا.
    
    حالت‌های کاری:
    1. اگر use_retriever_tool=True → از tool استفاده می‌کند (برای آینده)
    2. وگرنه → مستقیماً RAG را اجرا می‌کند (حالت فعلی)
    """
    q = state["question"]
    domain = state.get("domain", "")
    top_k = domain_top_k(domain)
    
    use_tool = state.get("use_retriever_tool", False)
    
    if use_tool:
        pass
    
    # حالت مستقیم (بدون tool calling)
    results = legal_rag_retrieve(
        query=q,
        method="auto",
        top_k=top_k,
        use_rerank=True,
        verbose=True,
    )
    context = format_results_for_llm(results, include_metadata=True)

    # preview کوتاه برای دیباگ
    preview = context[:2500]

    # متادیتای سبک برای لاگ/تحلیل
    docs_meta: List[Dict[str, Any]] = []
    for i, r in enumerate(results[:10], start=1):
        m = r.get("metadata", {}) if isinstance(r, dict) else {}
        docs_meta.append(
            {
                "i": i,
                "law": m.get("law_name"),
                "article_number": m.get("article_number"),
                "source_type": r.get("source_type"),
                "title": m.get("title"),
            }
        )

    return {
        "rag_results": results,
        "context": context,
        "context_preview": preview,
        "docs_meta": docs_meta,
    }
