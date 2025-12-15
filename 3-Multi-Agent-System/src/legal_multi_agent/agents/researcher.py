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
    ایجنت پژوهشگر: اجرای RAG یا استفاده از retriever_tool.
    """
    q = state["question"]
    domain = state.get("domain", "")
    top_k = domain_top_k(domain)
    
    use_tool = state.get("use_retriever_tool", False)
    
    if use_tool:
        tool_results = state.get("tool_results", {})
        
        # 👉 مرحله 2: اگر tool اجرا شده، نتیجه را برگردان و context را SET کن
        if "retriever_tool" in tool_results:
            retriever_result = tool_results["retriever_tool"]
            return {
                "rag_results": retriever_result.get("rag_results", []),
                "context": retriever_result.get("context", ""),  # ✅ کلیدی!
                "context_preview": retriever_result.get("context_preview", ""),
                "docs_meta": retriever_result.get("docs_meta", []),
            }
        
        # 👉 مرحله 1: اگر context خالی است و هنوز tool call نکردیم
        messages = state.get("messages", [])
        
        # چک کن که آیا قبلاً tool call کردیم
        already_called = False
        if messages:
            for msg in messages:
                if isinstance(msg, dict) and msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        if tc.get("name") == "retriever_tool":
                            already_called = True
                            break
        
        if not already_called:
            # ساخت tool call request
            new_message = {
                "role": "assistant",
                "content": f"جستجوی اسناد برای: {q[:50]}...",
                "tool_calls": [
                    {
                        "id": "retriever_001",
                        "name": "retriever_tool",
                        "arguments": {
                            "query": q,
                            "top_k": top_k,
                            "use_rerank": True,
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
    
    # 👉 حالت مستقیم (بدون tool calling)
    results = legal_rag_retrieve(
        query=q,
        method="auto",
        top_k=top_k,
        use_rerank=True,
        verbose=True,
    )
    context = format_results_for_llm(results, include_metadata=True)
    preview = context[:2500]

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
