from __future__ import annotations
from typing import Any, Dict, List

from langsmith import traceable

from legal_multi_agent.state.schemas import MASharedState
from legal_multi_agent.rag.pipeline import legal_rag_retrieve, format_results_for_llm


def build_retrieval_query(question: str, options_text: str) -> str:
    question = (question or "").strip()
    options_text = (options_text or "").strip()
    if options_text:
        return f"{question}\n\nگزینه‌ها:\n{options_text}"
    return question


@traceable(name="researcher_agent")
def researcher_agent(state: MASharedState) -> MASharedState:
    """
    ایجنت پژوهشگر: اجرای RAG یا استفاده از retriever_tool.
    بدون وابستگی به domain/category.
    """
    question = state["question"]
    options_text = state.get("options_text", "")
    query = build_retrieval_query(question, options_text)

    top_k = int(state.get("top_k", 6))          # قابل کنترل از بیرون
    use_rerank = bool(state.get("use_rerank", True))
    use_tool = bool(state.get("use_retriever_tool", False))

    if use_tool:
        tool_results = state.get("tool_results", {})

        # اگر tool قبلاً اجرا شده، context را ست کن
        if "retriever_tool" in tool_results:
            retriever_result = tool_results["retriever_tool"]
            return {
                "rag_results": retriever_result.get("rag_results", []),
                "context": retriever_result.get("context", ""),
                "context_preview": retriever_result.get("context_preview", ""),
                "docs_meta": retriever_result.get("docs_meta", []),
            }

        # اگر هنوز tool call نکردیم، درخواست tool call بساز
        messages = state.get("messages", [])

        already_called = False
        for msg in messages or []:
            if isinstance(msg, dict) and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    if tc.get("name") == "retriever_tool":
                        already_called = True
                        break

        if not already_called:
            new_message = {
                "role": "assistant",
                "content": f"جستجوی اسناد برای: {question[:50]}...",
                "tool_calls": [
                    {
                        "id": "retriever_001",
                        "name": "retriever_tool",
                        "arguments": {
                            "query": query,          # ✅ سوال + گزینه‌ها
                            "top_k": top_k,
                            "use_rerank": use_rerank,
                        },
                    }
                ],
            }

            messages_copy = (messages or []).copy()
            messages_copy.append(new_message)
            return {"messages": messages_copy}

        return {}

    # حالت مستقیم (بدون tool calling)
    results = legal_rag_retrieve(
        query=query,
        method="auto",
        top_k=top_k,
        use_rerank=use_rerank,
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
