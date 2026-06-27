from __future__ import annotations

from typing import Any, Dict, List
import json

from uuid import uuid4

from langsmith import traceable

from legal_multi_agent.state.schemas import MASharedState
from legal_multi_agent.rag.pipeline import legal_rag_retrieve, format_results_for_llm
from legal_multi_agent.utils.logger import log_debug, log_info


# ═══════════════════════════════════════════════════════════
# Helper: ساخت query ترکیبی برای RAG
# ═══════════════════════════════════════════════════════════
def build_retrieval_query(question: str, options_text: str) -> str:
    """ساخت query برای RAG با ترکیب سوال و گزینه‌ها."""
    question = (question or "").strip()
    options_text = (options_text or "").strip()
    if options_text:
        return f"{question}\n\nگزینه‌ها:\n{options_text}"
    return question


# ═══════════════════════════════════════════════════════════
# Helper: بررسی tool_call معلق retriever
# ═══════════════════════════════════════════════════════════
def _has_pending_retriever_call(messages: list) -> bool:
    if not messages:
        return False

    last_retriever_call_id = None
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                # ✅ فرمت صحیح OpenAI
                func = tc.get("function", {})
                if func.get("name") == "retriever_tool":
                    last_retriever_call_id = tc.get("id")
                    break
        if last_retriever_call_id:
            break

    if not last_retriever_call_id:
        return False

    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "tool":
            if msg.get("tool_call_id") == last_retriever_call_id:
                return False

    return True


# ═══════════════════════════════════════════════════════════
# Helper: اجرای مستقیم RAG pipeline
# ═══════════════════════════════════════════════════════════
def _execute_direct_rag(
    query: str,
    top_k: int,
    use_rerank: bool,
) -> Dict[str, Any]:
    """
    اجرای مستقیم RAG pipeline و بازگشت نتایج.
    این تابع کمکی است برای جلوگیری از تکرار کد.
    """
    log_debug("   🔍 Calling legal_rag_retrieve...")

    try:
        results = legal_rag_retrieve(
            query=query,
            method="auto",
            top_k=top_k,
            use_rerank=use_rerank,
            verbose=True,
        )
    except Exception as e:
        log_debug(f"   ❌ RAG ERROR: {e.__class__.__name__}: {str(e)}")
        return {
            "rag_results":      [],
            "context":          f"⚠️ خطا در بازیابی اسناد: {str(e)}",
            "context_preview":  f"⚠️ خطا در بازیابی اسناد: {str(e)[:200]}",
            "docs_meta":        [],
        }

    log_debug(f"   ✅ RAG returned {len(results)} results")

    context = format_results_for_llm(results, include_metadata=True)
    preview = context[:2500] + ("..." if len(context) > 2500 else "")

    docs_meta: List[Dict[str, Any]] = []
    for i, r in enumerate(results[:10], start=1):
        if not isinstance(r, dict):
            continue
        metadata = r.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        docs_meta.append({
            "index":          i,
            "law_name":       metadata.get("law_name"),
            "article_number": metadata.get("article_number"),
            "source_type":    r.get("source_type"),
            "title":          metadata.get("title"),
            "score":          r.get("score"),
        })

    return {
        "rag_results":     results,
        "context":         context,
        "context_preview": preview,
        "docs_meta":       docs_meta,
    }


# ═══════════════════════════════════════════════════════════
# Researcher Agent
# ═══════════════════════════════════════════════════════════
@traceable(name="researcher_agent")
def researcher_agent(state: MASharedState) -> MASharedState:
    """
    ایجنت پژوهشگر: اجرای RAG یا استفاده از retriever_tool.

    دو حالت عملکرد:
        1. use_retriever_tool=False: اجرای مستقیم RAG
        2. use_retriever_tool=True:  ارسال درخواست به tool_executor

    تغییرات نسبت به نسخه قبل:
        - پیام researcher در هر دو حالت در messages ذخیره می‌شود
        - new_message در tool calling دارای name و metadata است
        - پیام نتیجه بازیابی (خلاصه docs_meta) در messages ثبت می‌شود
    """
    log_debug("\n📚 ═══ RESEARCHER START ═══")

    question    = state["question"]
    options_text = state.get("options_text", "")
    query       = build_retrieval_query(question, options_text)
    messages: List[Dict[str, Any]] = list(state.get("messages") or [])

    log_debug(f"  📝 Query: {query[:80]}...")

    top_k       = int(state.get("top_k",  6) or 6)
    use_rerank  = bool(state.get("use_rerank", True))
    use_tool    = bool(state.get("use_retriever_tool", False))

    log_debug(f"  ⚙️ top_k={top_k}, rerank={use_rerank}, use_tool={use_tool}")

    # ═══════════════════════════════════════════════════════
    # حالت ۱: استفاده از Tool Calling
    # ═══════════════════════════════════════════════════════
    if use_tool:
        log_debug("  🔧 Mode: Tool calling")
        tool_results = state.get("tool_results") or {}

        # ── ۱.۱ نتیجه tool آمده → context را ست کن ─────────────────────
        if "retriever_tool" in tool_results:
            log_info("📚 Researcher: Processing retriever results")
            retriever_result = tool_results["retriever_tool"]

            if not isinstance(retriever_result, dict):
                log_debug("  ⚠️ Invalid result format → fallback to direct RAG")
                result = _execute_direct_rag(query, top_k, use_rerank)
                _append_researcher_result_message(messages, result, mode="fallback_direct")
                num  = len(result.get("rag_results", []))
                clen = len(result.get("context", ""))
                log_info(f"📚 Researcher: Retrieved {num} docs, context={clen} chars")
                log_debug("📚 ═══ RESEARCHER END ═══\n")
                result["messages"] = messages
                return result

            num_docs    = len(retriever_result.get("rag_results", []))
            context_len = len(retriever_result.get("context", ""))
            log_info(f"📚 Researcher: Retrieved {num_docs} docs, context={context_len} chars")
            log_debug(f"  📊 docs={num_docs}, context={context_len} chars")

            # ✅ ذخیره پیام نتیجه بازیابی در messages
            _append_researcher_result_message(
                messages, retriever_result, mode="tool_calling"
            )

            log_debug("📚 ═══ RESEARCHER END ═══\n")
            return {
                "rag_results":     retriever_result.get("rag_results", []),
                "context":         retriever_result.get("context", ""),
                "context_preview": retriever_result.get("context_preview", ""),
                "docs_meta":       retriever_result.get("docs_meta", []),
                "messages":        messages,
            }

        # ── ۱.۲ pending call در انتظار پاسخ ──────────────────────────────
        has_pending = _has_pending_retriever_call(messages)
        log_debug(f"  🔍 has_pending_retriever_call: {has_pending}")
        if has_pending:
            log_debug("  ⏳ Waiting for pending retriever call")
            log_debug("📚 ═══ RESEARCHER END ═══\n")
            return {}

        # ── ۱.۳ tool call جدید بساز ───────────────────────────────────────
        log_info("📚 Researcher: Requesting retriever tool")
        call_id = f"retriever_{uuid4().hex[:8]}"

        # ✅ پیام با name و metadata کامل — هماهنگ با reasoner و critic

        retrieval_request_message: Dict[str, Any] = {
            "role":    "assistant",
            "name":    "researcher",
            "content": (
                f"درخواست بازیابی اسناد حقوقی مرتبط با سؤال:\n"
                f"{question[:120]}{'...' if len(question) > 120 else ''}"
            ),
            "tool_calls": [
                {
                    "id":   call_id,
                    "type": "function",              # ✅ اضافه شد
                    "function": {                    # ✅ داخل function
                        "name": "retriever_tool",
                        "arguments": json.dumps(
                            {
                                "query":      query,
                                "top_k":      top_k,
                                "use_rerank": use_rerank,
                            },
                            ensure_ascii=False,
                        ),
                    },
                }
            ],
            "metadata": {
                "agent":      "researcher",
                "phase":      "request_retriever_tool",
                "tool_name":  "retriever_tool",
                "top_k":      top_k,
                "use_rerank": use_rerank,
            },
        }

        messages.append(retrieval_request_message)
        log_debug(f"  📤 Created retriever tool_call id={call_id}")
        log_debug("📚 ═══ RESEARCHER END ═══\n")
        return {"messages": messages}

    # ═══════════════════════════════════════════════════════
    # حالت ۲: اجرای مستقیم RAG
    # ═══════════════════════════════════════════════════════
    log_debug("  📖 Mode: Direct RAG execution")
    log_info("📚 Researcher: Executing direct RAG")

    result = _execute_direct_rag(query, top_k, use_rerank)

    num_docs    = len(result.get("rag_results", []))
    context_len = len(result.get("context", ""))
    log_info(f"📚 Researcher: Retrieved {num_docs} docs, context={context_len} chars")
    log_debug(f"  📊 docs={num_docs}, context={context_len} chars")

    # ✅ ذخیره پیام نتیجه بازیابی در messages
    _append_researcher_result_message(messages, result, mode="direct_rag")

    log_debug("📚 ═══ RESEARCHER END ═══\n")
    result["messages"] = messages
    return result


# ═══════════════════════════════════════════════════════════
# Helper: ذخیره پیام نتیجه researcher در messages
# ═══════════════════════════════════════════════════════════
def _append_researcher_result_message(
    messages: List[Dict[str, Any]],
    result: Dict[str, Any],
    mode: str,
) -> None:
    """
    یک پیام ساختاریافته از نتایج بازیابی researcher به messages اضافه می‌کند.
    این پیام برای trace مکالمه و خروجی CSV استفاده می‌شود.
    """
    docs_meta   = result.get("docs_meta", []) or []
    num_docs    = len(result.get("rag_results", []))
    context_len = len(result.get("context", ""))

    # خلاصه اسناد بازیابی‌شده برای خوانایی در CSV
    docs_summary_lines = []
    for doc in docs_meta[:10]:
        law  = doc.get("law_name") or "نامشخص"
        art  = doc.get("article_number") or "—"
        score = doc.get("score")
        score_str = f" | امتیاز: {score:.3f}" if isinstance(score, float) else ""
        docs_summary_lines.append(f"  - {law}، ماده {art}{score_str}")

    docs_summary = "\n".join(docs_summary_lines) if docs_summary_lines else "  (سندی بازیابی نشد)"

    content = (
        f"بازیابی اسناد انجام شد (حالت: {mode}).\n"
        f"تعداد اسناد: {num_docs} | طول context: {context_len} کاراکتر\n"
        f"اسناد بازیابی‌شده:\n{docs_summary}"
    )

    message: Dict[str, Any] = {
        "role":    "assistant",
        "name":    "researcher",
        "content": content,
        "metadata": {
            "agent":       "researcher",
            "phase":       "retrieval_result",
            "mode":        mode,
            "num_docs":    num_docs,
            "context_len": context_len,
            "docs_meta":   docs_meta,
        },
    }
    messages.append(message)