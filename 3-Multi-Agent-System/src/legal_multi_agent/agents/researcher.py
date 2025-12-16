from __future__ import annotations
from typing import Any, Dict, List
from uuid import uuid4

from langsmith import traceable

from legal_multi_agent.state.schemas import MASharedState
from legal_multi_agent.rag.pipeline import legal_rag_retrieve, format_results_for_llm
from legal_multi_agent.utils.logger import log_debug, log_info


def build_retrieval_query(question: str, options_text: str) -> str:
    """
    ساخت query برای RAG با ترکیب سوال و گزینه‌ها.
    """
    question = (question or "").strip()
    options_text = (options_text or "").strip()
    if options_text:
        return f"{question}\n\nگزینه‌ها:\n{options_text}"
    return question


def _has_pending_retriever_call(messages: list) -> bool:
    """
    بررسی اینکه آیا یک tool_call برای retriever_tool وجود دارد
    که هنوز پاسخ نگرفته است.
    """
    if not messages:
        return False
    
    # پیدا کردن آخرین tool_call برای retriever
    last_retriever_call_id = None
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                if tc.get("name") == "retriever_tool":
                    last_retriever_call_id = tc.get("id")
                    break
            if last_retriever_call_id:
                break
    
    if not last_retriever_call_id:
        return False
    
    # بررسی اینکه آیا پاسخ برای این call_id وجود دارد
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "tool":
            if msg.get("tool_call_id") == last_retriever_call_id:
                return False  # پاسخ پیدا شد، pending نیست
    
    return True  # tool_call وجود دارد ولی پاسخ نداریم


@traceable(name="researcher_agent")
def researcher_agent(state: MASharedState) -> MASharedState:
    """
    ایجنت پژوهشگر: اجرای RAG یا استفاده از retriever_tool.
    
    دو حالت عملکرد:
    1. use_retriever_tool=False: اجرای مستقیم RAG
    2. use_retriever_tool=True: ارسال درخواست به tool_executor
    """
    log_debug("\n📚 ═══ RESEARCHER START ═══")
    
    question = state["question"]
    options_text = state.get("options_text", "")
    query = build_retrieval_query(question, options_text)

    log_debug(f"   📝 Query: {query[:80]}...")

    # پارامترهای قابل کنترل از state
    top_k = int(state.get("top_k", 6) or 6)
    use_rerank = bool(state.get("use_rerank", True))
    use_tool = bool(state.get("use_retriever_tool", False))

    log_debug(f"   ⚙️  top_k={top_k}, rerank={use_rerank}, use_tool={use_tool}")

    # ═══════════════════════════════════════════════════════════
    # حالت 1: استفاده از Tool Calling
    # ═══════════════════════════════════════════════════════════
    if use_tool:
        log_debug("   🔧 Mode: Tool calling")
        
        tool_results = state.get("tool_results") or {}
        messages = state.get("messages") or []

        # 1.1) اگر tool نتیجه داده → context را ست کن
        if "retriever_tool" in tool_results:
            log_info("📚 Researcher: Processing retriever results")
            log_debug("   ✅ Retriever tool result found")
            
            retriever_result = tool_results["retriever_tool"]
            
            # اطمینان از اینکه نتیجه یک dict است
            if not isinstance(retriever_result, dict):
                log_debug("   ⚠️  Invalid result format, falling back to direct RAG")
                # fallback به حالت مستقیم
                return _execute_direct_rag(query, top_k, use_rerank)
            
            num_docs = len(retriever_result.get("rag_results", []))
            context_len = len(retriever_result.get("context", ""))
            log_info(f"📚 Researcher: Retrieved {num_docs} docs, context={context_len} chars")
            log_debug(f"   📊 Retrieved: {num_docs} documents")
            log_debug(f"   📄 Context length: {context_len} chars")
            log_debug("📚 ═══ RESEARCHER END ═══\n")
            
            return {
                "rag_results": retriever_result.get("rag_results", []),
                "context": retriever_result.get("context", ""),
                "context_preview": retriever_result.get("context_preview", ""),
                "docs_meta": retriever_result.get("docs_meta", []),
            }

        # 1.2) چک کردن pending call
        has_pending = _has_pending_retriever_call(messages)
        log_debug(f"   🔍 has_pending_retriever_call: {has_pending}")
        
        if has_pending:
            log_debug("   ⏳ Waiting for pending retriever call")
            log_debug("📚 ═══ RESEARCHER END ═══\n")
            # منتظر پاسخ tool هستیم - هیچ کاری نکن
            return {}

        # 1.3) اگر pending نداریم → tool call جدید بساز
        log_info("📚 Researcher: Requesting retriever tool")
        log_debug("   🆕 Creating new retriever tool_call")
        
        call_id = f"retriever_{uuid4().hex[:8]}"
        
        new_message = {
            "role": "assistant",
            "content": f"🔍 جستجوی اسناد حقوقی: {question[:60]}...",
            "tool_calls": [
                {
                    "id": call_id,
                    "name": "retriever_tool",
                    "arguments": {
                        "query": query,
                        "top_k": top_k,
                        "use_rerank": use_rerank,
                    },
                }
            ],
        }

        messages_copy = messages.copy()
        messages_copy.append(new_message)
        
        log_debug(f"   📤 Created tool_call with id: {call_id}")
        log_debug("📚 ═══ RESEARCHER END ═══\n")
        
        return {
            "messages": messages_copy,
        }

    # ═══════════════════════════════════════════════════════════
    # حالت 2: اجرای مستقیم RAG (بدون tool calling)
    # ═══════════════════════════════════════════════════════════
    log_debug("   📖 Mode: Direct RAG execution")
    log_info("📚 Researcher: Executing direct RAG")
    result = _execute_direct_rag(query, top_k, use_rerank)
    
    num_docs = len(result.get("rag_results", []))
    context_len = len(result.get("context", ""))
    log_info(f"📚 Researcher: Retrieved {num_docs} docs, context={context_len} chars")
    log_debug(f"   📊 Retrieved: {num_docs} documents")
    log_debug(f"   📄 Context length: {context_len} chars")
    log_debug("📚 ═══ RESEARCHER END ═══\n")
    
    return result


def _execute_direct_rag(query: str, top_k: int, use_rerank: bool) -> Dict[str, Any]:
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
        # در صورت خطا در RAG، یک نتیجه خالی با پیام خطا برگردان
        return {
            "rag_results": [],
            "context": f"⚠️ خطا در بازیابی اسناد: {str(e)}",
            "context_preview": f"⚠️ خطا در بازیابی اسناد: {str(e)[:200]}",
            "docs_meta": [],
        }
    
    log_debug(f"   ✅ RAG returned {len(results)} results")
    
    # فرمت کردن نتایج برای LLM
    context = format_results_for_llm(results, include_metadata=True)
    preview = context[:2500] + ("..." if len(context) > 2500 else "")

    # استخراج metadata از نتایج
    docs_meta: List[Dict[str, Any]] = []
    for i, r in enumerate(results[:10], start=1):
        if not isinstance(r, dict):
            continue
        
        metadata = r.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        
        docs_meta.append({
            "index": i,
            "law_name": metadata.get("law_name"),
            "article_number": metadata.get("article_number"),
            "source_type": r.get("source_type"),
            "title": metadata.get("title"),
            "score": r.get("score"),  # اضافه کردن score برای دیباگ
        })

    return {
        "rag_results": results,
        "context": context,
        "context_preview": preview,
        "docs_meta": docs_meta,
    }
