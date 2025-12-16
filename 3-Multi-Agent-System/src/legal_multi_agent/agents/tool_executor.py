from __future__ import annotations
from typing import Any, Dict, List

from langsmith import traceable

from legal_multi_agent.state.schemas import MASharedState
from legal_multi_agent.tools.retriever_tool import retrieve_documents
from legal_multi_agent.tools.option_verifier_tool import verify_options_direct


@traceable(name="tool_executor_node")
def tool_executor_node(state: MASharedState) -> MASharedState:
    """
    Node برای اجرای tools بر اساس درخواست‌های موجود در messages.
    
    این node:
    1. messages را چک می‌کند
    2. tool_calls را پیدا می‌کند
    3. tools را اجرا می‌کند
    4. نتایج را در tool_results و state fields مربوطه ذخیره می‌کند
    5. پیام‌های tool response را به messages اضافه می‌کند
    """
    messages = state.get("messages") or []
    if not messages:
        return {}
    
    # آخرین پیام را بررسی می‌کنیم
    last_message = messages[-1]
    
    if not isinstance(last_message, dict):
        return {}
    
    # اگر tool_calls وجود ندارد، کاری نداریم
    tool_calls = last_message.get("tool_calls")
    if not tool_calls or not isinstance(tool_calls, list):
        return {}
    
    # ═══════════════════════════════════════════════════════════
    # آماده‌سازی: کپی state fields
    # ═══════════════════════════════════════════════════════════
    tool_results = (state.get("tool_results") or {}).copy()
    new_messages = messages.copy()
    
    # برای ذخیره مستقیم در state (علاوه بر tool_results)
    state_updates: Dict[str, Any] = {}
    
    # ═══════════════════════════════════════════════════════════
    # اجرای هر tool_call
    # ═══════════════════════════════════════════════════════════
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("arguments", {})
        tool_id = tool_call.get("id")
        
        # اگر id نداریم، از نام استفاده می‌کنیم (fallback)
        if not tool_id:
            tool_id = tool_name or "unknown"
        
        if not tool_name:
            # tool نامعتبر - پیام خطا اضافه کن
            new_messages.append({
                "role": "tool",
                "tool_call_id": tool_id,
                "name": "unknown",
                "content": "⚠️ خطا: نام tool مشخص نیست",
            })
            continue
        
        # ───────────────────────────────────────────────────────
        # 1️⃣ اجرای retriever_tool
        # ───────────────────────────────────────────────────────
        if tool_name == "retriever_tool":
            try:
                query = tool_args.get("query", "")
                top_k = int(tool_args.get("top_k", 5) or 5)
                use_rerank = bool(tool_args.get("use_rerank", True))
                
                if not query:
                    raise ValueError("query خالی است")
                
                # اجرای tool
                result = retrieve_documents(
                    query=query,
                    top_k=top_k,
                    use_rerank=use_rerank,
                )
                
                if not isinstance(result, dict):
                    raise ValueError("نتیجه retriever_tool یک dict نیست")
                
                # ذخیره در tool_results
                tool_results["retriever_tool"] = result
                
                # ذخیره مستقیم در state (برای اطمینان از دسترسی supervisor)
                state_updates["rag_results"] = result.get("rag_results", [])
                state_updates["context"] = result.get("context", "")
                state_updates["context_preview"] = result.get("context_preview", "")
                state_updates["docs_meta"] = result.get("docs_meta", [])
                
                # ساخت پیام پاسخ
                num_docs = len(result.get("rag_results", []))
                preview = result.get("context_preview", "")[:500]
                
                content = (
                    f"✅ بازیابی اسناد موفق\n"
                    f"📊 تعداد اسناد: {num_docs}\n\n"
                    f"پیش‌نمایش:\n{preview}..."
                )
                
                new_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "name": tool_name,
                    "content": content,
                })
                
            except Exception as e:
                # در صورت خطا، یک پیام خطا اضافه کن
                error_msg = f"⚠️ خطا در اجرای retriever_tool: {str(e)}"
                
                tool_results["retriever_tool"] = {
                    "error": str(e),
                    "rag_results": [],
                    "context": "",
                    "context_preview": "",
                    "docs_meta": [],
                }
                
                new_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "name": tool_name,
                    "content": error_msg,
                })
        
        # ───────────────────────────────────────────────────────
        # 2️⃣ اجرای option_verifier_tool
        # ───────────────────────────────────────────────────────
        elif tool_name == "option_verifier_tool":
            try:
                question = tool_args.get("question", "")
                options_text = tool_args.get("options_text", "")
                sources = tool_args.get("sources", "")
                
                if not question:
                    raise ValueError("question خالی است")
                if not options_text:
                    raise ValueError("options_text خالی است")
                if not sources:
                    raise ValueError("sources خالی است")
                
                # اجرای tool
                result = verify_options_direct(
                    question=question,
                    options_text=options_text,
                    sources=sources,
                )
                
                if not isinstance(result, dict):
                    raise ValueError("نتیجه option_verifier_tool یک dict نیست")
                
                # ذخیره در tool_results
                tool_results["option_verifier_tool"] = result
                
                # ذخیره مستقیم در state
                state_updates["verifier_output"] = result
                
                # ساخت پیام پاسخ
                if result.get("scores") and isinstance(result["scores"], list):
                    summary_lines = [
                        f"✅ تحلیل {len(result['scores'])} گزینه انجام شد:",
                        ""
                    ]
                    
                    for score in result["scores"]:
                        if not isinstance(score, dict):
                            continue
                        
                        opt_num = score.get("option_number", "?")
                        support = score.get("support_level", "UNKNOWN")
                        reasoning = score.get("reasoning", "")[:100]
                        
                        summary_lines.append(
                            f"• گزینه {opt_num}: {support}"
                        )
                        summary_lines.append(f"  └─ {reasoning}...")
                        summary_lines.append("")
                    
                    recommended = result.get("recommended_answer", "?")
                    confidence = result.get("confidence", "?")
                    
                    summary_lines.append(f"💡 گزینه پیشنهادی: {recommended}")
                    summary_lines.append(f"📊 سطح اطمینان: {confidence}/5")
                    
                    content = "\n".join(summary_lines)
                else:
                    error = result.get("error", "نامشخص")
                    content = f"⚠️ خطا در تحلیل گزینه‌ها: {error}"
                
                new_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "name": tool_name,
                    "content": content,
                })
                
            except Exception as e:
                # در صورت خطا
                error_msg = f"⚠️ خطا در اجرای option_verifier_tool: {str(e)}"
                
                tool_results["option_verifier_tool"] = {
                    "error": str(e),
                    "scores": [],
                    "recommended_answer": None,
                    "confidence": 0,
                }
                
                new_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "name": tool_name,
                    "content": error_msg,
                })
        
        # ───────────────────────────────────────────────────────
        # 3️⃣ tool ناشناخته
        # ───────────────────────────────────────────────────────
        else:
            # tool ناشناخته - پیام خطا
            new_messages.append({
                "role": "tool",
                "tool_call_id": tool_id,
                "name": tool_name,
                "content": f"⚠️ خطا: tool '{tool_name}' شناخته نشده است",
            })
    
    # ═══════════════════════════════════════════════════════════
    # بازگشت تمام تغییرات
    # ═══════════════════════════════════════════════════════════
    return {
        "tool_results": tool_results,
        "messages": new_messages,
        **state_updates,  # شامل context, verifier_output و غیره
    }
