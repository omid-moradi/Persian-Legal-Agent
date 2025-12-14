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
    2. tool calls را پیدا می‌کند
    3. tools را اجرا می‌کند
    4. نتایج را در tool_results ذخیره می‌کند
    """
    messages = state.get("messages", [])
    if not messages:
        return {}
    
    # آخرین پیام را بررسی می‌کنیم
    last_message = messages[-1]
    
    # اگر tool_calls وجود ندارد، کاری نداریم
    if "tool_calls" not in last_message:
        return {}
    
    tool_calls = last_message.get("tool_calls", [])
    if not tool_calls:
        return {}
    
    # نتایج tools
    tool_results = state.get("tool_results", {}).copy()
    new_messages = messages.copy()
    
    for tool_call in tool_calls:
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("arguments", {})
        tool_id = tool_call.get("id", tool_name)
        
        result = None
        
        # اجرای retriever_tool
        if tool_name == "retriever_tool":
            query = tool_args.get("query")
            top_k = tool_args.get("top_k", 5)
            use_rerank = tool_args.get("use_rerank", True)
            
            result = retrieve_documents(
                query=query,
                top_k=top_k,
                use_rerank=use_rerank,
            )
            
            # ذخیره نتیجه
            tool_results["retriever_tool"] = result
            
            # اضافه کردن پیام نتیجه
            new_messages.append({
                "role": "tool",
                "tool_call_id": tool_id,
                "name": tool_name,
                "content": f"✓ {len(result.get('rag_results', []))} سند بازیابی شد.\n\n{result.get('context_preview', '')}",
            })
        
        # اجرای option_verifier_tool
        elif tool_name == "option_verifier_tool":
            question = tool_args.get("question")
            options_text = tool_args.get("options_text")
            sources = tool_args.get("sources")
            
            result = verify_options_direct(
                question=question,
                options_text=options_text,
                sources=sources,
            )
            
            # ذخیره نتیجه
            tool_results["option_verifier_tool"] = result
            
            # فرمت کردن نتیجه برای LLM
            if result.get("scores"):
                summary_lines = [
                    f"✓ تحلیل {len(result['scores'])} گزینه:",
                ]
                for score in result["scores"]:
                    summary_lines.append(
                        f"  گزینه {score['option_number']}: {score['support_level']} - {score['reasoning'][:80]}..."
                    )
                summary_lines.append(f"\n✓ گزینه پیشنهادی: {result.get('recommended_answer')}")
                summary_lines.append(f"✓ اطمینان: {result.get('confidence')}/5")
                
                content = "\n".join(summary_lines)
            else:
                content = f"خطا در تحلیل: {result.get('error', 'نامشخص')}"
            
            new_messages.append({
                "role": "tool",
                "tool_call_id": tool_id,
                "name": tool_name,
                "content": content,
            })
    
    return {
        "tool_results": tool_results,
        "messages": new_messages,
    }
