from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import json

from langsmith import traceable

from legal_multi_agent.state.schemas import MASharedState
from legal_multi_agent.tools.retriever_tool import retrieve_documents
from legal_multi_agent.tools.option_verifier_tool import verify_options_direct
from legal_multi_agent.utils.logger import log_debug, log_info


# ═══════════════════════════════════════════════════════════
# Helper: پیدا کردن آخرین پیام assistant با tool_calls حل‌نشده
# ═══════════════════════════════════════════════════════════
def _find_pending_tool_calls(
    messages: List[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    آخرین پیام assistant که tool_calls دارد و حداقل یک call آن
    هنوز پاسخ tool نگرفته را برمی‌گرداند.

    خروجی: (پیام assistant، لیست tool_call‌های حل‌نشده)
    """
    # جمع‌آوری همه tool_call_idهایی که پاسخ گرفته‌اند
    responded_ids = set()
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "tool":
            tc_id = msg.get("tool_call_id")
            if tc_id:
                responded_ids.add(tc_id)

    # پیدا کردن آخرین assistant که tool_call حل‌نشده دارد
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls")
        if not tool_calls or not isinstance(tool_calls, list):
            continue

        pending = [
            tc for tc in tool_calls
            if isinstance(tc, dict) and tc.get("id") not in responded_ids
        ]
        if pending:
            return msg, pending

    return None, []


# ═══════════════════════════════════════════════════════════
# Tool Executor Node
# ═══════════════════════════════════════════════════════════
@traceable(name="tool_executor_node")
def tool_executor_node(state: MASharedState) -> MASharedState:
    """
    Node برای اجرای tools بر اساس درخواست‌های موجود در messages.

    این node:
    1. آخرین tool_call حل‌نشده را در messages پیدا می‌کند
    2. tools را اجرا می‌کند
    3. نتایج را در tool_results و state fields مربوطه ذخیره می‌کند
    4. پیام‌های tool response را با metadata کامل به messages اضافه می‌کند

    تغییرات نسبت به نسخه قبل:
        - جستجو در کل messages (نه فقط last_message) — bugfix مهم
        - حذف confidence از پیام verifier
        - اضافه شدن metadata به پیام‌های tool response
        - import logger اضافه شد
    """
    log_debug("\n🔧 ═══ TOOL EXECUTOR START ═══")

    messages = list(state.get("messages") or [])
    if not messages:
        log_debug("  ⚠️ No messages found")
        log_debug("🔧 ═══ TOOL EXECUTOR END ═══\n")
        return {}

    # ── پیدا کردن tool_call‌های حل‌نشده در کل messages ──────────────────
    source_message, pending_tool_calls = _find_pending_tool_calls(messages)

    if not pending_tool_calls:
        log_debug("  ✅ No pending tool_calls found")
        log_debug("🔧 ═══ TOOL EXECUTOR END ═══\n")
        return {}

    log_debug(f"  🔍 Found {len(pending_tool_calls)} pending tool_call(s)")

    tool_results: Dict[str, Any] = (state.get("tool_results") or {}).copy()
    new_messages  = messages.copy()
    state_updates: Dict[str, Any] = {}

    # ═══════════════════════════════════════════════════════
    # اجرای هر tool_call حل‌نشده
    # ═══════════════════════════════════════════════════════
    for tool_call in pending_tool_calls:
        func      = tool_call.get("function", {}) or {}
        tool_name = func.get("name") or tool_call.get("name")          # backward compat
        _raw_args = func.get("arguments") or tool_call.get("arguments", {})
        tool_args = (
            json.loads(_raw_args)
            if isinstance(_raw_args, str)
            else (_raw_args or {})
        )
        tool_id   = tool_call.get("id") or tool_name or "unknown"

        log_debug(f"  ⚙️ Executing: {tool_name} (id={tool_id})")

        if not tool_name:
            new_messages.append({
                "role":         "tool",
                "tool_call_id": tool_id,
                "name":         "unknown",
                "content":      "⚠️ خطا: نام tool مشخص نیست",
                "metadata":     {"tool_name": "unknown", "status": "error"},
            })
            continue

        # ──────────────────────────────────────────────────
        # ۱. retriever_tool
        # ──────────────────────────────────────────────────
        if tool_name == "retriever_tool":
            try:
                query      = tool_args.get("query", "")
                top_k      = int(tool_args.get("top_k", 5) or 5)
                use_rerank = bool(tool_args.get("use_rerank", True))

                if not query:
                    raise ValueError("query خالی است")

                result = retrieve_documents(
                    query=query,
                    top_k=top_k,
                    use_rerank=use_rerank,
                )

                if not isinstance(result, dict):
                    raise ValueError("نتیجه retriever_tool یک dict نیست")

                tool_results["retriever_tool"] = result

                state_updates["rag_results"]     = result.get("rag_results", [])
                state_updates["context"]         = result.get("context", "")
                state_updates["context_preview"] = result.get("context_preview", "")
                state_updates["docs_meta"]       = result.get("docs_meta", [])

                num_docs = len(result.get("rag_results", []))
                preview  = result.get("context_preview", "")[:500]

                # خلاصه اسناد برای خوانایی در CSV
                docs_meta = result.get("docs_meta", []) or []
                docs_lines = []
                for doc in docs_meta[:6]:
                    law   = doc.get("law_name") or "نامشخص"
                    art   = doc.get("article_number") or "—"
                    score = doc.get("score")
                    score_str = f" | امتیاز: {score:.3f}" if isinstance(score, float) else ""
                    docs_lines.append(f"  - {law}، ماده {art}{score_str}")
                docs_summary = "\n".join(docs_lines) if docs_lines else "  (سندی یافت نشد)"

                content = (
                    f"✅ بازیابی اسناد موفق\n"
                    f"تعداد اسناد: {num_docs}\n"
                    f"اسناد بازیابی‌شده:\n{docs_summary}\n\n"
                    f"پیش‌نمایش context:\n{preview}..."
                )

                log_info(f"🔧 retriever_tool: {num_docs} docs retrieved")
                new_messages.append({
                    "role":         "tool",
                    "tool_call_id": tool_id,
                    "name":         tool_name,
                    "content":      content,
                    "metadata": {
                        "tool_name": tool_name,
                        "status":    "success",
                        "num_docs":  num_docs,
                        "docs_meta": docs_meta,
                    },
                })

            except Exception as e:
                log_debug(f"  ❌ retriever_tool error: {e}")
                error_msg = f"⚠️ خطا در اجرای retriever_tool: {str(e)}"
                tool_results["retriever_tool"] = {
                    "error":           str(e),
                    "rag_results":     [],
                    "context":         "",
                    "context_preview": "",
                    "docs_meta":       [],
                }
                new_messages.append({
                    "role":         "tool",
                    "tool_call_id": tool_id,
                    "name":         tool_name,
                    "content":      error_msg,
                    "metadata":     {"tool_name": tool_name, "status": "error", "error": str(e)},
                })

        # ──────────────────────────────────────────────────
        # ۲. option_verifier_tool
        # ──────────────────────────────────────────────────
        elif tool_name == "option_verifier_tool":
            try:
                question     = tool_args.get("question", "")
                options_text = tool_args.get("options_text", "")
                sources      = tool_args.get("sources", "")

                if not question:
                    raise ValueError("question خالی است")
                if not options_text:
                    raise ValueError("options_text خالی است")
                if not sources:
                    raise ValueError("sources خالی است")

                result = verify_options_direct(
                    question=question,
                    options_text=options_text,
                    sources=sources,
                )

                if not isinstance(result, dict):
                    raise ValueError("نتیجه option_verifier_tool یک dict نیست")

                tool_results["option_verifier_tool"] = result
                state_updates["verifier_output"]     = result

                # ✅ بدون confidence — فقط support_level و reasoning
                if result.get("scores") and isinstance(result["scores"], list):
                    summary_lines = [
                        f"✅ تحلیل {len(result['scores'])} گزینه انجام شد:",
                        "",
                    ]
                    for score in result["scores"]:
                        if not isinstance(score, dict):
                            continue
                        opt_num   = score.get("option_number", "?")
                        support   = score.get("support_level", "UNKNOWN")
                        reasoning = score.get("reasoning", "")[:200]
                        summary_lines.append(f"• گزینه {opt_num}: {support}")
                        summary_lines.append(f"  └─ {reasoning}...")
                        summary_lines.append("")

                    recommended = result.get("recommended_answer", "?")
                    summary_lines.append(f"💡 گزینه پیشنهادی verifier: {recommended}")
                    content = "\n".join(summary_lines)
                else:
                    error = result.get("error", "نامشخص")
                    content = f"⚠️ خطا در تحلیل گزینه‌ها: {error}"

                log_info(f"🔧 option_verifier_tool: recommended={result.get('recommended_answer')}")
                new_messages.append({
                    "role":         "tool",
                    "tool_call_id": tool_id,
                    "name":         tool_name,
                    "content":      content,
                    "metadata": {
                        "tool_name":          tool_name,
                        "status":             "success",
                        "recommended_answer": result.get("recommended_answer"),
                        "scores_count":       len(result.get("scores", [])),
                    },
                })

            except Exception as e:
                log_debug(f"  ❌ option_verifier_tool error: {e}")
                error_msg = f"⚠️ خطا در اجرای option_verifier_tool: {str(e)}"
                tool_results["option_verifier_tool"] = {
                    "error":              str(e),
                    "scores":             [],
                    "recommended_answer": None,
                }
                new_messages.append({
                    "role":         "tool",
                    "tool_call_id": tool_id,
                    "name":         tool_name,
                    "content":      error_msg,
                    "metadata":     {"tool_name": tool_name, "status": "error", "error": str(e)},
                })

        # ──────────────────────────────────────────────────
        # ۳. tool ناشناخته
        # ──────────────────────────────────────────────────
        else:
            log_debug(f"  ⚠️ Unknown tool: {tool_name}")
            new_messages.append({
                "role":         "tool",
                "tool_call_id": tool_id,
                "name":         tool_name,
                "content":      f"⚠️ خطا: tool '{tool_name}' شناخته نشده است",
                "metadata":     {"tool_name": tool_name, "status": "unknown_tool"},
            })

    log_debug(f"  ✅ Executed {len(pending_tool_calls)} tool(s)")
    log_debug("🔧 ═══ TOOL EXECUTOR END ═══\n")

    return {
        "tool_results": tool_results,
        "messages":     new_messages,
        **state_updates,
    }