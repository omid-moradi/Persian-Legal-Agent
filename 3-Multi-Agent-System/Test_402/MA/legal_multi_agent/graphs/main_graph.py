"""
گراف اصلی سیستم مولتی‌ایجنت برای MCQ حقوقی (با tool support)

این ماژول گراف اصلی LangGraph را تعریف می‌کند که workflow زیر را اجرا می‌کند:
1. Supervisor: تصمیم‌گیری مسیریابی
2. Researcher: بازیابی اسناد حقوقی (RAG)
3. Reasoner: استدلال و تولید پاسخ TOON
4. Critic: ارزیابی و بازبینی پاسخ
5. Tools: اجرای ابزارهای خارجی (retriever, verifier)
6. Finalize: نهایی‌سازی خروجی
"""

from __future__ import annotations
from typing import Any, Dict, Optional

from langgraph.graph import StateGraph, START, END

from legal_multi_agent.state.schemas import MASharedState
from legal_multi_agent.agents.supervisor import supervisor_agent
from legal_multi_agent.agents.researcher import researcher_agent
from legal_multi_agent.agents.reasoner import reasoner_agent
from legal_multi_agent.agents.critic import critic_agent
from legal_multi_agent.agents.tool_executor import tool_executor_node
from legal_multi_agent.utils.toon import extract_toon_answer
from legal_multi_agent.utils.logger import log_debug, log_info, log_error


def finalize_node(state: MASharedState) -> MASharedState:
    """
    نود نهایی‌سازی: draft_toon را به final_toon منتقل می‌کند.

    این نود فقط زمانی فراخوانی می‌شود که:
    - draft_toon موجود باشد
    - critic تأیید کرده باشد یا max_revisions رسیده باشد
    """
    log_debug("\n🟣 ═══ FINALIZE START ═══")

    draft_raw  = state.get("draft_raw", "")
    draft_toon = state.get("draft_toon")

    log_debug(f"  📄 draft_raw length: {len(draft_raw)}")
    log_debug(f"  🎯 draft_toon exists: {bool(draft_toon)}")

    # اگر draft_toon موجود نیست، سعی می‌کنیم از draft_raw استخراج کنیم
    if not draft_toon and draft_raw:
        log_debug("  ⚠️ draft_toon not found, extracting from draft_raw...")
        draft_toon = extract_toon_answer(draft_raw)
        if draft_toon:
            # ✅ بدون confidence
            log_debug(f"  ✅ Extracted: answer={draft_toon['answer']}")

    # اگر هنوز draft_toon نداریم، یک خطای fallback بساز
    if not draft_toon:
        log_debug("  ❌ ERROR: No draft_toon available!")
        draft_toon = {
            "explanation": "خطا: پاسخ نهایی تولید نشد",
            "answer":      "1",
        }
        draft_raw = "Error: No draft available"
        log_error("🟣 Finalize: ERROR - no draft available")
    else:
        # ✅ بدون confidence در لاگ
        log_info(f"🟣 Finalize: answer={draft_toon['answer']}")
        log_debug(f"  ✅ Finalizing answer: {draft_toon['answer']}")

    log_debug("🟣 ═══ FINALIZE END ═══\n")

    return {
        "final_raw":  draft_raw,
        "final_toon": draft_toon,
    }


def initialize_node(state: MASharedState) -> MASharedState:
    """
    نود اولیه‌سازی: تنظیمات پیش‌فرض را اعمال می‌کند.

    این نود اطمینان می‌دهد که:
    - max_revisions مقدار معتبر دارد
    - revision_count از 0 شروع می‌شود
    - total_steps از 0 شروع می‌شود
    - messages یک list است
    - tool_results یک dict است
    """
    log_debug("\n🔷 ═══ INITIALIZE START ═══")

    updates: Dict[str, Any] = {}

    if state.get("max_revisions") is None:
        updates["max_revisions"] = 2
        log_debug("  ✓ Set max_revisions = 2")

    if state.get("revision_count") is None:
        updates["revision_count"] = 0
        log_debug("  ✓ Set revision_count = 0")

    # ✅ مقداردهی اولیه total_steps
    if state.get("total_steps") is None:
        updates["total_steps"] = 0
        log_debug("  ✓ Set total_steps = 0")

    if state.get("messages") is None:
        updates["messages"] = []
        log_debug("  ✓ Initialized messages = []")

    if state.get("tool_results") is None:
        updates["tool_results"] = {}
        log_debug("  ✓ Initialized tool_results = {}")

    log_debug("🔷 ═══ INITIALIZE END ═══\n")

    return updates


def build_graph(
    enable_debug:  bool         = False,
    checkpointer:  Optional[Any] = None,
) -> StateGraph:
    """
    ساخت گراف کامل مولتی‌ایجنت (با tool execution support)

    Args:
        enable_debug: فعال‌سازی حالت دیباگ (لاگ‌های بیشتر)
        checkpointer: اختیاری - برای ذخیره وضعیت گراف (persistence)

    Returns:
        گراف کامپایل شده آماده برای اجرا

    Example:
        >>> graph = build_graph()
        >>> result = graph.invoke({
        ...     "question": "سوال",
        ...     "options_text": "گزینه‌ها",
        ...     "max_revisions": 2,
        ... })
    """
    workflow = StateGraph(MASharedState)

    # ═══════════════════════════════════════════════════════════
    # اضافه کردن نودها
    # ═══════════════════════════════════════════════════════════

    workflow.add_node("initialize", initialize_node)
    workflow.add_node("supervisor", supervisor_agent)
    workflow.add_node("researcher", researcher_agent)
    workflow.add_node("reasoner",   reasoner_agent)
    workflow.add_node("critic",     critic_agent)
    workflow.add_node("finalize",   finalize_node)
    workflow.add_node("tools",      tool_executor_node)

    # ═══════════════════════════════════════════════════════════
    # یال‌های ثابت
    # ═══════════════════════════════════════════════════════════

    workflow.add_edge(START,        "initialize")
    workflow.add_edge("initialize", "supervisor")

    workflow.add_edge("researcher", "supervisor")
    workflow.add_edge("reasoner",   "supervisor")
    workflow.add_edge("critic",     "supervisor")
    workflow.add_edge("finalize",   "supervisor")
    workflow.add_edge("tools",      "supervisor")

    # ═══════════════════════════════════════════════════════════
    # یال‌های شرطی: supervisor مسیرها را مدیریت می‌کند
    # ═══════════════════════════════════════════════════════════

    def route_supervisor(state: MASharedState) -> str:
        """تابع routing با error handling"""
        next_step = state.get("next")

        if not next_step:
            # ✅ همیشه لاگ می‌شود — نه فقط در enable_debug
            log_error("⚠️ route_supervisor: next is None, defaulting to researcher")
            return "researcher"

        valid_routes = {
            "researcher", "reasoner", "critic",
            "finalize", "tools", "FINISH",
        }

        if next_step not in valid_routes:
            # ✅ همیشه لاگ می‌شود
            log_error(f"⚠️ route_supervisor: مسیر نامعتبر '{next_step}' — به FINISH می‌رویم")
            return "FINISH"

        if enable_debug:
            log_debug(f"  🔀 route_supervisor → {next_step}")

        return next_step

    workflow.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "researcher": "researcher",
            "reasoner":   "reasoner",
            "critic":     "critic",
            "finalize":   "finalize",
            "tools":      "tools",
            "FINISH":     END,
        },
    )

    # ═══════════════════════════════════════════════════════════
    # کامپایل گراف
    # ═══════════════════════════════════════════════════════════

    compile_kwargs: Dict[str, Any] = {}

    if checkpointer:
        compile_kwargs["checkpointer"] = checkpointer

    if enable_debug:
        compile_kwargs["debug"] = True

    return workflow.compile(**compile_kwargs)


# ═══════════════════════════════════════════════════════════
# نمونه‌های گراف آماده برای استفاده
# ═══════════════════════════════════════════════════════════

# گراف پیش‌فرض (بدون debug، بدون persistence)
graph = build_graph()

# graph_debug = build_graph(enable_debug=True)
# from langgraph.checkpoint.memory import MemorySaver
# graph_persistent = build_graph(checkpointer=MemorySaver())


def run_graph(
    question:            str,
    options_text:        str,
    max_revisions:       int  = 2,
    use_option_verifier: bool = True,
    use_retriever_tool:  bool = False,
    recursion_limit:     int  = 50,
) -> Dict[str, Any]:
    """
    Helper function برای اجرای ساده گراف

    Args:
        question:            متن سوال
        options_text:        متن گزینه‌ها
        max_revisions:       حداکثر تعداد بازبینی
        use_option_verifier: استفاده از verifier tool
        use_retriever_tool:  استفاده از retriever tool
        recursion_limit:     حداکثر تعداد گام‌های گراف

    Returns:
        نتیجه نهایی state

    Example:
        >>> result = run_graph(
        ...     question="سوال",
        ...     options_text="1) گزینه یک\\n2) گزینه دو",
        ... )
        >>> print(result["final_toon"]["answer"])
    """
    initial_state = {
        "question":            question,
        "options_text":        options_text,
        "max_revisions":       max_revisions,
        "use_option_verifier": use_option_verifier,
        "use_retriever_tool":  use_retriever_tool,
    }

    config = {"recursion_limit": recursion_limit}

    return graph.invoke(initial_state, config)