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


# ═══════════════════════════════════════════════════════════
# نود اولیه‌سازی
# ═══════════════════════════════════════════════════════════

def initialize_node(state: MASharedState) -> MASharedState:
    """
    نود اولیه‌سازی: تنظیمات پیش‌فرض را اعمال می‌کند.

    ✅ FIX: verifier_output، draft_raw، draft_toon و critic_toon
    همیشه ریست می‌شوند تا بین سوالات مختلف تداخل نباشد.
    """
    log_debug("\n🔷 ═══ INITIALIZE START ═══")

    updates: Dict[str, Any] = {}

    # ── مقادیر شمارنده ──────────────────────────────────────
    if state.get("max_revisions") is None:
        updates["max_revisions"] = 2
        log_debug(" ✓ Set max_revisions = 2")

    # revision_count همیشه از صفر شروع می‌شود
    updates["revision_count"] = 0
    log_debug(" ✓ Reset revision_count = 0")

    updates["total_steps"] = 0
    log_debug(" ✓ Reset total_steps = 0")

    # ── ساختارهای داده ──────────────────────────────────────
    if state.get("messages") is None:
        updates["messages"] = []
        log_debug(" ✓ Initialized messages = []")

    if state.get("tool_results") is None:
        updates["tool_results"] = {}
        log_debug(" ✓ Initialized tool_results = {}")

    # ✅ FIX: ریست کردن state های قبلی — جلوگیری از تداخل بین سوالات
    updates["verifier_output"] = None
    updates["draft_raw"] = ""
    updates["draft_toon"] = None
    updates["critic_toon"] = None
    updates["critic_raw"] = ""
    updates["final_toon"] = None
    updates["final_raw"] = ""
    log_debug(" ✓ Reset verifier_output, draft_toon, critic_toon, final_toon")

    log_debug("🔷 ═══ INITIALIZE END ═══\n")
    return updates


# ═══════════════════════════════════════════════════════════
# نود نهایی‌سازی
# ═══════════════════════════════════════════════════════════

def finalize_node(state: MASharedState) -> MASharedState:
    """
    نود نهایی‌سازی: draft_toon را به final_toon منتقل می‌کند.

    ✅ FIX 1: اگر answer خالی بود، از draft_raw مجدداً extract می‌کند.
    ✅ FIX 2: fallback دیگر answer="1" نمی‌دهد — answer="" باقی می‌ماند.
    ✅ FIX 3: validate می‌کند که answer یکی از {"1","2","3","4"} باشد.
    """
    log_debug("\n🟣 ═══ FINALIZE START ═══")

    draft_raw  = state.get("draft_raw", "") or ""
    draft_toon = state.get("draft_toon")

    log_debug(f" 📄 draft_raw length  : {len(draft_raw)}")
    log_debug(f" 🎯 draft_toon exists : {bool(draft_toon)}")

    # ── مرحله ۱: اگر draft_toon نداریم از draft_raw استخراج کن ──
    if not draft_toon and draft_raw:
        log_debug(" ⚠️ draft_toon not found → extracting from draft_raw...")
        extracted = extract_toon_answer(draft_raw)
        if extracted and isinstance(extracted, dict):
            draft_toon = extracted
            log_debug(f" ✅ Extracted draft_toon: answer={draft_toon.get('answer')}")

    # ── مرحله ۲: validate کردن answer ────────────────────────
    VALID_ANSWERS = {"1", "2", "3", "4"}

    if draft_toon and isinstance(draft_toon, dict):
        answer = str(draft_toon.get("answer", "")).strip()

        # ✅ FIX: تبدیل اعداد فارسی به لاتین در صورت لزوم
        fa_to_en = str.maketrans("۱۲۳۴", "1234")
        answer = answer.translate(fa_to_en)

        if answer not in VALID_ANSWERS:
            # ✅ FIX: به جای فرض "1"، answer خالی می‌ماند و لاگ می‌شود
            log_error(f"🟣 Finalize: INVALID answer='{answer}' — keeping empty")
            log_info("🟣 Finalize: WARNING — no valid answer extracted")
            draft_toon = {
                "explanation": draft_toon.get("explanation", "پاسخ معتبر استخراج نشد"),
                "answer": "",   # ✅ خالی — نه "1" فرضی
            }
        else:
            draft_toon["answer"] = answer   # نسخه لاتین normalize شده
            log_info(f"🟣 Finalize: answer={answer} ✅")
            log_debug(f" ✅ Valid answer confirmed: {answer}")

    # ── مرحله ۳: اگر هنوز هیچ draft_toon نداریم ─────────────
    if not draft_toon:
        log_error("🟣 Finalize: ERROR — no draft_toon available at all!")
        draft_toon = {
            "explanation": "خطا: پاسخ نهایی تولید نشد — هیچ draft_toon موجود نیست",
            "answer": "",   # ✅ FIX: خالی به جای "1"
        }
        draft_raw = draft_raw or "Error: No draft available"

    log_debug("🟣 ═══ FINALIZE END ═══\n")

    return {
        "final_raw" : draft_raw,
        "final_toon": draft_toon,
    }


# ═══════════════════════════════════════════════════════════
# ساخت گراف
# ═══════════════════════════════════════════════════════════

def build_graph(
    enable_debug: bool = False,
    checkpointer: Optional[Any] = None,
) -> StateGraph:
    """
    ساخت گراف کامل مولتی‌ایجنت (با tool execution support)

    Args:
        enable_debug: فعال‌سازی حالت دیباگ (لاگ‌های بیشتر)
        checkpointer: اختیاری — برای ذخیره وضعیت گراف (persistence)

    Returns:
        گراف کامپایل‌شده آماده برای اجرا
    """
    workflow = StateGraph(MASharedState)

    # ── اضافه کردن نودها ────────────────────────────────────
    workflow.add_node("initialize", initialize_node)
    workflow.add_node("supervisor", supervisor_agent)
    workflow.add_node("researcher", researcher_agent)
    workflow.add_node("reasoner",   reasoner_agent)
    workflow.add_node("critic",     critic_agent)
    workflow.add_node("finalize",   finalize_node)
    workflow.add_node("tools",      tool_executor_node)

    # ── یال‌های ثابت ────────────────────────────────────────
    workflow.add_edge(START,        "initialize")
    workflow.add_edge("initialize", "supervisor")

    workflow.add_edge("researcher", "supervisor")
    workflow.add_edge("reasoner",   "supervisor")
    workflow.add_edge("critic",     "supervisor")
    workflow.add_edge("finalize",   "supervisor")
    workflow.add_edge("tools",      "supervisor")

    # ── یال‌های شرطی ────────────────────────────────────────
    def route_supervisor(state: MASharedState) -> str:
        """
        تابع routing با error handling.

        ✅ FIX: وقتی next=None است به FINISH می‌رود نه researcher
        تا از حلقه بی‌پایان جلوگیری شود.
        """
        next_step = state.get("next")

        if not next_step:
            # ✅ FIX: None → FINISH (نه researcher)
            log_error("⚠️ route_supervisor: next is None → FINISH")
            return "FINISH"

        valid_routes = {
            "researcher", "reasoner", "critic",
            "finalize", "tools", "FINISH",
        }

        if next_step not in valid_routes:
            log_error(f"⚠️ route_supervisor: مسیر نامعتبر '{next_step}' → FINISH")
            return "FINISH"

        if enable_debug:
            log_debug(f" 🔀 route_supervisor → {next_step}")

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

    # ── کامپایل گراف ────────────────────────────────────────
    compile_kwargs: Dict[str, Any] = {}
    if checkpointer:
        compile_kwargs["checkpointer"] = checkpointer
    if enable_debug:
        compile_kwargs["debug"] = True

    return workflow.compile(**compile_kwargs)


# ═══════════════════════════════════════════════════════════
# نمونه‌های گراف آماده برای استفاده
# ═══════════════════════════════════════════════════════════

graph = build_graph()
# graph_debug      = build_graph(enable_debug=True)
# from langgraph.checkpoint.memory import MemorySaver
# graph_persistent = build_graph(checkpointer=MemorySaver())


def run_graph(
    question: str,
    options_text: str,
    max_revisions: int = 2,
    use_option_verifier: bool = True,
    use_retriever_tool: bool = False,
    recursion_limit: int = 60,   # ✅ FIX: از 50 به 60 — برای revision loop کافی باشد
) -> Dict[str, Any]:
    """
    Helper function برای اجرای ساده گراف.

    Args:
        question          : متن سوال
        options_text      : متن گزینه‌ها
        max_revisions     : حداکثر تعداد بازبینی
        use_option_verifier: استفاده از verifier tool
        use_retriever_tool : استفاده از retriever tool
        recursion_limit   : حداکثر تعداد گام‌های گراف

    Returns:
        نتیجه نهایی state
    """
    initial_state = {
        "question":           question,
        "options_text":       options_text,
        "max_revisions":      max_revisions,
        "use_option_verifier": use_option_verifier,
        "use_retriever_tool":  use_retriever_tool,
    }

    config = {"recursion_limit": recursion_limit}
    return graph.invoke(initial_state, config)