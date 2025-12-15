"""
گراف اصلی سیستم مولتی‌ایجنت برای MCQ حقوقی (با tool support)
"""
from langgraph.graph import StateGraph, START, END

from legal_multi_agent.state.schemas import MASharedState
from legal_multi_agent.agents.supervisor import supervisor_agent
from legal_multi_agent.agents.researcher import researcher_agent
from legal_multi_agent.agents.reasoner import reasoner_agent
from legal_multi_agent.agents.critic import critic_agent
from legal_multi_agent.agents.tool_executor import tool_executor_node
from legal_multi_agent.utils.toon import extract_toon_answer


def finalize_node(state: MASharedState) -> MASharedState:
    """
    نود نهایی‌سازی: draft_toon را به final_toon منتقل می‌کند.
    """
    final_raw = state.get("draft_raw", "")
    final_toon = extract_toon_answer(final_raw)
    return {
        "final_raw": final_raw,
        "final_toon": final_toon,
    }


def build_graph() -> StateGraph:
    """
    ساخت گراف کامل مولتی‌ایجنت (با tool execution support)
    """
    workflow = StateGraph(MASharedState)

    # اضافه کردن نودها
    workflow.add_node("supervisor", supervisor_agent)
    workflow.add_node("researcher", researcher_agent)
    workflow.add_node("reasoner", reasoner_agent)
    workflow.add_node("critic", critic_agent)
    workflow.add_node("finalize", finalize_node)
    workflow.add_node("tools", tool_executor_node) 

    # یال‌های ثابت: همه به supervisor برمی‌گردند
    workflow.add_edge(START, "supervisor")
    workflow.add_edge("researcher", "supervisor")
    workflow.add_edge("reasoner", "supervisor")
    workflow.add_edge("critic", "supervisor")
    workflow.add_edge("finalize", "supervisor")
    workflow.add_edge("tools", "supervisor")  

    # یال‌های شرطی: supervisor مسیرها را مدیریت می‌کند
    workflow.add_conditional_edges(
        "supervisor",
        lambda s: s["next"],
        {
            "researcher": "researcher",
            "reasoner": "reasoner",
            "critic": "critic",
            "finalize": "finalize",
            "tools": "tools", 
            "FINISH": END,
        },
    )

    return workflow.compile()


# نمونه گراف آماده برای استفاده
graph = build_graph()
