from langsmith import traceable
from legal_multi_agent.state.schemas import MASharedState


@traceable(name="supervisor_agent")
def supervisor_agent(state: MASharedState) -> MASharedState:
    """
    Supervisor با قابلیت مدیریت tool execution
    """
    # 👉 اول: چک کردن tool calls در آخرین پیام
    messages = state.get("messages", [])
    if messages:
        last_msg = messages[-1]
        # اگر آخرین پیام tool call است
        if isinstance(last_msg, dict) and last_msg.get("tool_calls"):
            # بررسی که آیا این tool call پاسخ گرفته یا نه
            # با چک کردن اینکه آیا پیام tool با همان id وجود دارد
            tool_call_ids = {tc.get("id") for tc in last_msg.get("tool_calls", [])}
            
            # چک کردن پیام‌های بعدی برای tool response
            has_response = False
            for i in range(len(messages) - 1, -1, -1):
                msg = messages[i]
                if msg.get("role") == "tool" and msg.get("tool_call_id") in tool_call_ids:
                    has_response = True
                    break
            
            if not has_response:
                return {"next": "tools"}
    
    # 1) اگر context نداریم → researcher
    if not state.get("context"):
        return {"next": "researcher"}
    
    # 2) اگر draft نداریم → reasoner
    if not state.get("draft_toon"):
        return {"next": "reasoner"}
    
    # 3) اگر final داریم → تمام
    if state.get("final_toon"):
        return {"next": "FINISH"}
    
    # 4) اگر critic نداریم → critic
    if not state.get("critic_toon"):
        return {"next": "critic"}

    # 5) بررسی نیاز به revision
    needs = bool(state["critic_toon"]["needs_revision"])
    rc = int(state.get("revision_count", 0))
    mr = int(state.get("max_revisions", 2))

    if needs and rc < mr:
        return {"next": "reasoner"}

    return {"next": "finalize"}
