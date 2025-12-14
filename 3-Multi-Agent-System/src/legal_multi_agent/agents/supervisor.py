from langsmith import traceable
from legal_multi_agent.state.schemas import MASharedState


@traceable(name="supervisor_agent")
def supervisor_agent(state: MASharedState) -> MASharedState:
    """
    Supervisor با قابلیت مدیریت tool execution
    """
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
