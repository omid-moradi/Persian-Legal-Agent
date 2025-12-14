from langsmith import traceable
from legal_multi_agent.state.schemas import MASharedState

@traceable(name="supervisor_agent")
def supervisor_agent(state: MASharedState) -> MASharedState:
    if not state.get("context"):
        return {"next": "researcher"}
    if not state.get("draft_toon"):
        return {"next": "reasoner"}
    if state.get("final_toon"):
        return {"next": "FINISH"}
    if not state.get("critic_toon"):
        return {"next": "critic"}

    needs = bool(state["critic_toon"]["needs_revision"])
    rc = int(state.get("revision_count", 0))
    mr = int(state.get("max_revisions", 2))

    if needs and rc < mr:
        return {"next": "reasoner"}

    return {"next": "finalize"}
