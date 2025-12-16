from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from langsmith import traceable
from legal_multi_agent.state.schemas import MASharedState
from legal_multi_agent.utils.logger import log_debug, log_info


def _find_latest_pending_tool_call(
    messages: List[Dict[str, Any]],
    tool_results: Optional[Dict[str, Any]] = None
) -> Tuple[bool, Set[str]]:
    """
    پیدا کردن آخرین پیام assistant که tool_calls دارد و هنوز برای همه tool_call_idها
    پاسخ tool نیامده است.

    خروجی:
      (has_pending, pending_ids)
    """
    if not messages:
        return False, set()

    tool_results = tool_results or {}

    # از انتها به ابتدا: آخرین پیام assistant با tool_calls را پیدا می‌کنیم
    last_tc_index: Optional[int] = None
    last_tool_calls: List[Dict[str, Any]] = []

    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if not isinstance(msg, dict):
            continue
        tool_calls = msg.get("tool_calls")
        if tool_calls and isinstance(tool_calls, list):
            last_tc_index = i
            last_tool_calls = tool_calls
            break

    if last_tc_index is None or not last_tool_calls:
        return False, set()

    # tool_call_idهای درخواست شده
    requested_ids: Set[str] = set()
    # نگاشت id -> tool name (برای fail-safe)
    id_to_name: Dict[str, str] = {}

    for tc in last_tool_calls:
        if not isinstance(tc, dict):
            continue
        tc_id = tc.get("id")
        tc_name = tc.get("name")
        if tc_id:
            requested_ids.add(tc_id)
            if isinstance(tc_name, str) and tc_name:
                id_to_name[tc_id] = tc_name

    if not requested_ids:
        return False, set()

    # پاسخ‌های tool بعد از همان پیام را جمع می‌کنیم
    responded_ids: Set[str] = set()
    for msg in messages[last_tc_index + 1:]:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "tool":
            tci = msg.get("tool_call_id")
            if tci in requested_ids:
                responded_ids.add(tci)

    # fail-safe: اگر به هر دلیلی tool message ثبت نشده ولی tool_results ست شده،
    # آن tool_call را پاسخ‌گرفته فرض می‌کنیم تا supervisor در tools گیر نکند.
    for tc_id, tc_name in id_to_name.items():
        if tc_id in requested_ids and tc_id not in responded_ids:
            if tc_name in tool_results:
                responded_ids.add(tc_id)

    pending_ids = requested_ids - responded_ids
    return (len(pending_ids) > 0), pending_ids


@traceable(name="supervisor_agent")
def supervisor_agent(state: MASharedState) -> MASharedState:
    """
    Supervisor با قابلیت مدیریت tool execution (سازگار با schema جدید).

    اولویت تصمیم‌گیری:
      0) اگر خروجی نهایی داریم => FINISH
      1) اگر tool_call حل‌نشده داریم => tools
      2) اگر context نداریم => researcher
      3) اگر draft نداریم => reasoner
      4) اگر critic نداریم => critic
      5) اگر critic نیاز به اصلاح دارد و هنوز به سقف نرسیده‌ایم => reasoner
      6) در غیر این صورت => finalize
    """
    log_debug("\n🟢 ═══ SUPERVISOR START ═══")
    
    # نمایش state کلیدی
    log_debug(f"   📊 State overview:")
    log_debug(f"      - has_context: {bool(state.get('context'))}")
    log_debug(f"      - has_draft_toon: {bool(state.get('draft_toon'))}")
    log_debug(f"      - has_critic_toon: {bool(state.get('critic_toon'))}")
    log_debug(f"      - has_final_toon: {bool(state.get('final_toon'))}")
    log_debug(f"      - messages: {len(state.get('messages', []))}")
    log_debug(f"      - tool_results: {list(state.get('tool_results', {}).keys())}")
    
    # ═══════════════════════════════════════════════════════════
    # 0) اگر final داریم → تمام
    # ═══════════════════════════════════════════════════════════
    if state.get("final_toon"):
        log_info("🟢 Supervisor → FINISH")
        log_debug("   ✅ Decision: FINISH (final_toon exists)")
        log_debug("🟢 ═══ SUPERVISOR END ═══\n")
        return {"next": "FINISH"}

    messages = state.get("messages", []) or []
    tool_results = state.get("tool_results", {}) or {}

    # ═══════════════════════════════════════════════════════════
    # 1) اگر tool_call حل‌نشده داریم → tools
    # ═══════════════════════════════════════════════════════════
    has_pending, pending_ids = _find_latest_pending_tool_call(
        messages=messages,
        tool_results=tool_results,
    )
    
    if has_pending:
        log_info("🟢 Supervisor → tools")
        log_debug(f"   🔧 Decision: tools (pending tool_calls: {pending_ids})")
        log_debug("🟢 ═══ SUPERVISOR END ═══\n")
        return {"next": "tools"}

    # ═══════════════════════════════════════════════════════════
    # 2) اگر context نداریم → researcher
    # ═══════════════════════════════════════════════════════════
    if not state.get("context"):
        log_info("🟢 Supervisor → researcher")
        log_debug("   📚 Decision: researcher (no context)")
        log_debug("🟢 ═══ SUPERVISOR END ═══\n")
        return {"next": "researcher"}

    # ═══════════════════════════════════════════════════════════
    # 3) اگر draft نداریم → reasoner
    # ═══════════════════════════════════════════════════════════
    if not state.get("draft_toon"):
        log_info("🟢 Supervisor → reasoner")
        log_debug("   🤔 Decision: reasoner (no draft_toon)")
        log_debug("🟢 ═══ SUPERVISOR END ═══\n")
        return {"next": "reasoner"}

    # ═══════════════════════════════════════════════════════════
    # 4) اگر critic نداریم → critic
    # ═══════════════════════════════════════════════════════════
    critic_toon = state.get("critic_toon")
    if not critic_toon:
        log_info("🟢 Supervisor → critic")
        log_debug("   🔍 Decision: critic (no critic_toon)")
        log_debug("🟢 ═══ SUPERVISOR END ═══\n")
        return {"next": "critic"}

    # ═══════════════════════════════════════════════════════════
    # 5) بررسی نیاز به revision
    # ═══════════════════════════════════════════════════════════
    needs_revision = bool(critic_toon.get("needs_revision", False)) if isinstance(critic_toon, dict) else False
    revision_count = int(state.get("revision_count", 0) or 0)
    max_revisions = int(state.get("max_revisions", 2) or 2)

    log_debug(f"   🔄 Revision status:")
    log_debug(f"      - needs_revision: {needs_revision}")
    log_debug(f"      - revision_count: {revision_count}")
    log_debug(f"      - max_revisions: {max_revisions}")

    if needs_revision and revision_count < max_revisions:
        log_info(f"🟢 Supervisor → reasoner (revision {revision_count + 1}/{max_revisions})")
        log_debug(f"   🔄 Decision: reasoner (revision {revision_count + 1}/{max_revisions})")
        log_debug("🟢 ═══ SUPERVISOR END ═══\n")
        return {"next": "reasoner"}

    # ═══════════════════════════════════════════════════════════
    # 6) در غیر این صورت → finalize
    # ═══════════════════════════════════════════════════════════
    log_info("🟢 Supervisor → finalize")
    log_debug("   ✅ Decision: finalize (all checks passed)")
    log_debug("🟢 ═══ SUPERVISOR END ═══\n")
    return {"next": "finalize"}
