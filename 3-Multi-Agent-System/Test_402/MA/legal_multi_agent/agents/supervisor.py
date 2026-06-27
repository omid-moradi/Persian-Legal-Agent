from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from langsmith import traceable
from legal_multi_agent.state.schemas import MASharedState
from legal_multi_agent.utils.logger import log_debug, log_info


# ═══════════════════════════════════════════════════════════
# Helper: پیدا کردن tool_call حل‌نشده
# ═══════════════════════════════════════════════════════════

def _find_latest_pending_tool_call(
    messages: List[Dict[str, Any]],
    tool_results: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Set[str]]:
    """
    پیدا کردن آخرین پیام assistant که tool_calls دارد و هنوز برای همه
    tool_call_idها پاسخ tool نیامده است.

    خروجی:
        (has_pending, pending_ids)
    """
    if not messages:
        return False, set()

    tool_results = tool_results or {}

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

    requested_ids: Set[str] = set()
    id_to_name: Dict[str, str] = {}

    for tc in last_tool_calls:
        if not isinstance(tc, dict):
            continue
        tc_id   = tc.get("id")
        # ✅ فرمت جدید OpenAI: name داخل function است
        tc_name = tc.get("function", {}).get("name") or tc.get("name")
        if tc_id:
            requested_ids.add(tc_id)
            if isinstance(tc_name, str) and tc_name:
                id_to_name[tc_id] = tc_name

    if not requested_ids:
        return False, set()

    responded_ids: Set[str] = set()
    for msg in messages[last_tc_index + 1:]:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "tool":
            tci = msg.get("tool_call_id")
            if tci in requested_ids:
                responded_ids.add(tci)

    # fail-safe: اگر tool message ثبت نشده ولی tool_results ست شده
    for tc_id, tc_name in id_to_name.items():
        if tc_id in requested_ids and tc_id not in responded_ids:
            if tc_name in tool_results:
                responded_ids.add(tc_id)

    pending_ids = requested_ids - responded_ids
    return (len(pending_ids) > 0), pending_ids


# ═══════════════════════════════════════════════════════════
# Helper: ثبت تصمیم supervisor در messages
# ═══════════════════════════════════════════════════════════

def _supervisor_message(decision: str, reason: str) -> Dict[str, Any]:
    """پیام ساختاریافته supervisor برای ذخیره در تاریخچه مکالمه."""
    return {
        "role":    "assistant",
        "name":    "supervisor",
        "content": f"[تصمیم Supervisor] ← {decision} | دلیل: {reason}",
        "metadata": {
            "agent":    "supervisor",
            "decision": decision,
            "reason":   reason,
        },
    }


# ═══════════════════════════════════════════════════════════
# Supervisor Agent
# ═══════════════════════════════════════════════════════════

@traceable(name="supervisor_agent")
def supervisor_agent(state: MASharedState) -> MASharedState:
    """
    Supervisor با قابلیت مدیریت tool execution.

    اولویت تصمیم‌گیری:
        0) اگر total_steps از سقف مطلق گذشته              → FINISH (emergency)
        1) اگر final_toon داریم                             → FINISH
        2) اگر tool_call حل‌نشده داریم                     → tools
        3) اگر context نداریم                              → researcher
        4) اگر draft_toon نداریم                           → reasoner
        5) اگر critic_toon نداریم                          → critic
        6) اگر critic نیاز به revision دارد و سقف نرسیده  → reasoner (+reset critic)
        7) در غیر این صورت                                 → finalize

    اصلاحات نسبت به نسخه قبل:
        - total_steps برای محافظت از loop بی‌نهایت در کل workflow  (bugfix)
        - revision_count در هنگام ارسال به reasoner افزایش می‌یابد  (bugfix)
        - critic_toon و draft_toon پیش از revision پاک می‌شوند      (bugfix)
        - draft_raw نگه داشته می‌شود تا reasoner پاسخ قبلی را ببیند (bugfix)
        - هر تصمیم در messages ثبت می‌شود برای traceability کامل
        - محافظت از loop بی‌نهایت با hard-cap روی revision_count
    """
    log_debug("\n🟢 ═══ SUPERVISOR START ═══")

    messages: List[Dict[str, Any]] = list(state.get("messages") or [])
    tool_results: Dict[str, Any]   = state.get("tool_results", {}) or {}
    total_steps                    = int(state.get("total_steps", 0) or 0)
    MAX_TOTAL_STEPS                = 20

    log_debug(f"  📊 State overview:")
    log_debug(f"    - has_context:     {bool(state.get('context'))}")
    log_debug(f"    - has_draft_toon:  {bool(state.get('draft_toon'))}")
    log_debug(f"    - has_critic_toon: {bool(state.get('critic_toon'))}")
    log_debug(f"    - has_final_toon:  {bool(state.get('final_toon'))}")
    log_debug(f"    - revision_count:  {state.get('revision_count', 0)}")
    log_debug(f"    - total_steps:     {total_steps}/{MAX_TOTAL_STEPS}")
    log_debug(f"    - messages:        {len(messages)}")
    log_debug(f"    - tool_results:    {list(tool_results.keys())}")

    # ── گام ۰: سقف مطلق workflow — محافظت از loop بی‌نهایت ────────────────
    if total_steps >= MAX_TOTAL_STEPS:
        reason = (
            f"سقف مطلق workflow ({MAX_TOTAL_STEPS} گام) رسیده شد — خروج اجباری | "
            f"آخرین وضعیت: context={bool(state.get('context'))} | "
            f"draft={bool(state.get('draft_toon'))} | "
            f"critic={bool(state.get('critic_toon'))}"
        )
        log_info("🔴 Supervisor → FINISH (emergency: max total_steps exceeded)")
        log_debug(f"  🚨 Emergency FINISH ({reason})")
        log_debug("🟢 ═══ SUPERVISOR END ═══\n")
        messages.append(_supervisor_message("FINISH (emergency)", reason))
        return {
            "next":        "FINISH",
            "messages":    messages,
            "total_steps": total_steps + 1,
        }

    # ── گام ۱: خروجی نهایی موجود است ──────────────────────────────────────
    if state.get("final_toon"):
        reason = "final_toon موجود است — workflow کامل شد"
        log_info("🟢 Supervisor → FINISH")
        log_debug(f"  ✅ Decision: FINISH ({reason})")
        log_debug("🟢 ═══ SUPERVISOR END ═══\n")
        messages.append(_supervisor_message("FINISH", reason))
        return {
            "next":        "FINISH",
            "messages":    messages,
            "total_steps": total_steps + 1,
        }

    # ── گام ۲: tool_call حل‌نشده ──────────────────────────────────────────
    has_pending, pending_ids = _find_latest_pending_tool_call(
        messages=messages,
        tool_results=tool_results,
    )
    if has_pending:
        reason = f"tool_call در انتظار پاسخ: {pending_ids}"
        log_info("🟢 Supervisor → tools")
        log_debug(f"  🔧 Decision: tools ({reason})")
        log_debug("🟢 ═══ SUPERVISOR END ═══\n")
        messages.append(_supervisor_message("tools", reason))
        return {
            "next":        "tools",
            "messages":    messages,
            "total_steps": total_steps + 1,
        }

    # ── گام ۳: context وجود ندارد ─────────────────────────────────────────
    if not state.get("context"):
        reason = "context موجود نیست — نیاز به بازیابی اسناد از RAG"
        log_info("🟢 Supervisor → researcher")
        log_debug(f"  📚 Decision: researcher ({reason})")
        log_debug("🟢 ═══ SUPERVISOR END ═══\n")
        messages.append(_supervisor_message("researcher", reason))
        return {
            "next":        "researcher",
            "messages":    messages,
            "total_steps": total_steps + 1,
        }

    # ── گام ۴: draft_toon وجود ندارد ──────────────────────────────────────
    if not state.get("draft_toon"):
        reason = "draft_toon موجود نیست — نیاز به استدلال اولیه"
        log_info("🟢 Supervisor → reasoner")
        log_debug(f"  🤔 Decision: reasoner ({reason})")
        log_debug("🟢 ═══ SUPERVISOR END ═══\n")
        messages.append(_supervisor_message("reasoner", reason))
        return {
            "next":        "reasoner",
            "messages":    messages,
            "total_steps": total_steps + 1,
        }

    # ── گام ۵: critic_toon وجود ندارد ─────────────────────────────────────
    critic_toon = state.get("critic_toon")
    if not critic_toon:
        reason = "critic_toon موجود نیست — نیاز به ارزیابی توسط critic"
        log_info("🟢 Supervisor → critic")
        log_debug(f"  🔍 Decision: critic ({reason})")
        log_debug("🟢 ═══ SUPERVISOR END ═══\n")
        messages.append(_supervisor_message("critic", reason))
        return {
            "next":        "critic",
            "messages":    messages,
            "total_steps": total_steps + 1,
        }

    # ── گام ۶: بررسی نیاز به revision ─────────────────────────────────────
    needs_revision = (
        bool(critic_toon.get("needs_revision", False))
        if isinstance(critic_toon, dict)
        else False
    )
    revision_count = int(state.get("revision_count", 0) or 0)
    max_revisions  = int(state.get("max_revisions",  3) or 3)

    # hard-cap دفاعی: هرگز از max_revisions + 1 بیشتر نمی‌شود
    HARD_CAP = max(max_revisions + 1, 5)

    log_debug(f"  🔄 Revision status:")
    log_debug(f"    - needs_revision: {needs_revision}")
    log_debug(f"    - revision_count: {revision_count}")
    log_debug(f"    - max_revisions:  {max_revisions}")
    log_debug(f"    - hard_cap:       {HARD_CAP}")

    if needs_revision and revision_count < max_revisions and revision_count < HARD_CAP:
        new_revision_count = revision_count + 1
        reason = (
            f"بازبینی درخواست‌شده توسط critic — دور {new_revision_count} از {max_revisions} | "
            f"ایراد: {critic_toon.get('issue', 'نامشخص')} | "
            f"دستور اصلاح: {critic_toon.get('action', '')[:120]}"
        )
        log_info(f"🟢 Supervisor → reasoner (revision {new_revision_count}/{max_revisions})")
        log_debug(f"  🔄 Decision: reasoner ({reason})")
        log_debug("🟢 ═══ SUPERVISOR END ═══\n")

        messages.append(_supervisor_message("reasoner (revision)", reason))
        return {
            "next":           "reasoner",
            "revision_count": new_revision_count,
            "critic_toon":    None,   # ✅ پاک‌سازی critic برای دور بعد
            "draft_toon":     None,   # ✅ پاک‌سازی draft برای بازنویسی
            # draft_raw عمداً نگه داشته می‌شود تا reasoner پاسخ قبلی را ببیند
            "messages":       messages,
            "total_steps":    total_steps + 1,
        }

    # ── گام ۷: finalize ────────────────────────────────────────────────────
    if needs_revision and revision_count >= max_revisions:
        reason = (
            f"سقف بازبینی ({max_revisions} دور) رسیده شد — نهایی‌سازی با بهترین پاسخ موجود | "
            f"آخرین ایراد critic: {critic_toon.get('issue', 'نامشخص')}"
        )
    else:
        reason = "critic پاسخ را تأیید کرد — نیازی به بازبینی نیست"

    log_info("🟢 Supervisor → finalize")
    log_debug(f"  ✅ Decision: finalize ({reason})")
    log_debug("🟢 ═══ SUPERVISOR END ═══\n")

    messages.append(_supervisor_message("finalize", reason))
    return {
        "next":        "finalize",
        "messages":    messages,
        "total_steps": total_steps + 1,
    }