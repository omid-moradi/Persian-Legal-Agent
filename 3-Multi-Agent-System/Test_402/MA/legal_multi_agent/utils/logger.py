"""
سیستم logging با سطوح مختلف
"""
import os
from typing import Any

# ── سطوح logging ────────────────────────────────────────────
QUIET = 0   # فقط خطاها
INFO  = 1   # نتایج مهم
DEBUG = 2   # همه چیز

_LOG_LEVEL_MAP = {
    "QUIET": QUIET,
    "INFO":  INFO,
    "DEBUG": DEBUG,
}


def _get_log_level() -> int:
    """✅ هر بار از environment خوانده می‌شود — تغییر runtime اثر دارد."""
    level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    return _LOG_LEVEL_MAP.get(level_str, INFO)


# ── توابع logging ────────────────────────────────────────────

def log_debug(*args, **kwargs):
    """لاگ سطح DEBUG (همه چیز)"""
    if _get_log_level() >= DEBUG:
        print(*args, **kwargs)

def log_info(*args, **kwargs):
    """لاگ سطح INFO (نتایج مهم)"""
    if _get_log_level() >= INFO:
        print(*args, **kwargs)

def log_error(*args, **kwargs):
    """لاگ سطح ERROR (همیشه نمایش)"""
    print(*args, **kwargs)

def log_agent_start(agent_name: str):
    """شروع یک agent"""
    if _get_log_level() >= DEBUG:
        print(f"\n{'═'*50}")
        print(f"🔹 {agent_name} START")
        print(f"{'═'*50}")

def log_agent_end(agent_name: str):
    """پایان یک agent"""
    if _get_log_level() >= DEBUG:
        print(f"🔹 {agent_name} END\n")

def log_agent_result(agent_name: str, result: Any):
    """نتیجه مهم یک agent (همیشه در INFO نمایش)"""
    if _get_log_level() >= INFO:
        print(f"✓ {agent_name}: {result}")


# ── مقدار فعلی برای import مستقیم (backward compatibility) ──
CURRENT_LOG_LEVEL = _get_log_level()