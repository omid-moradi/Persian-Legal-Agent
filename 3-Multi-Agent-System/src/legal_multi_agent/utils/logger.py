"""
سیستم logging با سطوح مختلف
"""
import os
from typing import Any

# سطوح logging
QUIET = 0   # فقط خطاها
INFO = 1    # نتایج مهم
DEBUG = 2   # همه چیز

# خواندن از environment
_LOG_LEVEL_STR = os.getenv("LOG_LEVEL", "INFO").upper()
_LOG_LEVEL_MAP = {
    "QUIET": QUIET,
    "INFO": INFO,
    "DEBUG": DEBUG,
}
CURRENT_LOG_LEVEL = _LOG_LEVEL_MAP.get(_LOG_LEVEL_STR, INFO)


def log_debug(*args, **kwargs):
    """لاگ سطح DEBUG (همه چیز)"""
    if CURRENT_LOG_LEVEL >= DEBUG:
        print(*args, **kwargs)


def log_info(*args, **kwargs):
    """لاگ سطح INFO (نتایج مهم)"""
    if CURRENT_LOG_LEVEL >= INFO:
        print(*args, **kwargs)


def log_error(*args, **kwargs):
    """لاگ سطح ERROR (همیشه نمایش)"""
    print(*args, **kwargs)


def log_agent_start(agent_name: str):
    """شروع یک agent"""
    if CURRENT_LOG_LEVEL >= DEBUG:
        print(f"\n{'═'*50}")
        print(f"🔹 {agent_name} START")
        print(f"{'═'*50}")


def log_agent_end(agent_name: str):
    """پایان یک agent"""
    if CURRENT_LOG_LEVEL >= DEBUG:
        print(f"🔹 {agent_name} END\n")


def log_agent_result(agent_name: str, result: Any):
    """نتیجه مهم یک agent (همیشه در INFO نمایش)"""
    if CURRENT_LOG_LEVEL >= INFO:
        print(f"✓ {agent_name}: {result}")
