from __future__ import annotations

import logging
import time

from openarm_src.utils.config import DISABLE_RETRY

logger = logging.getLogger("openarm_gc.safety")

# ----------------------------------------------------------------------------
# 連線中的手臂清單 + 安全關力矩 / registry of connected arms + safe torque-off
# ----------------------------------------------------------------------------
_CONNECTED: list = []  # 每支 OpenArmLeader / OpenArmFollower 連上後加入，關閉後移除
_FINALIZED = False


def register_arm(arm):
    if arm not in _CONNECTED:
        _CONNECTED.append(arm)


def unregister_arm(arm):
    if arm in _CONNECTED:
        _CONNECTED.remove(arm)


def bus_connected(arm) -> bool:
    try:
        return bool(arm.bus.is_connected)
    except Exception:  # noqa: BLE001
        return False


def safe_disable(arm, label: str = ""):
    """對一支手臂重送 disable 數次；失敗只記錄，不往外丟例外。"""
    if not bus_connected(arm):
        return
    for i in range(DISABLE_RETRY):
        try:
            arm.bus.disable_torque()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[safe] {label or arm}: 關閉力矩第 {i + 1} 次失敗 ({e})")
        time.sleep(0.05)


def final_safety_net():
    """最後一道保險：任何還連著的手臂都關力矩。正常、Ctrl+C、出錯都會跑。"""
    global _FINALIZED
    if _FINALIZED:
        return
    _FINALIZED = True
    left = [a for a in list(_CONNECTED) if bus_connected(a)]
    if left:
        logger.warning(f"[safe] 結束時仍有 {len(left)} 支手臂沒被正常關閉 -> 現在補關（先回下垂再關力矩）")
    for arm in left:
        try:
            arm.disconnect()  # 已被 patch：會先回下垂、停執行緒、重送 disable
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[safe] {arm}: 正常關閉失敗 ({e}) -> 直接關力矩，請扶好手臂")
            gc = getattr(arm, "_gc", None)
            if gc is not None:
                gc.stop()
            safe_disable(arm)
        unregister_arm(arm)

