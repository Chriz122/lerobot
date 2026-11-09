from __future__ import annotations

import logging
import time
from typing import Any

from openarm_src.utils.config import (
    DEFAULT_MAX_STEP_DEG,
    DEFAULT_MAX_STEP_GRIPPER_DEG,
    DEFAULT_STEP_GUARD_ENABLE,
    STEP_GUARD_WARN_THROTTLE_S,
)

logger = logging.getLogger("openarm_gc.step_guard")


class StepJumpGuard:
    """DAgger 與即時控制中的動作跳變防護器 (Step Jump Guard / Rate Limiter)。

    防止：
      1. 人類介入 (CORRECTING) 切換回 Policy 自主運行 (AUTONOMOUS) 時，
         Policy 首拍輸出與人類放開姿態存在落差導致的劇烈階躍跳動 (Step Jump)。
      2. 遙操作 (Teleop) 斷訊、封包遺失或劇烈甩手時導致的瞬間突波。
      3. Policy 出現分佈外 (OOD) 或模型幻覺時輸出危險的大角度瞬變。
    """

    def __init__(
        self,
        max_step_deg: float | None = None,
        max_step_gripper: float | None = None,
        enabled: bool | None = None,
    ) -> None:
        if enabled is None:
            enabled = DEFAULT_STEP_GUARD_ENABLE
        self.enabled = bool(enabled)

        if max_step_deg is None:
            max_step_deg = DEFAULT_MAX_STEP_DEG
        self.max_step_deg = float(max_step_deg)

        if max_step_gripper is None:
            max_step_gripper = DEFAULT_MAX_STEP_GRIPPER_DEG
        self.max_step_gripper = float(max_step_gripper)

        self.last_action: dict[str, float] = {}
        self._last_warn_time: dict[str, float] = {}

    def sync_from_observation(self, obs: dict[str, Any]) -> None:
        """從當前機器人觀測值 (真實關節角度) 同步基準。

        在 DAgger 每次狀態切換（例如 PAUSED -> AUTONOMOUS、PAUSED -> CORRECTING）
        以及起始 homing 完成後呼叫，確保下一拍的動作是相對於機器人「當前真實姿態」進行平滑限制。
        """
        if not isinstance(obs, dict):
            return

        synced_count = 0
        for k, v in obs.items():
            # 先過濾鍵名，避免對相機影像陣列調用 .item()
            if not (k.endswith(".pos") or "joint_" in k or "gripper" in k):
                continue
            if isinstance(v, (int, float)):
                val = float(v)
            elif hasattr(v, "item") and getattr(v, "size", getattr(v, "numel", lambda: 1)()) == 1:
                val = float(v.item())
            else:
                continue
            self.last_action[k] = val
            synced_count += 1

        if synced_count > 0:
            logger.info(f"[step_guard] 已同步真實姿態基準 ({synced_count} 個關節/夾爪)")

    def clamp(self, action: dict[str, Any]) -> dict[str, Any]:
        """限制每個關節單步最大移動量，防止任何突發的 step jump。"""
        if not self.enabled or not isinstance(action, dict):
            return action

        clamped_action = dict(action)
        now = time.monotonic()

        for k, v in action.items():
            if isinstance(v, (int, float)):
                v_float = float(v)
            elif hasattr(v, "item") and getattr(v, "size", getattr(v, "numel", lambda: 1)()) == 1:
                v_float = float(v.item())
            else:
                continue

            if k in self.last_action:
                prev_v = self.last_action[k]
                limit = self.max_step_gripper if "gripper" in k else self.max_step_deg
                delta = v_float - prev_v
                if abs(delta) > limit:
                    # 進行步長箝制
                    clamped_v = prev_v + (limit if delta > 0 else -limit)
                    clamped_action[k] = clamped_v

                    # 節流告警記錄 (每個關節最多每 STEP_GUARD_WARN_THROTTLE_S 秒印一次，避免洗版)
                    if now - self._last_warn_time.get(k, 0) > STEP_GUARD_WARN_THROTTLE_S:
                        logger.warning(
                            f"[step_guard] [警告] 抑制關節步長跳變 ({k}): 目標 delta={delta:+.1f}° > 限制 {limit:.1f}°，"
                            f"已平滑限制為 {clamped_v:.1f}°"
                        )
                        self._last_warn_time[k] = now
                else:
                    clamped_action[k] = v_float
            else:
                # 首次出現的關節鍵，建立基準
                clamped_action[k] = v_float

            self.last_action[k] = clamped_action[k]

        return clamped_action

