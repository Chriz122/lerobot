from __future__ import annotations

import json
import logging
from pathlib import Path
import threading
from typing import Any

from openarm_src.utils.config import (
    DEFAULT_SLOT_NOISE_ENABLE,
    DEFAULT_TASK_NAME,
    DEFAULT_TRAJ_FPS,
    DEFAULT_URDF_PATH,
)
from openarm_src.utils.homing import get_leaders
from openarm_src.utils.slot_teacher import get_task_slot_path
from openarm_src.utils.step_guard import StepJumpGuard
from openarm_src.utils.trajectory_planner import generate_slot_placement_trajectory

logger = logging.getLogger("openarm_gc.hybrid_recorder")


def resolve_slots_config(task_or_path: str | None) -> tuple[dict[str, Any], Path]:
    """解析任務名稱或檔案路徑，載入槽位 Waypoint 設定。

    優先搜尋順序：
      1. 直接指定存在的檔案路徑
      2. openarm_src/tasks/<task_or_path>/tray_slots.json
      3. openarm_src/tasks/default/tray_slots.json
    """
    if task_or_path:
        p = Path(task_or_path)
        if p.is_file():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f), p

        task_p = get_task_slot_path(task_or_path)
        if task_p.is_file():
            with open(task_p, "r", encoding="utf-8") as f:
                return json.load(f), task_p

    # 預設路徑
    default_p = get_task_slot_path(DEFAULT_TASK_NAME)
    if default_p.is_file():
        with open(default_p, "r", encoding="utf-8") as f:
            return json.load(f), default_p

    raise FileNotFoundError(
        f"找不到托盤槽位設定檔！請先執行示教工具：\n"
        f"  python openarm_gc.py teach-slots --task {task_or_path or DEFAULT_TASK_NAME}\n"
        f"將會自動在 openarm_src/tasks/{task_or_path or DEFAULT_TASK_NAME}/tray_slots.json 建立配置。"
    )


def patch_hybrid_record(module, task_or_path: str | None = None):
    """為 lerobot_record 掛載切換式混合錄製功能 (Hybrid Teleop + Script)。

    運作流程：
      1. 平時處於 TELEOP 模式，由操作員手動操作 Leader 夾取茶包。
      2. 操作員按下鍵盤數字鍵 '1'~'9' 或 'Space'（順序循環）時，觸發自動放置腳本。
      3. 系統即時生成從當前抓取姿態至該槽位的 Min-Jerk 平滑軌跡（Approach -> Place -> Release -> Retract）。
      4. 執行期間維持 30 FPS 持續向 Dataset 寫入相機與關節數據，且 Leader 鏡像跟隨。
      5. 放置完成後夾爪張開並抬回安全高度，控制權無縫交回 Teleop，供操作員抓取下一個茶包。
    """
    slots_config, config_path = resolve_slots_config(task_or_path)
    slots = slots_config.get("slots", {})
    num_slots = len(slots)
    logger.info(f"[hybrid] 已載入任務 '{slots_config.get('task')}' 槽位配置 ({num_slots} 個槽位, 來源: {config_path})")

    orig_record_loop = module.record_loop

    def hybrid_record_loop(*args, **kwargs):
        # 取得傳入參數
        from lerobot.scripts.lerobot_record import _call_arguments
        bound = _call_arguments(orig_record_loop, args, kwargs)
        robot = bound.get("robot")
        teleop = bound.get("teleop")
        fps = bound.get("fps", DEFAULT_TRAJ_FPS)

        # 狀態管理
        state_lock = threading.Lock()
        pending_slot_id: str | None = None
        next_seq_idx = 1

        def on_key(name: str) -> None:
            nonlocal pending_slot_id, next_seq_idx
            key = name.lower()
            with state_lock:
                if key in [str(i) for i in range(1, 10)] and key in slots:
                    pending_slot_id = key
                    logger.info(f"[hybrid] 鍵盤觸發：選擇第 {key} 個槽位放置 Script")
                elif key == "space":
                    pending_slot_id = str(next_seq_idx)
                    logger.info(f"[hybrid] 空白鍵觸發：依序執行第 {pending_slot_id} 個槽位放置 Script")
                    next_seq_idx = (next_seq_idx % num_slots) + 1

        # 啟動鍵盤監聽
        from lerobot.utils.keyboard_input import create_key_listener
        listener = create_key_listener(
            on_key,
            controls_help="數字鍵 1~4=指定槽位放置, Space=依序放置, Right/n=下一集, r=重錄, Esc/q=退出",
        )

        leaders = get_leaders(teleop) if teleop is not None else []
        step_guard = StepJumpGuard()

        from openarm_src.utils.kinematics import CartesianPerturbator

        perturb_enabled = DEFAULT_SLOT_NOISE_ENABLE
        perturbator = (
            CartesianPerturbator(urdf_path=DEFAULT_URDF_PATH)
            if perturb_enabled
            else None
        )

        # 攔截 teleop.get_action 或在 record_loop 外層包覆
        # 由於 lerobot 的 record_loop 內部迴圈每 tick 呼叫 teleop.get_action()，
        # 我們可以直接透過 monkey-patch teleop.get_action 來無縫注入 Script 動作！
        orig_get_action = teleop.get_action if teleop is not None else None
        active_trajectory: list[dict[str, float]] = []
        traj_index = 0

        def hybrid_get_action():
            nonlocal active_trajectory, traj_index, pending_slot_id
            with state_lock:
                # 檢查是否有新觸發的槽位放置請求
                if pending_slot_id is not None and not active_trajectory:
                    target_slot = slots.get(pending_slot_id)
                    pending_slot_id = None
                    if target_slot:
                        obs = robot.get_observation()
                        curr_pose = {k: float(v) for k, v in obs.items() if k.endswith(".pos")}
                        active_trajectory = generate_slot_placement_trajectory(
                            curr_pose, target_slot, fps=fps, perturbator=perturbator
                        )
                        traj_index = 0
                        logger.info(f"[hybrid] 開始執行槽位自動放置 ({len(active_trajectory)} 幀 / {len(active_trajectory)/fps:.1f}s)...")

            # 若正在執行 Script 軌跡
            if active_trajectory and traj_index < len(active_trajectory):
                cmd = active_trajectory[traj_index]
                traj_index += 1

                # 讓 Leader 同步鏡像跟隨
                if leaders:
                    for prefix, leader in leaders:
                        leader_cmd = {
                            key[len(prefix):].removesuffix(".pos"): float(v)
                            for key, v in cmd.items()
                            if key.startswith(prefix)
                        }
                        if leader_cmd:
                            leader._gc.set_target(leader_cmd)

                # 軌跡結束時
                if traj_index >= len(active_trajectory):
                    logger.info("[hybrid] 槽位放置完成，釋放 Leader 控制權，恢復 Teleop 手動操作。")
                    active_trajectory = []
                    traj_index = 0
                    if leaders:
                        for _, leader in leaders:
                            leader._gc.set_target(None)
                    # 重新同步 step_guard 基準
                    step_guard.sync_from_observation(robot.get_observation())

                return cmd

            # 正常 Teleop 手動模式
            return orig_get_action() if orig_get_action is not None else {}

        if teleop is not None:
            teleop.get_action = hybrid_get_action

        try:
            return orig_record_loop(*args, **kwargs)
        finally:
            if listener is not None:
                listener.stop()
            if teleop is not None and orig_get_action is not None:
                teleop.get_action = orig_get_action

    module.record_loop = hybrid_record_loop
