# openarm_src/test/teleop_position.py
"""
主從遙操作擺位：用 leader（主臂）把 follower（從臂）帶到想要的位置，按 Enter 結束。

用途：
- teach 模式：把張開的夾爪擺到「閉合就能夾到方塊」的位置
- approach / compare 模式（加 --teleop）：把手臂帶到腕部相機看得到 Tag 的位置

流程：
1. leader 自然下垂（零點姿態）後連線，開啟重力補償
2. leader 自動走到 follower 目前的姿態（避免一開始跟隨時 follower 跳動）
3. 放開 leader，follower 即時跟隨（每拍有步長限制）
4. 按 Enter：follower 停在原位，leader 回下垂後關力矩
"""
from __future__ import annotations

import threading
import time
from typing import Optional


def default_ports(side: str) -> tuple[str, str]:
    """回傳 (follower 埠, leader 埠) 的預設值，來源為 openarm_src/utils/config.py 的環境變數設定。"""
    from openarm_src.utils import config as oa

    if side == "left":
        return oa.DEFAULT_FOLLOWER_LEFT_PORT, oa.DEFAULT_LEADER_LEFT_PORT
    return oa.DEFAULT_FOLLOWER_RIGHT_PORT, oa.DEFAULT_LEADER_RIGHT_PORT


def teleop_until_enter(robot, leader_port: str, leader_id: Optional[str] = None,
                       purpose: str = "把手臂帶到目標位置", rate_hz: float = 30.0,
                       align_time_s: float = 4.0) -> None:
    """主從遙操作直到使用者按 Enter。結束後 follower 保持在最後姿態。"""
    from lerobot.teleoperators.openarm_leader import OpenArmLeader, OpenArmLeaderConfig
    from openarm_src.utils.config import ARM_JOINTS, GRIPPER
    from openarm_src.utils.patches import install_patches
    from openarm_src.utils.step_guard import StepJumpGuard

    side = robot.side
    install_patches()      # leader 重力補償；中斷連線前先回下垂
    leader = OpenArmLeader(OpenArmLeaderConfig(port=leader_port, id=leader_id or f"openarm_{side}_leader"))
    leader._gc_side = side

    print(f"\n[Teleop] 主臂 {side} (port={leader_port})。LeRobot 連線時會把主臂「當下姿態」設為零點。")
    input("[Teleop] 請先讓主臂自然下垂，確認後按 Enter 連線: ")
    leader.connect()
    try:
        motors = list(ARM_JOINTS) + [GRIPPER]
        obs = robot.arm.get_observation()
        follower_pose = {m: float(obs[f"{m}.pos"]) for m in motors}

        print(f"[Teleop] 主臂將在 {align_time_s:.0f} 秒內走到從臂目前的姿態，請勿阻擋。")
        leader._gc.move_to(follower_pose, align_time_s)
        time.sleep(0.3)
        leader._gc.set_target(None)        # 放開 -> 回到重力補償，可用手帶動

        guard = StepJumpGuard(enabled=True)
        guard.sync_from_observation({f"{m}.pos": v for m, v in follower_pose.items()})

        stop = threading.Event()
        threading.Thread(target=lambda: (input(), stop.set()), daemon=True).start()
        print(f"[Teleop] 主從控制已開啟：請用主臂{purpose}，完成後按 Enter。")

        period = 1.0 / rate_hz
        while not stop.is_set():
            t0 = time.perf_counter()
            action = {k: float(v) for k, v in leader.get_action().items() if k.endswith(".pos") and v is not None}
            robot.arm.send_action(guard.clamp(action))
            time.sleep(max(period - (time.perf_counter() - t0), 0.0))

        # 從臂停在原位，並把內部命令角度對齊到實際角度
        time.sleep(0.3)
        robot.resync()
        print("[Teleop] 主從控制結束，從臂保持目前姿態；主臂回下垂後關閉力矩。")
    finally:
        leader.disconnect()
