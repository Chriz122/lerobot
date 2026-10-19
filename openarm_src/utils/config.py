from __future__ import annotations

import os

# ----------------------------------------------------------------------------
# 可調參數 / Tunables (index 0..6 = joint_1..joint_7)
# ----------------------------------------------------------------------------
ARM_JOINTS = [f"joint_{i}" for i in range(1, 8)]
GRIPPER = "gripper"

MOTOR_NAMES: dict[int, str] = {
    0x01: "joint_1",
    0x02: "joint_2",
    0x03: "joint_3",
    0x04: "joint_4",
    0x05: "joint_5",
    0x06: "joint_6",
    0x07: "joint_7",
    0x08: "gripper",
}

# LeRobot 角度 -> URDF 角度 的正負號。v1 URDF 與 LeRobot 的 joint limit 方向一致，預設全 +1。
# 若 check 時某關節「越補越重」，把該關節改成 -1。
JOINT_SIGN = {"left": [1, 1, 1, 1, 1, 1, 1], "right": [1, 1, 1, 1, 1, 1, 1]}

# 每關節重力補償倍率 (leader 末端是握把，質量與 URDF 的夾爪不同，可微調)
JOINT_GC_SCALE = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# 重力補償模式下的阻尼 (kp=0)。太大手感黏，太小容易晃。
GC_KD = [0.4, 0.4, 0.2, 0.2, 0.05, 0.05, 0.05]
GRIPPER_KD = 0.05

# 摩擦補償 (Nm)：順著移動方向「幫推」一點，抵消減速機的靜摩擦/黏滯感。
# tau_f = FRICTION_COMP * tanh(速度 / FRICTION_VEL_EPS)；太大會自己滑走，從小開始調。
FRICTION_COMP = [0.3, 0.3, 0.25, 0.25, 0.05, 0.05, 0.05]
# rad/s，速度低於此值時補償平滑地趨近 0。馬達回報的速度有雜訊，
# 太小 (例如 0.05) 會讓 tanh 變成近似 sign()，靜止/慢速時補償來回跳 -> 手感一格一格的。
FRICTION_VEL_EPS = 0.15

# homing / 抓住姿態時的 PD 增益 (比 follower 的 240 軟很多，碰到人也安全)
HOLD_KP = [60.0, 60.0, 40.0, 40.0, 8.0, 8.0, 8.0]
HOLD_KD = [3.0, 3.0, 1.5, 1.5, 0.3, 0.3, 0.3]
GRIPPER_HOLD_KP, GRIPPER_HOLD_KD = 5.0, 0.2

# 前饋力矩上限 (Nm) — 防止模型/符號錯誤時暴衝
TAU_LIMIT = [18.0, 18.0, 8.0, 8.0, 2.5, 2.5, 2.5]

CONTROL_HZ = 200.0
STATS_PERIOD = 5.0  # 秒

DISABLE_RETRY = int(os.environ.get("OPENARM_DISABLE_RETRY", "3"))
ENABLE_RETRY = max(int(os.environ.get("OPENARM_ENABLE_RETRY", "3")), 1)

# 力回饋參數 / Force feedback tunables
DEFAULT_FORCE_FEEDBACK_SCALE = 0.05
DEFAULT_FORCE_FEEDBACK_LIMIT = 0.2
DEFAULT_FORCE_FEEDBACK_ALPHA = 0.6
DEFAULT_FORCE_FEEDBACK_DEADBAND = 0.02


