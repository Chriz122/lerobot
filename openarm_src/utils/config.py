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

# 關節步長防跳保護參數 / Step Jump Guard tunables
DEFAULT_MAX_STEP_DEG = float(os.environ.get("OPENARM_MAX_STEP_DEG", "4.0"))          # 每拍最大容許關節移動度數 (30 FPS 下相當於 120 deg/s)
DEFAULT_MAX_STEP_GRIPPER_DEG = float(os.environ.get("OPENARM_MAX_STEP_GRIPPER", "15.0")) # 每拍夾爪最大容許移動度數
STEP_GUARD_WARN_THROTTLE_S = float(os.environ.get("OPENARM_WARN_THROTTLE_S", "2.0"))      # 告警節流間隔 (秒)

# 托盤槽位放置與夾爪擾動參數 / Slot placement & perturbation tunables
DEFAULT_SLOT_MAX_XY_MM = float(os.environ.get("OPENARM_SLOT_MAX_XY_MM", "4.0"))
DEFAULT_SLOT_MAX_Z_MM = float(os.environ.get("OPENARM_SLOT_MAX_Z_MM", "1.0"))
DEFAULT_SLOT_MAX_YAW_DEG = float(os.environ.get("OPENARM_SLOT_MAX_YAW_DEG", "2.5"))

DEFAULT_GRIPPER_MIN_DEG = float(os.environ.get("OPENARM_SLOT_GRIPPER_MIN_DEG", "30.0"))
DEFAULT_GRIPPER_MAX_DEG = float(os.environ.get("OPENARM_SLOT_GRIPPER_MAX_DEG", "60.0"))
DEFAULT_GRIPPER_SIGMA_DEG = float(os.environ.get("OPENARM_SLOT_GRIPPER_SIGMA_DEG", "10.0"))

# 任務與槽位示教參數 / Task & Slot Teaching tunables
DEFAULT_TASK_NAME = os.environ.get("OPENARM_TASK_NAME", "teabag_4slots")
DEFAULT_SLOTS_COUNT = int(os.environ.get("OPENARM_SLOTS_COUNT", "4"))
DEFAULT_APPROACH_OFFSET_DEG = float(os.environ.get("OPENARM_APPROACH_OFFSET_DEG", "12.0"))
DEFAULT_SLOT_FILENAME = "tray_slots.json"

# 軌跡生成與時間參數 / Trajectory Planning & Timing tunables
DEFAULT_TRAJ_FPS = int(os.environ.get("OPENARM_TRAJ_FPS", "30"))
DEFAULT_TRANSIT_TIME_S = float(os.environ.get("OPENARM_TRANSIT_TIME_S", "1.2"))
DEFAULT_PLACE_TIME_S = float(os.environ.get("OPENARM_PLACE_TIME_S", "0.8"))
DEFAULT_RELEASE_TIME_S = float(os.environ.get("OPENARM_RELEASE_TIME_S", "0.4"))
DEFAULT_RETRACT_TIME_S = float(os.environ.get("OPENARM_RETRACT_TIME_S", "0.8"))

# 硬體連線與 CAN 埠號參數 / Hardware Connection & CAN ports
DEFAULT_LEADER_ID = os.environ.get("OPENARM_LEADER_ID", "bi_leader")
DEFAULT_FOLLOWER_ID = os.environ.get("OPENARM_FOLLOWER_ID", "bi_follower")
DEFAULT_LEADER_LEFT_PORT = os.environ.get("OPENARM_LEADER_LEFT_PORT", "can2")
DEFAULT_LEADER_RIGHT_PORT = os.environ.get("OPENARM_LEADER_RIGHT_PORT", "can3")
DEFAULT_FOLLOWER_LEFT_PORT = os.environ.get("OPENARM_FOLLOWER_LEFT_PORT", "can1")
DEFAULT_FOLLOWER_RIGHT_PORT = os.environ.get("OPENARM_FOLLOWER_RIGHT_PORT", "can0")

DEFAULT_CAN_IFACES = [x.strip() for x in os.environ.get("OPENARM_CAN_IFACES", "can0,can1,can2,can3").split(",") if x.strip()]
DEFAULT_CAN_BITRATE = int(os.environ.get("OPENARM_CAN_BITRATE", "1000000"))
DEFAULT_CAN_DATA_BITRATE = int(os.environ.get("OPENARM_CAN_DATA_BITRATE", "5000000"))

# 運動學模型與 URDF 參數 / Kinematics & URDF tunables
DEFAULT_URDF_PATH = os.environ.get("OPENARM_URDF", r"C:\Users\chsh1\Desktop\v1.urdf")
PINOCCHIO_IK_DAMPING = float(os.environ.get("OPENARM_IK_DAMPING", "1e-4"))
OPENARM_UPPER_ARM_LEN_M = float(os.environ.get("OPENARM_UPPER_ARM_LEN_M", "0.35"))
OPENARM_FOREARM_LEN_M = float(os.environ.get("OPENARM_FOREARM_LEN_M", "0.33"))
OPENARM_GRIPPER_LEN_M = float(os.environ.get("OPENARM_GRIPPER_LEN_M", "0.15"))

# 安全與 Homing 參數 / Safety & Homing tunables
DEFAULT_TABLE_MARGIN_M = float(os.environ.get("OPENARM_TABLE_MARGIN", "0.05"))
DEFAULT_HOME_TIME_S = float(os.environ.get("OPENARM_HOME_TIME", "8.0"))
DEFAULT_PARK_TIME_S = float(os.environ.get("OPENARM_PARK_TIME", "3.0"))
DEFAULT_HOMING_RATE_HZ = float(os.environ.get("OPENARM_HOMING_RATE_HZ", "50.0"))
DEFAULT_HOME_AT = os.environ.get("OPENARM_HOME_AT", "record")
DEFAULT_RESET_FILE = os.environ.get("OPENARM_HOME", "reset.json")

# 評估與實機測試參數 / Evaluation tunables
DEFAULT_EVAL_TRIALS = int(os.environ.get("OPENARM_EVAL_TRIALS", "10"))
DEFAULT_EVAL_DURATION_S = float(os.environ.get("OPENARM_EVAL_DURATION_S", "45.0"))
DEFAULT_EVAL_LOG_FILE = os.environ.get("OPENARM_EVAL_LOG_FILE", "eval_results.csv")

# WebRTC 實時串流
DEFAULT_WEBRTC_PORT = int(os.environ.get("OPENARM_WEBRTC_PORT", "8080"))

# 重力補償與力回饋通用擴充
DEFAULT_GC_SCALE = float(os.environ.get("OPENARM_GC_SCALE", "1.0"))
DEFAULT_KD_SCALE = float(os.environ.get("OPENARM_KD_SCALE", "1.0"))
DEFAULT_FRICTION_SCALE = float(os.environ.get("OPENARM_FRICTION_SCALE", "0.0"))
DEFAULT_FF_SIGN = float(os.environ.get("OPENARM_FF_SIGN", "-1.0"))

