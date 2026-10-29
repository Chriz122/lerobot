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

# ----------------------------------------------------------------------------
# 環境變數預設值 / Environment Variable Defaults
# ----------------------------------------------------------------------------
DEFAULT_CAN_IFACES = "can0,can1,can2,can3"
DEFAULT_CAN_SETUP = True
DEFAULT_CAN_DOWN = False

DEFAULT_PARK = True
DEFAULT_PARK_TIME = 3.0

DEFAULT_HOME_TIME = 8.0
DEFAULT_HOME_AT = "record"
DEFAULT_TABLE_MARGIN = 0.05

DEFAULT_PROFILE = True
DEFAULT_LEADER_TRACK = True
DEFAULT_GC_STATS = True

DEFAULT_GC_SCALE = 1.0
DEFAULT_KD_SCALE = 1.0
DEFAULT_FRICTION_SCALE = 0.0

DEFAULT_DISABLE_RETRY = 3
DEFAULT_ENABLE_RETRY = 3

DEFAULT_FORCE_FEEDBACK_SCALE = 0.05
DEFAULT_FORCE_FEEDBACK_LIMIT = 0.2
DEFAULT_FORCE_FEEDBACK_ALPHA = 0.6
DEFAULT_FORCE_FEEDBACK_DEADBAND = 0.02
DEFAULT_FORCE_FEEDBACK_SIGN = -1.0

# 模組層級常數 (向後相容)
DISABLE_RETRY = int(os.environ.get("OPENARM_DISABLE_RETRY", str(DEFAULT_DISABLE_RETRY)))
ENABLE_RETRY = max(int(os.environ.get("OPENARM_ENABLE_RETRY", str(DEFAULT_ENABLE_RETRY))), 1)


# ----------------------------------------------------------------------------
# 環境變數存取函式 / Environment Variable Accessors
# ----------------------------------------------------------------------------

def get_env_urdf() -> str | None:
    """取得 OPENARM_URDF 路徑。"""
    return os.environ.get("OPENARM_URDF")


def get_env_home() -> str | None:
    """取得 OPENARM_HOME 路徑 (reset.json 或 home.json)。"""
    return os.environ.get("OPENARM_HOME")


def set_env_home(path: str) -> None:
    """設定 OPENARM_HOME 路徑。"""
    os.environ["OPENARM_HOME"] = path


def get_env_home_time(default: float | None = None) -> float:
    """取得 OPENARM_HOME_TIME (移動至起始姿態秒數)。"""
    val = os.environ.get("OPENARM_HOME_TIME")
    if val is not None:
        return float(val)
    return default if default is not None else DEFAULT_HOME_TIME


def get_env_home_at() -> str:
    """取得 OPENARM_HOME_AT ('record' 或 'reset')。"""
    return os.environ.get("OPENARM_HOME_AT", DEFAULT_HOME_AT)


def get_env_table_z() -> float | None:
    """取得 OPENARM_TABLE_Z (桌面世界座標高度)。若未設定回傳 None。"""
    val = os.environ.get("OPENARM_TABLE_Z")
    return float(val) if val is not None else None


def get_env_table_margin() -> float:
    """取得 OPENARM_TABLE_MARGIN (桌面防撞安全餘裕，預設 0.05m)。"""
    return float(os.environ.get("OPENARM_TABLE_MARGIN", str(DEFAULT_TABLE_MARGIN)))


def get_env_can_ifaces() -> list[str]:
    """取得 OPENARM_CAN_IFACES 清單 (預設 can0,can1,can2,can3)。"""
    raw = os.environ.get("OPENARM_CAN_IFACES", DEFAULT_CAN_IFACES)
    return [x.strip() for x in raw.split(",") if x.strip()]


def get_env_can_setup() -> bool:
    """取得 OPENARM_CAN_SETUP (預設 True，為 '0' 時不執行 setup)。"""
    return os.environ.get("OPENARM_CAN_SETUP", "1") != "0"


def get_env_can_down() -> bool:
    """取得 OPENARM_CAN_DOWN (預設 False，為 '1' 時關閉連線時把介面 down)。"""
    return os.environ.get("OPENARM_CAN_DOWN", "0") == "1"


def get_env_park() -> bool:
    """取得 OPENARM_PARK (關閉連線時是否先回到下垂姿態，預設 True)。"""
    return os.environ.get("OPENARM_PARK", "1") != "0"


def get_env_park_time(default: float | None = None) -> float:
    """取得 OPENARM_PARK_TIME (回到下垂姿態秒數，預設 3.0)。"""
    val = os.environ.get("OPENARM_PARK_TIME")
    if val is not None:
        return float(val)
    return default if default is not None else DEFAULT_PARK_TIME


def get_env_leader_track() -> bool:
    """取得 OPENARM_LEADER_TRACK (DAgger 推論時 Leader 是否跟隨 Follower，預設 True)。"""
    return os.environ.get("OPENARM_LEADER_TRACK", "1") != "0"


def get_env_profile() -> bool:
    """取得 OPENARM_PROFILE (是否啟用迴圈效能分析，預設 True)。"""
    return os.environ.get("OPENARM_PROFILE", "1") != "0"


def get_env_leader_side() -> str | None:
    """取得 OPENARM_LEADER_SIDE ('left' 或 'right'，單臂 leader 專用)。"""
    return os.environ.get("OPENARM_LEADER_SIDE")


def get_env_gc_scale() -> float:
    """取得 OPENARM_GC_SCALE (重力補償總倍率，預設 1.0)。"""
    return float(os.environ.get("OPENARM_GC_SCALE", str(DEFAULT_GC_SCALE)))


def get_env_kd_scale() -> float:
    """取得 OPENARM_KD_SCALE (阻尼倍率，預設 1.0)。"""
    return float(os.environ.get("OPENARM_KD_SCALE", str(DEFAULT_KD_SCALE)))


def get_env_friction_scale() -> float:
    """取得 OPENARM_FRICTION_SCALE (摩擦補償倍率，預設 0.0)。"""
    return float(os.environ.get("OPENARM_FRICTION_SCALE", str(DEFAULT_FRICTION_SCALE)))


def get_env_friction_eps() -> float:
    """取得 OPENARM_FRICTION_EPS (摩擦補償速度閾值，預設 0.15 rad/s)。"""
    return float(os.environ.get("OPENARM_FRICTION_EPS", str(FRICTION_VEL_EPS)))


def get_env_control_hz() -> float:
    """取得 OPENARM_CONTROL_HZ (Leader 控制迴圈頻率，預設 200.0 Hz)。"""
    return float(os.environ.get("OPENARM_CONTROL_HZ", str(CONTROL_HZ)))


def get_env_gc_stats() -> bool:
    """取得 OPENARM_GC_STATS (是否印出 Leader 控制統計，預設 True)。"""
    return os.environ.get("OPENARM_GC_STATS", "1") != "0"


def get_env_disable_retry() -> int:
    """取得 OPENARM_DISABLE_RETRY (關閉力矩重試次數，預設 3)。"""
    return int(os.environ.get("OPENARM_DISABLE_RETRY", str(DEFAULT_DISABLE_RETRY)))


def get_env_enable_retry() -> int:
    """取得 OPENARM_ENABLE_RETRY (啟動力矩重試次數，預設 3)。"""
    return max(int(os.environ.get("OPENARM_ENABLE_RETRY", str(DEFAULT_ENABLE_RETRY))), 1)


def get_env_ff_scale() -> float:
    """取得 OPENARM_FF_SCALE (雙向力回饋比例，預設 0.05)。"""
    return float(os.environ.get("OPENARM_FF_SCALE", str(DEFAULT_FORCE_FEEDBACK_SCALE)))


def set_env_ff_scale(val: float | str) -> None:
    """設定 OPENARM_FF_SCALE。"""
    os.environ["OPENARM_FF_SCALE"] = str(val)


def get_env_ff_sign() -> float:
    """取得 OPENARM_FF_SIGN (雙向力回饋正負號，預設 -1.0)。"""
    return float(os.environ.get("OPENARM_FF_SIGN", str(DEFAULT_FORCE_FEEDBACK_SIGN)))


def get_env_ff_limit() -> float:
    """取得 OPENARM_FF_LIMIT (雙向力回饋力矩上限 Nm，預設 0.2)。"""
    return abs(float(os.environ.get("OPENARM_FF_LIMIT", str(DEFAULT_FORCE_FEEDBACK_LIMIT))))


def get_env_ff_alpha() -> float:
    """取得 OPENARM_FF_ALPHA (雙向力回饋低通濾波權重 0~1，預設 0.6)。"""
    raw = float(os.environ.get("OPENARM_FF_ALPHA", str(DEFAULT_FORCE_FEEDBACK_ALPHA)))
    return min(max(raw, 0.0), 1.0)


def get_env_ff_deadband() -> float:
    """取得 OPENARM_FF_DEADBAND (雙向力回饋死區 Nm，預設 0.02)。"""
    return abs(float(os.environ.get("OPENARM_FF_DEADBAND", str(DEFAULT_FORCE_FEEDBACK_DEADBAND))))
