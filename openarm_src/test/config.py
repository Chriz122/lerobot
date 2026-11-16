# openarm_src/test/config.py
"""
OpenArm + RealSense (D455 + D405*2) + AprilTag Cube 實驗配置檔。

單位慣例（整個套件一致）：
- 影像特徵 s = [u(px), v(px), w(px), psi(deg)]
    u, v : Tag 中心像素座標
    w    : Tag 視邊長（sqrt(像素面積)），等效於深度：w = f * L / Z
    psi  : Tag 在影像中的旋轉角，對 90 度取模（方塊四面對稱）
- 控制命令 a = [dx(mm), dy(mm), dz(mm), dyaw(deg)]：相機座標系下的單步位移增量
- 關節角：內部一律用 rad（URDF 慣例），送到 LeRobot 時才轉成 deg
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class CameraConfig:
    name: str
    camera_type: str            # 'D455' 或 'D405'
    serial_number: str = ""     # 空字串代表由 pyrealsense2 自動挑選（多台相機時務必填寫）
    width: int = 848
    height: int = 480
    fps: int = 30
    enable_depth: bool = False  # IBVS 只用彩色影像；深度由 Tag 視邊長推得


@dataclass
class TargetObjectConfig:
    name: str = "AprilTagCube"
    cube_size_meters: float = 0.05           # 5cm x 5cm x 5cm 立方體
    tag_family: str = "tag36h11"
    tag_id: int = 0
    # Tag「黑色外框」的實際邊長。若列印時留了白邊，這裡要填黑框邊長而不是 5cm，
    # 否則由 w 推得的深度會等比例偏掉（示教模式不受影響）。
    tag_size_meters: float = 0.05


@dataclass
class HandEyeConfig:
    """腕部相機相對 TCP 的名義外參 T_tcp_cam（相機座標：x 右、y 下、z 光軸向前）。

    只需要「大致正確」：殘差模型會學名義值與實際值的差異。
    但若誤差太大（例如軸向差 90 度），請先量測修正或改用示教模式。
    """
    translation_m: Tuple[float, float, float] = (0.0, -0.04, -0.08)
    rpy_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0)   # R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    # TCP 座標系中「夾爪前進（接近物體）」的方向
    approach_axis_tcp: Tuple[float, float, float] = (0.0, 0.0, 1.0)


@dataclass
class RobotSafetyConfig:
    num_joints: int = 7                      # OpenArm 單臂 7 軸（不含夾爪）
    side: str = "left"
    max_delta_q_deg: float = 2.0             # 單步、單關節最大增量（超過則整個向量等比例縮小）
    control_rate_hz: float = 20.0
    settle_time_s: float = 0.25              # 試探動作後等待手臂靜止的時間（真機）
    max_tracking_lag_deg: float = 6.0        # 命令角與實際角差距上限，避免命令越積越遠
    max_tcp_travel_m: float = 0.45           # 單次任務 TCP 可離開起點的最大距離
    ik_damping: float = 0.02                 # 阻尼最小二乘的阻尼項
    # 與 lerobot OpenArmFollower 的預設軟限位一致 (deg)
    joint_limits_deg: Dict[str, Tuple[Tuple[float, float], ...]] = field(default_factory=lambda: {
        "left": ((-75, 75), (-90, 9), (-85, 85), (0, 135), (-85, 85), (-40, 40), (-80, 80)),
        "right": ((-75, 75), (-9, 90), (-85, 85), (0, 135), (-85, 85), (-40, 40), (-80, 80)),
    })
    limit_margin_deg: float = 1.0
    # 夾爪角度 (deg)。OpenArm follower 軟限位為 [-65, 0]，0 為閉合。上機前請先確認正負號。
    gripper_open_deg: float = -55.0
    gripper_closed_deg: float = 0.0
    gripper_step_deg: float = 3.0
    gripper_hold_threshold_deg: float = 6.0  # 閉合後仍離全閉這麼遠 => 判定夾到東西


@dataclass
class IBVSConfig:
    gain_lambda: float = 0.5                 # IBVS 比例增益
    # 收斂門檻（三項都要滿足，且連續 settle_frames 幀）
    center_tol_px: float = 3.0
    size_tol_px: float = 3.0
    yaw_tol_deg: float = 2.0
    settle_frames: int = 3
    max_steps_per_episode: int = 220
    # 單步命令上限
    max_step_xy_mm: float = 4.0
    max_step_z_mm: float = 4.0
    max_step_yaw_deg: float = 1.5
    # 誤差範數中 1 度偏轉角等效幾個像素
    yaw_weight_px_per_deg: float = 2.0
    # 守門：候選命令預測誤差至少要比 IBVS 好 guard_margin_px + guard_margin_rel * ||e||
    guard_margin_px: float = 0.05
    guard_margin_rel: float = 0.002
    # 實測監看：採用殘差命令後，實際誤差連續幾步沒有下降就停用殘差（該回合）
    watchdog_strikes: int = 3
    watchdog_min_decrease_px: float = 0.0
    # 發散中止：誤差超過 起始誤差 * ratio + offset 就停下（任何方法都適用）
    diverge_ratio: float = 1.5
    diverge_offset_px: float = 40.0
    # 深度閘門：中心誤差 <= lo 時全速前進，>= hi 時完全不前進（先對準再靠近，避免 Tag 出畫面）
    depth_gate_lo_px: float = 40.0
    depth_gate_hi_px: float = 130.0
    border_margin_px: float = 12.0           # Tag 角點離畫面邊緣小於此值時不再前進
    max_lost_frames: int = 15                # 連續遺失 Tag 幾幀就中止
    # 預備抓取位置：相機到 Tag 表面的距離
    standoff_depth_m: float = 0.16
    desired_yaw_deg: float = 0.0             # 夾爪相對方塊的期望偏轉角（對 90 度取模）


@dataclass
class GraspConfig:
    grasp_depth_below_tag_m: float = 0.02    # TCP 最後停在 Tag 表面下方多深（夾在方塊中段）
    descend_step_mm: float = 3.0
    lift_height_m: float = 0.08
    max_descent_m: float = 0.20              # 開迴路下降的硬上限


@dataclass
class ProbeConfig:
    """安全微幅試探 (dithering) 參數"""
    n_samples: int = 120
    amp_xy_mm: float = 2.5
    amp_z_mm: float = 2.5
    amp_yaw_deg: float = 1.2
    box_center_px: float = 60.0              # 試探時 Tag 中心允許離起點多遠
    box_size_ratio: float = 0.12             # 試探時 Tag 視邊長允許變化比例
    val_fraction: float = 0.2                # 留作獨立驗證的比例


@dataclass
class SimConfig:
    """離線模擬中「真實世界」與名義模型的差異（控制器看不到這些值）"""
    focal_px: float = 430.0
    handeye_rot_err_deg: Tuple[float, float, float] = (3.0, -2.0, 15.0)  # roll, pitch, yaw
    handeye_trans_err_m: Tuple[float, float, float] = (0.004, -0.003, 0.005)
    link_scale_err: float = 0.04             # 真實連桿比名義長 4%
    joint_tracking: float = 0.9              # 每步只走到命令的 90%（模擬追蹤落後）
    joint_noise_deg: float = 0.004
    pixel_noise_px: float = 0.15
    start_height_m: float = 0.30             # 初始相機到 Tag 的大約距離
    xy_range_m: float = 0.045                # 方塊相對初始光軸的隨機水平偏移
    yaw_range_deg: float = 30.0
    grasp_xy_tol_mm: float = 8.0             # 模擬中判定「夾得到」的幾何容差
    grasp_z_tol_mm: float = 12.0
    grasp_yaw_tol_deg: float = 10.0


@dataclass
class SystemConfig:
    cameras: Dict[str, CameraConfig] = field(default_factory=lambda: {
        "ego": CameraConfig(name="ego_cam", camera_type="D455"),
        "wrist_left": CameraConfig(name="wrist_left_cam", camera_type="D405"),
        "wrist_right": CameraConfig(name="wrist_right_cam", camera_type="D405"),
    })
    target: TargetObjectConfig = field(default_factory=TargetObjectConfig)
    robot: RobotSafetyConfig = field(default_factory=RobotSafetyConfig)
    ibvs: IBVSConfig = field(default_factory=IBVSConfig)
    handeye: HandEyeConfig = field(default_factory=HandEyeConfig)
    grasp: GraspConfig = field(default_factory=GraspConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    table_z_m: Optional[float] = None        # 桌面在基座座標的高度；None 則由 OPENARM_TABLE_Z 讀取
    table_margin_m: float = 0.01
