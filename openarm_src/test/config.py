# openarm_src/test/config.py
"""
OpenArm + RealSense (D455 + D405*2) + AprilTag Cube 實驗配置檔
"""
from dataclasses import dataclass, field
from typing import Dict, Tuple
import numpy as np


@dataclass
class CameraConfig:
    # 相機型號與預設解析度
    name: str
    camera_type: str            # 'D455' 或 'D405'
    serial_number: str = ""     # 空字串代表自動由型號偵測
    width: int = 848            # D405 推薦 848x480 或 1280x720, D455 推薦 848x480 或 1280x720
    height: int = 480
    fps: int = 30
    enable_depth: bool = True


@dataclass
class TargetObjectConfig:
    name: str = "AprilTagCube"
    cube_size_meters: float = 0.05           # 5cm x 5cm x 5cm 立方體
    tag_family: str = "tag36h11"             # AprilTag 家族
    tag_id: int = 0                          # 目標 Tag ID
    tag_size_meters: float = 0.05            # Tag 邊長 5cm x 5cm (滿版貼合 5cm 方塊表面)


@dataclass
class RobotSafetyConfig:
    num_joints: int = 6                      # OpenArm 關節數 (預設 6 軸)
    max_delta_q_rad: float = 0.035           # 單步最大關節增量 (約 2.0 度，保護安全)
    joint_lower_limits: list = field(default_factory=lambda: [-2.8, -1.8, -2.8, -1.8, -2.8, -2.8])
    joint_upper_limits: list = field(default_factory=lambda: [2.8, 1.8, 2.8, 1.8, 2.8, 2.8])
    control_rate_hz: float = 20.0            # 控制迴路更新頻率 (20 Hz)


@dataclass
class IBVSConfig:
    gain_lambda: float = 0.65                # IBVS 比例增益 (保守值，防止超調)
    error_threshold_px: float = 3.0          # 視覺特徵收斂門檻 (像素)
    guard_improvement_margin: float = 0.8    # 收斂守門員之最低改善門檻 delta (px)
    max_steps_per_episode: int = 25          # 單次伺服最大步數
    # 期望特徵點 (Desired Feature s*): [u*, v*] 在 wrist 相機畫面正中心 (848/2, 480/2)
    desired_center_u: float = 424.0
    desired_center_v: float = 240.0
    desired_tag_area: float = 12000.0        # 期望特徵大小 (對應抓取預備距離約 10~12cm)


@dataclass
class SystemConfig:
    cameras: Dict[str, CameraConfig] = field(default_factory=lambda: {
        "ego": CameraConfig(name="ego_cam", camera_type="D455", width=848, height=480, fps=30),
        "wrist_left": CameraConfig(name="wrist_left_cam", camera_type="D405", width=848, height=480, fps=30),
        "wrist_right": CameraConfig(name="wrist_right_cam", camera_type="D405", width=848, height=480, fps=30),
    })
    target: TargetObjectConfig = field(default_factory=TargetObjectConfig)
    robot: RobotSafetyConfig = field(default_factory=RobotSafetyConfig)
    ibvs: IBVSConfig = field(default_factory=IBVSConfig)
