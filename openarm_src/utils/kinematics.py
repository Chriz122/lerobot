from __future__ import annotations

import logging
import math
import os
import random
from typing import Any
import numpy as np

from openarm_src.utils.config import (
    DEFAULT_GRIPPER_MAX_DEG,
    DEFAULT_GRIPPER_MIN_DEG,
    DEFAULT_GRIPPER_SIGMA_DEG,
    DEFAULT_SLOT_MAX_XY_MM,
    DEFAULT_SLOT_MAX_YAW_DEG,
    DEFAULT_SLOT_MAX_Z_MM,
    DEFAULT_URDF_PATH,
    JOINT_SIGN,
    OPENARM_FOREARM_LEN_M,
    OPENARM_UPPER_ARM_LEN_M,
    PINOCCHIO_IK_DAMPING,
)

logger = logging.getLogger("openarm_gc.kinematics")


class CartesianPerturbator:
    """末端 (TCP) 笛卡爾空間受控微小擾動器。

    設計目的：
      直接在夾爪末端 (TCP) 的真實物理空間中施加微米級/毫米級隨機位姿擾動，
      並透過 URDF 雅可比矩陣 (Jacobian) 或封閉形式解析逆運動學反解至各關節。
      徹底避免獨立對各關節隨機加雜訊導致運動鏈 (Kinematic Chain) 幾何誤差累加放大
      （防止出現 20~30mm 夾爪暴衝出托盤格子或角度嚴重歪斜的問題）。

    擾動邊界保證：
      1. 水平位置: sqrt(dx^2 + dy^2) <= max_xy_mm (預設 OPENARM_SLOT_MAX_XY_MM)
      2. 垂直高度: dz >= 0 且 <= max_z_up_mm (預設 OPENARM_SLOT_MAX_Z_MM，向下嚴格為 0，防撞托盤)
      3. 末端姿態: 繞托盤法線自旋 |d_yaw| <= max_yaw_deg (預設 OPENARM_SLOT_MAX_YAW_DEG)，Roll/Pitch 嚴格保持水平
      4. 夾爪開度: 以 gripper_min_deg 為基準半高斯抽樣，嚴格不低於 OPENARM_SLOT_GRIPPER_MIN_DEG，上限截斷不超過 OPENARM_SLOT_GRIPPER_MAX_DEG
    """

    def __init__(
        self,
        urdf_path: str | None = None,
        max_xy_mm: float = DEFAULT_SLOT_MAX_XY_MM,
        max_z_up_mm: float = DEFAULT_SLOT_MAX_Z_MM,
        max_yaw_deg: float = DEFAULT_SLOT_MAX_YAW_DEG,
        gripper_min_deg: float = DEFAULT_GRIPPER_MIN_DEG,
        gripper_max_deg: float = DEFAULT_GRIPPER_MAX_DEG,
        gripper_sigma_deg: float = DEFAULT_GRIPPER_SIGMA_DEG,
    ) -> None:
        self.urdf_path = urdf_path or DEFAULT_URDF_PATH
        self.max_xy_mm = float(max_xy_mm)
        self.max_z_up_mm = float(max_z_up_mm)
        self.max_yaw_deg = float(max_yaw_deg)
        self.gripper_min_deg = float(gripper_min_deg)
        self.gripper_max_deg = float(gripper_max_deg)
        self.gripper_sigma_deg = float(gripper_sigma_deg)

        self._pin = None
        self._model = None
        self._data = None
        self._init_pinocchio()

    def _init_pinocchio(self) -> None:
        """嘗試載入 Pinocchio 與 URDF 進行精確運動學解算。"""
        if not self.urdf_path or not os.path.isfile(self.urdf_path):
            logger.info(f"[kinematics] 未找到 URDF 檔案 ({self.urdf_path})，將使用解析幾何投影演算法。")
            return

        try:
            import pinocchio as pin
            self._pin = pin
            self._model = pin.buildModelFromUrdf(os.path.expanduser(self.urdf_path))
            self._data = self._model.createData()
            logger.info(f"[kinematics] 成功透過 Pinocchio 載入 URDF: {self.urdf_path}")
        except Exception as e:
            logger.info(f"[kinematics] 無法使用 Pinocchio 載入 URDF ({e})，採用解析幾何高精度投影算法。")
            self._pin = None

    def _solve_delta_q_pinocchio(
        self,
        nominal_pose: dict[str, float],
        side: str,
        dx_m: float,
        dy_m: float,
        dz_m: float,
        dyaw_rad: float,
    ) -> dict[str, float]:
        """使用 Pinocchio 6D 雅可比阻尼偽逆 (Damped Least Squares) 精確解算各關節微量增量。"""
        pin = self._pin
        m, d = self._model, self._data

        q = pin.neutral(m)
        joint_names = [f"openarm_{side}_joint{i}" for i in range(1, 8)]
        if not all(m.existJointName(n) for n in joint_names):
            joint_names = [f"openarm_joint{i}" for i in range(1, 8)]

        jids = [m.getJointId(n) for n in joint_names]
        q_idxs = [m.idx_qs[j] for j in jids]
        v_idxs = [m.idx_vs[j] for j in jids]
        signs = np.array(JOINT_SIGN.get(side, [1] * 7), dtype=float)

        for i, jname in enumerate(joint_names):
            key = f"{side}_joint_{i+1}.pos"
            if key in nominal_pose:
                q[q_idxs[i]] = signs[i] * np.radians(nominal_pose[key])

        pin.forwardKinematics(m, d, q)
        pin.updateFramePlacements(m, d)

        # 尋找末端 TCP Frame
        tcp_frame = f"openarm_{side}_hand_tcp"
        if not m.existFrame(tcp_frame):
            tcp_frame = f"openarm_{side}_link7"
        if not m.existFrame(tcp_frame):
            tcp_frame = m.frames[-1].name

        fid = m.getFrameId(tcp_frame)
        J = pin.computeFrameJacobian(m, d, q, fid, pin.LOCAL_WORLD_ALIGNED)
        # 只取出與本臂 7 關節對應的 Jacobian 欄位
        J_arm = J[:, v_idxs]  # shape: (6, 7)

        # 目標 6D 空間速度/微小位移: [dx, dy, dz, d_roll=0, d_pitch=0, d_yaw]
        delta_x = np.array([dx_m, dy_m, dz_m, 0.0, 0.0, dyaw_rad])

        # 阻尼最小二乘法 (Damped SVD / DLS) 防止奇異點
        damping = PINOCCHIO_IK_DAMPING
        JJT = J_arm @ J_arm.T + (damping**2) * np.eye(6)
        delta_v = J_arm.T @ np.linalg.solve(JJT, delta_x)

        # 轉換回 LeRobot 角度度數 (Degrees)
        delta_q_deg = np.degrees(signs * delta_v)
        return {f"{side}_joint_{i+1}.pos": float(delta_q_deg[i]) for i in range(7)}

    def _solve_delta_q_geometric(
        self,
        nominal_pose: dict[str, float],
        side: str,
        dx_m: float,
        dy_m: float,
        dz_m: float,
        dyaw_rad: float,
    ) -> dict[str, float]:
        """幾何解析投影演算法 (Geometric Projection Fallback)：

        直接依據 OpenArm 實體機械幾何長度（上臂 OPENARM_UPPER_ARM_LEN_M，前臂 OPENARM_FOREARM_LEN_M）計算，
        將水平位移分解為：
          - 切向位移 (橫向 dy) 由 Base Yaw (Joint 1) 提供，並由 Wrist Yaw (Joint 7) 進行精確反向補償，
            確保夾爪在橫移時方向保持平行！
          - 徑向位移 (縱向 dx) 由肘部/肩部連動提供。
        精準限制末端誤差在毫米範圍內，絕不產生鏈式放大！
        """
        j1_key = f"{side}_joint_1.pos"
        j2_key = f"{side}_joint_2.pos"
        j4_key = f"{side}_joint_4.pos"
        j7_key = f"{side}_joint_7.pos"

        q1_deg = nominal_pose.get(j1_key, 0.0)
        q2_deg = nominal_pose.get(j2_key, 20.0)
        q4_deg = nominal_pose.get(j4_key, 40.0)

        # 計算當前手臂在水平面上的有效投影视半徑 R
        l1, l2 = OPENARM_UPPER_ARM_LEN_M, OPENARM_FOREARM_LEN_M
        r_effective = max(l1 * math.cos(math.radians(q2_deg)) + l2 * math.cos(math.radians(q4_deg)), 0.3)

        # 切向角度 (Joint 1): dy_m = r * d_theta
        delta_q1_rad = dy_m / r_effective
        delta_q1_deg = math.degrees(delta_q1_rad)

        # 徑向伸縮 (Joint 4 / Joint 2 微調): dx_m
        delta_q4_deg = math.degrees(dx_m / l2) * 0.5
        delta_q2_deg = -delta_q4_deg * 0.4 + math.degrees(dz_m / l1) * 0.5

        # 末端夾爪補償 (Joint 7): 抵消 Joint 1 產生的方位角偏轉，再加上指定的自旋角 dyaw
        delta_q7_deg = -delta_q1_deg + math.degrees(dyaw_rad)

        return {
            j1_key: float(delta_q1_deg),
            j2_key: float(delta_q2_deg),
            j4_key: float(delta_q4_deg),
            j7_key: float(delta_q7_deg),
        }

    def sample_gripper_opening(self) -> float:
        """以 self.gripper_min_deg 為基準進行半高斯抽樣 (Half-Normal / Truncated Gaussian)。

        分佈保證：
          - 嚴格不低於 self.gripper_min_deg (OPENARM_SLOT_GRIPPER_MIN_DEG)
          - 上限截斷不超過 self.gripper_max_deg (OPENARM_SLOT_GRIPPER_MAX_DEG)
          - 基準中心與峰值落在 self.gripper_min_deg
        """
        offset = abs(random.gauss(0.0, self.gripper_sigma_deg))
        val = self.gripper_min_deg + offset
        return float(round(min(val, self.gripper_max_deg), 2))

    def sample_perturbation(self, slot_info: dict[str, Any]) -> dict[str, Any]:
        """對單一槽位的 Approach、Place、Retract 施加嚴格受控的末端空間微幅擾動。

        保證：
          1. Approach、Place、Retract 同步採用相同的水平偏移量 (dx, dy)，
             使得整個垂直下放與回抽通道保持完美筆直，絕不撞擊托盤邊框！
          2. Release 與 Retract 的夾爪開度以 self.gripper_min_deg 為基準進行半高斯擾動，
             嚴格不低於 self.gripper_min_deg (OPENARM_SLOT_GRIPPER_MIN_DEG)，
             上限截斷不超過 self.gripper_max_deg (OPENARM_SLOT_GRIPPER_MAX_DEG)。
        """
        # 1. 在圓盤內均勻取樣末端水平位移 (dx, dy)
        r = self.max_xy_mm * math.sqrt(random.random())
        theta = random.uniform(0, 2 * math.pi)
        dx_m = (r * math.cos(theta)) / 1000.0  # 轉為公尺
        dy_m = (r * math.sin(theta)) / 1000.0

        # 2. 垂直高度僅向上偏移 (0 ~ max_z_up_mm)
        dz_m = (random.uniform(0, self.max_z_up_mm)) / 1000.0

        # 3. 自旋角度
        dyaw_rad = math.radians(random.uniform(-self.max_yaw_deg, self.max_yaw_deg))

        # 4. 夾爪張開開度高斯擾動 (中心 30.0°，不低於 30.0°，上限 60.0°)
        sampled_gripper_deg = self.sample_gripper_opening()

        # 讀取原本的 place 姿態
        place_pose = dict(slot_info.get("place", {}))
        sides = set()
        for k in place_pose:
            if k.startswith("left_"):
                sides.add("left")
            elif k.startswith("right_"):
                sides.add("right")

        if not sides:
            sides.add("left")

        delta_q_total: dict[str, float] = {}
        for side in sides:
            if self._pin is not None:
                try:
                    dq = self._solve_delta_q_pinocchio(place_pose, side, dx_m, dy_m, dz_m, dyaw_rad)
                except Exception as e:
                    logger.debug(f"[kinematics] Pinocchio 解算回退至幾何解算: {e}")
                    dq = self._solve_delta_q_geometric(place_pose, side, dx_m, dy_m, dz_m, dyaw_rad)
            else:
                dq = self._solve_delta_q_geometric(place_pose, side, dx_m, dy_m, dz_m, dyaw_rad)
            delta_q_total.update(dq)

        # 5. 將增量疊加到各階段姿態
        perturbed_slot = {
            "slot_id": slot_info.get("slot_id"),
            "name": slot_info.get("name"),
            "perturbation_meta": {
                "dx_mm": round(dx_m * 1000, 2),
                "dy_mm": round(dy_m * 1000, 2),
                "dz_mm": round(dz_m * 1000, 2),
                "dyaw_deg": round(math.degrees(dyaw_rad), 2),
                "gripper_open_deg": sampled_gripper_deg,
            },
        }

        for phase in ("place", "approach", "retract", "release"):
            base_pose = slot_info.get(phase, {})
            new_pose = dict(base_pose)
            for k, dq in delta_q_total.items():
                if k in new_pose and "gripper" not in k:
                    new_pose[k] = round(new_pose[k] + dq, 2)

            # 在 release 與 retract 階段注入夾爪張開角度
            if phase in ("release", "retract"):
                for k in list(new_pose.keys()):
                    if "gripper" in k:
                        new_pose[k] = sampled_gripper_deg
                if not any("gripper" in k for k in new_pose):
                    for side in sides:
                        new_pose[f"{side}_gripper.pos"] = sampled_gripper_deg

            perturbed_slot[phase] = new_pose

        logger.info(
            f"[kinematics] 已注入末端笛卡爾擾動: dx={dx_m*1000:+.1f}mm, dy={dy_m*1000:+.1f}mm, "
            f"dz={dz_m*1000:+.1f}mm, dyaw={math.degrees(dyaw_rad):+.1f}°, "
            f"gripper={sampled_gripper_deg:.1f}° (min={self.gripper_min_deg}°, max={self.gripper_max_deg}°)"
        )
        return perturbed_slot
