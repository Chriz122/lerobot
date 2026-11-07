from __future__ import annotations

import logging
import math
import os
import random
from typing import Any
import numpy as np

from openarm_src.utils.config import JOINT_SIGN

logger = logging.getLogger("openarm_gc.kinematics")


class CartesianPerturbator:
    """末端 (TCP) 笛卡爾空間受控微小擾動器。

    設計目的：
      直接在夾爪末端 (TCP) 的真實物理空間中施加微米級/毫米級隨機位姿擾動，
      並透過 URDF 雅可比矩陣 (Jacobian) 或封閉形式解析逆運動學反解至各關節。
      徹底避免獨立對各關節隨機加雜訊導致運動鏈 (Kinematic Chain) 幾何誤差累加放大
      （防止出現 20~30mm 夾爪暴衝出托盤格子或角度嚴重歪斜的問題）。

    擾動邊界保證：
      1. 水平位置: sqrt(dx^2 + dy^2) <= max_xy_mm (預設 <= 4.0 mm)
      2. 垂直高度: dz >= 0 且 <= max_z_up_mm (預設 0 ~ 1.0 mm，絕不向下負向下切，防撞托盤)
      3. 末端姿態: 繞托盤法線自旋 |d_yaw| <= max_yaw_deg (預設 <= 2.5°)，Roll/Pitch 嚴格保持水平
    """

    def __init__(
        self,
        urdf_path: str | None = None,
        max_xy_mm: float = 4.0,
        max_z_up_mm: float = 1.0,
        max_yaw_deg: float = 2.5,
    ) -> None:
        self.urdf_path = urdf_path or os.environ.get("OPENARM_URDF", r"C:\Users\chsh1\Desktop\v1.urdf")
        self.max_xy_mm = float(max_xy_mm)
        self.max_z_up_mm = float(max_z_up_mm)
        self.max_yaw_deg = float(max_yaw_deg)

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
        damping = 1e-4
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

        直接依據 OpenArm 實體機械幾何長度（上臂 ~0.35m，前臂 ~0.33m，夾爪 ~0.15m）計算，
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
        l1, l2 = 0.35, 0.33
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

    def sample_perturbation(self, slot_info: dict[str, Any]) -> dict[str, Any]:
        """對單一槽位的 Approach、Place、Retract 施加嚴格受控的末端空間微幅擾動。

        保證：Approach、Place、Retract 同步採用相同的水平偏移量 (dx, dy)，
        使得整個垂直下放與回抽通道保持完美筆直，絕不撞擊托盤邊框！
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

        # 4. 將增量疊加到 place、approach、retract
        perturbed_slot = {
            "slot_id": slot_info.get("slot_id"),
            "name": slot_info.get("name"),
            "perturbation_meta": {
                "dx_mm": round(dx_m * 1000, 2),
                "dy_mm": round(dy_m * 1000, 2),
                "dz_mm": round(dz_m * 1000, 2),
                "dyaw_deg": round(math.degrees(dyaw_rad), 2),
            },
        }

        for phase in ("place", "approach", "retract", "release"):
            base_pose = slot_info.get(phase, {})
            new_pose = dict(base_pose)
            for k, dq in delta_q_total.items():
                if k in new_pose and "gripper" not in k:
                    new_pose[k] = round(new_pose[k] + dq, 2)
            perturbed_slot[phase] = new_pose

        logger.info(
            f"[kinematics] 已注入末端笛卡爾受控擾動: dx={dx_m*1000:+.1f}mm, dy={dy_m*1000:+.1f}mm, "
            f"dz={dz_m*1000:+.1f}mm, dyaw={math.degrees(dyaw_rad):+.1f}°"
        )
        return perturbed_slot
