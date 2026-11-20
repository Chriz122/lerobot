"""
單臂運動學（Pinocchio + OPENARM_URDF）與手眼外參工具。

角度慣例：
- 內部一律用 URDF 關節角 (rad)；LeRobot 觀測／命令是 deg，兩者差 JOINT_SIGN。
- HandEye.r 把相機座標的向量轉到 TCP 座標（相機座標：x 右、y 下、z 光軸向前）。
"""
from __future__ import annotations

import os
from typing import Sequence

import numpy as np

from openarm_src.utils.config import JOINT_SIGN

ARM_JOINTS = [f"joint_{i}" for i in range(1, 8)]

# 與 lerobot OpenArmFollower 預設軟限位一致 (deg, LeRobot 角度)，再內縮 LIMIT_MARGIN_DEG
JOINT_LIMITS_DEG = {
    "left": ((-75, 75), (-90, 9), (-85, 85), (0, 135), (-85, 85), (-40, 40), (-80, 80)),
    "right": ((-75, 75), (-9, 90), (-85, 85), (0, 135), (-85, 85), (-40, 40), (-80, 80)),
}
LIMIT_MARGIN_DEG = 1.0


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def rot_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    k = skew(axis)
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def rpy_to_matrix(rpy_deg: Sequence[float]) -> np.ndarray:
    """R = Rz(yaw) @ Ry(pitch) @ Rx(roll)"""
    r, p, y = np.radians(np.asarray(rpy_deg, dtype=float))
    return (rot_axis(np.array([0.0, 0.0, 1.0]), y) @ rot_axis(np.array([0.0, 1.0, 0.0]), p)
            @ rot_axis(np.array([1.0, 0.0, 0.0]), r))


def matrix_to_rpy(r: np.ndarray) -> np.ndarray:
    pitch = np.arcsin(np.clip(-r[2, 0], -1.0, 1.0))
    return np.degrees([np.arctan2(r[2, 1], r[2, 2]), pitch, np.arctan2(r[1, 0], r[0, 0])])


def rotation_angle_deg(r: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0))))


def kabsch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """找旋轉 Q 使 Q @ a_i ≈ b_i（a, b 為 N x 3）。"""
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    return vt.T @ np.diag([1.0, 1.0, d]) @ u.T


def dls_solve(jac: np.ndarray, twist: np.ndarray, damping: float) -> np.ndarray:
    """阻尼最小二乘：dq = J^T (J J^T + d^2 I)^-1 twist"""
    jjt = jac @ jac.T + (damping ** 2) * np.eye(jac.shape[0])
    return jac.T @ np.linalg.solve(jjt, twist)


class ArmModel:
    """OpenArm 單臂（URDF 中 openarm_<side>_joint1..7 與 openarm_<side>_hand_tcp）。"""

    def __init__(self, side: str, urdf_path: str | None = None):
        import pinocchio as pin

        urdf_path = urdf_path or os.environ.get("OPENARM_URDF")
        if not urdf_path or not os.path.isfile(os.path.expanduser(urdf_path)):
            raise RuntimeError("找不到 URDF：請設定 OPENARM_URDF")
        self.pin = pin
        self.side = side
        self.model = pin.buildModelFromUrdf(os.path.expanduser(urdf_path))
        self.data = self.model.createData()
        jids = [self.model.getJointId(f"openarm_{side}_joint{i}") for i in range(1, 8)]
        self.q_idx = [self.model.idx_qs[j] for j in jids]
        self.v_idx = [self.model.idx_vs[j] for j in jids]
        self.fid = self.model.getFrameId(f"openarm_{side}_hand_tcp")
        self.signs = np.array(JOINT_SIGN[side], dtype=float)
        lim = np.array(JOINT_LIMITS_DEG[side], dtype=float)
        self.lower_deg = lim[:, 0] + LIMIT_MARGIN_DEG
        self.upper_deg = lim[:, 1] - LIMIT_MARGIN_DEG

    # LeRobot deg <-> URDF rad
    def deg_to_q(self, deg: np.ndarray) -> np.ndarray:
        return np.radians(np.asarray(deg, dtype=float)) * self.signs

    def q_to_deg(self, q: np.ndarray) -> np.ndarray:
        return np.degrees(np.asarray(q, dtype=float) * self.signs)

    def _full_q(self, q: np.ndarray) -> np.ndarray:
        full = self.pin.neutral(self.model)
        for i, idx in enumerate(self.q_idx):
            full[idx] = float(q[i])
        return full

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.pin.framesForwardKinematics(self.model, self.data, self._full_q(q))
        m = self.data.oMf[self.fid]
        return np.array(m.rotation), np.array(m.translation)

    def jacobian_local(self, q: np.ndarray) -> np.ndarray:
        jac = self.pin.computeFrameJacobian(self.model, self.data, self._full_q(q), self.fid,
                                            self.pin.ReferenceFrame.LOCAL)
        return np.array(jac)[:, self.v_idx]

    def tcp_twist_to_dq(self, q: np.ndarray, twist: np.ndarray, damping: float = 0.02) -> np.ndarray:
        """TCP 座標 6 維位移 [m, m, m, rad, rad, rad] -> 關節增量 (rad)，會避開軟限位。

        直接把超限的關節截掉會讓其他關節的解不再對應原本的位移（例如 TCP 莫名旋轉），
        所以把「會越過限位」的關節從雅可比移除後重解。
        """
        jac = self.jacobian_local(q)
        deg = self.q_to_deg(q)
        free = np.ones(jac.shape[1], dtype=bool)
        dq = np.zeros(jac.shape[1])
        for _ in range(jac.shape[1]):
            dq = dls_solve(jac * free, twist, damping) * free
            new = deg + np.degrees(dq * self.signs)
            ddeg = new - deg
            viol = free & (((new < self.lower_deg) & (ddeg < 0)) | ((new > self.upper_deg) & (ddeg > 0)))
            if not viol.any():
                break
            free &= ~viol
        return dq

    def tcp_translation_to_dq(self, q: np.ndarray, v_tcp: np.ndarray, damping: float = 0.02) -> np.ndarray:
        """TCP 座標下的純平移 (m) -> 關節增量 (rad)，姿態保持不變。"""
        return self.tcp_twist_to_dq(q, np.concatenate([v_tcp, np.zeros(3)]), damping)

    def orientation_error(self, q: np.ndarray, r_ref: np.ndarray) -> np.ndarray:
        """從目前姿態轉到 r_ref 的旋轉向量（TCP 座標，rad）。"""
        r, _ = self.fk(q)
        return np.array(self.pin.log3(r.T @ r_ref))

    def clip_deg(self, deg: np.ndarray) -> np.ndarray:
        return np.clip(deg, self.lower_deg, self.upper_deg)
