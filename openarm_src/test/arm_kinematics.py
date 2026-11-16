# openarm_src/test/arm_kinematics.py
"""
手臂運動學：把「相機座標系下的位移增量」轉成 7 軸關節增量。

兩種後端，介面相同（fk / jacobian_local）：
- PinocchioArmModel：讀 OPENARM_URDF，真機必須使用。
- ApproxArmModel   ：純 numpy 的 7 軸 S-R-S 近似模型，只供離線模擬，幾何「不是」OpenArm 的真實尺寸。
"""
from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple

import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def rot_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues 旋轉公式"""
    k = skew(axis)
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def rpy_to_matrix(rpy_deg: Sequence[float]) -> np.ndarray:
    r, p, y = np.radians(np.asarray(rpy_deg, dtype=float))
    rx = rot_axis(np.array([1.0, 0.0, 0.0]), r)
    ry = rot_axis(np.array([0.0, 1.0, 0.0]), p)
    rz = rot_axis(np.array([0.0, 0.0, 1.0]), y)
    return rz @ ry @ rx


def rotation_angle_deg(r: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0))))


class ApproxArmModel:
    """7 軸 S-R-S 近似手臂（零位時手臂自然下垂，TCP 的 z 軸沿手臂向外）。"""

    n = 7
    is_exact = False

    def __init__(self, upper_arm_m: float = 0.35, forearm_m: float = 0.33, hand_m: float = 0.15):
        z = np.array([0.0, 0.0, 1.0])
        y = np.array([0.0, 1.0, 0.0])
        self.axes0 = np.array([z, y, z, y, z, y, z])
        shoulder = np.zeros(3)
        elbow = np.array([0.0, 0.0, -upper_arm_m])
        wrist = np.array([0.0, 0.0, -upper_arm_m - forearm_m])
        self.points0 = np.array([shoulder, shoulder, shoulder, elbow, wrist, wrist, wrist])
        self.tcp0 = np.array([0.0, 0.0, -upper_arm_m - forearm_m - hand_m])
        self.r_tcp0 = np.diag([1.0, -1.0, -1.0])   # TCP z 軸指向 -Z（沿手臂向外）
        self.lower = np.radians([-170.0, -120.0, -170.0, -150.0, -170.0, -120.0, -170.0])
        self.upper = -self.lower

    def _chain(self, q: np.ndarray):
        """回傳 (各關節軸在基座座標的方向, 各關節軸上一點, R_tcp, p_tcp)"""
        r = np.eye(3)
        p = np.zeros(3)   # 累積變換 x -> r @ x + p
        axes, points = [], []
        for i in range(self.n):
            axes.append(r @ self.axes0[i])
            points.append(r @ self.points0[i] + p)
            ri = rot_axis(self.axes0[i], float(q[i]))
            # 繞「過 points0[i] 的軸」旋轉：x -> ri @ (x - c) + c
            p = r @ (self.points0[i] - ri @ self.points0[i]) + p
            r = r @ ri
        return np.array(axes), np.array(points), r @ self.r_tcp0, r @ self.tcp0 + p

    def fk(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        _, _, r_tcp, p_tcp = self._chain(np.asarray(q, dtype=float))
        return r_tcp, p_tcp

    def jacobian_local(self, q: np.ndarray) -> np.ndarray:
        """6x7 幾何雅可比，[線速度; 角速度] 以 TCP 座標表示。"""
        axes, points, r_tcp, p_tcp = self._chain(np.asarray(q, dtype=float))
        jac = np.zeros((6, self.n))
        for i in range(self.n):
            jac[:3, i] = r_tcp.T @ np.cross(axes[i], p_tcp - points[i])
            jac[3:, i] = r_tcp.T @ axes[i]
        return jac


class PinocchioArmModel:
    """以 URDF + Pinocchio 計算的精確模型。q 為 URDF 慣例 (rad)，順序 joint1..joint7。"""

    n = 7
    is_exact = True

    def __init__(self, urdf_path: str, side: str = "left"):
        import pinocchio as pin

        self.pin = pin
        self.model = pin.buildModelFromUrdf(os.path.expanduser(urdf_path))
        self.data = self.model.createData()
        names = [f"openarm_{side}_joint{i}" for i in range(1, 8)]
        if not all(self.model.existJointName(n) for n in names):
            names = [f"openarm_joint{i}" for i in range(1, 8)]
        if not all(self.model.existJointName(n) for n in names):
            raise RuntimeError(f"URDF 中找不到 {side} 手臂的 joint1..7：{urdf_path}")
        jids = [self.model.getJointId(n) for n in names]
        self.q_idx = [self.model.idx_qs[j] for j in jids]
        self.v_idx = [self.model.idx_vs[j] for j in jids]
        frame = next(
            (f for f in (f"openarm_{side}_hand_tcp", "openarm_hand_tcp", f"openarm_{side}_link7", "openarm_link7")
             if self.model.existFrame(f)),
            None,
        )
        if frame is None:
            raise RuntimeError("URDF 中找不到 TCP frame (openarm_*_hand_tcp / link7)")
        self.tcp_frame = frame
        self.fid = self.model.getFrameId(frame)
        self.lower = np.array([self.model.lowerPositionLimit[i] for i in self.q_idx])
        self.upper = np.array([self.model.upperPositionLimit[i] for i in self.q_idx])

    def _full_q(self, q: np.ndarray) -> np.ndarray:
        full = self.pin.neutral(self.model)
        for i, idx in enumerate(self.q_idx):
            full[idx] = float(q[i])
        return full

    def fk(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        full = self._full_q(q)
        self.pin.framesForwardKinematics(self.model, self.data, full)
        placement = self.data.oMf[self.fid]
        return np.array(placement.rotation), np.array(placement.translation)

    def jacobian_local(self, q: np.ndarray) -> np.ndarray:
        full = self._full_q(q)
        jac = self.pin.computeFrameJacobian(self.model, self.data, full, self.fid, self.pin.ReferenceFrame.LOCAL)
        return np.array(jac)[:, self.v_idx]


def load_arm_model(side: str = "left", urdf_path: Optional[str] = None, allow_approx: bool = True):
    """有 URDF + pinocchio 就用精確模型；否則（僅限模擬）退回近似模型。"""
    urdf_path = urdf_path or os.environ.get("OPENARM_URDF")
    if urdf_path and os.path.isfile(os.path.expanduser(urdf_path)):
        try:
            model = PinocchioArmModel(urdf_path, side)
            print(f"[Kinematics] 使用 Pinocchio + URDF ({model.tcp_frame})")
            return model
        except ImportError:
            print("[Kinematics] 未安裝 pinocchio。")
    if not allow_approx:
        raise RuntimeError(
            "真機模式需要精確運動學：請安裝 pinocchio 並設定 OPENARM_URDF=/path/to/openarm.urdf"
        )
    print("[Kinematics] 使用近似 7 軸模型（僅供離線模擬，非 OpenArm 真實幾何）")
    return ApproxArmModel()


def dls_solve(jac: np.ndarray, twist: np.ndarray, damping: float) -> np.ndarray:
    """阻尼最小二乘：dq = J^T (J J^T + d^2 I)^-1 twist"""
    jjt = jac @ jac.T + (damping ** 2) * np.eye(jac.shape[0])
    return jac.T @ np.linalg.solve(jjt, twist)


class HandEye:
    """T_tcp_cam：相機在 TCP 座標系中的位姿。"""

    def __init__(self, translation_m: Sequence[float], rpy_deg: Sequence[float],
                 approach_axis_tcp: Sequence[float] = (0.0, 0.0, 1.0)):
        self.p = np.asarray(translation_m, dtype=float)
        self.r = rpy_to_matrix(rpy_deg)
        axis = np.asarray(approach_axis_tcp, dtype=float)
        self.approach = axis / np.linalg.norm(axis)

    def camera_pose(self, r_tcp: np.ndarray, p_tcp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return r_tcp @ self.r, p_tcp + r_tcp @ self.p

    def tcp_to_camera_twist_map(self) -> np.ndarray:
        """6x6：把 TCP 座標的 twist 轉成相機座標的 twist。
        v_c = R^T (v_e + w_e x p),  w_c = R^T w_e
        """
        x = np.zeros((6, 6))
        x[:3, :3] = self.r.T
        x[:3, 3:] = -self.r.T @ skew(self.p)
        x[3:, 3:] = self.r.T
        return x

    def point_on_approach_axis(self, depth_m: float) -> Tuple[np.ndarray, float]:
        """TCP 接近軸上、相機深度為 depth_m 的那一點。
        回傳 (該點在相機座標的位置, 該點沿接近軸離 TCP 原點的距離 a)。
        """
        n_c = self.r.T @ self.approach
        p_c = -self.r.T @ self.p
        if abs(n_c[2]) < 1e-6:
            raise ValueError("相機光軸與夾爪接近軸垂直，無法定義預備位置；請檢查 hand-eye 設定")
        a = (depth_m - p_c[2]) / n_c[2]
        return a * n_c + p_c, float(a)


def camera_twist_to_joint_delta(model, q: np.ndarray, twist_cam: np.ndarray,
                                handeye: HandEye, damping: float) -> np.ndarray:
    """相機座標系下的 6 維位移增量 [m, m, m, rad, rad, rad] -> 關節增量 (rad)。"""
    jac_cam = handeye.tcp_to_camera_twist_map() @ model.jacobian_local(q)
    return dls_solve(jac_cam, np.asarray(twist_cam, dtype=float), damping)


def tcp_twist_to_joint_delta(model, q: np.ndarray, twist_tcp: np.ndarray, damping: float) -> np.ndarray:
    """TCP 座標系下的 6 維位移增量 -> 關節增量 (rad)。"""
    return dls_solve(model.jacobian_local(q), np.asarray(twist_tcp, dtype=float), damping)


def solve_ik(model, r_target: np.ndarray, p_target: np.ndarray, q_seed: np.ndarray,
             iters: int = 300, damping: float = 0.05, tol: float = 1e-5) -> Tuple[np.ndarray, bool]:
    """簡單的數值逆運動學（只在模擬中用來擺初始姿態）。"""
    q = np.asarray(q_seed, dtype=float).copy()
    for _ in range(iters):
        r, p = model.fk(q)
        dp_local = r.T @ (p_target - p)
        r_err = r.T @ r_target
        angle = np.arccos(np.clip((np.trace(r_err) - 1.0) / 2.0, -1.0, 1.0))
        if angle < 1e-9:
            w_local = np.zeros(3)
        else:
            w_local = angle / (2.0 * np.sin(angle)) * np.array(
                [r_err[2, 1] - r_err[1, 2], r_err[0, 2] - r_err[2, 0], r_err[1, 0] - r_err[0, 1]]
            )
        err = np.concatenate([dp_local, w_local])
        if np.linalg.norm(err) < tol:
            return q, True
        step = dls_solve(model.jacobian_local(q), err, damping)
        q = np.clip(q + np.clip(step, -0.2, 0.2), model.lower, model.upper)
    return q, False
