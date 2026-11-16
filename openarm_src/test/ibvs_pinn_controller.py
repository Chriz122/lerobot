# openarm_src/test/ibvs_pinn_controller.py
"""
IBVS + Physics-Informed Learned Residual + Convergence Guard 控制器（4 自由度版）。

對應《IBVS_PINN_學生快速驗證_繁體中文.docx》第 2~4 節，差別只在維度：
套件示範是 2 維 (u, v)；要「靠近」方塊必須再控制深度與偏轉角，因此擴成 4 維。

    特徵   s = [u, v, w, psi]            (px, px, px, deg)
    命令   a = [dx, dy, dz, dyaw]        (mm, mm, mm, deg)，相機座標系下的單步增量
    (A) e      = s - s*
    (B) u_IBVS = -lambda * J_phys^+ * e
    (C) J_t    = J_phys + dJ_theta，其中 dJ_theta = J_phys * M_theta
    (D) u_res  = -lambda * J_t^+ * e - u_IBVS
    (E) u_nom  = u_IBVS + u_res
    守門：||e + J_t u_nom|| + delta < ||e + J_t u_IBVS|| 才採用 u_nom，否則退回 u_IBVS
    實測監看：守門只是「模型預測」；另外監看實際量到的誤差，採用殘差後連續沒下降就停用殘差

J_phys 是針孔相機模型的解析 interaction matrix（只跟 s、深度、焦距有關）。
殘差寫成乘法形式 J_phys * M 是因為 J_phys 隨深度變化約 1/Z：手眼外參、運動學的誤差
大致是「命令被扭曲了多少」(M)，與深度無關，這樣在遠處試探學到的修正才能用到近處。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from config import IBVSConfig
from tag_detector import wrap_quarter_turn

FEATURE_DIM = 4
ACTION_DIM = 4


@dataclass
class Intrinsics:
    f: float
    cx: float
    cy: float
    width: int
    height: int

    @staticmethod
    def from_matrix(camera_matrix: np.ndarray, width: int, height: int) -> "Intrinsics":
        k = np.asarray(camera_matrix, dtype=float)
        return Intrinsics(f=0.5 * (k[0, 0] + k[1, 1]), cx=k[0, 2], cy=k[1, 2], width=width, height=height)


@dataclass
class ControlOutput:
    action: np.ndarray          # 最終命令 [mm, mm, mm, deg]（已限幅）
    used_residual: bool
    error: np.ndarray           # 原始誤差 e = s - s*
    error_norm: float           # 加權誤差範數（像素等效）
    u_ibvs: np.ndarray
    u_res: np.ndarray
    depth_m: float


def feature_error(s: np.ndarray, s_star: np.ndarray) -> np.ndarray:
    e = np.asarray(s, dtype=float) - np.asarray(s_star, dtype=float)
    e[3] = wrap_quarter_turn(e[3])
    return e


def depth_from_size(side_px: float, focal_px: float, tag_size_m: float) -> float:
    """w = f * L / Z  =>  Z = f * L / w"""
    return focal_px * tag_size_m / max(float(side_px), 1e-6)


def interaction_matrix(s: np.ndarray, depth_m: float, intr: Intrinsics) -> np.ndarray:
    """解析 interaction matrix：ds = J_phys @ a。

    由標準點特徵 interaction matrix 取 (vx, vy, vz, wz) 四欄，再加上兩列：
        w   = f L / Z   =>  dw   = (w / Z) * dz
        psi             =>  dpsi = -dyaw
    """
    u, v, w, _ = s
    z_mm = max(depth_m, 0.03) * 1000.0
    xb, yb = u - intr.cx, v - intr.cy
    k = np.pi / 180.0
    return np.array([
        [-intr.f / z_mm, 0.0, xb / z_mm, yb * k],
        [0.0, -intr.f / z_mm, yb / z_mm, -xb * k],
        [0.0, 0.0, w / z_mm, 0.0],
        [0.0, 0.0, 0.0, -1.0],
    ])


def action_to_camera_twist(action: np.ndarray) -> np.ndarray:
    """[mm, mm, mm, deg] -> 相機座標 6 維位移增量 [m, m, m, rad, rad, rad]"""
    a = np.asarray(action, dtype=float)
    return np.array([a[0] / 1000.0, a[1] / 1000.0, a[2] / 1000.0, 0.0, 0.0, np.radians(a[3])])


class RidgeResidual:
    """M 為常數 4x4：最小化 sum ||W (ds - J (I + M) a)||^2 + beta ||M||^2（有解析解）。"""

    def __init__(self, beta: float = 0.5):
        self.beta = beta
        self.m = np.zeros((ACTION_DIM, ACTION_DIM))

    def fit(self, jac: np.ndarray, actions: np.ndarray, targets: np.ndarray,
            weights: np.ndarray, inputs: np.ndarray) -> None:
        rows, rhs = [], []
        for j, a, r in zip(jac, actions, targets):
            # J M a = kron(a^T, J) vec(M)，vec 為 column-major
            rows.append(np.kron(a[None, :], weights[:, None] * j))
            rhs.append(weights * r)
        design = np.vstack(rows)
        target = np.concatenate(rhs)
        sol = np.linalg.solve(design.T @ design + self.beta * np.eye(design.shape[1]), design.T @ target)
        self.m = sol.reshape(ACTION_DIM, ACTION_DIM, order="F")

    def correction(self, inputs: np.ndarray) -> np.ndarray:
        return self.m


class TorchResidual:
    """M_theta(phi)：小型 MLP，輸入 phi = [x/f, y/f, ln(Z/Z*)]。需要 PyTorch。"""

    def __init__(self, beta: float = 0.02, epochs: int = 400, seed: int = 0):
        import torch

        self.torch = torch
        torch.manual_seed(seed)
        self.beta = beta
        self.epochs = epochs
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, 32), torch.nn.Tanh(),
            torch.nn.Linear(32, 32), torch.nn.Tanh(),
            torch.nn.Linear(32, ACTION_DIM * ACTION_DIM),
        )
        # 最後一層歸零：訓練前 M = 0，控制器等同純 IBVS
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)
        self.input_lo = -np.inf * np.ones(3)
        self.input_hi = np.inf * np.ones(3)

    def fit(self, jac: np.ndarray, actions: np.ndarray, targets: np.ndarray,
            weights: np.ndarray, inputs: np.ndarray) -> None:
        torch = self.torch
        # 記下試探涵蓋的輸入範圍；之後超出範圍的輸入一律夾回邊界，不讓 MLP 外插
        self.input_lo, self.input_hi = inputs.min(axis=0), inputs.max(axis=0)
        j_t = torch.tensor(jac, dtype=torch.float32)
        a_t = torch.tensor(actions, dtype=torch.float32)
        r_t = torch.tensor(targets, dtype=torch.float32)
        w_t = torch.tensor(weights, dtype=torch.float32)
        x_t = torch.tensor(inputs, dtype=torch.float32)
        optim = torch.optim.Adam(self.net.parameters(), lr=5e-3)
        for _ in range(self.epochs):
            m = self.net(x_t).reshape(-1, ACTION_DIM, ACTION_DIM)
            pred = torch.bmm(torch.bmm(j_t, m), a_t.unsqueeze(-1)).squeeze(-1)
            loss = ((w_t * (pred - r_t)) ** 2).mean() + self.beta * (m ** 2).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()

    def correction(self, inputs: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            clipped = np.clip(np.asarray(inputs, dtype=float), self.input_lo, self.input_hi)
            x = self.torch.tensor(clipped.astype(np.float32)).unsqueeze(0)
            return self.net(x).numpy().reshape(ACTION_DIM, ACTION_DIM).astype(float)


class IBVSPINNController:
    def __init__(self, config: IBVSConfig, tag_size_m: float, learner: str = "ridge"):
        self.cfg = config
        self.tag_size_m = tag_size_m
        self.learner_type = learner
        self.weights = np.array([1.0, 1.0, 1.0, config.yaw_weight_px_per_deg])
        self.limits = np.array([config.max_step_xy_mm, config.max_step_xy_mm,
                                config.max_step_z_mm, config.max_step_yaw_deg])
        self.model = RidgeResidual() if learner == "ridge" else TorchResidual()
        self.is_trained = False
        self.is_trusted = False
        self.fit_report: Dict[str, float] = {}
        self.reset_episode()

    def reset_episode(self) -> None:
        """每回合開始時呼叫：清除實測監看狀態。"""
        self._last_used = False
        self._last_norm: Optional[float] = None
        self._strikes = 0
        self.residual_disabled = False

    def _watch_outcome(self, e_norm: float) -> None:
        """實測監看（補強模型預測式守門）：採用殘差命令後，實際量到的誤差若連續
        watchdog_strikes 次沒有下降，本回合就停用殘差、只用 IBVS。"""
        if self._last_used and self._last_norm is not None:
            if e_norm > self._last_norm - self.cfg.watchdog_min_decrease_px:
                self._strikes += 1
            else:
                self._strikes = 0
            if self._strikes >= self.cfg.watchdog_strikes:
                self.residual_disabled = True
        self._last_norm = e_norm

    # ---------------------------------------------------------------- 基本量
    def weighted_norm(self, e: np.ndarray) -> float:
        return float(np.linalg.norm(self.weights * e))

    def bound_action(self, a: np.ndarray) -> np.ndarray:
        """等比例縮小，使每個分量都在單步上限內（保持方向）。"""
        ratio = float(np.max(np.abs(a) / self.limits))
        return a / ratio if ratio > 1.0 else a

    def residual_inputs(self, s: np.ndarray, depth_m: float, intr: Intrinsics) -> np.ndarray:
        return np.array([
            (s[0] - intr.cx) / intr.f,
            (s[1] - intr.cy) / intr.f,
            np.log(max(depth_m, 1e-3) / self.cfg.standoff_depth_m),
        ])

    def is_converged(self, e: np.ndarray) -> bool:
        return bool(
            np.hypot(e[0], e[1]) <= self.cfg.center_tol_px
            and abs(e[2]) <= self.cfg.size_tol_px
            and abs(e[3]) <= self.cfg.yaw_tol_deg
        )

    def depth_gate(self, e: np.ndarray, corners: Optional[np.ndarray], intr: Intrinsics) -> float:
        """0~1：中心還沒對準、或 Tag 已貼近畫面邊緣時，不讓相機繼續前進。"""
        cfg = self.cfg
        center_err = float(np.hypot(e[0], e[1]))
        gate = np.clip((cfg.depth_gate_hi_px - center_err) / (cfg.depth_gate_hi_px - cfg.depth_gate_lo_px), 0.0, 1.0)
        if corners is not None and e[2] < 0.0:   # e_w < 0 代表還要靠近（Tag 會變大）
            m = cfg.border_margin_px
            near_border = (
                corners[:, 0].min() < m or corners[:, 1].min() < m
                or corners[:, 0].max() > intr.width - m or corners[:, 1].max() > intr.height - m
            )
            if near_border:
                gate = 0.0
        return float(gate)

    # ---------------------------------------------------------------- 殘差訓練
    def fit_residual(self, s_hist: np.ndarray, a_hist: np.ndarray, ds_hist: np.ndarray,
                     intr: Intrinsics, val_fraction: float = 0.2, seed: int = 0) -> Dict[str, float]:
        """
        L_train = || ds_obs - (J_phys + dJ_theta) a ||^2 + beta || dJ_theta ||^2
        並保留獨立驗證資料，比較「純物理」與「物理 + 殘差」的影像位移預測誤差。
        """
        s_hist = np.asarray(s_hist, dtype=float)
        a_hist = np.asarray(a_hist, dtype=float)
        ds_hist = np.asarray(ds_hist, dtype=float)
        n = len(s_hist)
        if n < 20:
            raise ValueError(f"試探樣本太少 ({n} 筆)，無法訓練殘差模型")

        depths = np.array([depth_from_size(s[2], intr.f, self.tag_size_m) for s in s_hist])
        jac = np.array([interaction_matrix(s, z, intr) for s, z in zip(s_hist, depths)])
        inputs = np.array([self.residual_inputs(s, z, intr) for s, z in zip(s_hist, depths)])
        phys_pred = np.einsum("nij,nj->ni", jac, a_hist)
        targets = ds_hist - phys_pred            # 觀測位移 - 物理模型預測

        order = np.random.default_rng(seed).permutation(n)
        n_val = max(int(round(n * val_fraction)), 4)
        val, train = order[:n_val], order[n_val:]

        self.model.fit(jac[train], a_hist[train], targets[train], self.weights, inputs[train])
        self.is_trained = True

        err_phys, err_learn = [], []
        for i in val:
            j_hat = jac[i] @ (np.eye(ACTION_DIM) + self.model.correction(inputs[i]))
            err_phys.append(self.weighted_norm(ds_hist[i] - phys_pred[i]))
            err_learn.append(self.weighted_norm(ds_hist[i] - j_hat @ a_hist[i]))
        self.fit_report = {
            "n_train": float(len(train)),
            "n_val": float(len(val)),
            "val_mae_physics": float(np.mean(err_phys)),
            "val_mae_learned": float(np.mean(err_learn)),
        }
        # 學習後在獨立資料上沒有比物理模型好 => 不信任，guarded 模式一律退回 IBVS
        self.is_trusted = self.fit_report["val_mae_learned"] < self.fit_report["val_mae_physics"]
        print(f"[PINN Controller] {self.learner_type} 殘差訓練完成："
              f"訓練 {len(train)} 筆 / 驗證 {len(val)} 筆；"
              f"驗證集影像位移誤差 physics={self.fit_report['val_mae_physics']:.3f} "
              f"learned={self.fit_report['val_mae_learned']:.3f} (像素等效/筆)"
              f"{'' if self.is_trusted else '  -> 殘差未通過驗證，guarded 將一律退回 IBVS'}")
        return self.fit_report

    def estimated_jacobian(self, s: np.ndarray, depth_m: float, intr: Intrinsics) -> np.ndarray:
        """J_t = J_phys + dJ_theta = J_phys (I + M_theta)"""
        j_phys = interaction_matrix(s, depth_m, intr)
        if not self.is_trained:
            return j_phys
        m = self.model.correction(self.residual_inputs(s, depth_m, intr))
        return j_phys @ (np.eye(ACTION_DIM) + m)

    # ---------------------------------------------------------------- 控制律
    def compute_action(self, s: np.ndarray, s_star: np.ndarray, intr: Intrinsics,
                       mode: str = "guarded", corners: Optional[np.ndarray] = None) -> ControlOutput:
        """mode: 'ibvs' | 'learned' | 'guarded'"""
        e = feature_error(s, s_star)                              # (A)
        e_norm = self.weighted_norm(e)
        depth_m = depth_from_size(s[2], intr.f, self.tag_size_m)

        e_ctrl = e.copy()
        e_ctrl[2] *= self.depth_gate(e, corners, intr)

        lam = self.cfg.gain_lambda
        j_phys = interaction_matrix(s, depth_m, intr)
        u_ibvs = self.bound_action(-lam * (np.linalg.pinv(j_phys) @ e_ctrl))   # (B)
        zero = np.zeros(ACTION_DIM)

        if mode == "guarded":
            self._watch_outcome(e_norm)
        self._last_used = False
        if mode == "ibvs" or not self.is_trained or (mode == "guarded" and self.residual_disabled):
            return ControlOutput(u_ibvs, False, e, e_norm, u_ibvs, zero, depth_m)

        j_hat = self.estimated_jacobian(s, depth_m, intr)         # (C)
        u_res = self.bound_action(-lam * (np.linalg.pinv(j_hat) @ e_ctrl)) - u_ibvs   # (D)
        u_nom = u_ibvs + u_res                                    # (E)

        if mode == "learned":
            return ControlOutput(u_nom, True, e, e_norm, u_ibvs, u_res, depth_m)

        # 守門：用學到的 J_t 預測下一步誤差，候選沒有明顯比 IBVS 好就退回 IBVS
        pred_nom = self.weighted_norm(e_ctrl + j_hat @ u_nom)
        pred_ibvs = self.weighted_norm(e_ctrl + j_hat @ u_ibvs)
        ctrl_norm = self.weighted_norm(e_ctrl)
        margin = self.cfg.guard_margin_px + self.cfg.guard_margin_rel * ctrl_norm
        if self.is_trusted and pred_nom + margin < pred_ibvs and pred_nom < ctrl_norm:
            self._last_used = True
            return ControlOutput(u_nom, True, e, e_norm, u_ibvs, u_res, depth_m)
        return ControlOutput(u_ibvs, False, e, e_norm, u_ibvs, zero, depth_m)
