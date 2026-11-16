# openarm_src/test/ibvs_pinn_controller.py
"""
IBVS + Physics-Informed Learned Residual + Convergence Guard 控制器。
嚴格對應《IBVS_PINN_學生快速驗證_繁體中文.docx》公式與架構。
"""
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from config import IBVSConfig


class TorchResidualMLP(nn.Module):
    """用於學習雅可比矩陣殘差 ΔJ_theta(q) 的輕量 MLP"""
    def __init__(self, in_features: int = 6, out_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, out_dim)
        )

    def forward(self, q):
        return self.net(q)


class IBVSPINNController:
    def __init__(self, config: IBVSConfig, num_active_joints: int = 2,
                 focal_px: float = 615.0, learner: str = "ridge"):
        self.cfg = config
        self.num_joints = num_active_joints
        self.focal_px = focal_px
        self.learner_type = learner

        # 預設名義物理幾何雅可比 (J_phys, 2 x num_active_joints)
        # 影像座標 [u, v] 與末端前兩自由度之間的基準映射
        self.J_phys = np.array([
            [12.5, 0.4],
            [0.3, 11.8]
        ], dtype=np.float32)

        # 學習器狀態
        self.torch_model: Optional[TorchResidualMLP] = None
        self.ridge_w: Optional[np.ndarray] = None
        self.is_trained = False

        if learner == "torch":
            self.torch_model = TorchResidualMLP(in_features=num_active_joints, out_dim=2 * num_active_joints)

    def compute_interaction_matrix(self, u_px: float, v_px: float,
                                    depth_m: float, cx: float, cy: float) -> np.ndarray:
        """
        標準解析 Image Interaction Matrix Ls (點特徵):
        [u_dot, v_dot]^T = Ls * [vx, vy, vz, wx, wy, wz]^T
        """
        Z = max(0.05, depth_m)
        x = (u_px - cx) / self.focal_px
        y = (v_px - cy) / self.focal_px
        Ls = np.array([
            [-self.focal_px / Z, 0.0, x * self.focal_px / Z, x * y * self.focal_px, -(self.focal_px + x**2 * self.focal_px), y * self.focal_px],
            [0.0, -self.focal_px / Z, y * self.focal_px / Z, self.focal_px + y**2 * self.focal_px, -x * y * self.focal_px, -x * self.focal_px]
        ], dtype=np.float32)
        return Ls

    def fit_residual(self, q_history: np.ndarray, u_history: np.ndarray,
                     ds_history: np.ndarray, epochs: int = 200):
        """
        訓練物理殘差模型：
        L_train = || Δs_obs - (J_phys + ΔJ_θ) * Δq ||^2 + β * ||ΔJ_θ||^2
        """
        q_history = np.asarray(q_history, dtype=np.float32)
        u_history = np.asarray(u_history, dtype=np.float32)
        ds_history = np.asarray(ds_history, dtype=np.float32)
        n_samples = len(q_history)

        if self.learner_type == "ridge":
            # 建立多項式特徵矩陣
            features = np.hstack([np.ones((n_samples, 1)), np.sin(q_history), np.cos(q_history)])
            k = features.shape[1]
            design = (features[:, :, None] * u_history[:, None, :]).reshape(n_samples, -1)
            # 目標：真實觀測位移減去物理名義預測
            nominal_pred = u_history @ self.J_phys.T
            targets = ds_history - nominal_pred
            ridge = 0.15
            sol = np.linalg.solve(design.T @ design + ridge * np.eye(design.shape[1]), design.T @ targets)
            self.ridge_w = sol.reshape(k, 2, self.num_joints)
            self.is_trained = True
            print(f"[PINN Controller] Ridge 殘差訓練完成 (樣本數={n_samples})")

        elif self.learner_type == "torch":
            optimizer = torch.optim.Adam(self.torch_model.parameters(), lr=0.01)
            q_t = torch.tensor(q_history, dtype=torch.float32)
            u_t = torch.tensor(u_history, dtype=torch.float32)
            ds_t = torch.tensor(ds_history, dtype=torch.float32)
            j_phys_t = torch.tensor(self.J_phys, dtype=torch.float32)

            for _ in range(epochs):
                delta_j = self.torch_model(q_t).reshape(-1, 2, self.num_joints)
                j_hat = j_phys_t[None, :, :] + delta_j
                ds_pred = torch.bmm(j_hat, u_t.unsqueeze(-1)).squeeze(-1)
                loss = ((ds_pred - ds_t)**2).mean() + 0.0003 * (delta_j**2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            self.is_trained = True
            print(f"[PINN Controller] Torch MLP 殘差訓練完成 (樣本數={n_samples})")

    def get_estimated_jacobian(self, q: np.ndarray) -> np.ndarray:
        """取得當前狀態下修正後的 Jacobian: J_t = J_phys + ΔJ_theta(q)"""
        if not self.is_trained:
            return self.J_phys.copy()

        q_active = q[:self.num_joints]
        if self.learner_type == "ridge":
            feat = np.hstack([1.0, np.sin(q_active), np.cos(q_active)])
            correction = np.einsum('k,kjo->oj', feat, self.ridge_w)
            return self.J_phys + correction
        elif self.learner_type == "torch":
            with torch.no_grad():
                q_t = torch.tensor(q_active, dtype=torch.float32).unsqueeze(0)
                correction = self.torch_model(q_t).numpy().reshape(2, self.num_joints)
                return self.J_phys + correction
        return self.J_phys.copy()

    def compute_action(self, current_feature: np.ndarray, desired_feature: np.ndarray,
                       q_current: np.ndarray, mode: str = "guarded") -> Tuple[np.ndarray, bool, float]:
        """
        計算控制增量命令：
        - mode: 'ibvs' | 'learned' | 'guarded'
        回傳: (action, is_residual_used, error_norm)
        """
        e = (current_feature - desired_feature).astype(np.float32) # e_t = s_t - s*
        e_norm = float(np.linalg.norm(e))

        # (B) 基準解析 IBVS 命令: u_IBVS = -lambda * J_phys^+ * e
        j_phys_pinv = np.linalg.pinv(self.J_phys)
        u_ibvs = -self.cfg.gain_lambda * (j_phys_pinv @ e)

        if mode == "ibvs" or not self.is_trained:
            return u_ibvs, False, e_norm

        # (C) 修正後 Jacobian: J_t = J_phys + ΔJ_theta
        j_hat = self.get_estimated_jacobian(q_current)
        j_hat_pinv = np.linalg.pinv(j_hat)

        # (D) 合併候選命令: u_nom = -lambda * J_hat^+ * e
        u_nom = -self.cfg.gain_lambda * (j_hat_pinv @ e)

        if mode == "learned":
            return u_nom, True, e_norm

        # (E) Convergence Guard 守門機制驗證:
        # 預測: ||e_t + J_t * u_nom|| + delta < ||e_t + J_t * u_ibvs||
        pred_error_nom = np.linalg.norm(e + j_hat @ u_nom)
        pred_error_ibvs = np.linalg.norm(e + j_hat @ u_ibvs)
        margin = self.cfg.guard_improvement_margin + 0.005 * e_norm

        # 滿足收斂改善條件才採納候選修正；否則安全退回純 IBVS
        if (pred_error_nom + margin < pred_error_ibvs) and (pred_error_nom < e_norm):
            return u_nom, True, e_norm
        else:
            return u_ibvs, False, e_norm
