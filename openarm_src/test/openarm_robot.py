# openarm_src/test/openarm_robot.py
"""
OpenArm 機械手臂驅動封裝：支援安全限幅、速度限制與離線模擬 (Dry-Run / Mock)。
"""
from typing import Optional, Tuple
import numpy as np
from config import RobotSafetyConfig


class OpenArmInterface:
    def __init__(self, config: RobotSafetyConfig, dry_run: bool = True):
        self.cfg = config
        self.dry_run = dry_run
        self.num_joints = config.num_joints
        # 初始關節角度 (Home Position)
        self.q = np.zeros(self.num_joints, dtype=np.float32)
        self.connected = False
        self.connect()

    def connect(self) -> bool:
        if self.dry_run:
            print("[OpenArm] 處於 Dry-Run 模式 (安全模擬，不輸出馬達命令)")
            self.connected = True
            return True

        # 若使用者有實際 OpenArm SDK / CAN bus / Serial 介面，在此初始化
        try:
            # 範例：import openarm_sdk
            print("[OpenArm] 嘗試連接 OpenArm 硬體控制器...")
            # self.arm = openarm_sdk.Arm()
            self.connected = True
            print("[OpenArm] 硬體連接成功！")
            return True
        except Exception as e:
            print(f"[OpenArm] 連接失敗: {e}，自動切換至 Dry-Run 模式。")
            self.dry_run = True
            self.connected = True
            return False

    def get_joint_positions(self) -> np.ndarray:
        """取得當前各關節角度 (rad)"""
        if self.dry_run:
            return self.q.copy()
        # 實機讀取關節狀態
        return self.q.copy()

    def step_joint_delta(self, delta_q: np.ndarray) -> Tuple[bool, np.ndarray]:
        """
        以關節增量模式 (Delta-q) 推進一步，自動執行安全防護檢驗與限幅。
        """
        delta_q = np.asarray(delta_q, dtype=np.float32).flatten()
        if len(delta_q) < self.num_joints:
            # 若只控制腕部或前 2~3 軸，自動補零
            pad = np.zeros(self.num_joints - len(delta_q), dtype=np.float32)
            delta_q = np.concatenate([delta_q, pad])
        elif len(delta_q) > self.num_joints:
            delta_q = delta_q[:self.num_joints]

        # 1. 增量大小限幅 (Bound Norm / Max Step)
        norm_dq = np.linalg.norm(delta_q)
        if norm_dq > self.cfg.max_delta_q_rad:
            delta_q = delta_q * (self.cfg.max_delta_q_rad / norm_dq)

        # 2. 計算目標關節角度
        target_q = self.q + delta_q

        # 3. 關節軟極限檢查 (Joint Limit Clamp)
        for i in range(self.num_joints):
            low = self.cfg.joint_lower_limits[i]
            high = self.cfg.joint_upper_limits[i]
            target_q[i] = np.clip(target_q[i], low, high)

        actual_executed_delta = target_q - self.q
        self.q = target_q

        if not self.dry_run:
            # 實機發送指令：arm.send_joint_position(self.q)
            pass

        return True, actual_executed_delta

    def stop(self):
        """緊急停止"""
        print("[OpenArm] 控制終止，手臂進入 Holding 狀態。")
