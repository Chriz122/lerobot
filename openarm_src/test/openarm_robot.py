# openarm_src/test/openarm_robot.py
"""
OpenArm 單臂介面：關節增量命令 + 安全檢查。

- dry_run=True ：命令送進 SimWorld（離線模擬，不碰任何馬達）
- dry_run=False：透過 lerobot 的 OpenArmFollower 以 CAN 控制真實手臂

內部關節角一律是 URDF 慣例的 rad；只有在與 LeRobot 交換資料時才轉成 deg。
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from config import RobotSafetyConfig

_REPO_ROOT = Path(__file__).resolve().parents[2]


class OpenArmInterface:
    def __init__(self, config: RobotSafetyConfig, model, dry_run: bool = True, sim=None,
                 port: Optional[str] = None, table_z_m: Optional[float] = None, table_margin_m: float = 0.01):
        self.cfg = config
        self.model = model
        self.dry_run = dry_run
        self.sim = sim
        self.port = port
        self.side = config.side
        self.num_joints = config.num_joints
        self.table_margin_m = table_margin_m
        if table_z_m is None and "OPENARM_TABLE_Z" in os.environ:
            table_z_m = float(os.environ["OPENARM_TABLE_Z"])
        self.table_z_m = table_z_m

        self.arm = None
        self.signs = np.ones(self.num_joints)
        self.connected = False
        self.q_cmd = np.zeros(self.num_joints)
        self.gripper_cmd_deg = config.gripper_open_deg
        self.p_start: Optional[np.ndarray] = None
        self.last_block_reason = ""
        self._set_limits()
        self.connect()

    # ------------------------------------------------------------------ 連線
    def _set_limits(self) -> None:
        if self.dry_run:
            self.lower, self.upper = self.model.lower.copy(), self.model.upper.copy()
            return
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        from openarm_src.utils.config import JOINT_SIGN

        self.signs = np.array(JOINT_SIGN[self.side], dtype=float)
        limits = np.array(self.cfg.joint_limits_deg[self.side], dtype=float)
        margin = self.cfg.limit_margin_deg
        lo = np.radians(limits[:, 0] + margin) * self.signs
        hi = np.radians(limits[:, 1] - margin) * self.signs
        self.lower, self.upper = np.minimum(lo, hi), np.maximum(lo, hi)

    def connect(self) -> None:
        if self.dry_run:
            if self.sim is None:
                raise ValueError("Dry-Run 模式需要傳入 SimWorld")
            print("[OpenArm] Dry-Run 模式（離線模擬，不輸出任何馬達命令）")
        else:
            if not self.port:
                raise ValueError("真機模式需要指定 CAN 埠，例如 --port can1")
            from lerobot.robots.openarm_follower import OpenArmFollower, OpenArmFollowerConfig
            from openarm_src.utils.can_mgr import can_setup_and_check
            from openarm_src.utils.patches import patch_follower_park

            patch_follower_park()   # 中斷連線前先慢慢回下垂姿態，再關力矩
            try:
                can_setup_and_check()
            except Exception as e:  # noqa: BLE001
                print(f"[OpenArm] CAN 自動設定略過: {e}")
            cfg = OpenArmFollowerConfig(port=self.port, side=self.side, id=f"openarm_{self.side}_follower")
            self.arm = OpenArmFollower(cfg)
            print(f"[OpenArm] 連線 {self.side} 手臂 (port={self.port}) ...")
            self.arm.connect()
            print("[OpenArm] 連線成功")
        self.connected = True
        self.resync()

    def resync(self) -> None:
        """把命令角度對齊到目前實際角度，並送出一次保持命令。"""
        self.q_cmd = self.get_joint_positions()
        self.gripper_cmd_deg = self.get_gripper_deg()
        self._send()

    def mark_start(self) -> None:
        """記下任務起點，用來限制 TCP 總位移。"""
        _, self.p_start = self.model.fk(self.q_cmd)

    # ------------------------------------------------------------------ 讀取
    def get_joint_positions(self) -> np.ndarray:
        """目前關節角 (rad, URDF 慣例)"""
        if self.dry_run:
            return self.sim.measured_joints()
        obs = self.arm.get_observation()
        deg = np.array([float(obs[f"joint_{i}.pos"]) for i in range(1, self.num_joints + 1)])
        return np.radians(deg) * self.signs

    def get_gripper_deg(self) -> float:
        if self.dry_run:
            return float(self.sim.gripper_deg)
        return float(self.arm.get_observation()["gripper.pos"])

    def tcp_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """以名義運動學模型算出的 TCP 位姿（基座座標）"""
        return self.model.fk(self.get_joint_positions())

    # ------------------------------------------------------------------ 命令
    def _send(self) -> None:
        if self.dry_run:
            self.sim.command_joints(self.q_cmd)
            return
        deg = np.degrees(self.q_cmd * self.signs)
        action = {f"joint_{i + 1}.pos": float(deg[i]) for i in range(self.num_joints)}
        action["gripper.pos"] = float(self.gripper_cmd_deg)
        self.arm.send_action(action)

    def step_joint_delta(self, delta_q: np.ndarray) -> Tuple[bool, np.ndarray]:
        """送出一步關節增量 (rad)。回傳 (是否執行, 實際送出的增量)。

        依序做：單步限幅 -> 命令不得領先實際角度太多 -> 關節軟限位 -> 桌面高度 -> TCP 總位移。
        後兩項不通過時整步取消（手臂保持原位），原因記在 last_block_reason。
        """
        delta_q = np.asarray(delta_q, dtype=float).flatten()
        if delta_q.shape[0] != self.num_joints or not np.all(np.isfinite(delta_q)):
            self.last_block_reason = "關節增量維度錯誤或含非有限值"
            return False, np.zeros(self.num_joints)

        max_step = np.radians(self.cfg.max_delta_q_deg)
        peak = float(np.max(np.abs(delta_q)))
        if peak > max_step:
            delta_q = delta_q * (max_step / peak)

        q_meas = self.get_joint_positions()
        lag = np.radians(self.cfg.max_tracking_lag_deg)
        base = np.clip(self.q_cmd, q_meas - lag, q_meas + lag)
        target = np.clip(base + delta_q, self.lower, self.upper)

        _, p_tcp = self.model.fk(target)
        if self.table_z_m is not None and p_tcp[2] < self.table_z_m + self.table_margin_m:
            self.last_block_reason = (
                f"TCP 高度 {p_tcp[2]:.3f} m 低於桌面 + 餘裕 {self.table_z_m + self.table_margin_m:.3f} m"
            )
            return False, np.zeros(self.num_joints)
        if self.p_start is not None and np.linalg.norm(p_tcp - self.p_start) > self.cfg.max_tcp_travel_m:
            self.last_block_reason = f"TCP 離起點超過 {self.cfg.max_tcp_travel_m:.2f} m"
            return False, np.zeros(self.num_joints)

        executed = target - self.q_cmd
        self.q_cmd = target
        self.last_block_reason = ""
        self._send()
        return True, executed

    def set_gripper(self, target_deg: float, wait: bool = True) -> float:
        """夾爪慢慢移到目標角度，回傳最後實際角度。"""
        step = self.cfg.gripper_step_deg
        while abs(target_deg - self.gripper_cmd_deg) > 1e-6:
            diff = target_deg - self.gripper_cmd_deg
            self.gripper_cmd_deg += float(np.clip(diff, -step, step))
            if self.dry_run:
                self.sim.set_gripper(self.gripper_cmd_deg)
            else:
                self._send()
                time.sleep(1.0 / self.cfg.control_rate_hz)
        if not self.dry_run and wait:
            time.sleep(0.5)
        return self.get_gripper_deg()

    def wait_step(self) -> None:
        """真機：等一個控制週期。模擬：不需要等。"""
        if not self.dry_run:
            time.sleep(1.0 / self.cfg.control_rate_hz)

    def settle(self) -> None:
        """等手臂追上命令並靜止後再取像（試探、示教、下降前使用）。"""
        if self.dry_run:
            for _ in range(5):
                self.sim.command_joints(self.q_cmd)
        else:
            time.sleep(self.cfg.settle_time_s)

    def move_to_joints(self, q_target: np.ndarray, duration_s: float = 3.0) -> None:
        """以 min-jerk 關節插值回到指定姿態（用來讓每回合從相同初始狀態開始）。"""
        q_target = np.clip(np.asarray(q_target, dtype=float), self.lower, self.upper)
        q_from = self.q_cmd.copy()
        n = 1 if self.dry_run else max(int(duration_s * self.cfg.control_rate_hz), 1)
        for k in range(1, n + 1):
            x = k / n
            blend = 10 * x**3 - 15 * x**4 + 6 * x**5
            self.q_cmd = q_from + blend * (q_target - q_from)
            self._send()
            self.wait_step()
        self.settle()

    def go_home_pose(self, duration_s: Optional[float] = None) -> bool:
        """真機：以 min-jerk 回到 reset.json 的起始姿態（最後一個 waypoint；只動 7 個關節，夾爪不變）。"""
        if self.dry_run or self.arm is None:
            return False
        from openarm_src.utils.config import DEFAULT_HOME_PATH, DEFAULT_HOME_TIME_S
        from openarm_src.utils.homing import load_waypoints

        if not DEFAULT_HOME_PATH or not os.path.isfile(DEFAULT_HOME_PATH):
            print(f"[OpenArm] 找不到起始姿態檔 ({DEFAULT_HOME_PATH})，略過回起始姿態。")
            return False
        home = load_waypoints(DEFAULT_HOME_PATH)[-1]
        keys = [f"{self.side}_joint_{i}.pos" for i in range(1, self.num_joints + 1)]
        if not all(k in home for k in keys):
            print(f"[OpenArm] {DEFAULT_HOME_PATH} 沒有 {self.side} 手臂的完整關節角，略過回起始姿態。")
            return False
        q_home = np.radians([home[k] for k in keys]) * self.signs

        # 先對齊命令角度，再檢查整段路徑不會低於桌面
        self.q_cmd = self.get_joint_positions()
        if self.table_z_m is not None:
            for x in np.linspace(0.0, 1.0, 21):
                _, p_tcp = self.model.fk(self.q_cmd + x * (q_home - self.q_cmd))
                if p_tcp[2] < self.table_z_m + self.table_margin_m:
                    print("[OpenArm] 回起始姿態的路徑會低於桌面，已略過。")
                    return False
        print(f"[OpenArm] 回到起始姿態 ({os.path.basename(DEFAULT_HOME_PATH)}) ...")
        self.move_to_joints(q_home, duration_s or DEFAULT_HOME_TIME_S)
        return True

    def stop(self) -> None:
        """結束：真機先回 reset.json 起始姿態，再回下垂姿態並關力矩（後者由 patch_follower_park 處理）。"""
        if self.dry_run or self.arm is None:
            return
        try:
            self.go_home_pose()
        except Exception as e:  # noqa: BLE001
            print(f"[OpenArm] 回起始姿態失敗: {e}")
        print("[OpenArm] 結束控制，手臂回下垂姿態後關閉力矩 ...")
        try:
            self.arm.disconnect()
        except Exception as e:  # noqa: BLE001
            print(f"[OpenArm] 中斷連線出錯: {e}")
            from openarm_src.utils.safety import safe_disable

            safe_disable(self.arm)
        self.connected = False
