# openarm_src/test/sim_world.py
"""
離線模擬世界（Dry-Run 用）：一支 7 軸手臂、一台腕部相機、桌上一顆貼了 AprilTag 的 5cm 方塊。

「真實世界」刻意與控制器使用的名義模型不同（連桿長度、手眼外參都有誤差），
控制器看不到這些真值，只能透過相機影像觀察結果 —— 這就是殘差模型要學的東西。
影像是依 3D 幾何投影後貼圖產生的，偵測器跑的是真正的 AprilTag 偵測流程。
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from arm_kinematics import ApproxArmModel, HandEye, rot_axis, rpy_to_matrix, solve_ik
from config import SystemConfig

_TAG_TEXTURE_PX = 400


class SimWorld:
    def __init__(self, cfg: SystemConfig, camera_name: str = "wrist_left", seed: int = 0):
        self.cfg = cfg
        self.camera_name = camera_name
        self.rng = np.random.default_rng(seed)
        sc = cfg.sim

        # 真實手臂：連桿比名義模型長一點
        scale = 1.0 + sc.link_scale_err
        self.true_model = ApproxArmModel(0.35 * scale, 0.33 * scale, 0.15)

        # 真實手眼外參 = 名義值 + 誤差
        he = cfg.handeye
        self.true_handeye = HandEye(
            np.asarray(he.translation_m) + np.asarray(sc.handeye_trans_err_m), he.rpy_deg, he.approach_axis_tcp
        )
        self.true_handeye.r = self.true_handeye.r @ rpy_to_matrix(sc.handeye_rot_err_deg)

        cam = cfg.cameras[camera_name]
        self.width, self.height = cam.width, cam.height
        self.camera_matrix = np.array(
            [[sc.focal_px, 0.0, cam.width / 2.0], [0.0, sc.focal_px, cam.height / 2.0], [0.0, 0.0, 1.0]]
        )

        # AprilTag 貼圖（黑框外緣 = tag_size_meters）
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        tag = cv2.aruco.generateImageMarker(dictionary, cfg.target.tag_id, _TAG_TEXTURE_PX)
        self.tag_texture = cv2.cvtColor(tag, cv2.COLOR_GRAY2BGR)
        self.background = self._make_background()

        # 初始姿態：夾爪朝下、位於桌面上方
        r_down = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
        seed_q = np.radians([0.0, -40.0, 0.0, 110.0, 0.0, -70.0, 0.0])
        self.q_home, ok = solve_ik(self.true_model, r_down, np.array([0.30, 0.0, -0.36]), seed_q)
        if not ok:
            raise RuntimeError("模擬初始姿態求解失敗")

        self.q = self.q_home.copy()
        self.gripper_deg = cfg.robot.gripper_open_deg
        self.cube_center = np.zeros(3)       # Tag 中心（方塊頂面中心）
        self.cube_yaw = 0.0
        self.attached = False
        self._attach_offset = np.zeros(3)
        self.reset(np.zeros(2), 0.0)

    # ------------------------------------------------------------------ 狀態
    def _make_background(self) -> np.ndarray:
        """淺色桌面加一點紋理雜訊（Tag 黑框外需要淺色區域才偵測得到）。"""
        noise = np.random.default_rng(7).normal(0.0, 4.0, size=(self.height, self.width, 1))
        return np.clip(205.0 + noise, 0, 255).astype(np.uint8).repeat(3, axis=2)

    def reset(self, cube_offset_xy_m: np.ndarray, cube_yaw_deg: float) -> None:
        """手臂回初始姿態；方塊放在初始相機光軸正下方 + 指定偏移。"""
        self.q = self.q_home.copy()
        self.gripper_deg = self.cfg.robot.gripper_open_deg
        self.attached = False
        _, p_cam = self.camera_pose()
        self.cube_center = np.array([
            p_cam[0] + cube_offset_xy_m[0],
            p_cam[1] + cube_offset_xy_m[1],
            p_cam[2] - self.cfg.sim.start_height_m,
        ])
        self.cube_yaw = float(np.radians(cube_yaw_deg))
        self._rest_z = float(self.cube_center[2])
        self.table_z = self._rest_z - self.cfg.target.cube_size_meters

    def place_tcp_at_grasp_pose(self) -> None:
        """模擬「人把夾爪擺到閉合就能夾到的位置」（示教的第一步）。"""
        c, s = np.cos(self.cube_yaw), np.sin(self.cube_yaw)
        r_target = np.array([[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, -1.0]])
        p_target = self.cube_center - np.array([0.0, 0.0, self.cfg.grasp.grasp_depth_below_tag_m])
        q, ok = solve_ik(self.true_model, r_target, p_target, self.q)
        if not ok:
            raise RuntimeError("模擬示教姿態求解失敗")
        self.q = q

    def tcp_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.true_model.fk(self.q)

    def camera_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.true_handeye.camera_pose(*self.tcp_pose())

    def command_joints(self, q_target: np.ndarray) -> None:
        """位置控制：每步只追到命令的一部分，並帶少量雜訊。"""
        sc = self.cfg.sim
        noise = self.rng.normal(0.0, np.radians(sc.joint_noise_deg), size=self.q.shape)
        self.q = self.q + sc.joint_tracking * (np.asarray(q_target, dtype=float) - self.q) + noise
        self.q = np.clip(self.q, self.true_model.lower, self.true_model.upper)
        if self.attached:
            _, p_tcp = self.tcp_pose()
            self.cube_center = p_tcp + self._attach_offset

    def measured_joints(self) -> np.ndarray:
        return self.q.copy()

    # ------------------------------------------------------------------ 影像
    def tag_corners_world(self) -> np.ndarray:
        half = self.cfg.target.tag_size_meters / 2.0
        local = np.array([[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]])
        rz = rot_axis(np.array([0.0, 0.0, 1.0]), self.cube_yaw)
        return local @ rz.T + self.cube_center

    def render(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """回傳 (彩色影像, 內參矩陣, 畸變係數)"""
        img = self.background.copy()
        r_cam, p_cam = self.camera_pose()
        pts = (self.tag_corners_world() - p_cam) @ r_cam      # = R^T (x - p)
        if np.all(pts[:, 2] > 0.03):
            k = self.camera_matrix
            px = np.stack([k[0, 0] * pts[:, 0] / pts[:, 2] + k[0, 2],
                           k[1, 1] * pts[:, 1] / pts[:, 2] + k[1, 2]], axis=1)
            px += self.rng.normal(0.0, self.cfg.sim.pixel_noise_px, size=px.shape)
            if np.all(np.abs(px) < 5000):
                n = float(_TAG_TEXTURE_PX)
                src = np.array([[0, 0], [n, 0], [n, n], [0, n]], dtype=np.float32)
                homography = cv2.getPerspectiveTransform(src, px.astype(np.float32))
                cv2.warpPerspective(self.tag_texture, homography, (self.width, self.height), dst=img,
                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
        img = cv2.GaussianBlur(img, (3, 3), 0)   # 模擬鏡頭的輕微模糊
        return img, self.camera_matrix.copy(), np.zeros(5)

    # ------------------------------------------------------------------ 夾取判定
    def grasp_error(self) -> Dict[str, float]:
        """TCP 相對「理想夾取位姿」的幾何誤差（只有模擬才量得到）。"""
        r_tcp, p_tcp = self.tcp_pose()
        ideal_z = self.cube_center[2] - self.cfg.grasp.grasp_depth_below_tag_m
        x_axis = r_tcp[:, 0]
        tcp_yaw = np.degrees(np.arctan2(x_axis[1], x_axis[0]))
        yaw_err = (tcp_yaw - np.degrees(self.cube_yaw) + 45.0) % 90.0 - 45.0
        approach = r_tcp @ self.true_handeye.approach
        tilt = np.degrees(np.arccos(np.clip(-approach[2], -1.0, 1.0)))
        return {
            "xy_mm": float(np.linalg.norm(p_tcp[:2] - self.cube_center[:2]) * 1000.0),
            "z_mm": float((p_tcp[2] - ideal_z) * 1000.0),
            "yaw_deg": float(yaw_err),
            "tilt_deg": float(tilt),
        }

    def set_gripper(self, deg: float) -> None:
        self.gripper_deg = float(deg)

    def try_grasp(self) -> bool:
        """夾爪閉合時呼叫：幾何誤差都在容差內才算夾住，之後方塊跟著 TCP 移動。"""
        sc = self.cfg.sim
        err = self.grasp_error()
        ok = (
            err["xy_mm"] <= sc.grasp_xy_tol_mm
            and abs(err["z_mm"]) <= sc.grasp_z_tol_mm
            and abs(err["yaw_deg"]) <= sc.grasp_yaw_tol_deg
        )
        if ok:
            _, p_tcp = self.tcp_pose()
            self._attach_offset = self.cube_center - p_tcp
            self.attached = True
            # 夾到東西：夾爪停在半路
            self.gripper_deg = self.cfg.robot.gripper_closed_deg + 0.4 * (
                self.cfg.robot.gripper_open_deg - self.cfg.robot.gripper_closed_deg
            )
        return ok

    def lifted_height_m(self) -> Optional[float]:
        if not self.attached:
            return None
        return float(self.cube_center[2] - self._rest_z)
