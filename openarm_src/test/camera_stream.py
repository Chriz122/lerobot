# openarm_src/test/camera_stream.py
"""
Multi-Camera Stream 模組：管理 Intel RealSense D455 (Ego) 與 D405 x 2 (Wrist Left/Right)。
具備實體相機驅動 (pyrealsense2) 與離線模擬器 (MockMultiCameraStream)，支援隨時測試。
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import cv2
import numpy as np
from config import CameraConfig, SystemConfig


@dataclass
class FrameData:
    color: np.ndarray
    depth: Optional[np.ndarray] = None
    camera_matrix: Optional[np.ndarray] = None
    dist_coeffs: Optional[np.ndarray] = None


class RealSenseMultiCamera:
    """管理真實 Intel RealSense D455 與 D405 相機串流"""
    def __init__(self, camera_configs: Dict[str, CameraConfig]):
        self.configs = camera_configs
        self.pipelines = {}
        self.intrinsics = {}
        self.is_connected = False
        self._init_cameras()

    def _init_cameras(self):
        try:
            import pyrealsense2 as rs
        except ImportError:
            print("[Camera] 未安裝 pyrealsense2 套件。若需實機串流，請執行: pip install pyrealsense2")
            return

        ctx = rs.context()
        devices = ctx.query_devices()
        print(f"[Camera] 偵測到 {len(devices)} 台 RealSense 設備。")
        for dev in devices:
            name = dev.get_info(rs.camera_info.name)
            serial = dev.get_info(rs.camera_info.serial_number)
            print(f"  - 發現設備: {name} (序號: {serial})")

        # 啟動各相機
        for key, cfg in self.configs.items():
            pipe = rs.pipeline()
            rs_cfg = rs.config()
            if cfg.serial_number:
                rs_cfg.enable_device(cfg.serial_number)

            rs_cfg.enable_stream(rs.stream.color, cfg.width, cfg.height, rs.format.bgr8, cfg.fps)
            if cfg.enable_depth:
                rs_cfg.enable_stream(rs.stream.depth, cfg.width, cfg.height, rs.format.z16, cfg.fps)

            try:
                profile = pipe.start(rs_cfg)
                self.pipelines[key] = pipe
                # 取得內參
                color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
                intr = color_stream.get_intrinsics()
                K = np.array([
                    [intr.fx, 0, intr.ppx],
                    [0, intr.fy, intr.ppy],
                    [0, 0, 1]
                ], dtype=np.float32)
                dist = np.array(intr.coeffs, dtype=np.float32)
                self.intrinsics[key] = (K, dist)
                print(f"[Camera] 成功啟動相機 [{key}] ({cfg.camera_type}): {cfg.width}x{cfg.height}@{cfg.fps}fps")
            except Exception as e:
                print(f"[Camera] 啟動相機 [{key}] 失敗: {e}")

        if len(self.pipelines) > 0:
            self.is_connected = True

    def get_frames(self) -> Dict[str, FrameData]:
        results = {}
        for key, pipe in self.pipelines.items():
            try:
                frames = pipe.wait_for_frames(timeout_ms=1000)
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color_frame:
                    continue
                color_img = np.asanyarray(color_frame.get_data())
                depth_img = np.asanyarray(depth_frame.get_data()) if depth_frame else None
                K, dist = self.intrinsics.get(key, (None, None))
                results[key] = FrameData(color=color_img, depth=depth_img, camera_matrix=K, dist_coeffs=dist)
            except Exception as e:
                print(f"[Camera] 讀取相機 [{key}] 影格逾時或出錯: {e}")
        return results

    def stop(self):
        for key, pipe in self.pipelines.items():
            try:
                pipe.stop()
            except Exception:
                pass
        self.pipelines.clear()
        self.is_connected = False


class MockMultiCameraStream:
    """離線模擬環境相機串流：合成出帶有 AprilTag 的 Cube 影像，供離線演算法與控制閉迴路驗證"""
    def __init__(self, sys_config: SystemConfig):
        self.cfg = sys_config
        self.tag_family = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        # 產生一個 AprilTag 圖案
        self.tag_img = np.zeros((120, 120), dtype=np.uint8)
        cv2.aruco.generateImageMarker(self.tag_family, id=0, sidePixels=120, img=self.tag_img, borderBits=1)

    def generate_synthetic_frames(self, current_pixel_error: np.ndarray,
                                   depth_m: float = 0.22) -> Dict[str, FrameData]:
        """依據當前模擬特徵誤差動態生成影像"""
        frames = {}
        # 內參矩陣模擬
        fx, fy = 615.0, 615.0
        cx, cy = self.cfg.ibvs.desired_center_u, self.cfg.ibvs.desired_center_v
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        dist = np.zeros(5, dtype=np.float32)

        for cam_key, cam_cfg in self.cfg.cameras.items():
            img = np.full((cam_cfg.height, cam_cfg.width, 3), 45, dtype=np.uint8)
            # 繪製桌面工作區網格
            for gx in range(0, cam_cfg.width, 50):
                cv2.line(img, (gx, 0), (gx, cam_cfg.height), (55, 55, 55), 1)
            for gy in range(0, cam_cfg.height, 50):
                cv2.line(img, (0, gy), (cam_cfg.width, gy), (55, 55, 55), 1)

            # 計算 Tag 在當前相機畫面的中心
            if "wrist" in cam_key:
                # wrist 相機隨手臂運動而改變誤差
                u_c = int(cx + current_pixel_error[0])
                v_c = int(cy + current_pixel_error[1])
                tag_scale = 90
            else:
                # ego (D455) 俯視視角，Tag 位置相對穩定
                u_c = int(cam_cfg.width / 2 + current_pixel_error[0] * 0.3)
                v_c = int(cam_cfg.height / 2 + current_pixel_error[1] * 0.3)
                tag_scale = 55

            # 貼上 5x5x5 cube 底座與 AprilTag
            cube_half = int(tag_scale * 0.7)
            top_left = (max(0, u_c - cube_half), max(0, v_c - cube_half))
            bot_right = (min(cam_cfg.width - 1, u_c + cube_half), min(cam_cfg.height - 1, v_c + cube_half))
            cv2.rectangle(img, top_left, bot_right, (180, 140, 60), -1) # 木質/黃色 Cube

            # 貼上 AprilTag
            resized_tag = cv2.resize(self.tag_img, (tag_scale, tag_scale))
            tag_bgr = cv2.cvtColor(resized_tag, cv2.COLOR_GRAY2BGR)
            th, tw = tag_bgr.shape[:2]
            y1 = max(0, v_c - th // 2)
            y2 = min(cam_cfg.height, y1 + th)
            x1 = max(0, u_c - tw // 2)
            x2 = min(cam_cfg.width, x1 + tw)
            if (y2 - y1) == th and (x2 - x1) == tw:
                img[y1:y2, x1:x2] = tag_bgr

            # 建立合成深度圖 (毫米)
            depth_img = np.full((cam_cfg.height, cam_cfg.width), int(depth_m * 1000), dtype=np.uint16)
            frames[cam_key] = FrameData(color=img, depth=depth_img, camera_matrix=K, dist_coeffs=dist)

        return frames
