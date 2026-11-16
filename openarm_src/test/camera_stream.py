# openarm_src/test/camera_stream.py
"""
Multi-Camera Stream 模組：管理 Intel RealSense D455 (Ego) 與 D405 x 2 (Wrist Left/Right)。
- RealSenseMultiCamera：實體相機 (pyrealsense2)
- SimCameraStream     ：離線模擬，影像由 SimWorld 依 3D 幾何算出
兩者介面相同：get_frames() -> {相機名稱: FrameData}
"""
from dataclasses import dataclass
from typing import Dict, Iterable, Optional
import numpy as np
from config import CameraConfig


@dataclass
class FrameData:
    color: np.ndarray
    depth: Optional[np.ndarray] = None
    camera_matrix: Optional[np.ndarray] = None
    dist_coeffs: Optional[np.ndarray] = None


class RealSenseMultiCamera:
    """管理真實 Intel RealSense D455 與 D405 相機串流"""
    def __init__(self, camera_configs: Dict[str, CameraConfig], only: Optional[Iterable[str]] = None):
        # only：只開啟指定的相機（IBVS 只需要一台腕部相機，少開可省 USB 頻寬）
        names = set(only) if only is not None else set(camera_configs)
        self.configs = {k: v for k, v in camera_configs.items() if k in names}
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


class SimCameraStream:
    """離線模擬相機：只有裝在手臂上的那台腕部相機有畫面。"""
    def __init__(self, world):
        self.world = world
        self.is_connected = True

    def get_frames(self) -> Dict[str, FrameData]:
        color, camera_matrix, dist = self.world.render()
        return {self.world.camera_name: FrameData(color=color, camera_matrix=camera_matrix, dist_coeffs=dist)}

    def stop(self):
        self.is_connected = False
