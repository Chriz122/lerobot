# openarm_src/test/tag_detector.py
"""
AprilTag 偵測器模組：支援 tag36h11，提供特徵點計算與可視化。
使用 OpenCV 原生 cv2.aruco 模組，不依賴外部編譯套件。
"""
from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np


@dataclass
class TagDetectionResult:
    detected: bool
    tag_id: int = -1
    center: np.ndarray = None        # (2,) [u, v]
    corners: np.ndarray = None       # (4, 2) 4個角點座標
    area: float = 0.0                # Tag 像素面積
    rvec: Optional[np.ndarray] = None# 旋轉向量 (solvePnP)
    tvec: Optional[np.ndarray] = None# 平移向量 (solvePnP) [x, y, z] 相對相機 (公尺)
    depth_m: float = 0.0             # 估計或測距之深度 (公尺)


class AprilTagDetector:
    def __init__(self, tag_family: str = "tag36h11", tag_size_m: float = 0.05):
        self.tag_size_m = tag_size_m
        
        # 建立 OpenCV ArUco AprilTag 字典
        aruco_dict_id = getattr(cv2.aruco, 'DICT_APRILTAG_36h11', cv2.aruco.DICT_APRILTAG_36h11)
        self.dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
        
        # 建立偵測器參數
        if hasattr(cv2.aruco, 'DetectorParameters'):
            self.parameters = cv2.aruco.DetectorParameters()
        else:
            self.parameters = cv2.aruco.DetectorParameters_create()
            
        # 針對高精度小 tag 優化參數
        self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        
        # 支援 OpenCV 4.7+ 的 ArucoDetector 物件
        if hasattr(cv2.aruco, 'ArucoDetector'):
            self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
        else:
            self.detector = None

        # 3D 標籤角點模型 (用於 solvePnP)
        s = tag_size_m / 2.0
        self.obj_points = np.array([
            [-s,  s, 0.0],
            [ s,  s, 0.0],
            [ s, -s, 0.0],
            [-s, -s, 0.0]
        ], dtype=np.float32)

    def detect(self, color_img: np.ndarray, camera_matrix: Optional[np.ndarray] = None,
               dist_coeffs: Optional[np.ndarray] = None, target_id: Optional[int] = None) -> TagDetectionResult:
        """
        在單張彩色影像中偵測 AprilTag
        """
        if color_img is None or color_img.size == 0:
            return TagDetectionResult(detected=False)

        gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY) if len(color_img.shape) == 3 else color_img

        if self.detector is not None:
            corners_list, ids, _ = self.detector.detectMarkers(gray)
        else:
            corners_list, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.parameters)

        if ids is None or len(ids) == 0:
            return TagDetectionResult(detected=False)

        ids = ids.flatten()
        selected_idx = 0
        if target_id is not None:
            matches = np.where(ids == target_id)[0]
            if len(matches) > 0:
                selected_idx = matches[0]
            else:
                return TagDetectionResult(detected=False)

        detected_id = int(ids[selected_idx])
        corners = corners_list[selected_idx][0].astype(np.float32) # (4, 2)
        center = np.mean(corners, axis=0) # (2,) [u, v]
        area = float(cv2.contourArea(corners.astype(np.int32)))

        rvec, tvec, depth_m = None, None, 0.0
        if camera_matrix is not None:
            if dist_coeffs is None:
                dist_coeffs = np.zeros(5, dtype=np.float32)
            success, rvec, tvec = cv2.solvePnP(
                self.obj_points, corners, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            if success:
                depth_m = float(tvec[2, 0])

        return TagDetectionResult(
            detected=True,
            tag_id=detected_id,
            center=center,
            corners=corners,
            area=area,
            rvec=rvec,
            tvec=tvec,
            depth_m=depth_m
        )

    def draw_overlay(self, image: np.ndarray, res: TagDetectionResult,
                     desired_center: Optional[Tuple[float, float]] = None,
                     camera_matrix: Optional[np.ndarray] = None,
                     dist_coeffs: Optional[np.ndarray] = None) -> np.ndarray:
        """
        在畫面上繪製角點、中心、誤差箭頭與 3D 姿態軸
        """
        annotated = image.copy()
        if not res.detected:
            cv2.putText(annotated, "Tag NOT Detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return annotated

        # 畫 Tag 四周框線
        pts = res.corners.astype(np.int32)
        cv2.polylines(annotated, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        # 標出四個角點編號
        colors = [(255, 0, 0), (0, 255, 0), (0, 255, 255), (255, 0, 255)]
        for i, pt in enumerate(pts):
            cv2.circle(annotated, tuple(pt), 4, colors[i], -1)

        # 畫中心點
        cu, cv = int(res.center[0]), int(res.center[1])
        cv2.drawMarker(annotated, (cu, cv), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)

        # 若有設定目標中心，繪製目標十字與誤差連線
        if desired_center is not None:
            du, dv = int(desired_center[0]), int(desired_center[1])
            cv2.drawMarker(annotated, (du, dv), (255, 255, 0), markerType=cv2.MARKER_TILTED_CROSS, markerSize=18, thickness=2)
            cv2.arrowedLine(annotated, (cu, cv), (du, dv), (0, 255, 255), 2, tipLength=0.2)
            err_px = np.linalg.norm([cu - du, cv - dv])
            cv2.putText(annotated, f"Error: {err_px:.1f}px", (cu + 10, cv - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 畫 3D 姿態坐標軸
        if res.rvec is not None and res.tvec is not None and camera_matrix is not None:
            cv2.drawFrameAxes(annotated, camera_matrix, dist_coeffs, res.rvec, res.tvec, self.tag_size_m * 0.8)

        # 顯示深度與 ID 資訊
        info_str = f"Tag ID: {res.tag_id} | Depth: {res.depth_m*100:.1f}cm" if res.depth_m > 0 else f"Tag ID: {res.tag_id}"
        cv2.putText(annotated, info_str, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return annotated
