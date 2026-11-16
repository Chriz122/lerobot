# openarm_src/test/check_cameras.py
"""
真機工作流 - 步驟 1 工具：相機自動偵測與 AprilTag 5x5cm 辨識預覽
執行方式：python openarm_src/test/check_cameras.py
"""
import time
import cv2
import numpy as np

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False


def list_connected_devices():
    print("==================================================")
    print(" [1/3] RealSense 設備檢測")
    print("==================================================")
    if not REALSENSE_AVAILABLE:
        print("未安裝 pyrealsense2。請先執行: pip install pyrealsense2")
        return []

    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        print("警告: 尚未偵測到任何 RealSense 設備，請檢查 USB 3.0 線材接線！")
        return []

    device_info = []
    print(f"成功偵測到 {len(devices)} 台設備：")
    for i, dev in enumerate(devices):
        name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        usb_type = dev.get_info(rs.camera_info.usb_type_descriptor) if hasattr(rs.camera_info, 'usb_type_descriptor') else "Unknown"
        print(f"  [{i}] {name:<25} | 序號: {serial:<15} | 介面: USB {usb_type}")
        device_info.append({"name": name, "serial": serial})
    print("提示: 請將上述序號填入 openarm_src/test/config.py 的 cameras 區段。\n")
    return device_info


def preview_tag_detection(tag_size_m: float = 0.05):
    print("==================================================")
    print(" [2/3] AprilTag (5x5cm) 辨識與相機畫面預覽")
    print("==================================================")
    print("說明: 按 'q' 鍵退出預覽。")

    # 建立 AprilTag 偵測器
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    if hasattr(cv2.aruco, 'ArucoDetector'):
        detector = cv2.aruco.ArucoDetector(aruco_dict)
    else:
        detector = None

    # 嘗試開啟可用相機
    cap = None
    if REALSENSE_AVAILABLE:
        try:
            pipe = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
            pipe.start(cfg)
            print("成功啟動 RealSense 彩色串流...")
        except Exception as e:
            print(f"無法直接啟動 RealSense 管道 ({e})，嘗試使用 OpenCV VideoCapture(0)...")
            pipe = None
            cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(0)

    try:
        while True:
            if REALSENSE_AVAILABLE and 'pipe' in locals() and pipe is not None:
                frames = pipe.wait_for_frames(1000)
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                frame = np.asanyarray(color_frame.get_data())
            elif cap is not None and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
            else:
                print("無可用視訊來源。")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if detector is not None:
                corners, ids, _ = detector.detectMarkers(gray)
            else:
                corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)

            display = frame.copy()
            h, w = display.shape[:2]
            # 繪製畫面中心十字
            cv2.drawMarker(display, (w//2, h//2), (255, 255, 0), cv2.MARKER_CROSS, 20, 2)

            if ids is not None and len(ids) > 0:
                cv2.polylines(display, [corners[0].astype(np.int32)], True, (0, 255, 0), 2)
                c = np.mean(corners[0][0], axis=0).astype(int)
                cv2.circle(display, tuple(c), 4, (0, 0, 255), -1)
                cv2.putText(display, f"AprilTag ID:{ids[0][0]} Detected (Size: 5x5cm)",
                            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display, "Waiting for AprilTag (tag36h11, 5x5cm)...",
                            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Real-Robot Tag Verification (Press 'q' to exit)", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        if REALSENSE_AVAILABLE and 'pipe' in locals() and pipe is not None:
            pipe.stop()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    list_connected_devices()
