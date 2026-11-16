# openarm_src/test/data_collector.py
"""
微幅安全試探資料收集器 (Safe Dithering Probe)：
依照指南要求，在安全範圍內施加微小關節增量 Δq，觀測 AprilTag 影像位移 Δs_obs，
以建立 Physics-Informed Residual 訓練集，避免損壞機構。
"""
import argparse
import time
import numpy as np
from config import SystemConfig
from tag_detector import AprilTagDetector
from camera_stream import RealSenseMultiCamera, MockMultiCameraStream
from openarm_robot import OpenArmInterface


def collect_probe_dataset(n_samples: int = 150, dry_run: bool = True,
                          active_camera: str = "wrist_left",
                          out_path: str = "probe_data.npz"):
    cfg = SystemConfig()
    detector = AprilTagDetector(tag_size_m=cfg.target.tag_size_meters)
    robot = OpenArmInterface(cfg.robot, dry_run=dry_run)

    if dry_run:
        cam_stream = MockMultiCameraStream(cfg)
    else:
        cam_stream = RealSenseMultiCamera(cfg.cameras)

    q_list, dq_list, ds_list = [], [], []
    sim_error = np.array([35.0, -25.0], dtype=np.float32)

    print(f"\n[DataCollector] 開始微幅安全試探收集 ({n_samples} 筆)...")
    print("安全規則：僅在目前姿態附近施加微幅抖動 (±0.015 rad)，保持無接觸影像對準。")

    rng = np.random.default_rng(42)

    for i in range(n_samples):
        q_before = robot.get_joint_positions()[:2]

        # 讀取試探前影像特徵
        if dry_run:
            frames = cam_stream.generate_synthetic_frames(sim_error)
        else:
            frames = cam_stream.get_frames()

        frame_data = frames.get(active_camera)
        if frame_data is None:
            continue
        res_before = detector.detect(frame_data.color, frame_data.camera_matrix, frame_data.dist_coeffs)
        if not res_before.detected:
            continue
        s_before = res_before.center.copy()

        # 生成微幅安全增量 (約 0.5 ~ 1.0 度)
        dq_cmd = rng.uniform(-0.015, 0.015, size=2).astype(np.float32)
        robot.step_joint_delta(dq_cmd)

        if dry_run:
            # 模擬真實環境微幅非線性映射 (加雜訊)
            true_J = np.array([[15.2 + 1.2*np.sin(q_before[0]), 1.1],
                               [0.8, 14.1 + 0.9*np.cos(q_before[1])]])
            sim_error += true_J @ dq_cmd + rng.normal(0, 0.2, size=2)
            time.sleep(0.01)
            frames_after = cam_stream.generate_synthetic_frames(sim_error)
        else:
            time.sleep(0.05) # 等待機械手臂微動平穩
            frames_after = cam_stream.get_frames()

        frame_after = frames_after.get(active_camera)
        if frame_after is None:
            continue
        res_after = detector.detect(frame_after.color, frame_after.camera_matrix, frame_after.dist_coeffs)
        if not res_after.detected:
            continue
        s_after = res_after.center.copy()

        # 觀測到位移量
        ds_obs = s_after - s_before

        q_list.append(q_before)
        dq_list.append(dq_cmd)
        ds_list.append(ds_obs)

        # 復原試探動作 (保持在工作區中心)
        robot.step_joint_delta(-dq_cmd)
        if dry_run:
            sim_error -= true_J @ dq_cmd

        if (i + 1) % 25 == 0 or i == n_samples - 1:
            print(f"  進度: {i+1}/{n_samples} 筆完成")

    # 儲存資料集
    q_arr = np.array(q_list)
    dq_arr = np.array(dq_list)
    ds_arr = np.array(ds_list)
    np.savez_compressed(out_path, q=q_arr, dq=dq_arr, ds=ds_arr)
    print(f"[DataCollector] 資料集收集完成，共儲存 {len(q_arr)} 筆樣本至: {out_path}\n")
    return q_arr, dq_arr, ds_arr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=120, help="試探採樣數")
    parser.add_argument("--dry-run", action="store_true", default=True, help="使用模擬 Dry-Run")
    parser.add_argument("--real", dest="dry_run", action="store_false", help="連接真實機器人與相機")
    parser.add_argument("--camera", default="wrist_left", choices=["wrist_left", "wrist_right", "ego"])
    parser.add_argument("--out", default="probe_data.npz")
    args = parser.parse_args()

    collect_probe_dataset(args.samples, args.dry_run, args.camera, args.out)
