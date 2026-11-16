# openarm_src/test/main_experiment.py
"""
OpenArm 實驗主程式：三種方法 (IBVS / Learned / Guarded) 閉迴路伺服實驗對比。
硬體目標配置：
- Arm: OpenArm (6-DOF)
- Ego Camera: Intel RealSense D455
- Wrist Cameras: Intel RealSense D405 x 2
- Target: 5x5x5 cm Cube (貼有 tag36h11 AprilTag)
"""
import argparse
import csv
import os
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np

from config import SystemConfig
from tag_detector import AprilTagDetector
from camera_stream import RealSenseMultiCamera, MockMultiCameraStream
from openarm_robot import OpenArmInterface
from ibvs_pinn_controller import IBVSPINNController
from data_collector import collect_probe_dataset


def run_single_episode(method: str, controller: IBVSPINNController,
                       robot: OpenArmInterface, cam_stream,
                       detector: AprilTagDetector, cfg: SystemConfig,
                       active_cam_name: str, initial_pixel_offset: np.ndarray,
                       is_dry_run: bool = True, save_frames: bool = False,
                       episode_id: int = 0) -> dict:
    """執行單次完整的視覺伺服 Episode"""
    desired_center = np.array([cfg.ibvs.desired_center_u, cfg.ibvs.desired_center_v], dtype=np.float32)
    current_sim_offset = initial_pixel_offset.copy()

    error_history = []
    used_residual_count = 0
    success = False

    # 若為模擬模式，定義環境真實非線性 Jacobian (存在標定偏差)
    true_J_env = np.array([
        [15.8 + 1.2 * np.sin(0.4 * initial_pixel_offset[0] / 30.0), 0.9],
        [0.7, 14.6 + 1.1 * np.cos(0.4 * initial_pixel_offset[1] / 30.0)]
    ])

    for step in range(cfg.ibvs.max_steps_per_episode):
        q = robot.get_joint_positions()[:2]

        # 1. 取得影像
        if is_dry_run:
            frames = cam_stream.generate_synthetic_frames(current_sim_offset)
        else:
            frames = cam_stream.get_frames()

        frame_data = frames.get(active_cam_name)
        if frame_data is None:
            time.sleep(0.02)
            continue

        # 2. 偵測 AprilTag
        res = detector.detect(frame_data.color, frame_data.camera_matrix, frame_data.dist_coeffs)
        if not res.detected:
            print(f"[{method}] Step {step}: Tag 遺失，維持當前姿態等待。")
            error_history.append(error_history[-1] if error_history else 50.0)
            continue

        current_center = res.center
        e_norm = np.linalg.norm(current_center - desired_center)
        error_history.append(float(e_norm))

        # 3. 檢查是否收斂成功
        if e_norm <= cfg.ibvs.error_threshold_px:
            success = True
            break

        # 4. 控制器計算增量
        delta_q, used_res, _ = controller.compute_action(
            current_feature=current_center,
            desired_feature=desired_center,
            q_current=q,
            mode=method
        )
        if used_res:
            used_residual_count += 1

        # 5. 機械手臂執行動作 (含限幅防護)
        robot.step_joint_delta(delta_q)

        # 6. 更新狀態
        if is_dry_run:
            # 模擬下一時刻位移 (真實物理雅可比 + 微小觀測雜訊)
            current_sim_offset += true_J_env @ delta_q + np.random.normal(0, 0.15, size=2)
        else:
            time.sleep(1.0 / cfg.robot.control_rate_hz)

    final_error = error_history[-1] if error_history else 999.0
    return {
        "method": method,
        "success": success,
        "final_error_px": final_error,
        "steps_taken": len(error_history),
        "residual_accepted_pct": 100.0 * used_residual_count / max(1, len(error_history)),
        "error_curve": error_history
    }


def main():
    parser = argparse.ArgumentParser(description="OpenArm IBVS + PINN Residual 實驗驗證")
    parser.add_argument("--learner", choices=["ridge", "torch"], default="ridge", help="物理殘差學習器類型")
    parser.add_argument("--episodes", type=int, default=15, help="每種方法測試的回合數")
    parser.add_argument("--dry-run", action="store_true", default=True, help="使用 Mock 模擬 (預設 True)")
    parser.add_argument("--real", dest="dry_run", action="store_false", help="連接真實機器人與相機硬體")
    parser.add_argument("--camera", default="wrist_left", choices=["wrist_left", "wrist_right", "ego"],
                        help="主要用於 IBVS 視覺伺服的相機 (預設腕部 wrist_left D405)")
    parser.add_argument("--out-dir", default="experiment_outputs", help="結果圖表輸出目錄")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = SystemConfig()
    detector = AprilTagDetector(tag_size_m=cfg.target.tag_size_meters)
    robot = OpenArmInterface(cfg.robot, dry_run=args.dry_run)

    print("=================================================================")
    print(" ACT / VLA Last-Mile: OpenArm IBVS + PINN 實驗快速驗證平台")
    print(f" - 硬體配置: OpenArm 6-DOF, Ego(D455), Wrist(D405*2)")
    print(f" - 待夾物: 5x5x5 cm Cube (貼有 {cfg.target.tag_family} Tag ID: {cfg.target.tag_id})")
    print(f" - 運行模式: {'離線 Dry-Run 模擬 (安全無馬達)' if args.dry_run else '真實硬體即時控制'}")
    print(f" - 主伺服相機: {args.camera}")
    print(f" - 學習器: {args.learner}")
    print("=================================================================\n")

    # 1. 準備相機串流
    if args.dry_run:
        cam_stream = MockMultiCameraStream(cfg)
    else:
        cam_stream = RealSenseMultiCamera(cfg.cameras)

    # 2. 收集試探資料並訓練 Physics-Informed Residual
    probe_file = os.path.join(args.out_dir, "probe_data.npz")
    print("步驟 1/3: 執行安全小步長試探 (Dithering) 以訓練殘差模型...")
    q_data, dq_data, ds_data = collect_probe_dataset(
        n_samples=120, dry_run=args.dry_run, active_camera=args.camera, out_path=probe_file
    )

    controller = IBVSPINNController(cfg.ibvs, num_active_joints=2, learner=args.learner)
    controller.fit_residual(q_data, dq_data, ds_data)

    # 3. 測試三種控制方法
    methods = ["ibvs", "learned", "guarded"]
    results = {m: [] for m in methods}
    example_curves = {}

    rng = np.random.default_rng(2026)
    # 產生多個測試初始偏差 (±50 像素左右)
    initial_offsets = rng.uniform(-55.0, 55.0, size=(args.episodes, 2)).astype(np.float32)

    print("\n步驟 2/3: 開始三種控制方法的對比測試 (公平使用相同初始狀態)...")
    for method in methods:
        print(f"  正在執行方法: [{method.upper()}] ...")
        for ep in range(args.episodes):
            ep_res = run_single_episode(
                method=method,
                controller=controller,
                robot=robot,
                cam_stream=cam_stream,
                detector=detector,
                cfg=cfg,
                active_cam_name=args.camera,
                initial_pixel_offset=initial_offsets[ep],
                is_dry_run=args.dry_run,
                episode_id=ep
            )
            results[method].append(ep_res)
            if ep == 0:
                example_curves[method] = ep_res["error_curve"]

    # 4. 統計成果與指標
    print("\n步驟 3/3: 彙整實驗指標與報表...")
    summary_rows = []
    print("\n" + "="*75)
    print(f"{'控制方法':<12} | {'成功率 (<=3px)':<14} | {'平均最終誤差':<14} | {'平均耗用步數':<12} | {'殘差採納率':<10}")
    print("-" * 75)

    for m in methods:
        ep_list = results[m]
        succ_pct = 100.0 * np.mean([1 if r["success"] else 0 for r in ep_list])
        mean_err = float(np.mean([r["final_error_px"] for r in ep_list]))
        mean_steps = float(np.mean([r["steps_taken"] for r in ep_list]))
        guard_pct = float(np.mean([r["residual_accepted_pct"] for r in ep_list]))
        print(f"{m.upper():<12} | {succ_pct:6.1f}%{'':<7} | {mean_err:8.3f} px{'':<5} | {mean_steps:6.1f} 步{'':<5} | {guard_pct:6.1f}%")
        summary_rows.append({
            "method": m,
            "success_rate_pct": succ_pct,
            "mean_final_error_px": mean_err,
            "mean_steps": mean_steps,
            "residual_accepted_pct": guard_pct
        })
    print("=" * 75 + "\n")

    # 5. 儲存 CSV
    csv_path = os.path.join(args.out_dir, "experiment_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "success_rate_pct", "mean_final_error_px", "mean_steps", "residual_accepted_pct"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[報表儲存] 指標彙整已儲存至: {csv_path}")

    # 6. 繪製誤差收斂曲線圖
    plot_path = os.path.join(args.out_dir, "experiment_error_curves.png")
    plt.figure(figsize=(8, 5))
    colors = {"ibvs": "#d9534f", "learned": "#f0ad4e", "guarded": "#5cb85c"}
    labels = {"ibvs": "Baseline (Pure IBVS)", "learned": "IBVS + Learned Residual", "guarded": "IBVS + Residual + Convergence Guard"}

    for m in methods:
        curve = example_curves.get(m, [])
        plt.plot(range(len(curve)), curve, marker="o", label=labels[m], color=colors[m], linewidth=2)

    plt.axhline(y=cfg.ibvs.error_threshold_px, color="gray", linestyle="--", label=f"Success Threshold ({cfg.ibvs.error_threshold_px}px)")
    plt.xlabel("Control Steps", fontsize=11)
    plt.ylabel("Image Feature Error ||s - s*|| (pixels)", fontsize=11)
    plt.title("OpenArm Last-Mile Visual Servoing Comparison (Same Initial Offset)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"[曲線儲存] 誤差收斂曲線圖已儲存至: {plot_path}\n")

    if not args.dry_run:
        cam_stream.stop()
        robot.stop()


if __name__ == "__main__":
    main()
