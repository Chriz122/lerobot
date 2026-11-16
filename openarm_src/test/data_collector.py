# openarm_src/test/data_collector.py
"""
微幅安全試探資料收集器 (Safe Dithering Probe)。

依指南第 3 節：在目前姿態附近施加微小命令 a，觀測 AprilTag 特徵變化 ds_obs，
建立 (s, a, ds_obs) 資料集來訓練 Jacobian 殘差。真正的 J_real 不必直接量到。

做法是「有邊界的隨機漫步」：每步隨機取一個小命令；若預測會讓 Tag 離開起點附近的
安全框（中心偏移、視邊長變化都有上限），就改成往起點拉回的命令。
這樣資料涵蓋起點附近一小塊區域，而不是只有單一點。

單獨執行（離線模擬）： python data_collector.py
真機：                 python data_collector.py --real --side left --port can1
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ibvs_pinn_controller import depth_from_size, feature_error, interaction_matrix


def collect_probe_dataset(approach, n_samples: Optional[int] = None, seed: int = 42,
                          out_path: Optional[str] = None, verbose: bool = True) -> Dict[str, np.ndarray]:
    cfg = approach.cfg
    pc = cfg.probe
    n_samples = n_samples or pc.n_samples
    robot = approach.robot
    rng = np.random.default_rng(seed)
    amp = np.array([pc.amp_xy_mm, pc.amp_xy_mm, pc.amp_z_mm, pc.amp_yaw_deg])

    start = approach.observe_average(5)
    if start is None:
        raise RuntimeError("試探起點看不到 Tag：請先把手臂移到腕部相機看得到方塊的位置")
    s0, intr = start.s, start.intr
    if verbose:
        print(f"[DataCollector] 開始微幅安全試探 ({n_samples} 筆)，單步上限 "
              f"±{pc.amp_xy_mm}mm / ±{pc.amp_yaw_deg}deg，保持無接觸。")

    s_list, a_list, ds_list = [], [], []
    attempts = 0
    while len(s_list) < n_samples and attempts < n_samples * 3:
        attempts += 1
        before = approach.observe()
        if before is None:
            robot.settle()
            continue

        depth = depth_from_size(before.s[2], intr.f, cfg.target.tag_size_meters)
        j_phys = interaction_matrix(before.s, depth, intr)
        action = rng.uniform(-amp, amp)
        predicted = before.s + j_phys @ action
        drift = feature_error(predicted, s0)
        outside = (
            np.hypot(drift[0], drift[1]) > pc.box_center_px
            or abs(drift[2]) > pc.box_size_ratio * s0[2]
            or abs(drift[3]) > 12.0
        )
        if outside:
            # 往起點拉回，但仍保留一點隨機性
            pull = -np.linalg.pinv(j_phys) @ feature_error(before.s, s0)
            scale = float(np.max(np.abs(pull) / amp))
            action = pull / scale if scale > 1.0 else pull
            action = action + 0.3 * rng.uniform(-amp, amp)

        if not approach.move_camera(action):
            if verbose:
                print(f"  試探命令被安全檢查擋下: {robot.last_block_reason}")
            continue
        robot.settle()                      # 等手臂靜止再取像，避免時間不同步
        after = approach.observe()
        if after is None:
            continue

        s_list.append(before.s)
        a_list.append(action)
        ds_list.append(feature_error(after.s, before.s))
        if verbose and (len(s_list) % 30 == 0 or len(s_list) == n_samples):
            print(f"  進度: {len(s_list)}/{n_samples}")

    data = {
        "s": np.array(s_list), "a": np.array(a_list), "ds": np.array(ds_list),
        "intr": np.array([intr.f, intr.cx, intr.cy, intr.width, intr.height], dtype=float),
    }
    if out_path:
        np.savez_compressed(out_path, **data)
        if verbose:
            print(f"[DataCollector] 共 {len(s_list)} 筆樣本，已儲存至 {out_path}")
    return data


if __name__ == "__main__":
    import argparse
    import os

    from main_experiment import add_hardware_args, build_system

    parser = argparse.ArgumentParser(description="微幅安全試探資料收集")
    add_hardware_args(parser)
    parser.add_argument("--samples", type=int, default=None, help="試探樣本數（預設見 config.ProbeConfig）")
    parser.add_argument("--out", default=os.path.join("experiment_outputs", "probe_data.npz"))
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    system = build_system(args)
    try:
        collect_probe_dataset(system.approach, args.samples, out_path=args.out)
    finally:
        system.close()
