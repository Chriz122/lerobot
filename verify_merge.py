#!/usr/bin/env python3
"""
驗證多個單集 LeRobot 資料集合併後，資料是否完整、影像是否對得上原始資料。

用法：
    python verify_merge.py <原始資料夾> <合併後 repo_id>
例：
    python verify_merge.py ~/.cache/huggingface/lerobot/JJC489/Openarm_dataset JJC489/openarm_bottle_tray

檢查項目（每一集）：
  1. 幀數：合併後的集長度 == 原始集長度
  2. 數值：observation.state / action 逐行完全相同
  3. 影片時間範圍：(to_timestamp - from_timestamp) * fps ≈ 幀數
  4. 影像內容：抽樣幀（第一幀、中間、最後一幀 + 隨機幾幀）三台相機逐像素比對
     → 抓得到「時間戳偏移錯誤導致拿到別的畫面」與解碼失敗
"""
import sys
import random
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

N_RANDOM = 5          # 每集額外隨機抽幾幀
PIXEL_TOL = 2 / 255   # 影像平均絕對差容許值（直接串接不重編碼時應為 0）
random.seed(0)


def ep_meta(meta, i):
    e = meta.episodes[i]
    return dict(e) if not isinstance(e, dict) else e


def main():
    src = Path(sys.argv[1]).expanduser()
    merged_repo = sys.argv[2]
    eps = sorted(d for d in src.iterdir() if d.is_dir() and d.name.startswith("episode_"))

    merged = LeRobotDataset(merged_repo)
    cams = merged.meta.video_keys
    fps = merged.fps
    print(f"合併資料集：{merged.meta.total_episodes} 集 {merged.meta.total_frames} 幀；原始 {len(eps)} 集")
    if merged.meta.total_episodes != len(eps):
        print("✗ 集數不符，停止")
        return

    n_bad = 0
    total_src = 0
    for i, d in enumerate(eps):
        orig = LeRobotDataset(repo_id=f"local/{d.name}", root=d)
        problems = []
        em = ep_meta(merged.meta, i)
        start, end = int(em["dataset_from_index"]), int(em["dataset_to_index"])
        n_m, n_o = end - start, len(orig)
        total_src += n_o

        # 1. 幀數
        if n_m != n_o:
            problems.append(f"幀數 合併={n_m} 原始={n_o}")

        # 2. 數值逐行比對
        n = min(n_m, n_o)
        for col in ("observation.state", "action"):
            a = np.stack(merged.hf_dataset.select(range(start, start + n))[col])
            b = np.stack(orig.hf_dataset.select(range(n))[col])
            if not np.array_equal(a, b):
                problems.append(f"{col} 數值不同（最大差 {np.abs(a - b).max():.4g}）")

        # 3. 影片時間範圍
        for cam in cams:
            try:
                dur = em[f"videos/{cam}/to_timestamp"] - em[f"videos/{cam}/from_timestamp"]
                if abs(dur * fps - n_o) > 1.5:
                    problems.append(f"{cam} 影片長度 {dur * fps:.1f} 幀 vs {n_o}")
            except KeyError:
                pass

        # 4. 影像抽樣比對
        idxs = sorted({0, n // 2, n - 1, *random.sample(range(n), min(N_RANDOM, n))})
        for k in idxs:
            try:
                fm = merged[start + k]
                fo = orig[k]
            except Exception as e:
                problems.append(f"frame {k} 解碼失敗: {type(e).__name__}: {e}")
                break
            for cam in cams:
                diff = (fm[cam].float() - fo[cam].float()).abs().mean().item()
                if diff > PIXEL_TOL:
                    problems.append(f"frame {k} {cam} 影像不同（平均差 {diff:.4f}）")

        mark = "✗" if problems else "✓"
        n_bad += bool(problems)
        print(f"{mark} merged ep {i:2d} ← {d.name}  {n_m} 幀  抽查 {len(idxs)} 幀")
        for p in problems:
            print(f"    {p}")

    print(f"\n原始總幀數 {total_src}，合併總幀數 {merged.meta.total_frames}"
          + ("  ✓" if total_src == merged.meta.total_frames else "  ✗ 不符"))
    print(f"總結：{len(eps) - n_bad}/{len(eps)} 集通過" + ("" if n_bad else "，可以上傳"))


if __name__ == "__main__":
    main()
