#!/usr/bin/env python3
"""離線比較 OWLv2 與 YOLOE-26-seg：在錄好的腕部相機影片上跑兩個偵測器，存對照圖並印統計。

不需要機器人。每台相機取 --frames 張（平均分布在整支影片），每張圖左邊 OWLv2、右邊 YOLOE（綠色 = 瓶子遮罩）。
框的顏色：綠 = 指定瓶子（label 0，分數 >= 觸發門檻）、黃 = 只認出是 bottle、灰 = 分數低於觸發門檻。

用法（~/lerobot，conda activate lerobot）：
  python -m openarm_src.grasp.compare_detectors \\
      --dataset ~/.cache/huggingface/lerobot/JJC489/openarm_bottle_tray \\
      --query-left "coca cola bottle" --query-right "blue bottle"

輸出：openarm_src/grasp/runs/compare_<時間>/<相機>_<偵測器>.png 與 summary.txt
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import cv2
import numpy as np

from openarm_src.grasp.detector import DEFAULT_YOLOE_MODEL, make_detector

HERE = os.path.dirname(os.path.abspath(__file__))


def read_frames(path: str, n: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in np.linspace(0, total - 1, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return frames


def draw(rgb: np.ndarray, boxes, trigger: float, title: str) -> np.ndarray:
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for b in boxes:
        if b.mask is not None and b.mask.shape == img.shape[:2]:
            img[b.mask] = (0.55 * img[b.mask] + 0.45 * np.array([0, 255, 0])).astype(np.uint8)
    for b in boxes:
        col = (160, 160, 160) if b.score < trigger else ((0, 255, 0) if b.label == 0 else (0, 220, 255))
        cv2.rectangle(img, (int(b.x0), int(b.y0)), (int(b.x1), int(b.y1)), col, 2)
        cv2.putText(img, f"{b.score:.2f}", (int(b.x0) + 2, int(b.y0) + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    cv2.putText(img, title, (6, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return img


def grid(tiles: list[np.ndarray], cols: int) -> np.ndarray:
    h, w = tiles[0].shape[:2]
    rows = (len(tiles) + cols - 1) // cols
    out = np.zeros((rows * h, cols * w, 3), np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        out[r * h:(r + 1) * h, c * w:(c + 1) * w] = t
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="離線比較 OWLv2 與 YOLOE-26-seg")
    ap.add_argument("--dataset", default=os.path.expanduser("~/.cache/huggingface/lerobot/JJC489/openarm_bottle_tray"))
    ap.add_argument("--query-left", default="coca cola bottle")
    ap.add_argument("--query-right", default="blue bottle")
    ap.add_argument("--frames", type=int, default=24, help="每台相機取幾張")
    ap.add_argument("--threshold", type=float, default=0.10, help="偵測器輸出門檻（= --strategy.detector_threshold）")
    ap.add_argument("--trigger", type=float, default=0.25, help="觸發門檻（= --strategy.trigger_score）")
    ap.add_argument("--yoloe-model", default=DEFAULT_YOLOE_MODEL)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    queries, frames = {}, {}
    for side in ("left", "right"):
        vids = sorted(glob.glob(os.path.join(a.dataset, "videos", f"observation.images.wrist_{side}", "*", "*.mp4")))
        if not vids:
            continue
        q = getattr(a, f"query_{side}")
        cam = f"wrist_{side}"
        queries[cam] = [q] if q.strip() == "bottle" else [q, "bottle"]
        frames[cam] = read_frames(vids[0], a.frames)
        print(f"[compare] {cam}：{vids[0]}，取 {len(frames[cam])} 張，查詢 {queries[cam]}")
    if not queries:
        raise SystemExit(f"{a.dataset} 裡找不到腕部相機影片")
    out_dir = a.out or os.path.join(HERE, "runs", time.strftime("compare_%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    lines = []
    for backend, model in (("owlv2", None), ("yoloe", a.yoloe_model)):
        print(f"[compare] 載入 {backend} ...")
        det = make_detector(backend, queries, a.threshold, model)
        det.detect({c: f[0] for c, f in frames.items()})            # 暖機
        for cam, imgs in frames.items():
            lat, n_trig, n_any, n_boxes, tiles = [], 0, 0, 0, []
            for k, rgb in enumerate(imgs):
                t0 = time.perf_counter()
                boxes = det.detect({cam: rgb})[cam]
                lat.append((time.perf_counter() - t0) * 1000)
                n_trig += any(b.label == 0 and b.score >= a.trigger for b in boxes)
                n_any += bool(boxes)
                n_boxes += len(boxes)
                tiles.append(draw(rgb, boxes, a.trigger, f"{backend} #{k}"))
            path = os.path.join(out_dir, f"{cam}_{backend}.png")
            cv2.imwrite(path, grid(tiles, 4))
            line = (f"{cam:11s} {backend:6s}｜單台相機推論 {np.mean(lat):5.1f} ms（最慢 {np.max(lat):5.1f}）｜"
                    f"可觸發（指定瓶子 ≥ {a.trigger}）{n_trig}/{len(imgs)}｜有任何框 {n_any}/{len(imgs)}｜"
                    f"平均每張 {n_boxes / len(imgs):.1f} 個框")
            print(line)
            lines.append(line)
        del det
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[compare] 對照圖與統計：{out_dir}")


if __name__ == "__main__":
    main()
