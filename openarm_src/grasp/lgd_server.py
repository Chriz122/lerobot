#!/usr/bin/env python3
"""
lgd_server.py — Grasp-Anything (LGD, CVPR 2024) 語言條件夾取偵測服務。

LGD 需要 torch 1.12 / transformers 4.8（ALBEF + CLIP），與 lerobot 環境不相容，
所以獨立跑在 `lgd` conda 環境，rollout 端透過本機 socket（multiprocessing.connection）呼叫。

啟動（另開一個終端機，常駐）：
    conda activate lgd
    python ~/lerobot/openarm_src/grasp/lgd_server.py \
        --weights ~/third_party/weights/CVPR_2024_weights/model_lgd_grasp_anything++

請求：{"rgb": HxWx3 uint8 (RGB), "query": "bottle", "steps": int, "n": int, "roi": (x0, y0, size) 或 None}
回應：{"grasps": [(u, v, angle_rad, width_px, quality), ...]（原圖像素座標）, "ms": 推論毫秒}
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing.connection import Listener

import cv2
import numpy as np

LGD_ROOT = os.path.expanduser(os.environ.get("LGD_ROOT", "~/third_party/LGD"))
INPUT_SIZE = 224
DEFAULT_PORT = 6010
AUTHKEY = b"openarm-lgd"


def load_model(weights: str):
    sys.path.insert(0, LGD_ROOT)
    os.chdir(LGD_ROOT)          # ALBEF 設定檔以相對路徑讀取
    import torch

    net = torch.load(weights, map_location="cuda")
    net.eval()
    return torch, net


def make_diffusion(steps: int):
    """原始評估用 1000 步；用 respacing 縮短（模型直接預測 x0，步數少時仍可用）。"""
    from diffusion import gaussian_diffusion as gd
    from diffusion.respace import SpacedDiffusion, space_timesteps

    return SpacedDiffusion(
        use_timesteps=space_timesteps(1000, [steps]), betas=gd.get_named_beta_schedule("cosine", 1000),
        model_mean_type=gd.ModelMeanType.START_X, model_var_type=gd.ModelVarType.FIXED_SMALL,
        loss_type=gd.LossType.MSE, rescale_timesteps=False,
    )


def preprocess(rgb: np.ndarray, roi):
    """裁出正方形 ROI（預設為畫面中央最大正方形）並縮到 224，回傳張量與座標還原參數。"""
    h, w = rgb.shape[:2]
    if roi is None:
        size = min(h, w)
        x0, y0 = (w - size) // 2, (h - size) // 2
    else:
        x0, y0, size = (int(v) for v in roi)
        size = min(size, h, w)
        x0 = int(np.clip(x0, 0, w - size))
        y0 = int(np.clip(y0, 0, h - size))
    crop = cv2.resize(rgb[y0:y0 + size, x0:x0 + size], (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    img = crop.astype(np.float32) / 255.0
    img -= img.mean()                                    # 與 LGD 訓練時的 Image.normalise 相同
    return img.transpose(2, 0, 1)[None], (x0, y0, size / INPUT_SIZE)


def detect(torch, net, diffusion, rgb, query, n, roi):
    from inference.post_process import post_process_output
    from utils.dataset_processing import grasp

    x, (x0, y0, scale) = preprocess(rgb, roi)
    img = torch.from_numpy(x).cuda()
    shape = (1, 1, INPUT_SIZE, INPUT_SIZE)
    idx = torch.zeros(1).cuda()
    with torch.no_grad():
        pos = diffusion.p_sample_loop(net, shape, None, img, [query], 0.4, idx)
        q_img, ang_img, width_img = post_process_output(pos, net.cos_output_str, net.sin_output_str,
                                                        net.width_output_str)
    out = []
    for g in grasp.detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=n):
        r, c = g.center
        out.append((x0 + c * scale, y0 + r * scale, float(g.angle), float(g.length * scale),
                    float(q_img[int(r), int(c)])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="~/third_party/weights/CVPR_2024_weights/model_lgd_grasp_anything++")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--steps", type=int, default=50, help="預設擴散步數（請求可覆寫）")
    a = ap.parse_args()

    torch, net = load_model(os.path.expanduser(a.weights))
    cache = {a.steps: make_diffusion(a.steps)}
    print(f"[LGD] 模型已載入，監聽 127.0.0.1:{a.port}（擴散 {a.steps} 步）", flush=True)
    with Listener(("127.0.0.1", a.port), authkey=AUTHKEY) as listener:
        while True:
            with listener.accept() as conn:
                while True:
                    try:
                        req = conn.recv()
                    except EOFError:
                        break
                    steps = int(req.get("steps") or a.steps)
                    if steps not in cache:
                        cache[steps] = make_diffusion(steps)
                    t0 = time.perf_counter()
                    grasps = detect(torch, net, cache[steps], req["rgb"], req.get("query", "bottle"),
                                    int(req.get("n", 3)), req.get("roi"))
                    conn.send({"grasps": grasps, "ms": (time.perf_counter() - t0) * 1000.0})


if __name__ == "__main__":
    main()
