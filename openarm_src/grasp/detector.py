"""
OWLv2 開放詞彙瓶子偵測（腕部相機）。

AsyncDetector 在「獨立行程」推論：影像前處理與模型 forward 都很吃 Python（GIL），
放在控制迴圈同一個行程會拖慢 ACT 的 fps。主行程只保留每台相機最新一幀，
彩色影像送到偵測行程，深度圖留在主行程，結果回來時再配回同一幀的深度。
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import threading
import time
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "google/owlv2-base-patch16-ensemble"


@dataclass
class Box:
    x0: float
    y0: float
    x1: float
    y1: float
    score: float

    @property
    def center(self) -> np.ndarray:
        return np.array([(self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0])

    @property
    def size(self) -> np.ndarray:
        return np.array([self.x1 - self.x0, self.y1 - self.y0])


@dataclass
class Detection:
    camera: str
    frame_id: int
    t: float                                  # 影像擷取時間 (perf_counter)
    boxes: list[Box]
    depth_m: np.ndarray | None = None         # HxW (m)，與彩色影像像素對齊（D405）
    image_hw: tuple[int, int] = (0, 0)


@dataclass
class _Pending:
    frame_id: int
    t: float
    rgb: np.ndarray
    depth_m: np.ndarray | None


class BottleDetector:
    """同步介面：detect({cam: rgb}) -> {cam: [Box]}。"""

    def __init__(self, queries: dict[str, str], threshold: float = 0.2, model_name: str = DEFAULT_MODEL,
                 device: str = "cuda"):
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.torch = torch
        self.queries = queries
        self.threshold = threshold
        self.device = device
        self.dtype = torch.float16 if device.startswith("cuda") else torch.float32
        # torchvision 後端：前處理（放大到 960x960）在 GPU 上做，CPU 版每張約 10 ms
        try:
            self.processor = Owlv2Processor.from_pretrained(model_name, backend="torchvision")
        except (TypeError, ValueError):
            self.processor = Owlv2Processor.from_pretrained(model_name)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_name, torch_dtype=self.dtype).to(device).eval()

    def detect(self, images: dict[str, np.ndarray]) -> dict[str, list[Box]]:
        torch = self.torch
        names = list(images)
        texts = [[f"a photo of a {self.queries[n]}"] for n in names]
        try:
            inputs = self.processor(text=texts, images=[images[n] for n in names], return_tensors="pt",
                                    device=self.device)
        except TypeError:
            inputs = self.processor(text=texts, images=[images[n] for n in names], return_tensors="pt")
        inputs = {k: (v.to(self.device, self.dtype) if v.is_floating_point() else v.to(self.device))
                  for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model(**inputs)
        sizes = torch.tensor([[max(images[n].shape[:2])] * 2 for n in names])   # OWLv2 會補成正方形
        results = self.processor.post_process_grounded_object_detection(
            out, threshold=self.threshold, target_sizes=sizes)
        boxes = {}
        for n, res in zip(names, results):
            h, w = images[n].shape[:2]
            items = []
            for score, box in zip(res["scores"].float().tolist(), res["boxes"].float().tolist()):
                x0, y0, x1, y1 = box
                x0, x1 = np.clip([x0, x1], 0, w - 1)
                y0, y1 = np.clip([y0, y1], 0, h - 1)
                if x1 - x0 > 2 and y1 - y0 > 2:
                    items.append(Box(x0, y0, x1, y1, score))
            boxes[n] = sorted(items, key=lambda b: -b.score)
        return boxes


def _detector_worker(conn, queries, threshold, model_name):
    """偵測行程：收 {cam: rgb}，回 {cam: [(x0, y0, x1, y1, score), ...]}。"""
    try:
        det = BottleDetector(queries, threshold, model_name)
        det.detect({n: np.zeros((240, 424, 3), np.uint8) for n in queries})   # 暖機
        conn.send(("ready", None))
    except Exception as e:  # noqa: BLE001
        conn.send(("error", repr(e)))
        return
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            return
        if msg is None:
            return
        try:
            out = det.detect(msg)
            conn.send(("ok", {n: [(b.x0, b.y0, b.x1, b.y1, b.score) for b in bs] for n, bs in out.items()}))
        except Exception as e:  # noqa: BLE001
            conn.send(("error", repr(e)))


class AsyncDetector:
    """submit() 放最新一幀，latest() 取最新結果；推論在獨立行程，限制 max_hz。"""

    def __init__(self, queries: dict[str, str], threshold: float = 0.15, model_name: str = DEFAULT_MODEL,
                 max_hz: float = 10.0, startup_timeout_s: float = 120.0):
        ctx = mp.get_context("spawn")
        self._conn, child = ctx.Pipe()
        self._proc = ctx.Process(target=_detector_worker, args=(child, queries, threshold, model_name),
                                 daemon=True, name="owlv2-detector")
        self._proc.start()
        t_start = time.perf_counter()
        while not self._conn.poll(0.5):
            if not self._proc.is_alive():
                raise RuntimeError(f"OWLv2 偵測行程啟動失敗（exit code {self._proc.exitcode}），請看上方錯誤訊息")
            if time.perf_counter() - t_start > startup_timeout_s:
                self._proc.terminate()
                raise RuntimeError("OWLv2 偵測行程啟動逾時")
        status, info = self._conn.recv()
        if status != "ready":
            raise RuntimeError(f"OWLv2 偵測行程啟動失敗：{info}")
        self.min_period = 1.0 / max_hz
        self._lock = threading.Lock()
        self._pending: dict[str, _Pending] = {}
        self._results: dict[str, Detection] = {}
        self._frame_ids: dict[str, int] = {}
        self._wake = threading.Event()
        self._stop = threading.Event()
        self.latency_ms = 0.0
        self._thread = threading.Thread(target=self._run, daemon=True, name="owlv2-io")
        self._thread.start()

    def submit(self, camera: str, rgb: np.ndarray, depth_m: np.ndarray | None) -> None:
        with self._lock:
            fid = self._frame_ids.get(camera, 0) + 1
            self._frame_ids[camera] = fid
            self._pending[camera] = _Pending(fid, time.perf_counter(), rgb, depth_m)
        self._wake.set()

    def latest(self, camera: str) -> Detection | None:
        with self._lock:
            return self._results.get(camera)

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=2.0)
        try:
            self._conn.send(None)
        except (OSError, BrokenPipeError):
            pass
        self._proc.join(timeout=3.0)
        if self._proc.is_alive():
            self._proc.terminate()

    def _run(self) -> None:
        last = 0.0
        while not self._stop.is_set():
            self._wake.wait(0.5)
            self._wake.clear()
            wait = self.min_period - (time.perf_counter() - last)
            if wait > 0:
                time.sleep(wait)
            with self._lock:
                batch, self._pending = self._pending, {}
            if not batch:
                continue
            last = t0 = time.perf_counter()
            try:
                self._conn.send({n: p.rgb for n, p in batch.items()})
                status, out = self._conn.recv()
            except (EOFError, OSError, BrokenPipeError):
                logger.error("[IBVS] OWLv2 偵測行程已結束")
                return
            if status != "ok":
                logger.error("[IBVS] OWLv2 偵測失敗：%s", out)
                continue
            self.latency_ms = (time.perf_counter() - t0) * 1000.0
            with self._lock:
                for n, p in batch.items():
                    boxes = [Box(*b) for b in out.get(n, [])]
                    self._results[n] = Detection(n, p.frame_id, p.t, boxes, p.depth_m, p.rgb.shape[:2])


def box_depth_m(det: Detection, box: Box, frac: float = 0.3) -> float | None:
    """框中央 frac 比例區域內有效深度的中位數 (m)。"""
    if det.depth_m is None:
        return None
    c = box.center
    hw = box.size * frac / 2.0
    y0, y1 = int(max(c[1] - hw[1], 0)), int(min(c[1] + hw[1] + 1, det.depth_m.shape[0]))
    x0, x1 = int(max(c[0] - hw[0], 0)), int(min(c[0] + hw[0] + 1, det.depth_m.shape[1]))
    patch = det.depth_m[y0:y1, x0:x1]
    valid = patch[(patch > 0.03) & (patch < 2.0)]
    if valid.size < max(5, patch.size // 10):
        return None
    return float(np.median(valid))
