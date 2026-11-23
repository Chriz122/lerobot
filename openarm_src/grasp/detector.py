"""
開放詞彙瓶子偵測（腕部相機）：OWLv2（預設）或 YOLOE-26-seg（偵測 + 實例分割，`detector=yoloe`）。

AsyncDetector 在「獨立行程」推論：影像前處理與模型 forward 都很吃 Python（GIL），
放在控制迴圈同一個行程會拖慢 ACT 的 fps。主行程只保留每台相機最新一幀，
彩色影像送到偵測行程，深度圖留在主行程，結果回來時再配回同一幀的深度。
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import threading
import time
import os
from dataclasses import dataclass, field, replace

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "google/owlv2-base-patch16-ensemble"
# YOLOE-26-seg：第一次使用時權重（約 30 MB）與文字編碼器 mobileclip2（約 240 MB）會下載到這個資料夾
DEFAULT_YOLOE_MODEL = "~/models/yoloe/yoloe-26s-seg.pt"
BACKENDS = ("owlv2", "yoloe")


@dataclass
class Box:
    x0: float
    y0: float
    x1: float
    y1: float
    score: float
    label: int = 0           # 符合第幾個查詢：0 = 指定的瓶子（例如 coca cola bottle），1 = 一般的 bottle
    mask: np.ndarray | None = field(default=None, repr=False, compare=False)   # YOLOE-seg：瓶子像素（HxW bool）

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


def _area(b: Box) -> float:
    return max(b.x1 - b.x0, 0.0) * max(b.y1 - b.y0, 0.0)


def _inter(a: Box, b: Box) -> float:
    w = min(a.x1, b.x1) - max(a.x0, b.x0)
    h = min(a.y1, b.y1) - max(a.y0, b.y0)
    return max(w, 0.0) * max(h, 0.0)


def merge_duplicate_boxes(boxes: list[Box], iou_thr: float = 0.5, contain_thr: float = 0.8,
                          keep_larger_min_ratio: float = 0.5) -> list[Box]:
    """合併同一支瓶子的重複框（OWLv2 的後處理沒有 NMS，不分查詢類別）。

    - 兩框 IoU > iou_thr：保留分數高的。
    - 小框有 contain_thr 以上落在大框裡（例如只框到標籤、瓶蓋）：保留大框，
      除非大框分數不到小框的 keep_larger_min_ratio 倍（大框多半是誤框，例如框到兩支瓶子）。
    - 被合併掉的框若符合指定查詢（label 0），保留下來的框也標成 label 0：同一支瓶子有一部分被認出是指定的瓶子。
    先丟掉「同時包住兩個左右分開的框」的大框（框到並排的兩支瓶子），以免把兩支瓶子併成一個。
    只看左右：瓶子是直立的，同一支瓶子的局部框（瓶蓋、標籤）是上下排，並排的兩支瓶子才會左右分開。
    回傳依分數由高到低排序。
    """
    def contains(big: Box, small: Box) -> bool:
        return _area(small) < _area(big) and _inter(big, small) > contain_thr * _area(small)

    def group_box(g: Box) -> bool:
        inner = [b for b in boxes if b is not g and contains(g, b)]
        return any(min(p.x1, q.x1) <= max(p.x0, q.x0) for i, p in enumerate(inner) for q in inner[i + 1:])

    kept: list[Box] = []
    for b in sorted((b for b in boxes if not group_box(b)), key=lambda x: -x.score):
        merged = False
        for i, k in enumerate(kept):
            inter = _inter(b, k)
            if inter <= 0.0:
                continue
            a_b, a_k = _area(b), _area(k)
            iou = inter / max(a_b + a_k - inter, 1e-9)
            contain = inter / max(min(a_b, a_k), 1e-9)
            if iou <= iou_thr and contain <= contain_thr:
                continue
            # k 的分數 >= b。IoU 重疊或 k 較大：留 k；b 較大（k 是被包住的局部框）且分數不太低：換成 b
            winner = k
            if iou <= iou_thr and a_b > a_k and b.score >= keep_larger_min_ratio * k.score:
                winner = replace(b, score=k.score)
            kept[i] = replace(winner, label=min(winner.label, b.label, k.label))
            merged = True
            break
        if not merged:
            kept.append(b)
    return sorted(kept, key=lambda x: -x.score)


class BottleDetector:
    """同步介面：detect({cam: rgb}) -> {cam: [Box]}。"""

    def __init__(self, queries: dict[str, str | list[str]], threshold: float = 0.2, model_name: str = DEFAULT_MODEL,
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

    def _query_list(self, name: str) -> list[str]:
        q = self.queries[name]
        return [q] if isinstance(q, str) else list(q)

    def detect(self, images: dict[str, np.ndarray]) -> dict[str, list[Box]]:
        torch = self.torch
        names = list(images)
        texts = [[f"a photo of a {q}" for q in self._query_list(n)] for n in names]
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
            labels = res.get("labels")
            labels = labels.tolist() if labels is not None else [0] * len(res["scores"])
            for score, box, label in zip(res["scores"].float().tolist(), res["boxes"].float().tolist(), labels):
                x0, y0, x1, y1 = box
                x0, x1 = np.clip([x0, x1], 0, w - 1)
                y0, y1 = np.clip([y0, y1], 0, h - 1)
                if x1 - x0 > 2 and y1 - y0 > 2:
                    items.append(Box(x0, y0, x1, y1, score, int(label)))
            boxes[n] = merge_duplicate_boxes(items)
        return boxes


class YoloeDetector:
    """YOLOE-26-seg（Ultralytics）開放詞彙偵測 + 實例分割。介面同 BottleDetector，Box.mask = 瓶子像素。

    YOLOE 每個框只給一個類別，所以「指定瓶子」與「一般 bottle」放在同一組類別會互搶：
    「bottle」分數總是比「coca cola bottle」高，指定瓶子幾乎不會被認出來（錄影上可觸發 4/60）；
    只放指定瓶子，近距離看不到標籤時又整支瓶子都偵測不到。
    做法與 OWLv2 的兩個查詢相同：兩個模型各自固定類別（切換類別每次約 140 ms，不能每幀切）：
      - 指定模型：類別 = 各相機要找的瓶子的聯集（例如 coca cola bottle、blue bottle）
          這台相機的瓶子 -> label 0（觸發只認這個），別台相機的瓶子 -> label 1
      - 一般模型：類別 = bottle -> label 1（追蹤時靠 3D 鎖定區分是哪一支）
    兩組框用 merge_duplicate_boxes 合併：同一支瓶子兩邊都有框時，保留 label 0。
    """

    def __init__(self, queries: dict[str, str | list[str]], threshold: float = 0.1,
                 model_name: str = DEFAULT_YOLOE_MODEL, device: str = "cuda"):
        from ultralytics import YOLOE

        self.queries = {n: ([q] if isinstance(q, str) else list(q)) for n, q in queries.items()}
        self.target = {n: qs[0] for n, qs in self.queries.items()}        # 每台相機要找的瓶子（第一個查詢）
        self.classes: list[str] = []
        for q in self.target.values():
            if q not in self.classes:
                self.classes.append(q)
        generic = sorted({q for qs in self.queries.values() for q in qs[1:]} - set(self.classes))
        path = os.path.expanduser(model_name)
        wdir = os.path.dirname(path) or "."
        os.makedirs(wdir, exist_ok=True)
        cwd = os.getcwd()
        os.chdir(wdir)        # Ultralytics 把權重與文字編碼器下載到目前資料夾
        try:
            self.model = YOLOE(os.path.basename(path))
            self.model.set_classes(self.classes, self.model.get_text_pe(self.classes))
            self.generic = None
            if generic:
                self.generic = YOLOE(os.path.basename(path))
                self.generic.set_classes(generic, self.generic.get_text_pe(generic))
        finally:
            os.chdir(cwd)
        self.threshold = threshold
        self.device = device

    def _run(self, model, images: list[np.ndarray]):
        # Ultralytics 把 numpy 影像當 BGR；retina_masks：遮罩與原圖同大小
        return model.predict([np.ascontiguousarray(im[..., ::-1]) for im in images], conf=self.threshold,
                             verbose=False, retina_masks=True, device=self.device)

    @staticmethod
    def _boxes(r, h: int, w: int, label_fn) -> list[Box]:
        items = []
        if r.boxes is None or not len(r.boxes):
            return items
        xyxy = r.boxes.xyxy.float().cpu().numpy()
        conf = r.boxes.conf.float().cpu().numpy()
        cls = r.boxes.cls.int().cpu().numpy()
        masks = (r.masks.data > 0.5).cpu().numpy() if r.masks is not None else None
        for i, ((x0, y0, x1, y1), sc, c) in enumerate(zip(xyxy, conf, cls)):
            x0, x1 = np.clip([x0, x1], 0, w - 1)
            y0, y1 = np.clip([y0, y1], 0, h - 1)
            m = masks[i] if masks is not None and masks[i].shape == (h, w) else None
            if x1 - x0 > 2 and y1 - y0 > 2:
                items.append(Box(float(x0), float(y0), float(x1), float(y1), float(sc), label_fn(int(c)), m))
        return items

    def detect(self, images: dict[str, np.ndarray]) -> dict[str, list[Box]]:
        names = list(images)
        imgs = [images[n] for n in names]
        res_t = self._run(self.model, imgs)
        res_g = self._run(self.generic, imgs) if self.generic is not None else [None] * len(names)
        boxes = {}
        for n, img, rt, rg in zip(names, imgs, res_t, res_g):
            h, w = img.shape[:2]
            mine = self.classes.index(self.target[n])
            items = self._boxes(rt, h, w, lambda c, mine=mine: 0 if c == mine else 1)
            if rg is not None:
                items += self._boxes(rg, h, w, lambda c: 1)
            boxes[n] = merge_duplicate_boxes(items)
        return boxes


def make_detector(backend: str, queries: dict[str, str | list[str]], threshold: float, model_name: str | None = None):
    """backend = "owlv2" 或 "yoloe"；model_name 空白 = 該 backend 的預設模型。"""
    if backend == "yoloe":
        return YoloeDetector(queries, threshold, model_name or DEFAULT_YOLOE_MODEL)
    if backend == "owlv2":
        return BottleDetector(queries, threshold, model_name or DEFAULT_MODEL)
    raise ValueError(f"未知的偵測器 {backend!r}（可用：{', '.join(BACKENDS)}）")


def _pack_mask(m: np.ndarray | None):
    return None if m is None else (np.packbits(m), m.shape)


def _unpack_mask(p) -> np.ndarray | None:
    if p is None:
        return None
    bits, shape = p
    return np.unpackbits(bits, count=int(np.prod(shape))).reshape(shape).astype(bool)


def _detector_worker(conn, queries, threshold, model_name, backend="owlv2"):
    """偵測行程：收 {cam: rgb}，回 {cam: [(x0, y0, x1, y1, score, label, 壓縮的遮罩), ...]}。"""
    try:
        det = make_detector(backend, queries, threshold, model_name)
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
            conn.send(("ok", {n: [(b.x0, b.y0, b.x1, b.y1, b.score, b.label, _pack_mask(b.mask)) for b in bs]
                              for n, bs in out.items()}))
        except Exception as e:  # noqa: BLE001
            conn.send(("error", repr(e)))


class AsyncDetector:
    """submit() 放最新一幀，latest() 取最新結果；推論在獨立行程，限制 max_hz。"""

    def __init__(self, queries: dict[str, str], threshold: float = 0.15, model_name: str | None = None,
                 max_hz: float = 10.0, startup_timeout_s: float = 300.0, backend: str = "owlv2"):
        if backend not in BACKENDS:
            raise ValueError(f"未知的偵測器 {backend!r}（可用：{', '.join(BACKENDS)}）")
        self.backend = backend
        ctx = mp.get_context("spawn")
        self._conn, child = ctx.Pipe()
        self._proc = ctx.Process(target=_detector_worker, args=(child, queries, threshold, model_name, backend),
                                 daemon=True, name=f"{backend}-detector")
        self._proc.start()
        t_start = time.perf_counter()
        while not self._conn.poll(0.5):
            if not self._proc.is_alive():
                raise RuntimeError(f"{backend} 偵測行程啟動失敗（exit code {self._proc.exitcode}），請看上方錯誤訊息")
            if time.perf_counter() - t_start > startup_timeout_s:
                self._proc.terminate()
                raise RuntimeError(f"{backend} 偵測行程啟動逾時")
        status, info = self._conn.recv()
        if status != "ready":
            raise RuntimeError(f"{backend} 偵測行程啟動失敗：{info}")
        self.min_period = 1.0 / max_hz
        self._lock = threading.Lock()
        self._pending: dict[str, _Pending] = {}
        self._results: dict[str, Detection] = {}
        self._frame_ids: dict[str, int] = {}
        self._wake = threading.Event()
        self._stop = threading.Event()
        self.latency_ms = 0.0
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"{backend}-io")
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
                logger.error("[IBVS] %s 偵測行程已結束", self.backend)
                return
            if status != "ok":
                logger.error("[IBVS] %s 偵測失敗：%s", self.backend, out)
                continue
            self.latency_ms = (time.perf_counter() - t0) * 1000.0
            with self._lock:
                for n, p in batch.items():
                    boxes = [Box(*b[:6], mask=_unpack_mask(b[6])) for b in out.get(n, [])]
                    self._results[n] = Detection(n, p.frame_id, p.t, boxes, p.depth_m, p.rgb.shape[:2])


def box_depth_m(det: Detection, box: Box, frac: float = 0.3) -> float | None:
    """框中央 frac 比例區域內有效深度的中位數 (m)。

    有分割遮罩（YOLOE-seg）時只取中央區域裡「瓶子的像素」，背景（桌面、托盤、後方物體）不會混進來；
    中央區域裡瓶子像素太少（例如框被切到），改用整個遮罩。
    """
    if det.depth_m is None:
        return None
    c = box.center
    hw = box.size * frac / 2.0
    y0, y1 = int(max(c[1] - hw[1], 0)), int(min(c[1] + hw[1] + 1, det.depth_m.shape[0]))
    x0, x1 = int(max(c[0] - hw[0], 0)), int(min(c[0] + hw[0] + 1, det.depth_m.shape[1]))
    patch = det.depth_m[y0:y1, x0:x1]
    if box.mask is not None and box.mask.shape == det.depth_m.shape:
        ok = (patch > 0.03) & (patch < 2.0)
        sel = patch[ok & box.mask[y0:y1, x0:x1]]
        if sel.size >= 5:
            return float(np.median(sel))
        whole = det.depth_m[box.mask]
        whole = whole[(whole > 0.03) & (whole < 2.0)]
        return float(np.median(whole)) if whole.size >= 20 else None
    valid = patch[(patch > 0.03) & (patch < 2.0)]
    if valid.size < max(5, patch.size // 10):
        return None
    return float(np.median(valid))
