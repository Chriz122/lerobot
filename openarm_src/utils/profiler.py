from __future__ import annotations

import logging
import time

from openarm_src.utils.config import STATS_PERIOD

logger = logging.getLogger("openarm_gc.profiler")


class LoopProfiler:
    """包住 robot.get_observation / robot.send_action / teleop.get_action，每 5 秒印出平均/最大耗時。
    get_observation 的呼叫頻率 ≈ record 主迴圈的實際 fps。"""

    def __init__(self):
        self.t0 = time.perf_counter()
        self.d: dict[str, list] = {}

    def attach(self, robot, teleop):
        for obj, name, label in (
            (robot, "get_observation", "obs"),
            (robot, "send_action", "send"),
            (teleop, "get_action", "teleop"),
        ):
            if obj is None or getattr(obj, f"_prof_{name}", False):
                continue
            fn = getattr(obj, name)

            def wrapped(*a, _fn=fn, _label=label, **k):
                t = time.perf_counter()
                try:
                    return _fn(*a, **k)
                finally:
                    self._add(_label, time.perf_counter() - t)

            setattr(obj, name, wrapped)
            setattr(obj, f"_prof_{name}", True)

    def _add(self, label, dt):
        n, s, m = self.d.get(label, (0, 0.0, 0.0))
        self.d[label] = (n + 1, s + dt, max(m, dt))
        now = time.perf_counter()
        if now - self.t0 >= STATS_PERIOD:
            el = now - self.t0
            parts = []
            if "obs" in self.d:
                parts.append(f"主迴圈 ≈ {self.d['obs'][0] / el:4.1f} Hz")
            for lb in ("obs", "teleop", "send"):
                if lb in self.d:
                    n, s, m = self.d[lb]
                    parts.append(f"{lb} 平均 {1e3 * s / n:6.1f} ms 最大 {1e3 * m:6.1f} ms")
            logger.info("[loop-stats] " + " | ".join(parts))
            self.t0, self.d = now, {}

