from __future__ import annotations

import logging
import os
import threading
import time
import numpy as np

from openarm_src.utils.config import (
    ARM_JOINTS,
    CONTROL_HZ,
    DEFAULT_FORCE_FEEDBACK_ALPHA,
    DEFAULT_FORCE_FEEDBACK_DEADBAND,
    DEFAULT_FORCE_FEEDBACK_LIMIT,
    DEFAULT_FORCE_FEEDBACK_SCALE,
    DISABLE_RETRY,
    ENABLE_RETRY,
    FRICTION_COMP,
    FRICTION_VEL_EPS,
    GC_KD,
    GRIPPER,
    GRIPPER_HOLD_KD,
    GRIPPER_HOLD_KP,
    GRIPPER_KD,
    HOLD_KD,
    HOLD_KP,
    JOINT_GC_SCALE,
    STATS_PERIOD,
    TAU_LIMIT,
)
from openarm_src.utils.gravity_model import GravityModel
from openarm_src.utils.homing import min_jerk

logger = logging.getLogger("openarm_gc.leader_controller")


class LeaderGC:
    def __init__(self, leader, side: str | None):
        self.leader = leader
        self.bus = leader.bus
        self.gm = GravityModel(os.environ["OPENARM_URDF"], side)
        self.scale = float(os.environ.get("OPENARM_GC_SCALE", "1.0"))
        self.kd_scale = float(os.environ.get("OPENARM_KD_SCALE", "1.0"))
        self.fric_scale = float(os.environ.get("OPENARM_FRICTION_SCALE", "0.0"))
        self.fric_eps = float(os.environ.get("OPENARM_FRICTION_EPS", str(FRICTION_VEL_EPS)))
        # 力回饋參數
        self.ff_scale = float(os.environ.get("OPENARM_FF_SCALE", DEFAULT_FORCE_FEEDBACK_SCALE))
        self.ff_sign = float(os.environ.get("OPENARM_FF_SIGN", "-1.0"))
        self.ff_limit = abs(float(os.environ.get("OPENARM_FF_LIMIT", DEFAULT_FORCE_FEEDBACK_LIMIT)))
        self.ff_alpha = min(
            max(float(os.environ.get("OPENARM_FF_ALPHA", DEFAULT_FORCE_FEEDBACK_ALPHA)), 0.0), 1.0
        )
        self.ff_deadband = abs(float(os.environ.get("OPENARM_FF_DEADBAND", DEFAULT_FORCE_FEEDBACK_DEADBAND)))
        self.hz = float(os.environ.get("OPENARM_CONTROL_HZ", str(CONTROL_HZ)))
        self.stats_on = os.environ.get("OPENARM_GC_STATS", "1") != "0"
        self.lock = threading.Lock()
        self.target: dict[str, float] | None = None  # None = 純重力補償
        self.feedback_source = None
        self.latest: dict[str, dict] = {}
        self.last_tau = np.zeros(7)
        self.last_feedback_tau = 0.0
        self.feedback_samples = 0
        self.feedback_max = 0.0
        self.external_max = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: Exception | None = None

    # --- public -------------------------------------------------------------
    def start(self):
        states = self.bus.sync_read_all_states()
        with self.lock:
            self.latest = {m: dict(s) for m, s in states.items()}
        failed = []
        for motor in self.bus.motors:
            enabled = False
            last_error = None
            for attempt in range(1, ENABLE_RETRY + 1):
                try:
                    self.bus.enable_torque([motor])
                    enabled = True
                    break
                except Exception as e:  # noqa: BLE001
                    last_error = e
                    logger.warning(
                        f"[gc] {self.bus.port}/{motor}: 啟動力矩第 {attempt}/{ENABLE_RETRY} 次失敗 ({e})"
                    )
                    time.sleep(0.05)
            if not enabled:
                failed.append(f"{self.bus.port}/{motor}: {last_error}")

        if failed:
            try:
                self.bus.disable_torque(num_retry=DISABLE_RETRY)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[gc] 啟用失敗後關閉力矩也失敗 ({e})")
            raise ConnectionError("Torque enable failed: " + "; ".join(failed))

        logger.info(f"[gc] 力矩已啟用: {self.bus.port} ({len(self.bus.motors)} 顆馬達)")
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"gc-{self.gm.side}")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
            if self._thread.is_alive():
                logger.warning(f"[gc] {self.bus.port}: 重力補償執行緒 3 秒內沒停，仍繼續關力矩")

    def set_target(self, target: dict[str, float] | None):
        with self.lock:
            self.target = dict(target) if target is not None else None

    # ---force_feedback--------------------------------------------------------
    def set_feedback_source(self, follower) -> None:
        """Use a follower arm's measured motor torque for haptic feedback."""
        try:
            states = follower.bus.sync_read_all_states([GRIPPER])
            missing = [m for m in (GRIPPER,) if m not in states or "torque" not in states[m]]
        except Exception as e:  # noqa: BLE001
            logger.warning("[gc] %s: 無法讀取從臂力矩以進行力回饋: %s", self.gm.side, e)
            return
        if missing:
            logger.warning(
                "[gc] %s: 力回饋不可用；從臂缺少 %s 的力矩資料",
                self.gm.side,
                ", ".join(missing),
            )
            return
        with self.lock:
            self.feedback_source = follower if self.ff_scale > 0 else None
        logger.info("[gc] %s: 已綁定從臂夾爪力矩來源", self.gm.side)

    def _force_feedback(self, follower) -> float:
        if follower is None or self.ff_scale <= 0:
            return 0.0

        states = follower.bus._last_known_states
        state = states.get(GRIPPER)
        if state is None or "torque" not in state:
            return 0.0

        measured = float(state["torque"])
        if not np.isfinite(measured):
            return 0.0

        if abs(measured) < self.ff_deadband:
            measured = 0.0
        self.external_max = max(self.external_max, abs(measured))
        feedback = self.ff_sign * self.ff_scale * measured
        return float(np.clip(feedback, -self.ff_limit, self.ff_limit))
    # -------------------------------------------------------------------------

    def positions(self) -> dict[str, float]:
        with self.lock:
            return {m: s["position"] for m, s in self.latest.items()}

    def states(self) -> dict[str, dict]:
        with self.lock:
            return {m: dict(s) for m, s in self.latest.items()}

    def move_to(self, goal: dict[str, float], duration: float):
        """平滑移動到 goal (LeRobot 角度, key = joint_1..gripper)，結束後保持 target (呼叫者決定何時放開)"""
        start = self.positions()
        n = max(int(duration * 50), 1)
        for k in range(n + 1):
            a = min_jerk(k / n)
            self.set_target({m: start[m] + a * (goal[m] - start[m]) for m in goal if m in start})
            time.sleep(duration / n)

    # --- loop ---------------------------------------------------------------
    def _run(self):
        period = 1.0 / self.hz
        next_t = time.perf_counter()
        # 統計 (診斷延遲/卡頓用)
        st_t0 = time.perf_counter()
        st_n, st_over = 0, 0
        st_bus_sum, st_bus_max, st_cyc_max = 0.0, 0.0, 0.0
        try:
            while not self._stop.is_set():
                t_cyc = time.perf_counter()
                with self.lock:
                    target = self.target
                    feedback_source = self.feedback_source
                cache = self.bus._last_known_states
                q = np.array([cache[m]["position"] for m in ARM_JOINTS])
                tau = self.scale * np.array(JOINT_GC_SCALE) * self.gm.torque(q)
                if self.fric_scale > 0 and target is None:
                    v = np.radians([cache[m]["velocity"] for m in ARM_JOINTS])
                    tau = tau + self.fric_scale * np.array(FRICTION_COMP) * np.tanh(v / self.fric_eps)
                # ---force_feedback--------------------------------------------------------
                if target is None:
                    feedback_tau = self._force_feedback(feedback_source)
                    feedback_tau = (1.0 - self.ff_alpha) * self.last_feedback_tau + self.ff_alpha * feedback_tau
                else:
                    feedback_tau = 0.0
                self.last_feedback_tau = feedback_tau
                if feedback_tau != 0.0:
                    self.feedback_samples += 1
                    self.feedback_max = max(self.feedback_max, abs(feedback_tau))
                # -------------------------------------------------------------------------
                tau = np.clip(tau, -np.array(TAU_LIMIT), TAU_LIMIT)
                self.last_tau = tau

                cmds = {}
                for i, m in enumerate(ARM_JOINTS):
                    if target is not None and m in target:
                        cmds[m] = (HOLD_KP[i], HOLD_KD[i], float(target[m]), 0.0, float(tau[i]))
                    else:
                        cmds[m] = (0.0, GC_KD[i] * self.kd_scale, float(q[i]), 0.0, float(tau[i]))
                if GRIPPER in self.bus.motors:
                    if target is not None and GRIPPER in target:
                        cmds[GRIPPER] = (GRIPPER_HOLD_KP, GRIPPER_HOLD_KD, float(target[GRIPPER]), 0.0, 0.0)
                    else:
                        g = cache[GRIPPER]["position"]
                        cmds[GRIPPER] = (
                            0.0,
                            GRIPPER_KD * self.kd_scale,
                            float(g),
                            0.0,
                            float(np.clip(feedback_tau, -TAU_LIMIT[-1], TAU_LIMIT[-1])),
                        )

                t_bus = time.perf_counter()
                self.bus._mit_control_batch(cmds)  # 回傳封包會更新 _last_known_states
                dt_bus = time.perf_counter() - t_bus

                with self.lock:
                    self.latest = {m: dict(s) for m, s in self.bus._last_known_states.items()}

                next_t += period
                sleep = next_t - time.perf_counter()
                overran = sleep <= 0
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    next_t = time.perf_counter()

                if self.stats_on:
                    st_n += 1
                    st_over += overran
                    st_bus_sum += dt_bus
                    st_bus_max = max(st_bus_max, dt_bus)
                    st_cyc_max = max(st_cyc_max, t_bus - t_cyc + dt_bus)
                    now = time.perf_counter()
                    if now - st_t0 >= STATS_PERIOD:
                        extra_ff = ""
                        if self.feedback_samples > 0:
                            extra_ff = f" | 力回饋 原始 {self.external_max:5.2f} Nm -> 指令 {self.feedback_max:5.2f} Nm, 取樣數 {self.feedback_samples}"
                        logger.info(
                            f"[gc-stats] {self.gm.side:5s} 迴圈 {st_n / (now - st_t0):5.0f} Hz (目標 {self.hz:.0f})"
                            f" | CAN 往返 平均 {1e3 * st_bus_sum / st_n:5.2f} ms, 最大 {1e3 * st_bus_max:5.2f} ms"
                            f" | 單次運算+通訊 最大 {1e3 * st_cyc_max:5.2f} ms | 超時 {st_over}/{st_n}"
                            f"{extra_ff}"
                        )
                        st_t0, st_n, st_over = now, 0, 0
                        st_bus_sum, st_bus_max, st_cyc_max = 0.0, 0.0, 0.0
        except Exception as e:  # noqa: BLE001
            self.error = e
            logger.exception(f"[gc] 控制執行緒崩潰 — 正在關閉 {self.bus.port} 的力矩")
            try:
                self.bus.disable_torque()
            except Exception:  # noqa: BLE001
                pass

