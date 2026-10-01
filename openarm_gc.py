#!/usr/bin/env python
"""
OpenArm (v1) leader 重力補償 + 起始姿態 (home) — LeRobot 外掛包裝
Gravity compensation for OpenArm leader arms + home pose before every episode.

不修改 LeRobot 原始碼：本腳本在執行時 monkey-patch
  - OpenArmLeader.connect / get_action / disconnect  -> 背景執行緒 (200 Hz) 送 MIT 指令
        kp=0, kd=小阻尼, tau_ff = Pinocchio 算出的重力力矩
  - lerobot_record.record_loop -> 每個「錄製」episode 開始前，leader + follower
        一起用 min-jerk 軌跡平滑移到 home.json，然後 leader 放開回到重力補償模式

用法 / Usage
------------
  # 0) 產生 URDF (在有 ROS 2 + openarm_description 的機器上)，並設定環境變數
  export OPENARM_URDF=~/openarm_bimanual.urdf

  # 1) 檢查重力補償方向 + 擺出起始姿態並存檔
  python openarm_gc.py check --id my_bimanual_leader --left-port can2 --right-port can3 --save home.json

  # 2) 錄資料 (參數與 lerobot-record 完全相同，只是把 lerobot-record 換成 python openarm_gc.py record)
  export OPENARM_HOME=home.json
  python openarm_gc.py record --robot.type=bi_openarm_follower ... --teleop.type=bi_openarm_leader ...

  # 2b) 只讓 follower 移到起始姿態 (不需要 leader)，按 Enter 後慢慢回下垂
  python openarm_gc.py home --id bi_follower --left-port can1 --right-port can0 --home reset.json

  # 3) 只做遙操作 (無 home)
  python openarm_gc.py teleoperate <lerobot-teleoperate 參數>

環境變數 / Env vars
  OPENARM_URDF        (必填) bimanual URDF 路徑，關節名 openarm_{left,right}_joint1..7
  OPENARM_HOME        home.json 路徑；不設則不做 homing
  OPENARM_HOME_TIME   移到 home 的時間 (秒)，預設 4
  OPENARM_TABLE_Z     (選填) 桌面高度 (URDF 世界座標, m)，設了會在 homing 前檢查碰桌
  OPENARM_TABLE_MARGIN 碰桌檢查的安全餘裕 (m)，預設 0.05
  OPENARM_PARK        1 (預設) = follower 結束時先慢慢回下垂再關力矩；0 = 原本行為 (直接關力矩掉下來)
  OPENARM_PARK_TIME   回下垂每段的秒數，預設 3
  OPENARM_HOME_AT     record (預設，開錄前回起始姿態) / reset (episode 結束就回)
  OPENARM_KD_SCALE    阻尼倍率，預設 1.0 (覺得黏就調小，0 = 無阻尼)
  OPENARM_FRICTION_SCALE 摩擦補償倍率，預設 0 (關閉)；建議從 0.5 開始試
  OPENARM_FRICTION_EPS 摩擦補償的速度平滑區 (rad/s)，預設 0.15；靜止時會抖就調大
  OPENARM_CONTROL_HZ  重力補償迴圈頻率，預設 200
  OPENARM_GC_SCALE    重力補償整體倍率，預設 1.0 (第一次測試建議 0.5)
  OPENARM_LEADER_SIDE 單臂 openarm_leader 時指定 left / right
  OPENARM_GC_STATS    1 (預設) = 每 5 秒印出重力補償迴圈實際頻率 / CAN 通訊時間；0 = 關閉
  OPENARM_PROFILE     1 (預設) = record 時每 5 秒印出主迴圈各步驟耗時；0 = 關閉
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time

import numpy as np

logger = logging.getLogger("openarm_gc")

# ----------------------------------------------------------------------------
# 可調參數 / Tunables (index 0..6 = joint_1..joint_7)
# ----------------------------------------------------------------------------
ARM_JOINTS = [f"joint_{i}" for i in range(1, 8)]
GRIPPER = "gripper"

# LeRobot 角度 -> URDF 角度 的正負號。v1 URDF 與 LeRobot 的 joint limit 方向一致，預設全 +1。
# 若 check 時某關節「越補越重」，把該關節改成 -1。
JOINT_SIGN = {"left": [1, 1, 1, 1, 1, 1, 1], "right": [1, 1, 1, 1, 1, 1, 1]}

# 每關節重力補償倍率 (leader 末端是握把，質量與 URDF 的夾爪不同，可微調)
JOINT_GC_SCALE = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# 重力補償模式下的阻尼 (kp=0)。太大手感黏，太小容易晃。
GC_KD = [0.4, 0.4, 0.2, 0.2, 0.05, 0.05, 0.05]
GRIPPER_KD = 0.05

# 摩擦補償 (Nm)：順著移動方向「幫推」一點，抵消減速機的靜摩擦/黏滯感。
# tau_f = FRICTION_COMP * tanh(速度 / FRICTION_VEL_EPS)；太大會自己滑走，從小開始調。
FRICTION_COMP = [0.3, 0.3, 0.25, 0.25, 0.05, 0.05, 0.05]
# rad/s，速度低於此值時補償平滑地趨近 0。馬達回報的速度有雜訊，
# 太小 (例如 0.05) 會讓 tanh 變成近似 sign()，靜止/慢速時補償來回跳 -> 手感一格一格的。
FRICTION_VEL_EPS = 0.15

# homing / 抓住姿態時的 PD 增益 (比 follower 的 240 軟很多，碰到人也安全)
HOLD_KP = [60.0, 60.0, 40.0, 40.0, 8.0, 8.0, 8.0]
HOLD_KD = [3.0, 3.0, 1.5, 1.5, 0.3, 0.3, 0.3]
GRIPPER_HOLD_KP, GRIPPER_HOLD_KD = 5.0, 0.2

# 前饋力矩上限 (Nm) — 防止模型/符號錯誤時暴衝
TAU_LIMIT = [18.0, 18.0, 8.0, 8.0, 2.5, 2.5, 2.5]

CONTROL_HZ = 200.0
STATS_PERIOD = 5.0  # 秒


# ----------------------------------------------------------------------------
# Gravity model (Pinocchio)
# ----------------------------------------------------------------------------
class GravityModel:
    def __init__(self, urdf_path: str, side: str | None):
        import pinocchio as pin

        self.pin = pin
        self.model = pin.buildModelFromUrdf(os.path.expanduser(urdf_path))
        self.data = self.model.createData()
        self.q = pin.neutral(self.model)
        self.side = side or "right"

        candidates = []
        if side:
            candidates.append([f"openarm_{side}_joint{i}" for i in range(1, 8)])
        candidates.append([f"openarm_joint{i}" for i in range(1, 8)])
        for names in candidates:
            if all(self.model.existJointName(n) for n in names):
                break
        else:
            raise ValueError(
                f"URDF 裡找不到 {candidates[0]}。現有關節: {list(self.model.names)}"
            )
        jids = [self.model.getJointId(n) for n in names]
        self.q_idx = [self.model.idx_qs[j] for j in jids]
        self.v_idx = [self.model.idx_vs[j] for j in jids]
        self.sign = np.array(JOINT_SIGN.get(self.side, [1] * 7), dtype=float)
        logger.info(f"[gc] {side}: using URDF joints {names}")

    def torque(self, q_deg: np.ndarray) -> np.ndarray:
        """q_deg: LeRobot 角度 (7,) -> 需要送給馬達的重力補償力矩 (Nm, 7)"""
        q_urdf = self.sign * np.radians(q_deg)
        self.q[self.q_idx] = q_urdf
        g = self.pin.computeGeneralizedGravity(self.model, self.data, self.q)
        return self.sign * g[self.v_idx]


def min_jerk(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


# ----------------------------------------------------------------------------
# Background controller attached to one OpenArmLeader
# ----------------------------------------------------------------------------
class LeaderGC:
    def __init__(self, leader, side: str | None):
        self.leader = leader
        self.bus = leader.bus
        self.gm = GravityModel(os.environ["OPENARM_URDF"], side)
        self.scale = float(os.environ.get("OPENARM_GC_SCALE", "1.0"))
        self.kd_scale = float(os.environ.get("OPENARM_KD_SCALE", "1.0"))
        self.fric_scale = float(os.environ.get("OPENARM_FRICTION_SCALE", "0.0"))
        self.fric_eps = float(os.environ.get("OPENARM_FRICTION_EPS", str(FRICTION_VEL_EPS)))
        self.hz = float(os.environ.get("OPENARM_CONTROL_HZ", str(CONTROL_HZ)))
        self.stats_on = os.environ.get("OPENARM_GC_STATS", "1") != "0"
        self.lock = threading.Lock()
        self.target: dict[str, float] | None = None  # None = 純重力補償
        self.latest: dict[str, dict] = {}
        self.last_tau = np.zeros(7)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: Exception | None = None

    # --- public -------------------------------------------------------------
    def start(self):
        states = self.bus.sync_read_all_states()
        with self.lock:
            self.latest = {m: dict(s) for m, s in states.items()}
        self.bus.enable_torque()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"gc-{self.gm.side}")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def set_target(self, target: dict[str, float] | None):
        with self.lock:
            self.target = dict(target) if target is not None else None

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
                cache = self.bus._last_known_states
                q = np.array([cache[m]["position"] for m in ARM_JOINTS])
                tau = self.scale * np.array(JOINT_GC_SCALE) * self.gm.torque(q)
                if self.fric_scale > 0 and target is None:
                    v = np.radians([cache[m]["velocity"] for m in ARM_JOINTS])
                    tau = tau + self.fric_scale * np.array(FRICTION_COMP) * np.tanh(v / self.fric_eps)
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
                        cmds[GRIPPER] = (0.0, GRIPPER_KD * self.kd_scale, float(g), 0.0, 0.0)

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
                        logger.info(
                            f"[gc-stats] {self.gm.side:5s} loop {st_n / (now - st_t0):5.0f} Hz (目標 {self.hz:.0f})"
                            f" | CAN 往返 avg {1e3 * st_bus_sum / st_n:5.2f} ms, max {1e3 * st_bus_max:5.2f} ms"
                            f" | 單次運算+通訊 max {1e3 * st_cyc_max:5.2f} ms | 超時 {st_over}/{st_n}"
                        )
                        st_t0, st_n, st_over = now, 0, 0
                        st_bus_sum, st_bus_max, st_cyc_max = 0.0, 0.0, 0.0
        except Exception as e:  # noqa: BLE001
            self.error = e
            logger.exception(f"[gc] control thread crashed — disabling torque on {self.bus.port}")
            try:
                self.bus.disable_torque()
            except Exception:  # noqa: BLE001
                pass


# ----------------------------------------------------------------------------
# Main-loop profiler (record 主迴圈各步驟耗時)
# ----------------------------------------------------------------------------
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
                    parts.append(f"{lb} avg {1e3 * s / n:6.1f} ms max {1e3 * m:6.1f} ms")
            logger.info("[loop-stats] " + " | ".join(parts))
            self.t0, self.d = now, {}


# ----------------------------------------------------------------------------
# Monkey patches
# ----------------------------------------------------------------------------
def install_patches():
    from lerobot.teleoperators.bi_openarm_leader import BiOpenArmLeader
    from lerobot.teleoperators.openarm_leader import OpenArmLeader

    if "OPENARM_URDF" not in os.environ:
        sys.exit("請先 export OPENARM_URDF=/path/to/openarm_bimanual.urdf")

    orig_bi_init = BiOpenArmLeader.__init__
    orig_connect = OpenArmLeader.connect
    orig_disconnect = OpenArmLeader.disconnect

    def bi_init(self, config):
        orig_bi_init(self, config)
        self.left_arm._gc_side = "left"
        self.right_arm._gc_side = "right"

    def connect(self, calibrate: bool = True):
        # 原本的 connect: 連線 -> (校正) -> disable torque -> set_zero_position
        # 注意：LeRobot 每次 connect 都會把「當下姿態」設為零點，所以啟動時 leader 必須自然下垂！
        orig_connect(self, calibrate)
        side = getattr(self, "_gc_side", None) or os.environ.get("OPENARM_LEADER_SIDE")
        self._gc = LeaderGC(self, side)
        self._gc.start()
        logger.info(f"[gc] {self} gravity compensation ON (side={side})")

    def get_action(self):
        gc = getattr(self, "_gc", None)
        if gc is None:
            raise RuntimeError("leader not connected")
        if gc.error is not None:
            raise RuntimeError(f"gravity-comp thread died: {gc.error}")
        states = gc.states()
        action = {}
        for m in self.bus.motors:
            s = states.get(m, {})
            action[f"{m}.pos"] = s.get("position")
            if self.config.use_velocity_and_torque:
                action[f"{m}.vel"] = s.get("velocity")
                action[f"{m}.torque"] = s.get("torque")
        return action

    def disconnect(self):
        gc = getattr(self, "_gc", None)
        if gc is not None and gc.error is None:
            # 先慢慢垂回零點 (重力中性姿態)，再關力矩，避免手臂直接摔下
            logger.info(f"[gc] {self}: returning to hanging pose before disabling torque")
            zero = {m: 0.0 for m in ARM_JOINTS}
            gc.move_to(zero, 3.0)
            time.sleep(0.3)
            gc.stop()
        orig_disconnect(self)

    BiOpenArmLeader.__init__ = bi_init
    _patch_follower_park()
    OpenArmLeader.connect = connect
    OpenArmLeader.get_action = get_action
    OpenArmLeader.disconnect = disconnect



def _park_path(current: dict[str, float]) -> list[dict[str, float]]:
    """回下垂(零點)的路徑：若有 reset.json，先反向經過中繼點(抬手姿態)，再回零點，避免掃到桌子。"""
    zero = {k: 0.0 for k in current}
    path = []
    home_path = os.environ.get("OPENARM_HOME")
    if home_path and os.path.exists(home_path):
        wps = load_waypoints(home_path)
        for wp in reversed(wps[:-1]):  # 中繼點反向 (不含最後的起始姿態)
            path.append({k: wp[k] for k in current if k in wp})
    path.append(zero)
    return path


def follower_park(arm, prefix: str = "", duration_per_seg: float | None = None):
    """單支 OpenArmFollower 慢慢回到下垂姿態 (MIT 位置控制)，之後才讓 LeRobot 關力矩。"""
    T = duration_per_seg or float(os.environ.get("OPENARM_PARK_TIME", "3"))
    present = arm.bus.sync_read("Present_Position")
    cur = {f"{prefix}{m}.pos": float(v) for m, v in present.items()}
    for seg in _park_path(cur):
        goal = {k: seg.get(k, cur[k]) for k in cur}
        n = max(int(T * 50), 1)
        for k in range(1, n + 1):
            a = min_jerk(k / n)
            cmd = {key[len(prefix):]: cur[key] + a * (goal[key] - cur[key]) for key in cur}
            arm.send_action(cmd)
            time.sleep(T / n)
        cur = goal


def _patch_follower_park():
    if os.environ.get("OPENARM_PARK", "1") == "0":
        return
    from lerobot.robots.bi_openarm_follower import BiOpenArmFollower
    from lerobot.robots.openarm_follower import OpenArmFollower

    orig_disc = OpenArmFollower.disconnect
    orig_bi_disc = BiOpenArmFollower.disconnect

    def disconnect(self):
        if self.bus.is_connected and self.config.disable_torque_on_disconnect and not getattr(self, "_parked", False):
            try:
                logger.info(f"[gc] {self}: parking to hanging pose before disabling torque")
                follower_park(self, getattr(self, "_park_prefix", ""))
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[gc] park failed ({e}) — disabling torque directly")
        return orig_disc(self)

    def bi_disconnect(self):
        # 兩支同時回下垂，而不是左手先回完才輪到右手
        arms = [("left_", self.left_arm), ("right_", self.right_arm)]
        try:
            logger.info("[gc] parking both follower arms to hanging pose ...")
            T = float(os.environ.get("OPENARM_PARK_TIME", "3"))
            cur = {}
            for pfx, arm in arms:
                for m, v in arm.bus.sync_read("Present_Position").items():
                    cur[f"{pfx}{m}.pos"] = float(v)
            for seg in _park_path(cur):
                goal = {k: seg.get(k, cur[k]) for k in cur}
                n = max(int(T * 50), 1)
                for k in range(1, n + 1):
                    a = min_jerk(k / n)
                    self.send_action({key: cur[key] + a * (goal[key] - cur[key]) for key in cur})
                    time.sleep(T / n)
                cur = goal
            for _, arm in arms:
                arm._parked = True
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[gc] park failed ({e}) — disabling torque directly")
        return orig_bi_disc(self)

    OpenArmFollower.disconnect = disconnect
    BiOpenArmFollower.disconnect = bi_disconnect

def _leaders(teleop):
    from lerobot.teleoperators.bi_openarm_leader import BiOpenArmLeader
    from lerobot.teleoperators.openarm_leader import OpenArmLeader

    if isinstance(teleop, BiOpenArmLeader):
        return [("left_", teleop.left_arm), ("right_", teleop.right_arm)]
    if isinstance(teleop, OpenArmLeader):
        return [("", teleop)]
    return []


def load_waypoints(path: str) -> list[dict[str, float]]:
    """reset.json 可以是單一姿態 {key: deg}，或 {"waypoints": [姿態1, 姿態2, ...]}（最後一個 = 起始姿態）"""
    with open(path) as f:
        data = json.load(f)
    wps = data["waypoints"] if isinstance(data, dict) and "waypoints" in data else [data]
    return [{k: float(v) for k, v in wp.items()} for wp in wps]


class TableGuard:
    """用 URDF 正向運動學檢查軌跡上手肘/手腕/夾爪尖端是否低於桌面，低於就不動。
    OPENARM_TABLE_Z = 桌面在 URDF 世界座標的高度 (m) = 0.698 - 肩膀(joint1 軸心)離桌面的高度"""

    FRAMES = ["link4", "link6", "hand_tcp"]

    def __init__(self, urdf: str, table_z: float, margin: float):
        import pinocchio as pin

        self.pin = pin
        self.m = pin.buildModelFromUrdf(os.path.expanduser(urdf))
        self.d = self.m.createData()
        self.limit = table_z + margin

    def lowest(self, pose: dict[str, float]) -> tuple[float, str]:
        q = self.pin.neutral(self.m)
        for key, deg in pose.items():
            side, _, rest = key.partition("_")
            if side not in ("left", "right") or not rest.startswith("joint_"):
                continue
            name = f"openarm_{side}_joint{rest.removeprefix('joint_').removesuffix('.pos')}"
            if self.m.existJointName(name):
                sign = JOINT_SIGN[side][int(name[-1]) - 1]
                q[self.m.idx_qs[self.m.getJointId(name)]] = sign * np.radians(deg)
        self.pin.framesForwardKinematics(self.m, self.d, q)
        best = (1e9, "")
        for side in ("left", "right"):
            for fr in self.FRAMES:
                n = f"openarm_{side}_{fr}"
                if self.m.existFrame(n):
                    z = float(self.d.oMf[self.m.getFrameId(n)].translation[2])
                    best = min(best, (z, n))
        return best

    def check(self, path: list[dict[str, float]]) -> bool:
        for i, pose in enumerate(path):
            z, name = self.lowest(pose)
            if z < self.limit:
                logger.error(f"[gc] 軌跡第 {i} 點 {name} 高度 {z:.3f} m < 桌面+餘裕 {self.limit:.3f} m -> 取消 homing")
                return False
        return True


def go_home(robot, teleop, waypoints: list[dict[str, float]], duration: float, guard: TableGuard | None = None):
    """leader 與 follower 依序經過每個 waypoint (min-jerk)，最後停在起始姿態後放開 leader。"""
    leaders = _leaders(teleop)
    obs = robot.get_observation()
    keys = sorted({k for wp in waypoints for k in wp})
    f_now = {k: float(obs[k]) for k in keys if k in obs}
    l_now = {}
    for prefix, leader in leaders:
        for m, p in leader._gc.positions().items():
            l_now[prefix + m + ".pos"] = p

    # 先把整條 follower 軌跡算出來，檢查碰桌
    seg_T = duration / len(waypoints)
    n = max(int(seg_T * 50), 1)
    plan = []  # (follower_cmd, leader_cmd)
    f_a, l_a = dict(f_now), dict(l_now)
    for wp in waypoints:
        f_b = {**f_a, **{k: v for k, v in wp.items() if k in f_a}}
        l_b = {**l_a, **{k: v for k, v in wp.items() if k in l_a}}
        for k in range(1, n + 1):
            a = min_jerk(k / n)
            plan.append((
                {key: f_a[key] + a * (f_b[key] - f_a[key]) for key in f_a},
                {key: l_a[key] + a * (l_b[key] - l_a[key]) for key in l_a},
            ))
        f_a, l_a = f_b, l_b

    # 碰桌檢查要用完整姿態：waypoint 沒寫到的關節用 follower 目前角度
    f_full = {k: float(v) for k, v in obs.items() if k.endswith(".pos")}
    if guard is not None and not guard.check([{**f_full, **p[0]} for p in plan[::5]] + [{**f_full, **plan[-1][0]}]):
        input("⚠ homing 可能撞桌，已取消。請用 leader 把手臂移到安全位置，再按 Enter 重試...")
        return go_home(robot, teleop, waypoints, duration, guard)

    logger.info(f"[gc] moving to start pose via {len(waypoints)} waypoint(s) ...")
    for f_cmd, l_cmd in plan:
        robot.send_action(f_cmd)
        for prefix, leader in leaders:
            leader._gc.set_target({
                key[len(prefix):].removesuffix(".pos"): v
                for key, v in l_cmd.items() if key.startswith(prefix)
            })
        time.sleep(seg_T / n)
    time.sleep(0.3)
    for _, leader in leaders:
        leader._gc.set_target(None)  # 放開 -> 回到重力補償
    logger.info("[gc] at start pose — released leader, recording starts")


def patch_record_loop(module):
    home_path = os.environ.get("OPENARM_HOME")
    waypoints, duration, guard = None, 0.0, None
    if not home_path:
        logger.warning("[gc] OPENARM_HOME 未設定 -> 不做 homing")
    else:
        waypoints = load_waypoints(home_path)
        duration = float(os.environ.get("OPENARM_HOME_TIME", str(3 * len(waypoints) + 1)))
        if "OPENARM_TABLE_Z" in os.environ:
            guard = TableGuard(os.environ["OPENARM_URDF"], float(os.environ["OPENARM_TABLE_Z"]),
                               float(os.environ.get("OPENARM_TABLE_MARGIN", "0.05")))
    when = os.environ.get("OPENARM_HOME_AT", "record")  # "record" 或 "reset"
    profiler = LoopProfiler() if os.environ.get("OPENARM_PROFILE", "1") != "0" else None
    orig = module.record_loop

    def record_loop(*args, **kwargs):
        if profiler is not None:
            profiler.attach(kwargs.get("robot"), kwargs.get("teleop"))
        is_recording = kwargs.get("dataset") is not None
        if waypoints and kwargs.get("teleop") is not None and (
            (when == "record" and is_recording) or (when == "reset" and not is_recording)
        ):
            go_home(kwargs["robot"], kwargs["teleop"], waypoints, duration, guard)
        return orig(*args, **kwargs)

    module.record_loop = record_loop


# ----------------------------------------------------------------------------
# check / capture-home mode
# ----------------------------------------------------------------------------
def run_check(argv):
    import argparse

    from lerobot.teleoperators.bi_openarm_leader import BiOpenArmLeader
    from lerobot.teleoperators.bi_openarm_leader.config_bi_openarm_leader import BiOpenArmLeaderConfig
    from lerobot.teleoperators.openarm_leader.config_openarm_leader import OpenArmLeaderConfigBase

    p = argparse.ArgumentParser()
    p.add_argument("--id", required=True, help="與 lerobot-record 的 --teleop.id 相同 (讀同一份 calibration)")
    p.add_argument("--left-port", default="can2")
    p.add_argument("--right-port", default="can3")
    p.add_argument("--save", default=None, help="按 Enter 時把目前 leader 姿態存成 json")
    p.add_argument("--waypoints", action="store_true",
                   help="每按一次 Enter 追加一個 waypoint (依序經過，最後一個 = 起始姿態)")
    a = p.parse_args(argv)

    install_patches()
    cfg = BiOpenArmLeaderConfig(
        id=a.id,
        left_arm_config=OpenArmLeaderConfigBase(port=a.left_port),
        right_arm_config=OpenArmLeaderConfigBase(port=a.right_port),
    )
    teleop = BiOpenArmLeader(cfg)
    print("\n⚠  啟動前請讓兩支 leader 自然下垂 (零點姿態)！")
    teleop.connect()
    print("重力補償已開啟。慢慢把手臂抬到水平：應該感覺『變輕、放手會停住』。")
    print("若某關節放手後往上飄 → 該關節 JOINT_SIGN 可能要改 -1，或 JOINT_GC_SCALE 調小。")
    if a.save:
        print(f"擺好你要的起始姿態後按 Enter 存到 {a.save}；Ctrl+C 離開。\n")

    import select

    wps: list[dict] = []

    try:
        while True:
            lines = []
            for side, arm in (("left", teleop.left_arm), ("right", teleop.right_arm)):
                pos = arm._gc.positions()
                tau = arm._gc.last_tau
                lines.append(
                    f"{side:5s} q=" + " ".join(f"{pos[m]:7.1f}" for m in ARM_JOINTS)
                    + f" | g={pos.get(GRIPPER, 0):6.1f} | tau=" + " ".join(f"{t:5.2f}" for t in tau)
                )
            print("\r" + "   ".join(lines), end="", flush=True)
            if a.save and select.select([sys.stdin], [], [], 0)[0]:
                sys.stdin.readline()
                home = {}
                for prefix, arm in (("left_", teleop.left_arm), ("right_", teleop.right_arm)):
                    for m, v in arm._gc.positions().items():
                        home[f"{prefix}{m}.pos"] = round(float(v), 2)
                if a.waypoints:
                    wps.append(home)
                    out = {"waypoints": wps}
                else:
                    out = home
                with open(a.save, "w") as f:
                    json.dump(out, f, indent=2)
                print(f"\n✔ saved pose #{len(wps) if a.waypoints else 1} -> {a.save}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print()
    finally:
        teleop.disconnect()



def run_home(argv):
    """只控制 follower：移到 reset.json 的起始姿態並保持，按 Enter 後慢慢回下垂再關力矩。"""
    import argparse

    from lerobot.robots.bi_openarm_follower import BiOpenArmFollower, BiOpenArmFollowerConfig
    from lerobot.robots.openarm_follower import OpenArmFollowerConfigBase

    p = argparse.ArgumentParser()
    p.add_argument("--id", required=True, help="與 lerobot-record 的 --robot.id 相同")
    p.add_argument("--left-port", default="can1")
    p.add_argument("--right-port", default="can0")
    p.add_argument("--home", default=os.environ.get("OPENARM_HOME"), help="reset.json 路徑")
    p.add_argument("--time", type=float, default=float(os.environ.get("OPENARM_HOME_TIME", "8")))
    a = p.parse_args(argv)
    if not a.home:
        sys.exit("請用 --home reset.json 或 export OPENARM_HOME=reset.json")
    os.environ["OPENARM_HOME"] = a.home

    _patch_follower_park()
    cfg = BiOpenArmFollowerConfig(
        id=a.id,
        left_arm_config=OpenArmFollowerConfigBase(port=a.left_port, side="left"),
        right_arm_config=OpenArmFollowerConfigBase(port=a.right_port, side="right"),
    )
    robot = BiOpenArmFollower(cfg)
    print("\n⚠  follower 將開啟力矩並移動，請淨空桌面、手放在急停旁。")
    robot.connect()
    try:
        go_home(robot, None, load_waypoints(a.home), a.time)
        input("\n✔ follower 已在起始姿態。按 Enter 讓它慢慢回到下垂並關閉力矩 ...")
    except KeyboardInterrupt:
        print()
    finally:
        robot.disconnect()

# ----------------------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2 or sys.argv[1] not in ("record", "teleoperate", "check", "home"):
        print(__doc__)
        sys.exit(1)
    mode, rest = sys.argv[1], sys.argv[2:]

    if mode == "check":
        run_check(rest)
        return
    if mode == "home":
        run_home(rest)
        return

    install_patches()
    sys.argv = [f"lerobot-{mode}"] + rest
    if mode == "record":
        from lerobot.scripts import lerobot_record

        patch_record_loop(lerobot_record)
        lerobot_record.main()
    else:
        from lerobot.scripts import lerobot_teleoperate

        lerobot_teleoperate.main()


if __name__ == "__main__":
    main()