from __future__ import annotations

import inspect
import logging
import os
import sys
import time

from openarm_src.utils.config import (
    ARM_JOINTS,
    DEFAULT_ARDUINO_PORT,
    DEFAULT_FF_SIGN,
    DEFAULT_FORCE_FEEDBACK_LIMIT,
    DEFAULT_FORCE_FEEDBACK_SCALE,
    DEFAULT_FORCE_MONITOR_ENABLE,
    DEFAULT_HOME_AT,
    DEFAULT_HOME_PATH,
    DEFAULT_LEADER_SIDE,
    DEFAULT_LEADER_TRACK_ENABLE,
    DEFAULT_PARK_ENABLE,
    DEFAULT_PARK_TIME_S,
    DEFAULT_PROFILE_ENABLE,
    DEFAULT_URDF_PATH,
    DEFAULT_WEBRTC_ENABLE,
    DEFAULT_WEBRTC_PORT,
)
from openarm_src.utils.homing import (
    get_leaders,
    go_home,
    load_home_config,
    load_waypoints,
    min_jerk,
)
from openarm_src.utils.leader_controller import LeaderGC
from openarm_src.utils.profiler import LoopProfiler
from openarm_src.utils.safety import bus_connected, register_arm, safe_disable, unregister_arm

logger = logging.getLogger("openarm_gc.patches")


def _call_arguments(function, args, kwargs):
    signature = inspect.signature(inspect.unwrap(function))
    bound = signature.bind_partial(*args, **kwargs).arguments
    if "kwargs" in bound:
        return dict(bound["kwargs"])
    return bound


def bind_force_feedback(robot, teleop) -> None:
    """連接 follower 與 leader：leader._gc.set_feedback_source(follower)"""
    ff_scale = DEFAULT_FORCE_FEEDBACK_SCALE
    if ff_scale <= 0:
        return

    bindings = []
    if all(hasattr(teleop, attr) for attr in ("left_arm", "right_arm")) and all(
        hasattr(robot, attr) for attr in ("left_arm", "right_arm")
    ):
        bindings.extend(
            (
                ("left", teleop.left_arm, robot.left_arm),
                ("right", teleop.right_arm, robot.right_arm),
            )
        )
    elif getattr(teleop, "_gc_side", None):
        bindings.append((teleop._gc_side, teleop, robot))

    if not bindings:
        logger.warning("[gc] 已啟用力回饋，但未找到對應的主臂/從臂配對")
        return
    for side, leader, follower in bindings:
        gc = getattr(leader, "_gc", None)
        if gc is None:
            logger.warning("[gc] %s 略過力回饋：主臂控制器未處於啟用狀態", side)
            continue
        gc.set_feedback_source(follower)
    logger.info(
        "[gc] 力回饋已開啟 (比例=%.3g, 正負號=%+.0f, 上限=%.3g Nm)",
        ff_scale,
        DEFAULT_FF_SIGN,
        DEFAULT_FORCE_FEEDBACK_LIMIT,
    )

DAMIAO_ERROR_CODES: dict[int, str] = {
    0x9: "欠壓保護 (Under Voltage - 電源接觸不良或電壓過低)",
    0xA: "過流保護 (Over Current - 機械卡死/堵轉或電流過大)",
    0xB: "MOS過溫保護 (MOS Over Temp - 驅動板過熱)",
    0xC: "線圈過溫保護 (Coil Over Temp - 馬達線圈過熱)",
    0xD: "通訊丟失保護 (Communication Lost - CAN 線拉扯瞬斷或超時)",
    0xE: "過載保護 (Overload - 長時間超負荷或堵轉)",
}

_MONITOR_PATCHED = False


def patch_gripper_monitor():
    """監控夾爪馬達的硬體狀態碼、CAN 通訊中斷、行程限制裁切與手腕姿態，精確診斷特定角度斷線閃紅燈。"""
    global _MONITOR_PATCHED
    if _MONITOR_PATCHED:
        return
    _MONITOR_PATCHED = True

    from lerobot.motors.damiao.damiao import DamiaoMotorsBus
    from lerobot.robots.openarm_follower import OpenArmFollower

    orig_process_resp = DamiaoMotorsBus._process_response
    orig_mit_batch = DamiaoMotorsBus._mit_control_batch
    orig_send_action = OpenArmFollower.send_action

    _last_warn_time: dict[tuple[str, str], float] = {}
    _missing_tracker: dict[tuple[str, str], int] = {}

    def wrapped_process_response(self, motor: str, msg):
        if hasattr(msg, "data") and len(msg.data) >= 8:
            err_code = msg.data[0] & 0x0F
            if not hasattr(self, "_last_err_codes"):
                self._last_err_codes = {}
            self._last_err_codes[motor] = err_code

            if not hasattr(self, "_last_recv_time"):
                self._last_recv_time = {}
            self._last_recv_time[motor] = time.perf_counter()

            if err_code in DAMIAO_ERROR_CODES:
                port = getattr(self, "port", "can")
                key = (port, f"{motor}_err_{err_code}")
                now = time.perf_counter()
                if now - _last_warn_time.get(key, 0.0) >= 1.0:
                    _last_warn_time[key] = now
                    states = getattr(self, "_last_known_states", {})
                    j5 = states.get("joint_5", {}).get("position", 0.0)
                    j6 = states.get("joint_6", {}).get("position", 0.0)
                    j7 = states.get("joint_7", {}).get("position", 0.0)
                    g_pos = states.get(motor, {}).get("position", 0.0)
                    t_mos = msg.data[6]
                    t_rot = msg.data[7]
                    desc = DAMIAO_ERROR_CODES[err_code]
                    logger.error(
                        f"[馬達硬體故障] {port}/{motor} 達妙馬達報錯閃紅燈！"
                        f"狀態碼 0x{err_code:X}: {desc} | "
                        f"手腕當前姿態: J5={j5:.1f}°, J6={j6:.1f}°, J7={j7:.1f}°, Gripper={g_pos:.1f}° | "
                        f"MOS溫={t_mos}°C, 線圈溫={t_rot}°C"
                    )

        return orig_process_resp(self, motor, msg)

    def wrapped_mit_control_batch(self, commands):
        if not commands or self.canbus is None:
            return orig_mit_batch(self, commands)

        orig_mit_batch(self, commands)

        now = time.perf_counter()
        port = getattr(self, "port", "can")
        last_times = getattr(self, "_last_recv_time", {})

        for motor in commands:
            motor_name = self._get_motor_name(motor)
            if motor_name != "gripper":
                continue

            last_t = last_times.get(motor_name)
            key = (port, motor_name)
            if last_t is None or (now - last_t > 0.15):
                count = _missing_tracker.get(key, 0) + 1
                _missing_tracker[key] = count
                if count >= 3:
                    warn_t = _last_warn_time.get(key, 0.0)
                    if now - warn_t >= 1.0:
                        _last_warn_time[key] = now
                        states = getattr(self, "_last_known_states", {})
                        j5 = states.get("joint_5", {}).get("position", 0.0)
                        j6 = states.get("joint_6", {}).get("position", 0.0)
                        j7 = states.get("joint_7", {}).get("position", 0.0)
                        g_pos = states.get("gripper", {}).get("position", 0.0)
                        dt_ms = (now - last_t) * 1000 if last_t else 0.0
                        logger.warning(
                            f"[夾爪通訊中斷] {port}/{motor_name} 遺失 CAN 回應 (已持續 {dt_ms:.0f}ms)！馬達可能斷電或斷線！"
                            f"手腕當前姿態: J5={j5:.1f}°, J6={j6:.1f}°, J7={j7:.1f}°, 夾爪={g_pos:.1f}° | "
                            f"請特別檢查手腕在此角度下，末端排線是否被拉扯或接頭鬆脫！"
                        )
            else:
                if _missing_tracker.get(key, 0) >= 3:
                    logger.info(f"[夾爪通訊恢復] {port}/{motor_name} CAN 通訊已恢復正常。")
                _missing_tracker[key] = 0

    def wrapped_send_action(self, action, *args, **kwargs):
        if "gripper.pos" in action and "gripper" in self.config.joint_limits:
            target_g = float(action["gripper.pos"])
            min_lim, max_lim = self.config.joint_limits["gripper"]
            if target_g < min_lim or target_g > max_lim:
                port = getattr(self, "port", "follower")
                key = (port, "gripper_clip")
                now = time.perf_counter()
                if now - _last_warn_time.get(key, 0.0) >= 1.0:
                    _last_warn_time[key] = now
                    clipped = max(min_lim, min(max_lim, target_g))
                    states = getattr(self.bus, "_last_known_states", {})
                    j5 = states.get("joint_5", {}).get("position", 0.0)
                    j6 = states.get("joint_6", {}).get("position", 0.0)
                    j7 = states.get("joint_7", {}).get("position", 0.0)
                    logger.warning(
                        f"[夾爪限位告警] {self}: 目標角度 {target_g:.1f}° 超出軟體行程限制 [{min_lim:.1f}°, {max_lim:.1f}°]，已被限制為 {clipped:.1f}°！"
                        f"手腕姿態: J5={j5:.1f}°, J6={j6:.1f}°, J7={j7:.1f}° | "
                        f"若機械已到物理死點仍持續施壓，馬達會因堵轉過流 (0xA) 觸發閃紅燈斷線保護！"
                    )

        return orig_send_action(self, action, *args, **kwargs)

    DamiaoMotorsBus._process_response = wrapped_process_response
    DamiaoMotorsBus._mit_control_batch = wrapped_mit_control_batch
    OpenArmFollower.send_action = wrapped_send_action
    logger.info("[gc] 夾爪與手腕硬體診斷 Logger 已啟用 (即時監控排線拉扯、CAN丟包、達妙錯誤碼與限位堵轉)")

_FOLLOWER_PATCHED = False


class _SkipPark(Exception):
    pass


def _park_path(current: dict[str, float]) -> list[dict[str, float]]:
    """回下垂(零點)的路徑：若有 reset.json，先反向經過中繼點(抬手姿態)，再回零點，避免掃到桌子。"""
    zero = {k: 0.0 for k in current}
    path = []
    home_path = DEFAULT_HOME_PATH
    if home_path and os.path.exists(home_path):
        wps = load_waypoints(home_path)
        for wp in reversed(wps[:-1]):  # 中繼點反向 (不含最後的起始姿態)
            path.append({k: wp[k] for k in current if k in wp})
    path.append(zero)
    return path


def follower_park(arm, prefix: str = "", duration_per_seg: float | None = None):
    """單支 OpenArmFollower 慢慢回到下垂姿態 (MIT 位置控制)，之後才讓 LeRobot 關力矩。"""
    T = duration_per_seg or DEFAULT_PARK_TIME_S
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


def bimanual_disconnect(self):
    """雙手設備 (BiOpenArmLeader / BiOpenArmFollower) 同步回位至下垂姿態後，安全中斷連線。"""
    park_on = DEFAULT_PARK_ENABLE
    arms = [("left_", self.left_arm), ("right_", self.right_arm)]
    is_leader = hasattr(self.left_arm, "_gc") or not hasattr(self, "send_action")
    label = "主臂" if is_leader else "從臂"

    try:
        if not park_on:
            raise _SkipPark()

        logger.info(f"[gc] 正在將雙{label}回位至下垂姿態...")
        T = DEFAULT_PARK_TIME_S
        cur = {}

        if is_leader:
            for pfx, arm in arms:
                gc = getattr(arm, "_gc", None)
                if gc is not None and gc.error is None:
                    pos = gc.positions()
                    for m in ARM_JOINTS:
                        if m in pos:
                            cur[f"{pfx}{m}.pos"] = float(pos[m])
        else:
            for pfx, arm in arms:
                for m, v in arm.bus.sync_read("Present_Position").items():
                    cur[f"{pfx}{m}.pos"] = float(v)

        if not cur:
            raise _SkipPark()

        for seg in _park_path(cur):
            goal = {k: seg.get(k, cur[k]) for k in cur}
            n = max(int(T * 50), 1)
            for k in range(1, n + 1):
                a = min_jerk(k / n)
                step_pose = {key: cur[key] + a * (goal[key] - cur[key]) for key in cur}
                if is_leader:
                    for pfx, arm in arms:
                        gc = getattr(arm, "_gc", None)
                        if gc is not None and gc.error is None:
                            gc.set_target({
                                key[len(pfx):].removesuffix(".pos"): v
                                for key, v in step_pose.items()
                                if key.startswith(pfx)
                            })
                else:
                    self.send_action(step_pose)
                time.sleep(T / n)
            cur = goal

        time.sleep(0.3)
        for _, arm in arms:
            arm._parked = True
    except _SkipPark:
        pass
    except Exception as e:  # noqa: BLE001
        for _, arm in arms:
            arm._parked = True
        logger.warning(f"[gc] 回位失敗 ({e}) — 直接關閉力矩")

    # 左右各自關：一支出錯不會讓另一支被跳過
    for _, arm in arms:
        if not bus_connected(arm):
            continue
        try:
            arm.disconnect()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[gc] {arm}: 中斷連線出錯 ({e})")
            safe_disable(arm)

def patch_follower_park():
    patch_gripper_monitor()
    global _FOLLOWER_PATCHED
    if _FOLLOWER_PATCHED:
        return
    _FOLLOWER_PATCHED = True
    park_on = DEFAULT_PARK_ENABLE
    from lerobot.robots.bi_openarm_follower import BiOpenArmFollower
    from lerobot.robots.openarm_follower import OpenArmFollower

    orig_disc = OpenArmFollower.disconnect
    orig_conn = OpenArmFollower.connect

    def connect(self, *a, **k):
        orig_conn(self, *a, **k)
        register_arm(self)
        self._parked = False

    def disconnect(self):
        if (
            park_on
            and self.bus.is_connected
            and self.config.disable_torque_on_disconnect
            and not getattr(self, "_parked", False)
        ):
            try:
                logger.info(f"[gc] {self}: 關閉力矩前先回到下垂姿態")
                follower_park(self, getattr(self, "_park_prefix", ""))
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[gc] 回位失敗 ({e}) — 直接關閉力矩")
        if self.config.disable_torque_on_disconnect:
            safe_disable(self)
        try:
            return orig_disc(self)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[gc] {self}: 中斷連線出錯 ({e})，再補送關閉力矩指令")
            safe_disable(self)
        finally:
            unregister_arm(self)

    OpenArmFollower.connect = connect
    OpenArmFollower.disconnect = disconnect
    BiOpenArmFollower.disconnect = bimanual_disconnect


def install_patches():
    patch_gripper_monitor()
    from lerobot.teleoperators.bi_openarm_leader import BiOpenArmLeader
    from lerobot.teleoperators.openarm_leader import OpenArmLeader

    if not DEFAULT_URDF_PATH:
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
        register_arm(self)
        self._parked = False
        side = getattr(self, "_gc_side", None) or DEFAULT_LEADER_SIDE
        self._gc = LeaderGC(self, side)
        self._gc.start()
        logger.info(f"[gc] {self} 重力補償已開啟 (側別={side})")

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
        if gc is not None:
            park_on = DEFAULT_PARK_ENABLE
            if gc.error is None and park_on and not getattr(self, "_parked", False):
                # 先慢慢垂回零點 (重力中性姿態)，再關力矩，避免手臂直接摔下
                try:
                    logger.info(f"[gc] {self}: 關閉力矩前先回到下垂姿態")
                    T = DEFAULT_PARK_TIME_S
                    gc.move_to({m: 0.0 for m in ARM_JOINTS}, T)
                    time.sleep(0.3)
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"[gc] {self}: 回下垂失敗 ({e})")
            gc.stop()  # 一定要先停掉 200 Hz 執行緒，才不會跟 disable 搶 bus
        safe_disable(self)  # 多送幾次 disable
        try:
            orig_disconnect(self)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[gc] {self}: 中斷連線出錯 ({e})，再補送關閉力矩指令")
            safe_disable(self)
        unregister_arm(self)

    def send_feedback(self, feedback):
        gc = getattr(self, "_gc", None)
        if gc is None:
            raise RuntimeError("gravity compensation is not active")

        goals = {}
        for key, value in feedback.items():
            if not key.endswith(".pos"):
                raise ValueError(f"OpenArm feedback only supports .pos keys, got {key!r}")
            motor = key.removesuffix(".pos")
            if motor not in self.bus.motors:
                raise ValueError(f"Unknown OpenArm motor in feedback: {motor!r}")
            goals[motor] = float(value)

        if goals:
            gc.set_target(goals)

    BiOpenArmLeader.__init__ = bi_init
    BiOpenArmLeader.disconnect = bimanual_disconnect
    patch_follower_park()
    OpenArmLeader.connect = connect
    OpenArmLeader.get_action = get_action
    OpenArmLeader.send_feedback = send_feedback
    OpenArmLeader.disconnect = disconnect


def patch_record_loop(module):
    waypoints, duration, guard = load_home_config()
    when = DEFAULT_HOME_AT  # "record" 或 "reset"
    profiler = LoopProfiler() if DEFAULT_PROFILE_ENABLE else None

    streamer = None
    if DEFAULT_WEBRTC_ENABLE:
        from openarm_src.utils.webrtc_streamer import get_webrtc_streamer
        port = DEFAULT_WEBRTC_PORT
        streamer = get_webrtc_streamer(port=port)
        streamer.start()

    orig = module.record_loop

    def record_loop(*args, **kwargs):
        bound = _call_arguments(orig, args, kwargs)
        robot = bound.get("robot")
        teleop = bound.get("teleop")
        if profiler is not None:
            profiler.attach(robot, teleop)
        if streamer is not None:
            streamer.attach(robot, teleop)
            is_recording = bound.get("dataset") is not None
            streamer.set_recording(is_recording)
        if robot is not None and teleop is not None:
            bind_force_feedback(robot, teleop)
        is_recording = bound.get("dataset") is not None
        if waypoints and teleop is not None and (
            (when == "record" and is_recording) or (when == "reset" and not is_recording)
        ):
            go_home(robot, teleop, waypoints, duration, guard)
        return orig(*args, **kwargs)

    module.record_loop = record_loop


def patch_teleop_loop(module):
    """Bind follower state to leader force feedback after both devices connect."""
    streamer = None
    if DEFAULT_WEBRTC_ENABLE:
        from openarm_src.utils.webrtc_streamer import get_webrtc_streamer
        port = DEFAULT_WEBRTC_PORT
        streamer = get_webrtc_streamer(port=port)
        streamer.start()

    arduino_reader = None
    if DEFAULT_FORCE_MONITOR_ENABLE:
        arduino_port = DEFAULT_ARDUINO_PORT
        if arduino_port:
            try:
                from openarm_src.test.test_gripper_force import ArduinoReader
                arduino_reader = ArduinoReader(port=arduino_port)
                arduino_reader.start()
                logger.info("[teleop] 已連線 Arduino 壓力感測器 (%s)", arduino_port)
            except Exception as e:
                logger.warning("[teleop] Arduino 連線失敗: %s", e)

    orig = module.teleop_loop

    def teleop_loop(*args, **kwargs):
        bound = _call_arguments(orig, args, kwargs)
        robot = bound.get("robot")
        teleop = bound.get("teleop")
        if streamer is not None:
            streamer.attach(robot, teleop)
            streamer.set_recording(False)
        if robot is not None and teleop is not None:
            bind_force_feedback(robot, teleop)

        # 掛載實時力矩與 Arduino 壓力印出
        if DEFAULT_FORCE_MONITOR_ENABLE and robot is not None:
            orig_send_action = robot.send_action
            last_print_t = [0.0]

            def send_action_with_force_monitor(action, *s_args, **s_kwargs):
                res = orig_send_action(action, *s_args, **s_kwargs)
                now = time.perf_counter()
                if now - last_print_t[0] >= 0.05:  # 20Hz 刷新頻率
                    last_print_t[0] = now
                    try:
                        parts = []
                        # 取得手臂各夾爪狀態
                        for pfx, arm in [("左", getattr(robot, "left_arm", None)), ("右", getattr(robot, "right_arm", None))]:
                            if arm is None and hasattr(robot, "bus"):
                                arm = robot
                                pfx = ""
                            if arm is not None and hasattr(arm, "bus"):
                                g_state = arm.bus._last_known_states.get("gripper", {})
                                t_val = g_state.get("torque", 0.0)
                                p_val = g_state.get("position", 0.0)
                                parts.append(f"{pfx}爪: pos={p_val:5.1f}°, tau={t_val:6.3f}Nm")
                            if pfx == "":
                                break

                        # 取得 Arduino 感測器電壓/力量
                        if arduino_reader is not None:
                            raw, volt, _ = arduino_reader.get_latest()
                            parts.append(f"SY40電壓={volt:5.2f}V (RAW={raw:4.0f})")

                        if parts:
                            print("\r[Force] " + " | ".join(parts) + "   ", end="", flush=True)
                    except Exception:
                        pass
                return res

            robot.send_action = send_action_with_force_monitor

        try:
            return orig(*args, **kwargs)
        finally:
            if arduino_reader is not None:
                arduino_reader.stop()

    module.teleop_loop = teleop_loop


def patch_rollout(module):
    """lerobot-rollout：
    1. 硬體連線完成 (build_rollout_context) 後、policy 開始動之前，leader + follower 一起移到 OPENARM_HOME 起始姿態。
    2. DAgger 推論時 (AUTONOMOUS)，讓 leader 即時鏡像跟隨 follower 移動。
    3. 人類介入時 (CORRECTING)，leader 立即釋放 target，回歸零力重力補償供手動示範。
    4. DAgger Step Jump 防階躍跳變保護：
       - 攔截 robot_wrapper.send_action 透過 StepJumpGuard 對關節角度進行單步最大變動量限制 (Rate Limiter)。
       - 在 DAgger 狀態切換（特別是 PAUSED -> AUTONOMOUS 人類交接回 Policy、PAUSED -> CORRECTING 人類接管）時，
         自動以當前實際關節角度同步基準，徹底杜絕 Policy 首拍或接管時的劇烈甩動/爆衝。
    """
    waypoints, duration, guard = load_home_config()
    orig = module.build_rollout_context
    track_leader = DEFAULT_LEADER_TRACK_ENABLE

    def build_rollout_context(*args, **kwargs):
        ctx = orig(*args, **kwargs)
        teleop = ctx.hardware.teleop
        robot_wrapper = ctx.hardware.robot_wrapper

        if waypoints and teleop is not None:
            # 用原始 robot (不經 ThreadSafeRobot)；此時 inference engine 還沒啟動，不會搶 bus
            go_home(robot_wrapper.inner, teleop, waypoints, duration, guard)

        # 建立 Step Jump 防跳躍保護器
        from openarm_src.utils.step_guard import StepJumpGuard

        step_guard = StepJumpGuard()
        try:
            step_guard.sync_from_observation(robot_wrapper.get_observation())
        except Exception as e:
            logger.warning(f"[step_guard] 初始姿態同步警告: {e}")

        leaders = get_leaders(teleop) if (track_leader and teleop is not None) else []
        _is_tracking = bool(leaders)

        # 1. 攔截 robot_wrapper.send_action：加入 Step Jump Guard 速率限制，並同步跟隨 leader
        orig_send_action = robot_wrapper.send_action

        def send_action_guarded(action):
            # 經過步長保護器抑制瞬變
            safe_action = step_guard.clamp(action)
            res = orig_send_action(safe_action)

            nonlocal _is_tracking
            if _is_tracking and isinstance(safe_action, dict) and leaders:
                for prefix, leader in leaders:
                    leader_cmd = {
                        key[len(prefix):].removesuffix(".pos"): float(v)
                        for key, v in safe_action.items()
                        if key.startswith(prefix)
                    }
                    if leader_cmd:
                        leader._gc.set_target(leader_cmd)
            return res

        robot_wrapper.send_action = send_action_guarded
        logger.info(
            f"[step_guard] 已啟用 DAgger 步長防跳保護 (max_step_deg={step_guard.max_step_deg}°, "
            f"max_step_gripper={step_guard.max_step_gripper}°)"
        )

        # 2. Hook DAgger 狀態機切換：同步當前實際姿態基準，避免交接瞬間跳變
        from lerobot.rollout.strategies.dagger import DAggerPhase, DAggerStrategy

        orig_apply_transition = DAggerStrategy._apply_transition

        def apply_transition_with_guard(self, old_phase, new_phase, *t_args, **t_kwargs):
            nonlocal _is_tracking
            if new_phase == DAggerPhase.CORRECTING:
                _is_tracking = False
                logger.info("[gc] 人類介入 (CORRECTING)：釋放 leader 目標，進入純重力補償手動模式")
                for _, leader in leaders:
                    leader._gc.set_target(None)
                # 同步基準為人類介入瞬間的真實角度
                try:
                    step_guard.sync_from_observation(robot_wrapper.get_observation())
                except Exception as e:
                    logger.warning(f"[step_guard] CORRECTING 姿態同步警告: {e}")

            elif new_phase == DAggerPhase.AUTONOMOUS:
                _is_tracking = bool(leaders)
                logger.info("[gc] 恢復自主 (AUTONOMOUS)：啟用 leader 跟隨 follower 並同步防跳躍基準")
                # 同步基準為人類結束示範時的真實角度，確保 Policy 啟動時首拍動作平滑過渡
                try:
                    step_guard.sync_from_observation(robot_wrapper.get_observation())
                except Exception as e:
                    logger.warning(f"[step_guard] AUTONOMOUS 姿態同步警告: {e}")

            elif new_phase == DAggerPhase.PAUSED:
                _is_tracking = False
                try:
                    step_guard.sync_from_observation(robot_wrapper.get_observation())
                except Exception as e:
                    logger.warning(f"[step_guard] PAUSED 姿態同步警告: {e}")

            return orig_apply_transition(self, old_phase, new_phase, *t_args, **t_kwargs)

        DAggerStrategy._apply_transition = apply_transition_with_guard
        if leaders:
            logger.info("[gc] 已啟用 DAgger Leader 鏡像跟隨 (OPENARM_LEADER_TRACK=1)")

        return ctx

    module.build_rollout_context = build_rollout_context
