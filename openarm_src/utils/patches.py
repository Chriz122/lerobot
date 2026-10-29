from __future__ import annotations

import inspect
import logging
import os
import sys
import time

from openarm_src.utils.config import (
    ARM_JOINTS,
    DEFAULT_FORCE_FEEDBACK_LIMIT,
    DEFAULT_FORCE_FEEDBACK_SCALE,
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
    ff_scale = float(os.environ.get("OPENARM_FF_SCALE", DEFAULT_FORCE_FEEDBACK_SCALE))
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
        float(os.environ.get("OPENARM_FF_SIGN", "-1.0")),
        abs(float(os.environ.get("OPENARM_FF_LIMIT", DEFAULT_FORCE_FEEDBACK_LIMIT))),
    )

DAMIAO_ERROR_CODES: dict[int, str] = {
    0x8: "過壓保護 (Over Voltage - 電壓異常突增)",
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


def bimanual_disconnect(self):
    """雙手設備 (BiOpenArmLeader / BiOpenArmFollower) 同步回位至下垂姿態後，安全中斷連線。"""
    park_on = os.environ.get("OPENARM_PARK", "1") != "0"
    arms = [("left_", self.left_arm), ("right_", self.right_arm)]
    is_leader = hasattr(self.left_arm, "_gc") or not hasattr(self, "send_action")
    label = "主臂" if is_leader else "從臂"

    try:
        if not park_on:
            raise _SkipPark()

        logger.info(f"[gc] 正在將雙{label}回位至下垂姿態...")
        T = float(os.environ.get("OPENARM_PARK_TIME", "3"))
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
    park_on = os.environ.get("OPENARM_PARK", "1") != "0"
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
        register_arm(self)
        self._parked = False
        side = getattr(self, "_gc_side", None) or os.environ.get("OPENARM_LEADER_SIDE")
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
            park_on = os.environ.get("OPENARM_PARK", "1") != "0"
            if gc.error is None and park_on and not getattr(self, "_parked", False):
                # 先慢慢垂回零點 (重力中性姿態)，再關力矩，避免手臂直接摔下
                try:
                    logger.info(f"[gc] {self}: 關閉力矩前先回到下垂姿態")
                    T = float(os.environ.get("OPENARM_PARK_TIME", "3"))
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
    when = os.environ.get("OPENARM_HOME_AT", "record")  # "record" 或 "reset"
    profiler = LoopProfiler() if os.environ.get("OPENARM_PROFILE", "1") != "0" else None

    streamer = None
    if os.environ.get("OPENARM_WEBRTC", "0") == "1":
        from openarm_src.utils.webrtc_streamer import get_webrtc_streamer
        port = int(os.environ.get("OPENARM_WEBRTC_PORT", "8080"))
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
    if os.environ.get("OPENARM_WEBRTC", "0") == "1":
        from openarm_src.utils.webrtc_streamer import get_webrtc_streamer
        port = int(os.environ.get("OPENARM_WEBRTC_PORT", "8080"))
        streamer = get_webrtc_streamer(port=port)
        streamer.start()

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
        return orig(*args, **kwargs)

    module.teleop_loop = teleop_loop


def patch_rollout(module):
    """lerobot-rollout：
    1. 硬體連線完成 (build_rollout_context) 後、policy 開始動之前，leader + follower 一起移到 OPENARM_HOME 起始姿態。
    2. DAgger 推論時 (AUTONOMOUS)，讓 leader 即時鏡像跟隨 follower 移動。
    3. 人類介入時 (CORRECTING)，leader 立即釋放 target，回歸零力重力補償供手動示範。
    """
    waypoints, duration, guard = load_home_config()
    orig = module.build_rollout_context
    track_leader = os.environ.get("OPENARM_LEADER_TRACK", "1") != "0"

    def build_rollout_context(*args, **kwargs):
        ctx = orig(*args, **kwargs)
        teleop = ctx.hardware.teleop
        if waypoints and teleop is not None:
            # 用原始 robot (不經 ThreadSafeRobot)；此時 inference engine 還沒啟動，不會搶 bus
            go_home(ctx.hardware.robot_wrapper.inner, teleop, waypoints, duration, guard)

        if track_leader and teleop is not None:
            leaders = get_leaders(teleop)
            if leaders:
                _is_tracking = True

                # 1. 攔截 robot_wrapper.send_action：讓 leader 同步跟隨 follower 目標角度
                orig_send_action = ctx.hardware.robot_wrapper.send_action

                def send_action_with_tracking(action):
                    res = orig_send_action(action)
                    nonlocal _is_tracking
                    if _is_tracking and isinstance(action, dict):
                        for prefix, leader in leaders:
                            leader_cmd = {
                                key[len(prefix):].removesuffix(".pos"): float(v)
                                for key, v in action.items()
                                if key.startswith(prefix)
                            }
                            if leader_cmd:
                                leader._gc.set_target(leader_cmd)
                    return res

                ctx.hardware.robot_wrapper.send_action = send_action_with_tracking

                # 2. Hook DAgger 狀態機切換：進入 CORRECTING 時立刻放開 leader
                from lerobot.rollout.strategies.dagger import DAggerPhase, DAggerStrategy

                orig_apply_transition = DAggerStrategy._apply_transition

                def apply_transition_with_tracking(self, old_phase, new_phase, *t_args, **t_kwargs):
                    nonlocal _is_tracking
                    if new_phase == DAggerPhase.CORRECTING:
                        _is_tracking = False
                        logger.info("[gc] 人類介入 (CORRECTING)：釋放 leader 目標，進入純重力補償手動模式")
                        for _, leader in leaders:
                            leader._gc.set_target(None)
                    elif new_phase == DAggerPhase.AUTONOMOUS:
                        _is_tracking = True
                        logger.info("[gc] 恢復自主 (AUTONOMOUS)：啟用 leader 跟隨 follower")
                    elif new_phase == DAggerPhase.PAUSED:
                        # 暫停時停止持續更新，但維持在最後位置
                        _is_tracking = False

                    return orig_apply_transition(self, old_phase, new_phase, *t_args, **t_kwargs)

                DAggerStrategy._apply_transition = apply_transition_with_tracking
                logger.info("[gc] 已啟用 DAgger Leader 鏡像跟隨 (OPENARM_LEADER_TRACK=1)")

        return ctx

    module.build_rollout_context = build_rollout_context

