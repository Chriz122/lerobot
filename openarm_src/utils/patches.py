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
    orig = module.record_loop

    def record_loop(*args, **kwargs):
        bound = _call_arguments(orig, args, kwargs)
        robot = bound.get("robot")
        teleop = bound.get("teleop")
        if profiler is not None:
            profiler.attach(robot, teleop)
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
    orig = module.teleop_loop

    def teleop_loop(*args, **kwargs):
        bound = _call_arguments(orig, args, kwargs)
        robot = bound.get("robot")
        teleop = bound.get("teleop")
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

