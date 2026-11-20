#!/usr/bin/env python3
"""
rollout_home.py — 在 lerobot-rollout 開始推論前，先把 OpenArm follower 平滑移到起始姿態。

不修改 LeRobot 原始碼，做法與 openarm_gc.py 相同（monkey-patch）：
  - BiOpenArmFollower.connect：連線後依 reset.json 的 waypoints 用 min-jerk 軌跡移到起始姿態，
    並在推論開始前等你按 Enter（讓你擺好瓶子、托盤）
  - follower disconnect：沿用 openarm_gc 的 park，結束時先慢慢回下垂再關力矩（不會直接掉下來）

用法（放在 ~/lerobot，與 openarm_gc.py、reset.json 同一個資料夾）：
    python rollout_home.py <與 lerobot-rollout 完全相同的參數>

環境變數：
    OPENARM_HOME       起始姿態檔，預設 reset.json
    OPENARM_HOME_TIME  每一段 waypoint 的移動秒數，預設 4
    OPENARM_PARK       1（預設）= 結束時回下垂再關力矩

夾爪加強（只在按 Enter 交給模型之後生效，歸位、回下垂不受影響）：
    OPENARM_GRIPPER_SCALE  夾爪目標角度乘上這個倍數，預設 1（不變）。
                           0° = 全閉、-65° = 全開，所以 0.8 會夾更緊（-20° → -16°），全開也變小（-65° → -52°）
    OPENARM_GRIPPER_KP     夾爪 kp，預設不改（config 為 25）；夾緊力 ≈ kp × 擠壓角度

防跳保護 StepJumpGuard（同樣只在推論期間；以交給模型那一刻的真實姿態為基準，限制每拍的變化量）：
    OPENARM_STEP_GUARD        1（預設）= 開啟；0 = 關閉
    OPENARM_MAX_STEP_DEG      每拍關節最多移動幾度，預設 4（30 fps ≈ 120°/s）
    OPENARM_MAX_STEP_GRIPPER  每拍夾爪最多移動幾度，預設 15
"""
import os
import sys
import time
import importlib
from importlib.metadata import entry_points

from openarm_src.utils import StepJumpGuard, load_waypoints, min_jerk, patch_follower_park
from openarm_src.utils.config import DEFAULT_GRIPPER_KP, DEFAULT_GRIPPER_SCALE

HOME_FILE = os.environ.get("OPENARM_HOME", "reset.json")
SEG_TIME = float(os.environ.get("OPENARM_HOME_TIME", "4"))
CTRL_HZ = 30
GRIPPER_KP_INDEX = 7  # position_kp 依序為 joint_1..joint_7, gripper


def _find_bi_follower_cls():
    for mod in ("lerobot.robots.bi_openarm_follower",
                "lerobot.robots.bi_openarm_follower.bi_openarm_follower"):
        try:
            m = importlib.import_module(mod)
            if hasattr(m, "BiOpenArmFollower"):
                return m.BiOpenArmFollower
        except ImportError:
            pass
    raise ImportError("找不到 BiOpenArmFollower 類別")


def _move_to(robot, goal: dict[str, float], duration: float):
    obs = robot.get_observation()
    start = {k: float(obs[k]) for k in goal}
    n = max(1, int(duration * CTRL_HZ))
    for i in range(1, n + 1):
        s = min_jerk(i / n)
        robot.send_action({k: start[k] + (goal[k] - start[k]) * s for k in goal})
        time.sleep(1.0 / CTRL_HZ)


def _arms(robot):
    return (robot.left_arm, robot.right_arm)


def _grip_boost_on(robot):
    """交給模型之後才開始：換掉夾爪 kp、讓 send_action 縮放夾爪目標角度。"""
    if DEFAULT_GRIPPER_KP is not None:
        robot._grip_orig_kp = [arm.config.position_kp for arm in _arms(robot)]
        for arm in _arms(robot):
            kp = arm.config.position_kp
            gains = list(kp) if isinstance(kp, list) else [kp] * (GRIPPER_KP_INDEX + 1)
            gains[GRIPPER_KP_INDEX] = DEFAULT_GRIPPER_KP
            arm.config.position_kp = gains  # 換成新的 list，原本的留著還原用
    robot._grip_boost = True
    if DEFAULT_GRIPPER_SCALE != 1.0 or DEFAULT_GRIPPER_KP is not None:
        kp_now = robot.left_arm.config.position_kp[GRIPPER_KP_INDEX]
        print(f"[gripper] 夾爪加強：目標角度 ×{DEFAULT_GRIPPER_SCALE:g}，kp = {kp_now:g}")


def _grip_boost_off(robot):
    """回下垂之前恢復原本的夾爪設定。"""
    robot._grip_boost = False
    orig = getattr(robot, "_grip_orig_kp", None)
    if orig is not None:
        for arm, kp in zip(_arms(robot), orig):
            arm.config.position_kp = kp
        robot._grip_orig_kp = None


def _guard_on(robot):
    """交給模型時建立防跳保護，基準 = 當下的真實關節角度（避免模型第一拍就暴衝）。"""
    guard = StepJumpGuard()
    robot._step_guard = None
    if not guard.enabled:
        print("[step_guard] 防跳保護已關閉（OPENARM_STEP_GUARD=0）")
        return
    guard.sync_from_observation(robot.get_observation())
    robot._step_guard = guard
    robot._guard_hits = 0
    print(f"[step_guard] 防跳保護：每拍關節 ≤ {guard.max_step_deg:g}°、夾爪 ≤ {guard.max_step_gripper:g}°")


def _guard_off(robot):
    if getattr(robot, "_step_guard", None) is not None:
        print(f"[step_guard] 本回合有 {robot._guard_hits} 拍動作被限幅")
    robot._step_guard = None


def install_patches():
    if DEFAULT_GRIPPER_SCALE <= 0:
        sys.exit(f"OPENARM_GRIPPER_SCALE 必須 > 0（目前 {DEFAULT_GRIPPER_SCALE}）")
    if DEFAULT_GRIPPER_KP is not None and not 0 < DEFAULT_GRIPPER_KP <= 500:
        sys.exit(f"OPENARM_GRIPPER_KP 必須在 (0, 500] 之間（目前 {DEFAULT_GRIPPER_KP}）")

    patch_follower_park()  # 結束時先回下垂再關力矩

    cls = _find_bi_follower_cls()
    orig_connect = cls.connect
    orig_send_action = cls.send_action
    orig_disconnect = cls.disconnect

    def send_action(self, action, *args, **kwargs):
        if DEFAULT_GRIPPER_SCALE != 1.0 and getattr(self, "_grip_boost", False):
            action = {
                k: float(v) * DEFAULT_GRIPPER_SCALE if k.endswith("gripper.pos") else v
                for k, v in action.items()
            }
        guard = getattr(self, "_step_guard", None)
        if guard is not None:  # 先縮放夾爪再限幅，跟真實角度的基準在同一個空間
            clamped = guard.clamp(action)
            if any(abs(float(clamped[k]) - float(action[k])) > 1e-6 for k in action if k.endswith(".pos")):
                self._guard_hits += 1
            action = clamped
        return orig_send_action(self, action, *args, **kwargs)

    def disconnect(self, *args, **kwargs):
        _guard_off(self)
        _grip_boost_off(self)
        return orig_disconnect(self, *args, **kwargs)

    def connect(self, *args, **kwargs):
        orig_connect(self, *args, **kwargs)
        waypoints = load_waypoints(HOME_FILE)
        input(f"\n⚠  follower 將移到起始姿態（{HOME_FILE}，{len(waypoints)} 段）。淨空桌面、手放急停旁，按 Enter 開始移動 ...")
        for i, wp in enumerate(waypoints, 1):
            print(f"[home] 移動到 waypoint {i}/{len(waypoints)} ...")
            _move_to(self, wp, SEG_TIME)
        time.sleep(0.5)
        input("\n✔ 已在起始姿態。擺好瓶子與托盤後按 Enter 開始推論 ...")
        print("[home] 交給模型控制\n")
        _grip_boost_on(self)
        _guard_on(self)

    cls.connect = connect
    cls.send_action = send_action
    cls.disconnect = disconnect


def main():
    install_patches()
    import openarm_src.grasp.ibvs_assist  # noqa: F401  註冊 --strategy.type=ibvs_assist（ACT + IBVS 夾取）

    ep = next(e for e in entry_points(group="console_scripts") if e.name == "lerobot-rollout")
    rollout_main = ep.load()
    sys.argv = ["lerobot-rollout"] + sys.argv[1:]
    sys.exit(rollout_main())


if __name__ == "__main__":
    main()