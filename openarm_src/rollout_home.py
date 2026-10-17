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
    OPENARM_GRIPPER_SMOOTH   夾爪指令平滑係數 0~1，預設 0 (關閉)；建議 0.3。越小越平順但反應越慢
    OPENARM_GRIPPER_DEADBAND 夾爪死區 (度)，預設 0 (關閉)；建議 3。變化小於此值就不動，避免來回抖
    OPENARM_ACTIVE_ARM 單臂測試：left / right = 只讓這支手臂照模型動，另一支固定在起始姿態；
                       不設 = 兩支都照模型動。模型仍輸入完整 16 維 state 與三台相機。
"""
import os
import sys
import time
import importlib
from importlib.metadata import entry_points

import openarm_gc as gc

HOME_FILE = os.environ.get("OPENARM_HOME", "reset.json")
ACTIVE_ARM = os.environ.get("OPENARM_ACTIVE_ARM", "").strip().lower()
GRIP_ALPHA = float(os.environ.get("OPENARM_GRIPPER_SMOOTH", "0"))
GRIP_DEADBAND = float(os.environ.get("OPENARM_GRIPPER_DEADBAND", "0"))
if ACTIVE_ARM not in ("", "left", "right"):
    sys.exit(f"OPENARM_ACTIVE_ARM 只能是 left / right，收到 {ACTIVE_ARM!r}")
SEG_TIME = float(os.environ.get("OPENARM_HOME_TIME", "4"))
CTRL_HZ = 30


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
        s = gc.min_jerk(i / n)
        robot.send_action({k: start[k] + (goal[k] - start[k]) * s for k in goal})
        time.sleep(1.0 / CTRL_HZ)


def install_patches():
    gc._patch_follower_park()  # 結束時先回下垂再關力矩

    cls = _find_bi_follower_cls()
    orig_connect = cls.connect
    orig_disconnect = cls.disconnect  # 已含 openarm_gc 的回下垂

    def connect(self, *args, **kwargs):
        orig_connect(self, *args, **kwargs)
        waypoints = gc.load_waypoints(HOME_FILE)
        input(f"\n⚠  follower 將移到起始姿態（{HOME_FILE}，{len(waypoints)} 段）。淨空桌面、手放急停旁，按 Enter 開始移動 ...")
        for i, wp in enumerate(waypoints, 1):
            print(f"[home] 移動到 waypoint {i}/{len(waypoints)} ...")
            _move_to(self, wp, SEG_TIME)
        time.sleep(0.5)

        if GRIP_ALPHA > 0 or GRIP_DEADBAND > 0:
            # 只濾夾爪：死區擋掉小幅來回，指數平滑壓掉突變；手臂關節不受影響
            grip_prev = {}
            send_raw = self.send_action
            self._grip_filter_on = True

            def send_action_grip_filter(action):
                if getattr(self, "_grip_filter_on", False) and isinstance(action, dict):
                    action = dict(action)
                    for k, v in action.items():
                        if not k.endswith("gripper.pos"):
                            continue
                        v, prev = float(v), grip_prev.get(k)
                        if prev is not None:
                            if abs(v - prev) < GRIP_DEADBAND:
                                v = prev
                            elif GRIP_ALPHA > 0:
                                v = prev + GRIP_ALPHA * (v - prev)
                        grip_prev[k] = v
                        action[k] = v
                return send_raw(action)

            self.send_action = send_action_grip_filter
            print(f"[gripper] 濾波開啟：smooth={GRIP_ALPHA} deadband={GRIP_DEADBAND}°")

        if ACTIVE_ARM:
            # 單臂測試：另一支鎖在起始姿態，模型對它的輸出全部換掉
            frozen = "right_" if ACTIVE_ARM == "left" else "left_"
            obs = self.get_observation()
            self._single_hold = {k: float(v) for k, v in obs.items()
                                 if k.startswith(frozen) and k.endswith(".pos")}
            orig_send = self.send_action

            def send_action_single_arm(action):
                hold = getattr(self, "_single_hold", None)
                if hold and isinstance(action, dict):
                    action = {**action, **hold}
                return orig_send(action)

            self.send_action = send_action_single_arm
            print(f"[single-arm] 只執行 {ACTIVE_ARM} 手臂；{frozen.rstrip('_')} 手臂固定在起始姿態")

        input("\n✔ 已在起始姿態。擺好瓶子與托盤後按 Enter 開始推論 ...")
        print("[home] 交給模型控制\n")

    def disconnect(self, *args, **kwargs):
        self._single_hold = None  # 結束時解除鎖定，兩支手臂才都能回下垂
        self._grip_filter_on = False
        return orig_disconnect(self, *args, **kwargs)

    cls.connect = connect
    cls.disconnect = disconnect


def main():
    install_patches()
    ep = next(e for e in entry_points(group="console_scripts") if e.name == "lerobot-rollout")
    rollout_main = ep.load()
    sys.argv = ["lerobot-rollout"] + sys.argv[1:]
    sys.exit(rollout_main())


if __name__ == "__main__":
    main()