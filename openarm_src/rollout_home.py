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
"""
import os
import sys
import time
import importlib
from importlib.metadata import entry_points

import openarm_src.openarm_gc as gc

HOME_FILE = os.environ.get("OPENARM_HOME", "reset.json")
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

    def connect(self, *args, **kwargs):
        orig_connect(self, *args, **kwargs)
        waypoints = gc.load_waypoints(HOME_FILE)
        input(f"\n⚠  follower 將移到起始姿態（{HOME_FILE}，{len(waypoints)} 段）。淨空桌面、手放急停旁，按 Enter 開始移動 ...")
        for i, wp in enumerate(waypoints, 1):
            print(f"[home] 移動到 waypoint {i}/{len(waypoints)} ...")
            _move_to(self, wp, SEG_TIME)
        time.sleep(0.5)
        input("\n✔ 已在起始姿態。擺好瓶子與托盤後按 Enter 開始推論 ...")
        print("[home] 交給模型控制\n")

    cls.connect = connect


def main():
    install_patches()
    ep = next(e for e in entry_points(group="console_scripts") if e.name == "lerobot-rollout")
    rollout_main = ep.load()
    sys.argv = ["lerobot-rollout"] + sys.argv[1:]
    sys.exit(rollout_main())


if __name__ == "__main__":
    main()