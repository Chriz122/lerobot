from __future__ import annotations

import json
import logging
import os
import time
import numpy as np

from openarm_src.utils.config import JOINT_SIGN

logger = logging.getLogger("openarm_gc.homing")


def min_jerk(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


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


def get_leaders(teleop):
    from lerobot.teleoperators.bi_openarm_leader import BiOpenArmLeader
    from lerobot.teleoperators.openarm_leader import OpenArmLeader

    if isinstance(teleop, BiOpenArmLeader):
        return [("left_", teleop.left_arm), ("right_", teleop.right_arm)]
    if isinstance(teleop, OpenArmLeader):
        return [("", teleop)]
    return []


def go_home(robot, teleop, waypoints: list[dict[str, float]], duration: float, guard: TableGuard | None = None):
    """leader 與 follower 依序經過每個 waypoint (min-jerk)，最後停在起始姿態後放開 leader。"""
    leaders = get_leaders(teleop)
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


def load_home_config():
    """讀 OPENARM_HOME / OPENARM_HOME_TIME / OPENARM_TABLE_Z，回傳 (waypoints, duration, guard)。"""
    home_path = os.environ.get("OPENARM_HOME")
    if not home_path:
        logger.warning("[gc] OPENARM_HOME 未設定 -> 不做 homing")
        return None, 0.0, None
    waypoints = load_waypoints(home_path)
    duration = float(os.environ.get("OPENARM_HOME_TIME", str(3 * len(waypoints) + 1)))
    guard = None
    if "OPENARM_TABLE_Z" in os.environ:
        guard = TableGuard(os.environ["OPENARM_URDF"], float(os.environ["OPENARM_TABLE_Z"]),
                           float(os.environ.get("OPENARM_TABLE_MARGIN", "0.05")))
    return waypoints, duration, guard

