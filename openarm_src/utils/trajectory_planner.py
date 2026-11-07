from __future__ import annotations

import logging
from typing import Any

from openarm_src.utils.homing import min_jerk

logger = logging.getLogger("openarm_gc.trajectory_planner")


def interpolate_poses(
    start_pose: dict[str, float],
    target_pose: dict[str, float],
    duration_s: float,
    fps: int = 30,
) -> list[dict[str, float]]:
    """以 Min-Jerk 最小加加速度曲線在兩個關節姿態間生成平滑軌跡點清單。"""
    steps = max(int(duration_s * fps), 1)
    trajectory = []

    # 確保所有鍵值都有數值
    keys = set(start_pose.keys()) | set(target_pose.keys())
    s_full = {k: float(start_pose.get(k, target_pose.get(k, 0.0))) for k in keys}
    t_full = {k: float(target_pose.get(k, start_pose.get(k, 0.0))) for k in keys}

    for step in range(1, steps + 1):
        alpha = min_jerk(step / steps)
        point = {k: s_full[k] + alpha * (t_full[k] - s_full[k]) for k in keys}
        trajectory.append(point)

    return trajectory


def generate_slot_placement_trajectory(
    current_pose: dict[str, float],
    slot_info: dict[str, Any],
    fps: int = 30,
    transit_time_s: float = 1.2,
    place_time_s: float = 0.8,
    release_time_s: float = 0.4,
    retract_time_s: float = 0.8,
    perturbator: Any | None = None,
) -> list[dict[str, float]]:
    """為指定槽位生成完整的動態放置軌跡。

    階段：
      1. Transit: 從當前抓取姿態平滑移動至該槽位上方接近點 (Approach)
      2. Place: 垂直緩慢下放至托盤工整放置點 (Place)
      3. Release: 原地張開夾爪放下茶包 (Release)
      4. Retract: 原地垂直抬升退回安全高度 (Retract)

    提供 perturbator (CartesianPerturbator)，將自動施加末端受控毫米級笛卡爾擾動，
    防止各關節誤差疊加放大導致夾爪偏出托盤。
    """
    if perturbator is not None:
        slot_info = perturbator.sample_perturbation(slot_info)

    approach_pose = slot_info.get("approach", {})
    place_pose = slot_info.get("place", {})
    release_pose = slot_info.get("release", {})
    retract_pose = slot_info.get("retract", {})

    trajectory: list[dict[str, float]] = []

    # 階段 1: 當前抓取姿態 -> Approach
    seg1 = interpolate_poses(current_pose, approach_pose, duration_s=transit_time_s, fps=fps)
    trajectory.extend(seg1)

    # 階段 2: Approach -> Place
    curr = seg1[-1] if seg1 else current_pose
    seg2 = interpolate_poses(curr, place_pose, duration_s=place_time_s, fps=fps)
    trajectory.extend(seg2)

    # 階段 3: Place 處張開夾爪 (Release)
    curr = seg2[-1] if seg2 else curr
    # release 姿態通常包含 place 的所有關節 + 夾爪最大開度
    release_full = {**curr, **release_pose}
    seg3 = interpolate_poses(curr, release_full, duration_s=release_time_s, fps=fps)
    trajectory.extend(seg3)

    # 階段 4: Release -> Retract
    curr = seg3[-1] if seg3 else curr
    # retract 姿態保持夾爪張開狀態並抬起
    retract_full = {**retract_pose}
    for k in curr:
        if "gripper" in k and k not in retract_full:
            retract_full[k] = curr[k]
    seg4 = interpolate_poses(curr, retract_full, duration_s=retract_time_s, fps=fps)
    trajectory.extend(seg4)

    return trajectory
