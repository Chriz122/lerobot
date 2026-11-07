import math
from pathlib import Path
from openarm_src.utils.slot_teacher import get_task_slot_path
from openarm_src.utils.trajectory_planner import (
    interpolate_poses,
    generate_slot_placement_trajectory,
)
from openarm_src.utils.hybrid_recorder import resolve_slots_config


def test_interpolate_poses():
    start = {"left_joint_1.pos": 0.0, "left_joint_2.pos": 10.0}
    target = {"left_joint_1.pos": 30.0, "left_joint_2.pos": 50.0}
    traj = interpolate_poses(start, target, duration_s=1.0, fps=30)
    assert len(traj) == 30
    assert math.isclose(traj[-1]["left_joint_1.pos"], 30.0, abs_tol=1e-3)
    assert math.isclose(traj[-1]["left_joint_2.pos"], 50.0, abs_tol=1e-3)


def test_generate_slot_placement_trajectory():
    curr = {
        "left_joint_1.pos": 5.0,
        "left_joint_2.pos": 15.0,
        "left_gripper.pos": 10.0,
    }
    slot_info = {
        "slot_id": 1,
        "approach": {"left_joint_1.pos": 20.0, "left_joint_2.pos": 10.0, "left_gripper.pos": 10.0},
        "place": {"left_joint_1.pos": 20.0, "left_joint_2.pos": 30.0, "left_gripper.pos": 10.0},
        "release": {"left_gripper.pos": 30.0},
        "retract": {"left_joint_1.pos": 20.0, "left_joint_2.pos": 10.0, "left_gripper.pos": 30.0},
    }
    traj = generate_slot_placement_trajectory(
        curr,
        slot_info,
        fps=30,
        transit_time_s=0.5,
        place_time_s=0.5,
        release_time_s=0.2,
        retract_time_s=0.5,
    )
    # Total time = 0.5 + 0.5 + 0.2 + 0.5 = 1.7s -> ~51 ticks
    assert len(traj) >= 45
    # 最後一個點應該是收回的狀態，且夾爪必須精確張開至 30.0
    final_point = traj[-1]
    assert math.isclose(final_point["left_gripper.pos"], 30.0, abs_tol=1e-3)
    assert math.isclose(final_point["left_joint_2.pos"], 10.0, abs_tol=1e-3)


def test_get_task_slot_path():
    path = get_task_slot_path("test_task_xyz")
    assert path.name == "tray_slots.json"
    assert "test_task_xyz" in str(path)
    # 清除空的測試目錄
    if path.parent.exists() and not any(path.parent.iterdir()):
        path.parent.rmdir()


if __name__ == "__main__":
    test_interpolate_poses()
    test_generate_slot_placement_trajectory()
    test_get_task_slot_path()
    print("ALL PLANNER TESTS PASSED!")
