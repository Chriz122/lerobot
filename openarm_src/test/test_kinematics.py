import math
from openarm_src.utils.kinematics import CartesianPerturbator
from openarm_src.utils.trajectory_planner import generate_slot_placement_trajectory


def test_cartesian_perturbator_bounds():
    perturbator = CartesianPerturbator(
        max_xy_mm=4.0,
        max_z_up_mm=1.0,
        max_yaw_deg=2.5,
    )

    slot_info = {
        "slot_id": 1,
        "name": "Slot_1",
        "approach": {"left_joint_1.pos": 10.0, "left_joint_2.pos": 8.0, "left_joint_4.pos": 40.0, "left_joint_7.pos": 0.0, "left_gripper.pos": 15.0},
        "place": {"left_joint_1.pos": 10.0, "left_joint_2.pos": 20.0, "left_joint_4.pos": 40.0, "left_joint_7.pos": 0.0, "left_gripper.pos": 15.0},
        "release": {"left_gripper.pos": 65.0},
        "retract": {"left_joint_1.pos": 10.0, "left_joint_2.pos": 8.0, "left_joint_4.pos": 40.0, "left_joint_7.pos": 0.0, "left_gripper.pos": 65.0},
    }

    # 執行多次取樣以驗證擾動範圍
    for _ in range(20):
        perturbed = perturbator.sample_perturbation(slot_info)
        meta = perturbed["perturbation_meta"]

        dx = meta["dx_mm"]
        dy = meta["dy_mm"]
        dz = meta["dz_mm"]
        dyaw = meta["dyaw_deg"]

        r_xy = math.sqrt(dx**2 + dy**2)
        assert r_xy <= 4.05, f"Horizontal displacement {r_xy} exceeded limit 4.0mm"
        assert 0.0 <= dz <= 1.05, f"Vertical height offset {dz} out of [0, 1.0mm]"
        assert abs(dyaw) <= 2.55, f"Yaw rotation {dyaw} exceeded limit 2.5°"

        # 檢查 Approach、Place、Retract 的 joint_1 關節角度變化量是否一致
        dq1_place = perturbed["place"]["left_joint_1.pos"] - slot_info["place"]["left_joint_1.pos"]
        dq1_approach = perturbed["approach"]["left_joint_1.pos"] - slot_info["approach"]["left_joint_1.pos"]
        dq1_retract = perturbed["retract"]["left_joint_1.pos"] - slot_info["retract"]["left_joint_1.pos"]

        assert math.isclose(dq1_place, dq1_approach, abs_tol=1e-3)
        assert math.isclose(dq1_place, dq1_retract, abs_tol=1e-3)


def test_trajectory_with_perturbator():
    perturbator = CartesianPerturbator(max_xy_mm=3.0, max_z_up_mm=0.5, max_yaw_deg=2.0)
    curr = {"left_joint_1.pos": 0.0, "left_joint_2.pos": 0.0, "left_gripper.pos": 15.0}
    slot_info = {
        "slot_id": 2,
        "approach": {"left_joint_1.pos": 15.0, "left_joint_2.pos": 5.0, "left_gripper.pos": 15.0},
        "place": {"left_joint_1.pos": 15.0, "left_joint_2.pos": 18.0, "left_gripper.pos": 15.0},
        "release": {"left_gripper.pos": 30.0},
        "retract": {"left_joint_1.pos": 15.0, "left_joint_2.pos": 5.0, "left_gripper.pos": 30.0},
    }

    traj = generate_slot_placement_trajectory(curr, slot_info, fps=30, perturbator=perturbator)
    assert len(traj) > 0
    # 夾爪在最後一幀必須張開
    assert math.isclose(traj[-1]["left_gripper.pos"], 30.0, abs_tol=1e-3)


if __name__ == "__main__":
    test_cartesian_perturbator_bounds()
    test_trajectory_with_perturbator()
    print("ALL KINEMATICS & PERTURBATION TESTS PASSED!")
