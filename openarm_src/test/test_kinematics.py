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


def test_gripper_gaussian_perturbation():
    # 測試預設配置 (由環境變數或 config 載入)
    perturbator = CartesianPerturbator()
    samples = [perturbator.sample_gripper_opening() for _ in range(1000)]

    min_limit = perturbator.gripper_min_deg
    max_limit = perturbator.gripper_max_deg
    sigma = perturbator.gripper_sigma_deg

    # 1. 嚴格不低於 OPENARM_SLOT_GRIPPER_MIN_DEG
    assert all(s >= min_limit for s in samples), f"Sampled gripper degree was lower than {min_limit}°!"
    # 2. 上限截斷不超過 OPENARM_SLOT_GRIPPER_MAX_DEG
    assert all(s <= max_limit for s in samples), f"Sampled gripper degree exceeded {max_limit}°!"
    # 3. 分佈特性：中心峰值在 min_limit 附近，大部分值 (半高斯 1-sigma 68%) 落在 [min_limit, min_limit + sigma]
    in_1sigma = [s for s in samples if min_limit <= s <= (min_limit + sigma)]
    ratio_1sigma = len(in_1sigma) / len(samples)
    assert 0.55 <= ratio_1sigma <= 0.80, f"Expected ~68% in [{min_limit}, {min_limit + sigma}], got {ratio_1sigma:.2%}"

    # 4. 測試自訂非預設參數 (例如 min=35.0, max=55.0, sigma=8.0) 驗證無寫死魔法數字
    custom_p = CartesianPerturbator(gripper_min_deg=35.0, gripper_max_deg=55.0, gripper_sigma_deg=8.0)
    c_samples = [custom_p.sample_gripper_opening() for _ in range(500)]
    assert all(s >= 35.0 for s in c_samples), "Custom min limit violated!"
    assert all(s <= 55.0 for s in c_samples), "Custom max limit violated!"


def test_trajectory_with_perturbator():
    perturbator = CartesianPerturbator(max_xy_mm=3.0, max_z_up_mm=0.5, max_yaw_deg=2.0)
    min_deg = perturbator.gripper_min_deg
    max_deg = perturbator.gripper_max_deg

    curr = {"left_joint_1.pos": 0.0, "left_joint_2.pos": 0.0, "left_gripper.pos": 15.0}
    slot_info = {
        "slot_id": 2,
        "approach": {"left_joint_1.pos": 15.0, "left_joint_2.pos": 5.0, "left_gripper.pos": 15.0},
        "place": {"left_joint_1.pos": 15.0, "left_joint_2.pos": 18.0, "left_gripper.pos": 15.0},
        "release": {"left_gripper.pos": min_deg},
        "retract": {"left_joint_1.pos": 15.0, "left_joint_2.pos": 5.0, "left_gripper.pos": min_deg},
    }

    # 有擾動時：最後一幀必須張開且 >= min_deg 且 <= max_deg
    traj = generate_slot_placement_trajectory(curr, slot_info, fps=30, perturbator=perturbator)
    assert len(traj) > 0
    final_gripper = traj[-1]["left_gripper.pos"]
    assert min_deg <= final_gripper <= max_deg, f"Final gripper angle {final_gripper} out of [{min_deg}, {max_deg}]"

    # 無擾動時：最後一幀嚴格等於 min_deg
    traj_det = generate_slot_placement_trajectory(curr, slot_info, fps=30, perturbator=None)
    assert math.isclose(traj_det[-1]["left_gripper.pos"], min_deg, abs_tol=1e-3)


if __name__ == "__main__":
    test_cartesian_perturbator_bounds()
    test_gripper_gaussian_perturbation()
    test_trajectory_with_perturbator()
    print("ALL KINEMATICS & PERTURBATION TESTS PASSED!")
