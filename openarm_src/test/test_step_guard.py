import math
from openarm_src.utils.step_guard import StepJumpGuard


def approx(a, b, tol=1e-5):
    return math.isclose(a, b, abs_tol=tol)


def test_step_guard_clamp_normal():
    guard = StepJumpGuard(max_step_deg=4.0, max_step_gripper=15.0, enabled=True)

    # Initial state
    guard.sync_from_observation({
        "left_joint_1.pos": 10.0,
        "left_joint_2.pos": 20.0,
        "left_gripper.pos": 50.0,
    })

    # Small movement: within 4.0 degrees
    act = {
        "left_joint_1.pos": 12.0,  # +2.0
        "left_joint_2.pos": 18.5,  # -1.5
        "left_gripper.pos": 60.0,  # +10.0 (gripper limit is 15.0)
    }
    clamped = guard.clamp(act)
    assert approx(clamped["left_joint_1.pos"], 12.0)
    assert approx(clamped["left_joint_2.pos"], 18.5)
    assert approx(clamped["left_gripper.pos"], 60.0)


def test_step_guard_large_jump_clamping():
    guard = StepJumpGuard(max_step_deg=4.0, max_step_gripper=15.0, enabled=True)

    guard.sync_from_observation({
        "left_joint_1.pos": 0.0,
        "left_joint_2.pos": 0.0,
        "left_gripper.pos": 0.0,
    })

    # Policy predicts a huge step jump (+20.0 degrees)
    act = {
        "left_joint_1.pos": 20.0,
        "left_joint_2.pos": -25.0,
        "left_gripper.pos": 50.0,
    }
    clamped = guard.clamp(act)

    # Should be clamped to +- max_step
    assert approx(clamped["left_joint_1.pos"], 4.0)
    assert approx(clamped["left_joint_2.pos"], -4.0)
    assert approx(clamped["left_gripper.pos"], 15.0)

    # Next step: policy still wants 20.0
    clamped2 = guard.clamp(act)
    assert approx(clamped2["left_joint_1.pos"], 8.0)
    assert approx(clamped2["left_joint_2.pos"], -8.0)
    assert approx(clamped2["left_gripper.pos"], 30.0)


def test_step_guard_dagger_transition_resync():
    """模擬 DAgger 人類介入後交接回 Policy 的場景"""
    guard = StepJumpGuard(max_step_deg=4.0, max_step_gripper=15.0, enabled=True)

    # 1. 機器人原本在 0 度
    guard.sync_from_observation({"left_joint_1.pos": 0.0})

    # 2. 人類手動介入 (CORRECTING) 把手臂移到了 45 度
    # 介入結束時切換回 AUTONOMOUS，resync 讀取當前真實姿態 (45 度)
    guard.sync_from_observation({"left_joint_1.pos": 45.0})

    # 3. Policy 剛啟動首拍預測了 30 度 (落差 15 度)
    policy_act = {"left_joint_1.pos": 30.0}

    # 4. clamp 應從 45 度出發，限制第一拍最多移到 41 度 (45 - 4.0)，而不是瞬間跳到 30 度！
    clamped = guard.clamp(policy_act)
    assert approx(clamped["left_joint_1.pos"], 41.0)


def test_step_guard_disabled():
    guard = StepJumpGuard(max_step_deg=4.0, max_step_gripper=15.0, enabled=False)
    guard.sync_from_observation({"left_joint_1.pos": 0.0})
    act = {"left_joint_1.pos": 100.0}
    clamped = guard.clamp(act)
    assert clamped["left_joint_1.pos"] == 100.0


if __name__ == "__main__":
    test_step_guard_clamp_normal()
    test_step_guard_large_jump_clamping()
    test_step_guard_dagger_transition_resync()
    test_step_guard_disabled()
    print("ALL TESTS PASSED!")

