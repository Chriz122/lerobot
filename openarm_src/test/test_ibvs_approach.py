"""IBVS + 殘差 + 守門「靠近方塊」流程的離線測試（純模擬，不需任何硬體）。

執行： pytest openarm_src/test/test_ibvs_approach.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import main_experiment as exp  # noqa: E402
from approach_pipeline import target_from_handeye  # noqa: E402
from arm_kinematics import ApproxArmModel, HandEye  # noqa: E402
from ibvs_pinn_controller import (  # noqa: E402
    ACTION_DIM,
    Intrinsics,
    depth_from_size,
    feature_error,
    interaction_matrix,
)
from tag_detector import wrap_quarter_turn  # noqa: E402


def make_system(perfect_model: bool = False, **overrides):
    args = argparse.Namespace(real=False, side="left", port=None, camera=None, serial=None,
                              learner="ridge", seed=3, retreat=0.10, samples=60, target="taught",
                              target_file="unused.json", yes=True)
    for key, value in overrides.items():
        setattr(args, key, value)
    system = exp.build_system(args)
    if perfect_model:
        # 讓「真實世界」與名義模型完全一致、且沒有雜訊，用來驗證解析式本身
        world = system.world
        world.true_model = ApproxArmModel()
        world.true_handeye = system.approach.handeye
        world.cfg.sim.joint_noise_deg = 0.0
        world.cfg.sim.pixel_noise_px = 0.0
        world.cfg.sim.joint_tracking = 1.0
        world.reset(np.zeros(2), 0.0)
        system.robot.resync()
    return system


def test_wrap_quarter_turn():
    assert wrap_quarter_turn(50.0) == pytest.approx(-40.0)
    assert wrap_quarter_turn(-50.0) == pytest.approx(40.0)
    assert wrap_quarter_turn(90.0) == pytest.approx(0.0)
    assert feature_error(np.array([0, 0, 0, 44.0]), np.array([0, 0, 0, -44.0]))[3] == pytest.approx(-2.0)


def test_approx_model_jacobian_matches_finite_difference():
    model = ApproxArmModel()
    q = np.random.default_rng(0).uniform(-1.0, 1.0, 7)
    jac = model.jacobian_local(q)
    r, p = model.fk(q)
    eps = 1e-6
    for i in range(7):
        dq = np.zeros(7)
        dq[i] = eps
        r2, p2 = model.fk(q + dq)
        lin = r.T @ (p2 - p) / eps
        s = r.T @ r2
        ang = np.array([s[2, 1] - s[1, 2], s[0, 2] - s[2, 0], s[1, 0] - s[0, 1]]) / (2 * eps)
        assert np.allclose(jac[:3, i], lin, atol=1e-4)
        assert np.allclose(jac[3:, i], ang, atol=1e-4)


def test_interaction_matrix_predicts_observed_feature_motion():
    """模型完全正確時，解析 interaction matrix 的預測要與影像實際變化一致（含正負號）。"""
    system = make_system(perfect_model=True)
    world, robot, approach = system.world, system.robot, system.approach
    # 角點偵測本身約有 0.5 px 的量測誤差，所以用較大的位移、容差取 1.2
    for axis, step in enumerate([8.0, 8.0, 8.0, 4.0]):
        world.reset(np.array([0.02, -0.015]), 12.0)
        robot.resync()
        before = approach.observe()
        action = np.zeros(ACTION_DIM)
        action[axis] = step
        jac = interaction_matrix(before.s, depth_from_size(before.s[2], before.intr.f, 0.05), before.intr)
        assert approach.move_camera_total(action)
        observed = feature_error(approach.observe().s, before.s)
        predicted = jac @ action
        assert np.allclose(observed, predicted, atol=1.2), (axis, observed, predicted)
        assert np.sign(observed[axis]) == np.sign(predicted[axis])


def test_handeye_target_rejects_tag_outside_image():
    system = make_system()
    intr = Intrinsics(430.0, 424.0, 240.0, 848, 480)
    far_camera = HandEye((0.0, -0.12, -0.08), (0.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        target_from_handeye(system.cfg, far_camera, intr)
    target = target_from_handeye(system.cfg, system.approach.handeye, intr)
    assert target.descent_m == pytest.approx(0.16 - 0.08 + 0.02)


@pytest.mark.parametrize("method", ["ibvs", "learned", "guarded"])
def test_full_approach_grasps_cube_in_simulation(method, tmp_path):
    system = make_system()
    target = exp.resolve_target(system)
    if method != "ibvs":
        exp.train_residual(system, str(tmp_path))
    system.world.reset(np.array([0.035, -0.03]), 25.0)
    system.robot.resync()
    system.robot.mark_start()
    result = system.approach.servo(target, method)
    assert result.success, result.reason
    grasp = system.approach.descend_grasp_lift(target)
    assert grasp.holding and grasp.lifted, (grasp.reason, grasp.sim_error)
    assert grasp.sim_error["xy_mm"] < 5.0


def test_guard_survives_a_wrong_residual_model():
    """殘差模型錯到會讓命令反向時：learned 會失敗，guarded 仍要收斂。"""
    system = make_system()
    target = exp.resolve_target(system)
    controller = system.approach.controller
    controller.model.m = -1.6 * np.eye(ACTION_DIM)      # J_hat = -0.6 J_phys：方向相反
    controller.is_trained = controller.is_trusted = True

    def run(method):
        system.world.reset(np.array([0.03, 0.02]), -15.0)
        system.robot.resync()
        system.robot.mark_start()
        return system.approach.servo(target, method)

    assert not run("learned").success
    guarded = run("guarded")
    assert guarded.success, guarded.reason


def test_safety_blocks_motion_below_table():
    system = make_system()
    robot = system.robot
    _, p = robot.model.fk(robot.q_cmd)
    robot.table_z_m = float(p[2]) - 0.02
    assert system.approach.move_along_approach(0.01)
    assert not system.approach.move_along_approach(0.05)
    assert "桌面" in robot.last_block_reason
