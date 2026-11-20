"""ibvs_assist 夾取狀態機的離線閉迴路測試（真實 URDF 運動學 + 合成腕部相機，不需硬體）。

執行：PYTHONPATH=$HOME/lerobot pytest openarm_src/grasp/test_grasp_servo.py
"""
import json
import os

import numpy as np
import pytest

pytest.importorskip("pinocchio")
if not os.environ.get("OPENARM_URDF"):
    pytest.skip("需要 OPENARM_URDF", allow_module_level=True)

from openarm_src.grasp.detector import Box, Detection  # noqa: E402
from openarm_src.grasp.kinematics import ArmModel, kabsch, rotation_angle_deg, rpy_to_matrix  # noqa: E402
from openarm_src.grasp.servo import (  # noqa: E402
    ArmGraspServo,
    bottle_in_tcp,
    Intrinsics,
    Phase,
    ServoConfig,
    TaughtTarget,
)

HERE = os.path.dirname(os.path.abspath(__file__))
INTR = Intrinsics(215.0, 215.0, 212.0, 120.0)
WH = (424, 240)
CAM_P = np.array([0.0, -0.04, -0.08])         # 相機在 TCP 座標的位置
CAM_R = rpy_to_matrix([-35.0, 0.0, 3.0])      # 相機相對 TCP 傾斜約 35 度（實測）
APPROACH = np.array([0.0, 0.0, 1.0])
DESCENT = 0.10


def home_q(model: ArmModel) -> np.ndarray:
    with open(os.path.join(HERE, "..", "reset.json"), encoding="utf-8") as f:
        wp = json.load(f)["waypoints"][-1]
    return model.deg_to_q([wp[f"{model.side}_joint_{i}.pos"] for i in range(1, 8)])


def move_tcp(model, q, delta, n=50):
    for _ in range(n):
        q = q + model.tcp_translation_to_dq(q, delta / n, damping=1e-3)
    return q


def camera_point(model, q, p_world):
    r, p = model.fk(q)
    return CAM_R.T @ (r.T @ (p_world - p) - CAM_P)


BOTTLE_R, BOTTLE_H = 0.033, 0.22   # 瓶子半徑、高度；夾取點在瓶身中段（瓶底上方 BOTTLE_H/2）


def project(model, q, p_world):
    pc = camera_point(model, q, p_world)
    return np.array([INTR.cx + INTR.fx * pc[0] / pc[2], INTR.cy + INTR.fy * pc[1] / pc[2]]), pc[2]


def make_detection(model, q, p_world, fid, noise_px=0.0, rng=None):
    """瓶子 = 以夾取點為中心的直立圓柱；框 = 圓柱外緣點投影的外接框，超出畫面就裁切（與 OWLv2 相同）。"""
    _, z = project(model, q, p_world)
    pts = []
    for dz in (-BOTTLE_H / 2, BOTTLE_H / 2):
        for a in np.linspace(0, 2 * np.pi, 12, endpoint=False):
            uv, zz = project(model, q, p_world + np.array([BOTTLE_R * np.cos(a), BOTTLE_R * np.sin(a), dz]))
            if zz > 0.02:
                pts.append(uv)
    pts = np.array(pts)
    if len(pts) == 0:
        return Detection("wrist_left", fid, 0.0, [], None, (WH[1], WH[0]))
    if rng is not None:
        pts = pts + rng.normal(0, noise_px, size=2)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    x0, x1 = np.clip([x0, x1], 0, WH[0] - 1)
    y0, y1 = np.clip([y0, y1], 0, WH[1] - 1)
    # D405 在框中心量到的是「那一段瓶身」的深度：找投影最接近框中心的瓶軸點，深度取它（減去半徑，打在表面）
    c = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    hs = np.linspace(-BOTTLE_H / 2, BOTTLE_H / 2, 89)
    proj = [project(model, q, p_world + np.array([0.0, 0.0, h])) for h in hs]
    k = int(np.argmin([np.linalg.norm(uv - c) for uv, _ in proj]))
    depth = np.full((WH[1], WH[0]), proj[k][1] - BOTTLE_R, dtype=np.float32)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return Detection("wrist_left", fid, 0.0, [], depth, (WH[1], WH[0]))
    return Detection("wrist_left", fid, 0.0, [Box(x0, y0, x1, y1, 0.5)], depth, (WH[1], WH[0]))


@pytest.fixture(scope="module")
def setup():
    model = ArmModel("left")
    q_grasp = home_q(model)
    _, p_grasp = model.fk(q_grasp)                    # 瓶子的夾取點 = 示教時的 TCP 位置
    q_teach = move_tcp(model, q_grasp, -APPROACH * DESCENT)
    # 與 bottle_teach.py 相同：示教位置看到的（可能被裁切的）框中心 + 深度 -> TCP 座標的瓶子點
    det = make_detection(model, q_teach, p_grasp, 0)
    (u, v), z = det.boxes[0].center, float(det.depth_m[0, 0])
    # 與 bottle_teach.py 相同：下降距離 = FK 量到的實際後退量（關節碰到限位時會比指令少）
    r_g, _ = model.fk(q_grasp)
    descent = float((r_g.T @ (p_grasp - model.fk(q_teach)[1])) @ APPROACH)
    target = TaughtTarget(u, v, z, WH, CAM_R, descent, APPROACH, "bottle",
                          bottle_in_tcp(INTR, CAM_R, CAM_P, u, v, z), CAM_P, float(model.fk(q_teach)[1][2]),
                          model.fk(q_teach)[0].T @ (p_grasp - model.fk(q_teach)[1]))
    return model, q_grasp, p_grasp, q_teach, target


def run(servo, model, q, p_grasp, steps=900, rng=None, noise_px=0.0):
    grip = -20.0
    fid = 0
    q_end_descend = None
    for k in range(steps):
        det = None
        if k % 2 == 0:                                 # 偵測約 15 Hz
            fid += 1
            det = make_detection(model, q, p_grasp, fid, noise_px, rng)
        cmd = servo.update(model.q_to_deg(q), grip, det, 1.0 / 30.0)
        if cmd is not None:
            q = model.deg_to_q([cmd[f"left_joint_{i}.pos"] for i in range(1, 8)])
            grip = cmd["left_gripper.pos"]
        if servo.phase == Phase.CLOSE and q_end_descend is None:
            q_end_descend = q.copy()
        if servo.phase in (Phase.DONE, Phase.ABORTED):
            break
    return q, grip, q_end_descend


@pytest.mark.parametrize("offset", [[0.04, 0.0, 0.0], [0.0, -0.02, 0.02], [-0.03, -0.03, 0.02]])
def test_servo_grasps_bottle_from_offset_start(setup, offset):
    model, _, p_grasp, q_teach, target = setup
    servo = ArmGraspServo("left", model, target, INTR, WH, ServoConfig())
    q0 = move_tcp(model, q_teach, np.array(offset))   # 起點：示教位置再偏移
    q, grip, q_desc = run(servo, model, q0, p_grasp, rng=np.random.default_rng(0), noise_px=0.5)
    assert servo.phase == Phase.DONE, servo.status()
    r_desc, p_tcp = model.fk(q_desc)
    # 下降會照示教路徑走回夾取點，再沿同一方向多深入 extra_depth_m
    d = target.descent_vector / np.linalg.norm(target.descent_vector)
    err = p_tcp - (p_grasp + r_desc @ d * ServoConfig().extra_depth_m)
    assert np.linalg.norm(err) < 0.008, np.round(err * 1000, 1)
    assert grip == pytest.approx(0.0)
    _, p_end = model.fk(q)
    assert p_end[2] - p_tcp[2] > 0.05                  # 有提起


def test_servo_waits_until_bottle_is_close(setup):
    model, _, p_grasp, q_teach, target = setup
    servo = ArmGraspServo("left", model, target, INTR, WH, ServoConfig(trigger_max_depth_m=0.05))
    run(servo, model, q_teach, p_grasp, steps=60)
    assert servo.phase == Phase.IDLE                    # 太遠不接手


def test_observe_only_never_commands(setup):
    model, _, p_grasp, q_teach, target = setup
    servo = ArmGraspServo("left", model, target, INTR, WH, ServoConfig(), observe_only=True)
    for k in range(30):
        assert servo.update(model.q_to_deg(q_teach), -20.0, make_detection(model, q_teach, p_grasp, k + 1),
                            1 / 30) is None


def test_aborts_and_hands_back_when_bottle_lost(setup):
    model, _, p_grasp, q_teach, target = setup
    servo = ArmGraspServo("left", model, target, INTR, WH, ServoConfig(lost_timeout_s=0.0))
    q = move_tcp(model, q_teach, np.array([0.03, 0.0, 0.0]))
    for k in range(5):
        servo.update(model.q_to_deg(q), -20.0, make_detection(model, q, p_grasp, k + 1), 1 / 30)
    assert servo.active
    empty = Detection("wrist_left", 99, 0.0, [], None, (WH[1], WH[0]))
    assert servo.update(model.q_to_deg(q), -20.0, empty, 1 / 30) is None
    assert servo.phase == Phase.ABORTED


def test_kabsch_recovers_handeye_rotation(setup):
    model, _, p_grasp, q_teach, _ = setup
    a, b = [], []
    for axis in range(3):
        for sign in (1.0, -1.0):
            d = np.zeros(3)
            d[axis] = sign * 0.015
            q1 = move_tcp(model, q_teach, d)
            r0, p0 = model.fk(q_teach)
            _, p1 = model.fk(q1)
            a.append(r0.T @ (p1 - p0))
            b.append(-(camera_point(model, q1, p_grasp) - camera_point(model, q_teach, p_grasp)))
    r = kabsch(np.array(a), np.array(b)).T
    assert rotation_angle_deg(CAM_R.T @ r) < 0.5


def test_no_trigger_when_bottle_out_of_view(setup):
    model, _, p_grasp, q_teach, target = setup
    servo = ArmGraspServo("left", model, target, INTR, WH, ServoConfig())
    q = move_tcp(model, q_teach, np.array([0.0, 0.3, 0.0]))      # 瓶子在畫面之外
    assert not make_detection(model, q, p_grasp, 1).boxes
    run(servo, model, q, p_grasp, steps=60)
    assert servo.phase == Phase.IDLE


def test_waits_for_arm_to_reach_before_closing(setup):
    """手臂落後命令時（重力、追蹤落後），要等追上才閉合夾爪。"""
    model, _, p_grasp, q_teach, target = setup
    servo = ArmGraspServo("left", model, target, INTR, WH, ServoConfig(reach_timeout_s=5.0))
    q, grip, fid = q_teach.copy(), -20.0, 0
    lagging = True
    for k in range(2000):
        det = None
        if k % 2 == 0:
            fid += 1
            det = make_detection(model, q, p_grasp, fid)
        cmd = servo.update(model.q_to_deg(q), grip, det, 1 / 30)
        if cmd is None:
            continue
        q_cmd = model.deg_to_q([cmd[f"left_joint_{i}.pos"] for i in range(1, 8)])
        if servo.phase == Phase.CLOSE and lagging:
            q = q + 0.05 * (q_cmd - q)                 # 很慢才追上
            if np.max(np.abs(np.degrees(q_cmd - q))) < 0.5:
                lagging = False
            assert cmd["left_gripper.pos"] < -50.0, "手臂還沒到位就開始閉合"
        else:
            q = q_cmd
        grip = cmd["left_gripper.pos"]
        if servo.phase == Phase.LIFT:
            break
    assert servo.phase == Phase.LIFT and not lagging


@pytest.mark.parametrize("back", [0.0, 0.01, 0.03])
def test_too_close_retreats_then_grasps(setup, back):
    """ACT 已經把夾爪帶到瓶子上（比示教距離還近）：接手後先後退，再伺服、下降、夾取。"""
    model, q_grasp, p_grasp, q_teach, target = setup
    servo = ArmGraspServo("left", model, target, INTR, WH, ServoConfig())
    phases = []
    q = move_tcp(model, q_grasp, -APPROACH * back)
    grip, fid, q_desc = -20.0, 0, None
    for k in range(1500):
        det = None
        if k % 2 == 0:
            fid += 1
            det = make_detection(model, q, p_grasp, fid)
        cmd = servo.update(model.q_to_deg(q), grip, det, 1 / 30)
        if cmd is not None:
            q = model.deg_to_q([cmd[f"left_joint_{i}.pos"] for i in range(1, 8)])
            grip = cmd["left_gripper.pos"]
        if not phases or phases[-1] != servo.phase:
            phases.append(servo.phase)
        if servo.phase == Phase.CLOSE and q_desc is None:
            q_desc = q.copy()
        if servo.phase in (Phase.DONE, Phase.ABORTED):
            break
    assert servo.phase == Phase.DONE, (servo.status(), phases)
    r_desc, p_tcp = model.fk(q_desc)
    d = target.descent_vector / np.linalg.norm(target.descent_vector)
    err = p_tcp - (p_grasp + r_desc @ d * ServoConfig().extra_depth_m)
    assert np.linalg.norm(err) < 0.008, np.round(err * 1000, 1)
