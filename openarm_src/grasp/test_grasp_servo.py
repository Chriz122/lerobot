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
# 真機 TCP 座標：x = 夾爪上方（側向夾取時朝上）、y = 兩指張開方向、z = 接近軸。相機裝在夾爪上方往下傾約 35°：
# 影像往右 = TCP +y、影像往下 ≈ TCP -x。真實相機再多轉 3°（安裝誤差），示教檔用名義值。
CAM_P = np.array([0.04, 0.0, -0.08])          # 相機在 TCP 座標的位置
CAM_R = rpy_to_matrix([-35.0, 0.0, 93.0])
APPROACH = np.array([0.0, 0.0, 1.0])
DESCENT = 0.10


def home_q(model: ArmModel) -> np.ndarray:
    with open(os.path.join(HERE, "..", "reset.json"), encoding="utf-8") as f:
        wp = json.load(f)["waypoints"][-1]
    return model.deg_to_q([wp[f"{model.side}_joint_{i}.pos"] for i in range(1, 8)])


def side_grasp_q(model: ArmModel) -> np.ndarray:
    """與真機相同的側向夾取姿態：夾爪水平朝前、TCP x 朝上（真機示教姿態 TCP x 的 z 分量約 0.96）、相機正立。"""
    q = home_q(model)
    _, p_des = model.fk(q)
    p_des = p_des - np.array([0.0, 0.0, 0.055])     # 真機示教高度 TCP z ≈ 0.39–0.40 m（起始姿態約 0.455 m）
    yaw = np.radians(0.0)                          # 真機示教時接近方向 ≈ 基座 +x
    a = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    x = np.array([0.0, 0.0, 1.0])
    r_des = np.column_stack([x, np.cross(a, x), a])
    for _ in range(400):
        r, p = model.fk(q)
        twist = np.concatenate([0.5 * (r.T @ (p_des - p)), 0.5 * model.orientation_error(q, r_des)])
        q = q + model.tcp_twist_to_dq(q, twist, damping=0.01)
    return q


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
    q_grasp = side_grasp_q(model)
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
    servo = ArmGraspServo("left", model, target, INTR, WH, ServoConfig(center_u=False))
    q0 = move_tcp(model, q_teach, np.array(offset))   # 起點：示教位置再偏移
    q, grip, q_desc = run(servo, model, q0, p_grasp, rng=np.random.default_rng(0), noise_px=0.5)
    assert servo.phase == Phase.DONE, servo.status()
    r_desc, p_tcp = model.fk(q_desc)
    # 下降會照示教路徑走回夾取點，再沿同一方向多深入 extra_depth_m
    d = target.descent_vector / np.linalg.norm(target.descent_vector)
    err = p_tcp - (p_grasp + r_desc @ d * ServoConfig(center_u=False).extra_depth_m)
    assert np.linalg.norm(err) < 0.008, np.round(err * 1000, 1)
    assert grip == pytest.approx(0.0)
    _, p_end = model.fk(q)
    assert p_end[2] - p_tcp[2] > ServoConfig().lift_m - 0.01   # 有提起


def test_servo_waits_until_bottle_is_close(setup):
    model, _, p_grasp, q_teach, target = setup
    servo = ArmGraspServo("left", model, target, INTR, WH, ServoConfig(trigger_max_depth_m=0.05, center_u=False))
    run(servo, model, q_teach, p_grasp, steps=60)
    assert servo.phase == Phase.IDLE                    # 太遠不接手


def test_observe_only_never_commands(setup):
    model, _, p_grasp, q_teach, target = setup
    servo = ArmGraspServo("left", model, target, INTR, WH, ServoConfig(center_u=False), observe_only=True)
    for k in range(30):
        assert servo.update(model.q_to_deg(q_teach), -20.0, make_detection(model, q_teach, p_grasp, k + 1),
                            1 / 30) is None


def test_aborts_and_hands_back_when_bottle_lost(setup):
    model, _, p_grasp, q_teach, target = setup
    servo = ArmGraspServo("left", model, target, INTR, WH, ServoConfig(lost_timeout_s=0.0, center_u=False))
    q = move_tcp(model, q_teach, np.array([0.0, -0.03, 0.0]))     # 瓶子偏一邊（TCP y = 左右）
    for k in range(5):
        cmd = servo.update(model.q_to_deg(q), -20.0, make_detection(model, q, p_grasp, k + 1), 1 / 30)
        if cmd is not None:
            q = model.deg_to_q([cmd[f"left_joint_{i}.pos"] for i in range(1, 8)])
    assert servo.active
    empty = Detection("wrist_left", 99, 0.0, [], None, (WH[1], WH[0]))
    # 中止後先往上抬 lift_m（夾爪維持張開），抬完才交回 ACT
    z0 = model.fk(q)[1][2]
    for k in range(300):
        cmd = servo.update(model.q_to_deg(q), -20.0, empty, 1 / 30)
        if cmd is None:
            break
        assert servo.aborting and servo.phase == Phase.LIFT
        q = model.deg_to_q([cmd[f"left_joint_{i}.pos"] for i in range(1, 8)])
    assert servo.phase == Phase.ABORTED and "跟丟" in servo.reason
    assert model.fk(q)[1][2] - z0 == pytest.approx(ServoConfig().lift_m, abs=0.004)


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
    servo = ArmGraspServo("left", model, target, INTR, WH, ServoConfig(center_u=False))
    q = move_tcp(model, q_teach, np.array([-0.3, 0.0, 0.0]))     # 往下 30 cm（TCP -x），瓶子在畫面之外
    assert not make_detection(model, q, p_grasp, 1).boxes
    run(servo, model, q, p_grasp, steps=60)
    assert servo.phase == Phase.IDLE


def test_waits_for_arm_to_reach_before_closing(setup):
    """手臂落後命令時（重力、追蹤落後），要等追上才閉合夾爪。"""
    model, _, p_grasp, q_teach, target = setup
    servo = ArmGraspServo("left", model, target, INTR, WH, ServoConfig(reach_timeout_s=5.0, center_u=False))
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
    servo = ArmGraspServo("left", model, target, INTR, WH, ServoConfig(center_u=False))
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
    err = p_tcp - (p_grasp + r_desc @ d * ServoConfig(center_u=False).extra_depth_m)
    assert np.linalg.norm(err) < 0.008, np.round(err * 1000, 1)


def test_center_u_aligns_bottle_to_image_center(setup):
    """預設 center_u：對準完成時瓶子框中心在畫面水平中央（兩指正中間）。"""
    model, _, p_grasp, q_teach, target = setup
    servo = ArmGraspServo("left", model, target, INTR, WH, ServoConfig())
    assert servo.u_star == pytest.approx(INTR.cx)
    # 把瓶子放到示教位置看起來「偏右 25 px」的地方（真機示教 u*=188 與畫面中心 212 也差約 25 px）
    r_t, _ = model.fk(q_teach)
    cam_x_w = (r_t @ CAM_R)[:, 0]
    u_now = make_detection(model, q_teach, p_grasp, 0).boxes[0].center[0]
    z = camera_point(model, q_teach, p_grasp)[2]
    p_grasp = p_grasp + cam_x_w * ((INTR.cx + 25.0) - u_now) / INTR.fx * z
    q, fid = move_tcp(model, q_teach, np.array([0.02, 0.0, 0.0])), 0
    for k in range(900):
        det = None
        if k % 2 == 0:
            fid += 1
            det = make_detection(model, q, p_grasp, fid)
        cmd = servo.update(model.q_to_deg(q), -20.0, det, 1 / 30)
        if cmd is not None:
            q = model.deg_to_q([cmd[f"left_joint_{i}.pos"] for i in range(1, 8)])
        if servo.phase == Phase.DESCEND:
            break
    assert servo.phase == Phase.DESCEND, servo.status()
    box = make_detection(model, q, p_grasp, 0).boxes[0]
    assert abs(box.center[0] - INTR.cx) < 3.0, box.center


def rotate_tcp(model, q, rpy_deg, n=60):
    """TCP 原地轉 rpy_deg（TCP 座標，R = Rz Ry Rx；TCP z = 接近軸，所以第三個值 = roll）。"""
    r0, p0 = model.fk(q)
    r_des = r0 @ rpy_to_matrix(rpy_deg)
    for _ in range(n):
        r, p = model.fk(q)
        twist = np.concatenate([0.5 * (r.T @ (p0 - p)), 0.5 * model.orientation_error(q, r_des)])
        q = q + model.tcp_twist_to_dq(q, twist, damping=1e-3)
    return q


@pytest.mark.parametrize("rpy", [[0.0, 0.0, 12.0], [8.0, 0.0, 0.0], [0.0, 15.0, 0.0], [5.0, -8.0, 10.0]])
def test_servo_corrects_roll_pitch_yaw(setup, rpy):
    """ACT 交出的姿態偏 roll / pitch / yaw：伺服時轉回示教的相對姿態，再沿示教路徑夾取。"""
    model, _, p_grasp, q_teach, target = setup
    import dataclasses
    tgt = dataclasses.replace(target, q_teach_deg=model.q_to_deg(q_teach))
    cfg = ServoConfig(center_u=False, axis_source="gravity")    # 合成相機的深度是整片常數，不能拿來擬合瓶軸
    servo = ArmGraspServo("left", model, tgt, INTR, WH, cfg)
    q0 = move_tcp(model, rotate_tcp(model, q_teach, rpy), np.array([0.02, 0.0, 0.0]))
    q, _, q_desc = run(servo, model, q0, p_grasp, steps=1500)
    assert servo.phase == Phase.DONE, servo.status()
    r_desc, p_tcp = model.fk(q_desc)
    r_teach, _ = model.fk(q_teach)
    up = np.array([0.0, 0.0, 1.0])
    # roll / pitch：瓶軸（垂直）在 TCP 座標回到示教時的方向
    assert np.degrees(np.arccos(np.clip((r_desc.T @ up) @ (r_teach.T @ up), -1, 1))) < 2.0
    d = tgt.descent_vector / np.linalg.norm(tgt.descent_vector)
    err = p_tcp - (p_grasp + r_desc @ d * cfg.extra_depth_m)
    assert np.linalg.norm(err) < 0.008, np.round(err * 1000, 1)


def test_orient_axes_empty_keeps_takeover_orientation(setup):
    """orient_axes="" = 舊行為：全程保持接手當下的姿態，只平移。"""
    model, _, p_grasp, q_teach, target = setup
    import dataclasses
    tgt = dataclasses.replace(target, q_teach_deg=model.q_to_deg(q_teach))
    servo = ArmGraspServo("left", model, tgt, INTR, WH, ServoConfig(center_u=False, orient_axes=""))
    q0 = rotate_tcp(model, q_teach, [0.0, 0.0, 6.0])
    _, _, q_desc = run(servo, model, q0, p_grasp, steps=1500)
    assert q_desc is not None, servo.status()
    assert rotation_angle_deg(model.fk(q0)[0].T @ model.fk(q_desc)[0]) < 1.0


def test_tracks_only_locked_bottle_not_nearby_distractor(setup):
    """20261123_193345 右臂：目標漏偵測的那幾幀，附近另一個框（像素上很近、實際上是別支瓶子）被當成目標，
    一路跳過去跟丟。鎖定基座座標的瓶子位置後，只追同一支瓶子。"""
    model, _, p_grasp, q_teach, target = setup
    cfg = ServoConfig(center_u=False, orient_axes="")
    servo = ArmGraspServo("left", model, target, INTR, WH, cfg)
    r_t, _ = model.fk(q_teach)
    up = np.array([0.0, 0.0, 1.0])
    x_h = r_t @ CAM_R[:, 0]
    x_h -= (x_h @ up) * up
    a_h = r_t @ APPROACH
    a_h -= (a_h @ up) * up
    # 別支瓶子：往右 3 cm、往後 8 cm（實際相距約 8.5 cm，畫面上卻在舊的 64 px 像素門檻內）
    p_other = p_grasp + 0.03 * x_h / np.linalg.norm(x_h) + 0.08 * a_h / np.linalg.norm(a_h)

    def detect(q, fid):
        d_t = make_detection(model, q, p_grasp, fid)
        d_o = make_detection(model, q, p_other, fid)
        depth = np.zeros((WH[1], WH[0]), np.float32)
        boxes = []
        if d_t.boxes:
            b = d_t.boxes[0]
            depth[int(b.y0):int(b.y1) + 1, int(b.x0):int(b.x1) + 1] = d_t.depth_m[0, 0]
            if fid % 3 or not servo.active:            # 接手後，目標每 3 幀漏偵測一次
                boxes.append(b)
        if d_o.boxes:
            b = d_o.boxes[0]
            c = b.center
            hw = b.size * 0.3 / 2 + 1
            depth[int(c[1] - hw[1]):int(c[1] + hw[1]) + 1, int(c[0] - hw[0]):int(c[0] + hw[0]) + 1] = \
                d_o.depth_m[0, 0]
            boxes.append(Box(b.x0, b.y0, b.x1, b.y1, 0.6, label=1))   # 一般 "bottle"，觸發時不認
        return Detection("wrist_left", fid, 0.0, boxes, depth, (WH[1], WH[0]))

    q = move_tcp(model, q_teach, np.array([0.03, 0.0, 0.0]))
    uv_t, _ = project(model, q, p_grasp)
    uv_o, _ = project(model, q, p_other)
    assert np.linalg.norm(uv_t - uv_o) < 0.15 * WH[0]   # 確實落在舊的像素門檻內
    grip, fid, q_desc = -20.0, 0, None
    for k in range(900):
        det = None
        if k % 2 == 0:
            fid += 1
            det = detect(q, fid)
        cmd = servo.update(model.q_to_deg(q), grip, det, 1.0 / 30.0)
        if cmd is not None:
            q = model.deg_to_q([cmd[f"left_joint_{i}.pos"] for i in range(1, 8)])
            grip = cmd["left_gripper.pos"]
        if servo.phase == Phase.CLOSE and q_desc is None:
            q_desc = q.copy()
        if servo.phase in (Phase.DONE, Phase.ABORTED):
            break
    assert servo.phase == Phase.DONE, servo.status()
    r_desc, p_tcp = model.fk(q_desc)
    d = target.descent_vector / np.linalg.norm(target.descent_vector)
    err = p_tcp - (p_grasp + r_desc @ d * cfg.extra_depth_m)
    assert np.linalg.norm(err) < 0.008, np.round(err * 1000, 1)
    assert all(r.get("lock_d_mm") is None or not r.get("locked") or r["lock_d_mm"] < 40 for r in servo.trace)


@pytest.mark.parametrize("side_off", [0.04, -0.04])
def test_nominal_handeye_centers_sideways_offset(setup, side_off):
    """20261123_194426 右臂：舊名義手眼 -35 0 0 把「影像往右」當成 TCP x（真機上朝上），左右對準變成前後移動，
    瓶子一直偏在畫面右邊、夾爪往旁邊跑。示教檔改用名義值 -35 0 90（真實相機多轉 3°）、全部預設參數，
    瓶子偏左 / 偏右 4 cm 都要能置中並夾到。"""
    from openarm_src.grasp.kinematics import rpy_to_matrix
    from openarm_src.grasp.servo import NOMINAL_CAM_P_TCP, NOMINAL_HANDEYE_RPY

    model, q_grasp, p_grasp, q_teach, _ = setup
    r_nom, p_nom = rpy_to_matrix(NOMINAL_HANDEYE_RPY), np.array(NOMINAL_CAM_P_TCP)
    det = make_detection(model, q_teach, p_grasp, 0)
    (u, v), z = det.boxes[0].center, float(det.depth_m[0, 0])
    r_t, p_t = model.fk(q_teach)
    r_g, _ = model.fk(q_grasp)
    target = TaughtTarget(u, v, z, WH, r_nom, float((r_g.T @ (p_grasp - p_t)) @ APPROACH), APPROACH, "bottle",
                          bottle_in_tcp(INTR, r_nom, p_nom, u, v, z), p_nom, float(p_t[2]), r_t.T @ (p_grasp - p_t),
                          model.q_to_deg(q_teach))
    cfg = ServoConfig()
    servo = ArmGraspServo("left", model, target, INTR, WH, cfg)
    q0 = move_tcp(model, q_teach, np.array([0.0, side_off, 0.0]))     # TCP y = 左右
    _, _, q_desc = run(servo, model, q0, p_grasp, rng=np.random.default_rng(0), noise_px=0.5)
    assert q_desc is not None, servo.status()
    r_desc, p_tcp = model.fk(q_desc)
    d = target.descent_vector / np.linalg.norm(target.descent_vector)
    err = p_tcp - (p_grasp + r_desc @ d * cfg.extra_depth_m)
    assert np.linalg.norm(err) < 0.008, np.round(err * 1000, 1)


def test_warns_when_handeye_swaps_left_right(setup, caplog):
    """舊名義手眼 -35 0 0 在真機姿態下會被偵測出來並警告。"""
    from openarm_src.grasp.kinematics import rpy_to_matrix

    model, _, _, q_teach, target = setup
    bad = TaughtTarget(**{**target.__dict__, "r_tcp_cam": rpy_to_matrix([-35.0, 0.0, 0.0]),
                          "q_teach_deg": model.q_to_deg(q_teach)})
    with caplog.at_level("WARNING"):
        ArmGraspServo("left", model, bad, INTR, WH, ServoConfig())
    assert "手眼旋轉不對" in caplog.text
    caplog.clear()
    good = TaughtTarget(**{**target.__dict__, "q_teach_deg": model.q_to_deg(q_teach)})
    with caplog.at_level("WARNING"):
        ArmGraspServo("left", model, good, INTR, WH, ServoConfig())
    assert "手眼旋轉不對" not in caplog.text
