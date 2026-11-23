"""姿態控制的純數學測試（不需 pinocchio / URDF）：瓶軸深度擬合、對準旋轉的方向。

執行：PYTHONPATH=$HOME/lerobot pytest openarm_src/grasp/test_orientation.py
"""
import numpy as np
import pytest

from openarm_src.grasp.detector import Box
from openarm_src.grasp.kinematics import rot_axis, rpy_to_matrix
from openarm_src.grasp.servo import Intrinsics, axis_alignment_rotation, bottle_axis_cam, wrap_angle

INTR = Intrinsics(215.0, 215.0, 212.0, 120.0)
WH = (424, 240)
R_B, H_B = 0.033, 0.22


def render_cylinder(center_cam: np.ndarray, axis_cam: np.ndarray):
    """相機座標下的圓柱（中心、軸向）-> (外接框, 深度圖 z)。"""
    us, vs = np.meshgrid(np.arange(WH[0]), np.arange(WH[1]))
    d = np.stack([(us - INTR.cx) / INTR.fx, (vs - INTR.cy) / INTR.fy, np.ones(us.shape)], -1)
    a = axis_cam / np.linalg.norm(axis_cam)
    o = -center_cam
    dp = d - (d @ a)[..., None] * a
    op = o - (o @ a) * a
    qa, qb, qc = (dp * dp).sum(-1), 2 * (dp @ op), op @ op - R_B ** 2
    disc = qb * qb - 4 * qa * qc
    t = np.where(disc > 0, (-qb - np.sqrt(np.maximum(disc, 0))) / (2 * qa), np.inf)
    hit = (disc > 0) & (t > 0) & (np.abs(o @ a + t * (d @ a)) < H_B / 2)
    depth = np.where(hit, t, 0.0).astype(np.float32)
    ys, xs = np.nonzero(hit)
    return Box(float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()), 0.5), depth


# 相機朝基座 +x 看（相機 x = -y、y = -z、z = +x），再繞相機 x 軸往下傾 35°
R_WORLD_CAM = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]) @ rpy_to_matrix([-35.0, 0.0, 0.0])
UP_CAM = R_WORLD_CAM.T @ np.array([0.0, 0.0, 1.0])            # 重力反方向（擬合初值）在相機座標


def add_noise(depth: np.ndarray, noise_m: float) -> np.ndarray:
    noise = np.random.default_rng(0).normal(0.0, noise_m, depth.shape)
    return np.where(depth > 0, depth + noise, 0.0).astype(np.float32)


@pytest.mark.parametrize("dist_m", [0.09, 0.15])
@pytest.mark.parametrize("noise_m", [0.0, 0.001])
@pytest.mark.parametrize("tilt_deg", [0.0, 8.0, -12.0, 20.0])
def test_bottle_axis_from_depth(tilt_deg, noise_m, dist_m):
    """相機往下傾 35°、近距離（瓶子上下被切掉）、1 mm 深度雜訊下，瓶軸誤差 < 0.5°（初值 = 重力，與真值差 tilt）。"""
    assert (R_WORLD_CAM @ np.array([0.0, 0.0, 1.0]))[2] < -0.5      # 光軸確實往下
    axis_w = rot_axis(np.array([1.0, 0.0, 0.0]), np.radians(tilt_deg)) @ np.array([0.0, 0.0, 1.0])
    axis_c = R_WORLD_CAM.T @ axis_w
    box, depth = render_cylinder(np.array([0.0, 0.0, dist_m]), axis_c)
    if noise_m:
        depth = add_noise(depth, noise_m)
    est = bottle_axis_cam(depth, box, INTR, float(np.median(depth[depth > 0])), UP_CAM)
    assert est is not None
    assert est @ UP_CAM > 0                                        # 與初值同向
    ang = np.degrees(np.arccos(min(1.0, float(est @ axis_c))))
    assert ang < 0.5, ang


def test_bottle_axis_ignores_background_wall():
    """框裡混到瓶子後方 4 cm 的平面（在深度帶內）時，仍量得到瓶軸。"""
    axis_c = UP_CAM
    box, depth = render_cylinder(np.array([0.0, 0.0, 0.12]), axis_c)
    # 平面：垂直於相機 z，在 z = 0.12 + 0.033 + 0.04 處，只填瓶子以外的像素
    depth = np.where(depth > 0, depth, 0.193).astype(np.float32)
    est = bottle_axis_cam(add_noise(depth, 0.0005), box, INTR, 0.12, UP_CAM, min_inlier_frac=0.4)
    assert est is not None
    assert np.degrees(np.arccos(min(1.0, float(est @ axis_c)))) < 1.0


def test_bottle_axis_rejects_plane():
    """框裡只有平面（沒有瓶子）時不採用。"""
    depth = np.full((WH[1], WH[0]), 0.12, np.float32)
    assert bottle_axis_cam(add_noise(depth, 0.0005), Box(150, 20, 260, 200, 0.5), INTR, 0.12, UP_CAM) is None


def test_bottle_axis_none_without_depth():
    depth = np.zeros((WH[1], WH[0]), np.float32)
    assert bottle_axis_cam(depth, Box(100, 20, 200, 200, 0.5), INTR, 0.1, UP_CAM) is None


def test_bottle_axis_is_fast_enough():
    """每次偵測都在控制迴圈裡跑，要遠小於一個控制週期（33 ms）。"""
    import time

    box, depth = render_cylinder(np.array([0.0, 0.0, 0.10]), UP_CAM)
    depth = add_noise(depth, 0.0005)
    t0 = time.perf_counter()
    for _ in range(10):
        bottle_axis_cam(depth, box, INTR, 0.10, UP_CAM)
    assert (time.perf_counter() - t0) / 10 < 0.015


def test_alignment_rotation_brings_axis_to_target():
    """依 axis_alignment_rotation 轉 TCP 後，世界固定的瓶軸在 TCP 座標等於目標。"""
    rng = np.random.default_rng(0)
    for _ in range(20):
        r_tcp = rpy_to_matrix(rng.uniform(-60.0, 60.0, 3))
        up_w = np.array([0.0, 0.0, 1.0])
        star = rot_axis(np.array([1.0, 0.0, 0.0]), rng.uniform(-0.3, 0.3)) @ (r_tcp.T @ up_w)
        w = axis_alignment_rotation(star, r_tcp.T @ up_w)
        ang = np.linalg.norm(w)
        r_new = r_tcp @ rot_axis(w / ang, ang) if ang > 1e-12 else r_tcp
        assert np.allclose(r_new.T @ up_w, star, atol=1e-9)


def test_wrap_angle():
    assert wrap_angle(np.radians(190)) == pytest.approx(np.radians(-170))
    assert wrap_angle(-np.radians(190)) == pytest.approx(np.radians(170))
