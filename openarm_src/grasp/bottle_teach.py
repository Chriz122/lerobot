#!/usr/bin/env python3
"""
bottle_teach.py — 為一支手臂示教瓶子夾取目標。產生 teach_<side>.json。

流程：
  1. 從臂連線（腕部 D405 開彩色 + 深度），主臂連線開主從控制
  2. 用主臂把「張開的夾爪」擺到「一閉合就能夾到瓶子」的位置，按 Enter
  3. 沿夾爪接近軸後退 --retreat，用 OWLv2 + 深度記錄瓶子位置與 TCP 高度；下降距離用 FK 量到的實際後退量
  4. 手眼旋轉：預設直接用名義值 --handeye-rpy -35 0 90（相機在夾爪上方往下傾約 35°；影像往右 = TCP +y），不需要校正。
     伺服的目標是「示教時相機看到的樣子」，手眼角度不準會在示教與執行之間抵銷（模擬中假設 0°～-45° 夾取誤差都 < 3 mm）。
     加 --measure-handeye 才會實際量測（TCP 沿 x / y / z 各前後移動 --calib-step）。
  5. 存檔；結束時從臂慢慢回下垂再關力矩

用法（在 ~/lerobot，conda activate lerobot）：
  export PYTHONPATH=$HOME/lerobot
  python -m openarm_src.grasp.bottle_teach --side left
  python -m openarm_src.grasp.bottle_teach --side right --query "cola bottle"

解析度要與 rollout 的 --robot.cameras 相同（預設 424x240）。
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time

import numpy as np

from openarm_src.grasp.detector import BottleDetector, Detection, box_depth_m
from openarm_src.grasp.kinematics import (
    ARM_JOINTS,
    ArmModel,
    kabsch,
    matrix_to_rpy,
    rotation_angle_deg,
    rpy_to_matrix,
)
from openarm_src.grasp.servo import NOMINAL_CAM_P_TCP, NOMINAL_HANDEYE_RPY, Intrinsics, TaughtTarget, bottle_in_tcp
from openarm_src.utils import config as oa

WRIST_SERIAL = {"left": "353322271140", "right": "353322271542"}
HERE = os.path.dirname(os.path.abspath(__file__))


class Arm:
    """單臂從臂 + 腕部相機（彩色 + 深度）。"""

    def __init__(self, side: str, port: str, serial: str, width: int, height: int):
        from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
        from lerobot.robots.openarm_follower import OpenArmFollower, OpenArmFollowerConfig
        from openarm_src.utils.patches import patch_follower_park

        patch_follower_park()   # 中斷連線前先慢慢回下垂再關力矩
        cam = RealSenseCameraConfig(serial_number_or_name=serial, width=width, height=height, fps=30, use_depth=True)
        self.side = side
        self.model = ArmModel(side)
        self.arm = OpenArmFollower(OpenArmFollowerConfig(port=port, side=side, id=f"openarm_{side}_follower",
                                                         cameras={"wrist": cam}))
        print(f"[Teach] 連線 {side} 從臂 (port={port})、腕部相機 {serial} ...")
        self.arm.connect()
        self.cam = self.arm.cameras["wrist"]
        import pyrealsense2 as rs

        i = self.cam.rs_profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.intr = Intrinsics(i.fx, i.fy, i.ppx, i.ppy)
        self.depth_scale = self.cam.rs_profile.get_device().first_depth_sensor().get_depth_scale()
        self.wh = (self.cam.width, self.cam.height)

    def joints_deg(self) -> np.ndarray:
        obs = self.arm.get_observation()
        return np.array([float(obs[f"{m}.pos"]) for m in ARM_JOINTS])

    def frame(self):
        rgb = self.cam.read_latest()
        depth = self.cam.read_latest_depth().squeeze(-1).astype(np.float32) * self.depth_scale
        return rgb, depth

    def hold(self) -> None:
        obs = self.arm.get_observation()
        self.arm.send_action({k: float(v) for k, v in obs.items() if k.endswith(".pos")})

    def move_tcp(self, delta_tcp: np.ndarray, speed: float = 0.015, rate: float = 30.0) -> None:
        """TCP 座標純平移（開迴路，小步送出），結束後等手臂靜止。"""
        q = self.model.deg_to_q(self.joints_deg())
        n = max(int(np.ceil(np.linalg.norm(delta_tcp) / (speed / rate))), 1)
        grip = float(self.arm.get_observation()["gripper.pos"])
        for _ in range(n):
            q = q + self.model.tcp_translation_to_dq(q, delta_tcp / n)
            deg = self.model.clip_deg(self.model.q_to_deg(q))
            q = self.model.deg_to_q(deg)
            self.arm.send_action({**{f"{m}.pos": float(deg[i]) for i, m in enumerate(ARM_JOINTS)},
                                  "gripper.pos": grip})
            time.sleep(1.0 / rate)
        self.settle(deg)

    def settle(self, target_deg: np.ndarray, tol_deg: float = 0.4, still_deg: float = 0.05,
               timeout_s: float = 3.0) -> None:
        """等到手臂接近命令角、而且不再移動（有重力時可能停在命令角附近，以「靜止」為準）。"""
        t0, prev = time.perf_counter(), self.joints_deg()
        while time.perf_counter() - t0 < timeout_s:
            time.sleep(0.1)
            now = self.joints_deg()
            if np.max(np.abs(now - prev)) < still_deg and (
                    np.max(np.abs(now - target_deg)) < tol_deg or time.perf_counter() - t0 > 0.8):
                break
            prev = now
        time.sleep(0.2)

    def move_joints(self, q_target: np.ndarray, duration_s: float = 3.0, rate: float = 30.0) -> None:
        start = self.joints_deg()
        goal = self.model.clip_deg(self.model.q_to_deg(q_target))
        grip = float(self.arm.get_observation()["gripper.pos"])
        n = max(int(duration_s * rate), 1)
        for k in range(1, n + 1):
            x = k / n
            s = 10 * x**3 - 15 * x**4 + 6 * x**5
            deg = start + s * (goal - start)
            self.arm.send_action({**{f"{m}.pos": float(deg[i]) for i, m in enumerate(ARM_JOINTS)}, "gripper.pos": grip})
            time.sleep(1.0 / rate)
        self.settle(goal)

    def tcp_pose(self):
        return self.model.fk(self.model.deg_to_q(self.joints_deg()))

    def disconnect(self) -> None:
        self.arm.disconnect()


def teleop_until_enter(arm: Arm, leader_port: str, purpose: str, rate_hz: float = 30.0) -> None:
    from lerobot.teleoperators.openarm_leader import OpenArmLeader, OpenArmLeaderConfig
    from openarm_src.utils.patches import install_patches
    from openarm_src.utils.step_guard import StepJumpGuard

    install_patches()
    leader = OpenArmLeader(OpenArmLeaderConfig(port=leader_port, id=f"openarm_{arm.side}_leader"))
    leader._gc_side = arm.side
    input(f"\n[Teleop] 主臂 {arm.side} (port={leader_port})：請先讓主臂自然下垂，按 Enter 連線: ")
    leader.connect()
    try:
        motors = list(ARM_JOINTS) + ["gripper"]
        obs = arm.arm.get_observation()
        pose = {m: float(obs[f"{m}.pos"]) for m in motors}
        print("[Teleop] 主臂 4 秒內走到從臂目前姿態，請勿阻擋。")
        leader._gc.move_to(pose, 4.0)
        time.sleep(0.3)
        leader._gc.set_target(None)
        guard = StepJumpGuard(enabled=True)
        guard.sync_from_observation({f"{m}.pos": v for m, v in pose.items()})
        stop = threading.Event()
        threading.Thread(target=lambda: (input(), stop.set()), daemon=True).start()
        print(f"[Teleop] 主從控制已開啟：請用主臂{purpose}，完成後按 Enter。")
        while not stop.is_set():
            t0 = time.perf_counter()
            action = {k: float(v) for k, v in leader.get_action().items() if k.endswith(".pos") and v is not None}
            arm.arm.send_action(guard.clamp(action))
            time.sleep(max(1.0 / rate_hz - (time.perf_counter() - t0), 0.0))
        time.sleep(0.3)
        arm.hold()
        print("[Teleop] 結束，從臂保持目前姿態；主臂回下垂後關力矩。")
    finally:
        leader.disconnect()


BORDER_PX = 4
MAX_BOX_FRAC = 0.6


def box_ok(b, wh) -> bool:
    """框完整在畫面內、且不會大到塞滿畫面（量手眼旋轉用：被切到的框，中心不會跟著瓶子移動）。"""
    w, h = wh
    inside = b.x0 > BORDER_PX and b.y0 > BORDER_PX and b.x1 < w - 1 - BORDER_PX and b.y1 < h - 1 - BORDER_PX
    return inside and b.size[0] < MAX_BOX_FRAC * w and b.size[1] < MAX_BOX_FRAC * h


def center_ok(b, wh) -> bool:
    """記錄 s* 用：框可以被畫面切到（近距離時直立瓶子幾乎一定超出畫面），只要中心在畫面內。"""
    c = b.center
    return BORDER_PX < c[0] < wh[0] - 1 - BORDER_PX and BORDER_PX < c[1] < wh[1] - 1 - BORDER_PX


def save_debug(arm: Arm, det: BottleDetector, name: str) -> str:
    import cv2

    rgb, _ = arm.frame()
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for b in det.detect({"wrist": rgb})["wrist"]:
        col = (0, 255, 0) if box_ok(b, arm.wh) else (0, 0, 255)
        cv2.rectangle(img, (int(b.x0), int(b.y0)), (int(b.x1), int(b.y1)), col, 2)
        cv2.putText(img, f"{b.score:.2f}", (int(b.x0) + 3, int(b.y0) + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    path = os.path.join(HERE, f"teach_debug_{arm.side}_{name}.png")
    cv2.imwrite(path, img)
    return path


def measure(arm: Arm, det: BottleDetector, n: int = 8, near: np.ndarray | None = None, full: bool = True):
    """多幀量測瓶子：回傳 (u, v, Z, 相機座標 3D 點)；看不到回傳 None。full=True 時只接受完整的框。"""
    samples = []
    for k in range(n * 3):
        rgb, depth = arm.frame()
        boxes = det.detect({"wrist": rgb})["wrist"]
        d = Detection("wrist", k, time.perf_counter(), boxes, depth, rgb.shape[:2])
        ref = near if near is not None else np.array(arm.wh, dtype=float) / 2.0
        ok = box_ok if full else center_ok
        cands = [(b, box_depth_m(d, b)) for b in boxes if b.score >= 0.2 and ok(b, arm.wh)]
        cands = [(b, z) for b, z in cands if z is not None]
        if cands:
            b, z = min(cands, key=lambda bz: np.linalg.norm(bz[0].center - ref))
            samples.append((b.center[0], b.center[1], z))
        if len(samples) >= n:
            break
        time.sleep(0.03)
    if len(samples) < max(n // 2, 1):
        return None
    u, v, z = np.median(np.array(samples), axis=0)
    return float(u), float(v), float(z), arm.intr.backproject(u, v, z)


def calibrate(arm: Arm, det: BottleDetector, approach: np.ndarray, step: float):
    """TCP 沿 x / y / z 各前後平移 step，量瓶子在相機座標的位移，Kabsch 解相機 -> TCP 旋轉。

    需要完整的瓶子框（被切到的框中心不會跟著動）：看不到完整瓶子就沿接近軸再退，最多 15 cm。
    手眼旋轉與手臂姿態無關，所以在較遠處量的結果可以直接用在示教位置。
    """
    extra = 0.0
    while measure(arm, det, n=5, full=True) is None:
        if extra >= 0.15:
            path = save_debug(arm, det, "calib")
            sys.exit(f"[Teach] 再退 {extra * 100:.0f} cm 仍看不到完整的瓶子（標註圖：{path}），無法量測手眼旋轉。"
                     "可改用名義值（不加 --measure-handeye，預設 --handeye-rpy -35 0 90）")
        arm.move_tcp(-approach * 0.02)
        extra += 0.02
    if extra:
        print(f"[Teach] 為了看到完整的瓶子，量測時再後退 {extra * 100:.0f} cm")
    input(f"\n[確認] 即將讓 TCP 沿 x / y / z 各前後移動 {step * 100:.1f} cm（6 次，每次回原位）。按 Enter: ")
    pairs = []
    for axis, name in enumerate("xyz"):
        for sign in (1.0, -1.0):
            delta = np.zeros(3)
            delta[axis] = sign * step
            label = f"TCP {'+' if sign > 0 else '-'}{name}"
            before = measure(arm, det)
            r0, p0 = arm.tcp_pose()
            arm.move_tcp(delta)
            r1, p1 = arm.tcp_pose()
            after = measure(arm, det, near=None if before is None else np.array(before[:2]))
            arm.move_tcp(-delta)
            if before is None or after is None:
                print(f"  [{label}] 看不到完整的瓶子，略過（標註圖：{save_debug(arm, det, label[-2:])}）")
                continue
            d_tcp = r0.T @ (p1 - p0)
            d_cam = after[3] - before[3]
            pairs.append((d_tcp, -d_cam))
            print(f"  [{label}] TCP 位移 {np.round(d_tcp * 1000, 1)} mm -> 瓶子相機座標位移 {np.round(d_cam * 1000, 1)} mm"
                  f"（TCP 轉了 {rotation_angle_deg(r0.T @ r1):.1f} 度）")
    if len(pairs) < 3 or np.linalg.matrix_rank(np.array([p[0] for p in pairs]), tol=step * 0.3) < 2:
        sys.exit(f"[Teach] 有效樣本只有 {len(pairs)} 筆（需要至少 3 筆、兩個以上方向），無法量測手眼旋轉。"
                 "可改用名義值（不加 --measure-handeye，預設 --handeye-rpy -35 0 90）")
    A = np.array([p[0] for p in pairs])
    B = np.array([p[1] for p in pairs])
    q = kabsch(A, B)
    resid = float(np.mean([np.degrees(np.arccos(np.clip(np.dot(q @ ai, bi) / (np.linalg.norm(ai) * np.linalg.norm(bi)),
                                                         -1, 1))) for ai, bi in zip(A, B)]))
    scale = float(np.median(np.linalg.norm(B, axis=1) / np.linalg.norm(A, axis=1)))
    print(f"  方向殘差平均 {resid:.1f} 度（< 5 佳，> 10 建議重做）；深度位移 / 手臂位移 = {scale:.2f}（應接近 1）")
    return q.T, {"handeye_source": "measured", "residual_deg": resid, "scale_ratio": scale, "n_samples": len(pairs)}


def main() -> None:
    ap = argparse.ArgumentParser(description="示教瓶子夾取目標")
    ap.add_argument("--side", choices=["left", "right"], required=True)
    ap.add_argument("--port", default=None)
    ap.add_argument("--leader-port", default=None)
    ap.add_argument("--serial", default=None, help="腕部 D405 序號（預設依 --side）")
    ap.add_argument("--width", type=int, default=424)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--query", default="bottle", help="OWLv2 查詢文字，例如 'blue bottle'、'cola bottle'")
    ap.add_argument("--retreat", type=float, default=0.10, help="示教位置沿接近軸後退多少 (m)，之後下降實際後退的距離")
    ap.add_argument("--handeye-rpy", type=float, nargs=3, default=list(NOMINAL_HANDEYE_RPY),
                    help="相機相對 TCP 的旋轉 (deg, R = Rz Ry Rx)，預設名義值 -35 0 90（不需校正）")
    ap.add_argument("--measure-handeye", action="store_true",
                    help="實際量測手眼旋轉（TCP 沿 x/y/z 各移動），取代 --handeye-rpy；一般不需要")
    ap.add_argument("--calib-step", type=float, default=0.015, help="手眼量測時每次移動距離 (m)")
    ap.add_argument("--approach", type=float, nargs=3, default=[0.0, 0.0, 1.0], help="TCP 座標中夾爪前進方向")
    ap.add_argument("--out", default=None, help="輸出檔（預設 openarm_src/grasp/teach_<side>.json）")
    a = ap.parse_args()

    side = a.side
    port = a.port or (oa.DEFAULT_FOLLOWER_LEFT_PORT if side == "left" else oa.DEFAULT_FOLLOWER_RIGHT_PORT)
    leader_port = a.leader_port or (oa.DEFAULT_LEADER_LEFT_PORT if side == "left" else oa.DEFAULT_LEADER_RIGHT_PORT)
    out = a.out or os.path.join(HERE, f"teach_{side}.json")
    approach = np.array(a.approach, dtype=float)
    approach /= np.linalg.norm(approach)

    print("[Teach] 載入 OWLv2 ...")
    det = BottleDetector({"wrist": a.query})
    arm = Arm(side, port, a.serial or WRIST_SERIAL[side], a.width, a.height)
    try:
        teleop_until_enter(arm, leader_port, "把「張開的夾爪」擺到一閉合就能夾到瓶子的位置")
        r_g, p_g = arm.tcp_pose()                     # 夾取位姿
        input(f"\n[確認] 即將沿接近軸後退 {a.retreat * 100:.0f} cm 並記錄瓶子位置。手放急停旁，按 Enter: ")
        arm.move_tcp(-approach * a.retreat)
        while True:
            base = measure(arm, det, n=10, full=False)
            if base is not None:
                break
            path = save_debug(arm, det, "retreat")
            print(f"\n[Teach] 找不到瓶子（或沒有深度）。標註圖：{path}。常見原因：太近塞滿畫面、--query 不對。")
            ans = input("  要再沿接近軸後退幾公分？（Enter 放棄）: ").strip()
            if not ans:
                sys.exit("[Teach] 已放棄")
            arm.move_tcp(-approach * float(ans) / 100.0)
        u, v, z, _ = base
        q_teach = arm.model.deg_to_q(arm.joints_deg())
        r_t, p_t = arm.model.fk(q_teach)
        # 下降距離用「實際」後退量：手臂有重力、追蹤落後時，實際會比指令少
        back = r_g.T @ (p_g - p_t)
        descent = float(back @ approach)
        side_err = float(np.linalg.norm(back - descent * approach))
        print(f"[Teach] s* = (u {u:.1f}, v {v:.1f}) px、深度 {z * 100:.1f} cm；實際後退 {descent * 100:.1f} cm"
              f"（偏離接近軸 {side_err * 1000:.0f} mm，TCP 轉了 {rotation_angle_deg(r_g.T @ r_t):.1f} 度）")
        if side_err > 0.01:
            print("  [注意] 後退時偏離接近軸超過 1 cm（關節碰到限位或手臂被擋住）。下降會照實際路徑走回夾取位姿，"
                  "但請確認這條路徑不會撞到瓶子。")
        descent_tcp = r_t.T @ (p_g - p_t)            # 示教位置 -> 夾取位姿，示教位置的 TCP 座標

        if a.measure_handeye:
            r, meta = calibrate(arm, det, approach, a.calib_step)
            print("[Teach] 回到示教位置 ...")
            arm.move_joints(q_teach)
        else:
            r = rpy_to_matrix(a.handeye_rpy)
            meta = {"handeye_source": "manual"}
            print(f"[Teach] 手眼旋轉用名義值 {a.handeye_rpy}（不需校正）")
        tilt = float(np.degrees(np.arccos(np.clip(r[:, 2] @ approach, -1, 1))))
        print(f"[Teach] 手眼旋轉 rpy = {np.round(matrix_to_rpy(r), 1)} deg；相機光軸與接近軸夾 {tilt:.0f} 度")

        cam_p = np.array(NOMINAL_CAM_P_TCP)
        target = TaughtTarget(u, v, z, arm.wh, r, descent, approach, a.query,
                              bottle_in_tcp(arm.intr, r, cam_p, u, v, z), cam_p, float(p_t[2]), descent_tcp)
        target.save(out, {"side": side, "optical_vs_approach_deg": tilt, "q_teach_deg": arm.joints_deg().tolist(),
                          "time": time.strftime("%Y-%m-%d %H:%M:%S"), **meta})
        print(f"[Teach] 已儲存 {out}")
    finally:
        arm.disconnect()


if __name__ == "__main__":
    main()
