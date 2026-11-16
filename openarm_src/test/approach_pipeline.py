# openarm_src/test/approach_pipeline.py
"""
最後一哩靠近流程：手臂如何靠近並夾起 5cm 方塊。

    [示教/計算 s*] -> [伺服 SERVO] -> [下降 DESCEND] -> [夾取 GRASP] -> [提起 LIFT]

- SERVO  ：閉迴路 IBVS，把 Tag 帶到期望特徵 s* = [u*, v*, w*, psi*]。
           w*（Tag 視邊長）對應「預備抓取距離」，所以收斂時手臂已在方塊正上方的固定高度、夾爪也轉正。
- DESCEND：太近時 Tag 會超出視野或失焦，所以最後一小段沿夾爪接近軸「開迴路」直線下降固定距離。
- GRASP / LIFT：閉合夾爪，由夾爪是否停在半路判斷有沒有夾到，再往回提起。

s* 有兩種來源：
- 示教 (taught)：把夾爪放在「閉合就能夾到」的位置，程式沿接近軸後退 d 再記錄影像特徵；
                 之後下降距離就是同一個 d。不需要手眼標定，最準。
- 手眼 (handeye)：由 config 的名義手眼外參推算。外參有誤差時，夾取位置會跟著偏。
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from arm_kinematics import HandEye, camera_twist_to_joint_delta, tcp_twist_to_joint_delta
from config import SystemConfig
from ibvs_pinn_controller import (
    IBVSPINNController,
    Intrinsics,
    action_to_camera_twist,
)
from tag_detector import AprilTagDetector, wrap_quarter_turn


@dataclass
class Observation:
    s: np.ndarray               # [u, v, w, psi]
    corners: np.ndarray         # (4, 2)
    intr: Intrinsics
    depth_pnp_m: float


@dataclass
class Target:
    s_star: np.ndarray
    descent_m: float            # 伺服收斂後要沿接近軸再前進多少
    source: str                 # 'taught' | 'handeye'

    def save(self, path: str, camera: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"s_star": [float(x) for x in self.s_star], "descent_m": float(self.descent_m),
                       "source": self.source, "camera": camera}, f, indent=2)

    @staticmethod
    def load(path: str) -> "Target":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return Target(np.array(data["s_star"], dtype=float), float(data["descent_m"]), data.get("source", "taught"))


@dataclass
class ServoResult:
    method: str
    success: bool
    reason: str
    steps: int
    final_error: np.ndarray
    final_error_norm: float
    error_curve: List[float] = field(default_factory=list)
    residual_used_pct: float = 0.0
    decrease_pct: float = 0.0
    mean_latency_ms: float = 0.0


@dataclass
class GraspResult:
    descended: bool
    holding: bool
    lifted: bool
    reason: str = ""
    sim_error: Dict[str, float] = field(default_factory=dict)


def target_from_handeye(cfg: SystemConfig, handeye: HandEye, intr: Intrinsics) -> Target:
    """由名義手眼外參推算 s* 與下降距離：TCP 接近軸要穿過 Tag 中心。"""
    depth = cfg.ibvs.standoff_depth_m
    point_cam, along_axis = handeye.point_on_approach_axis(depth)
    u = intr.cx + intr.f * point_cam[0] / point_cam[2]
    v = intr.cy + intr.f * point_cam[1] / point_cam[2]
    w = intr.f * cfg.target.tag_size_meters / depth
    # 讓 TCP 的 x 軸與方塊邊平行：TCP x 軸在影像中的方向 + 期望偏轉角
    x_in_cam = handeye.r.T @ np.array([1.0, 0.0, 0.0])
    psi = float(wrap_quarter_turn(np.degrees(np.arctan2(x_in_cam[1], x_in_cam[0])) + cfg.ibvs.desired_yaw_deg))
    half = w / 2.0 * 1.2 + cfg.ibvs.border_margin_px
    if not (half <= u <= intr.width - half and half <= v <= intr.height - half):
        raise ValueError(
            f"依手眼外參推得的期望位置 (u*={u:.0f}, v*={v:.0f}, w*={w:.0f}px) 會讓 Tag 超出畫面。"
            "請加大 ibvs.standoff_depth_m、修正 handeye 設定，或改用示教模式 (--mode teach)。"
        )
    return Target(np.array([u, v, w, psi]), along_axis + cfg.grasp.grasp_depth_below_tag_m, "handeye")


class LastMileApproach:
    def __init__(self, cfg: SystemConfig, robot, camera, detector: AprilTagDetector,
                 controller: IBVSPINNController, handeye: HandEye, camera_name: str):
        self.cfg = cfg
        self.robot = robot
        self.camera = camera
        self.detector = detector
        self.controller = controller
        self.handeye = handeye
        self.camera_name = camera_name

    # ------------------------------------------------------------------ 觀測
    def observe(self) -> Optional[Observation]:
        frame = self.camera.get_frames().get(self.camera_name)
        if frame is None or frame.camera_matrix is None:
            return None
        det = self.detector.detect(frame.color, frame.camera_matrix, frame.dist_coeffs,
                                   target_id=self.cfg.target.tag_id)
        if not det.detected:
            return None
        height, width = frame.color.shape[:2]
        return Observation(det.feature, det.corners.astype(float),
                           Intrinsics.from_matrix(frame.camera_matrix, width, height), det.depth_m)

    def observe_average(self, n_frames: int = 10) -> Optional[Observation]:
        """多幀平均，降低示教時的量測雜訊。"""
        samples = []
        for _ in range(n_frames * 3):
            obs = self.observe()
            if obs is not None:
                samples.append(obs)
            if len(samples) >= n_frames:
                break
        if len(samples) < max(n_frames // 2, 1):
            return None
        ref = samples[0]
        s = np.mean([o.s for o in samples], axis=0)
        s[3] = ref.s[3] + np.mean([wrap_quarter_turn(o.s[3] - ref.s[3]) for o in samples])
        return Observation(s, ref.corners, ref.intr, float(np.mean([o.depth_pnp_m for o in samples])))

    # ------------------------------------------------------------------ 動作
    def move_camera(self, action: np.ndarray) -> bool:
        """執行一步相機座標系命令 [mm, mm, mm, deg]。"""
        q = self.robot.get_joint_positions()
        dq = camera_twist_to_joint_delta(self.robot.model, q, action_to_camera_twist(action),
                                         self.handeye, self.cfg.robot.ik_damping)
        ok, _ = self.robot.step_joint_delta(dq)
        return ok

    def move_camera_total(self, total: np.ndarray) -> bool:
        """把一段較大的相機座標位移拆成多個小步執行（用來製造不同的初始狀態）。"""
        total = np.asarray(total, dtype=float)
        n = int(np.ceil(np.max(np.abs(total) / self.controller.limits)))
        for _ in range(max(n, 1)):
            if not self.move_camera(total / max(n, 1)):
                return False
            self.robot.wait_step()
        self.robot.settle()
        return True

    def move_along_approach(self, distance_m: float) -> bool:
        """沿夾爪接近軸開迴路直線移動；正值朝物體前進，負值後退。"""
        step_m = self.cfg.grasp.descend_step_mm / 1000.0
        n = max(int(np.ceil(abs(distance_m) / step_m)), 1)
        twist = np.concatenate([self.handeye.approach * (distance_m / n), np.zeros(3)])
        for _ in range(n):
            dq = tcp_twist_to_joint_delta(self.robot.model, self.robot.q_cmd, twist, self.cfg.robot.ik_damping)
            ok, _ = self.robot.step_joint_delta(dq)
            if not ok:
                return False
            self.robot.wait_step()
        self.robot.settle()
        return True

    # ------------------------------------------------------------------ 示教
    def snapshot(self, path: Optional[str] = None) -> Dict:
        """診斷用：抓一張影像，列出看到的所有 Tag，並把標註後的影像存檔。"""
        info: Dict = {"has_frame": False, "ids": [], "path": None, "size": None, "brightness": None}
        frame = self.camera.get_frames().get(self.camera_name)
        if frame is None or frame.color is None:
            return info
        ids, corners_list = self.detector.detect_all(frame.color)
        height, width = frame.color.shape[:2]
        info.update(has_frame=True, ids=ids, size=(width, height), brightness=float(np.mean(frame.color)))
        if path:
            image = frame.color.copy()
            cv2.drawMarker(image, (width // 2, height // 2), (255, 255, 0), cv2.MARKER_CROSS, 24, 2)
            for tag_id, corners in zip(ids, corners_list):
                pts = corners.reshape(-1, 2).astype(np.int32)
                cv2.polylines(image, [pts], True, (0, 255, 0), 2)
                cv2.putText(image, f"ID {tag_id}", (int(pts[0][0]), int(pts[0][1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            if cv2.imwrite(path, image):
                info["path"] = path
        return info

    def explain_no_tag(self, info: Dict) -> str:
        """把 snapshot 的結果翻成「為什麼看不到 Tag」的說明。"""
        want = self.cfg.target.tag_id
        if not info["has_frame"]:
            return f"相機 [{self.camera_name}] 沒有回傳影像：請確認 --serial 與 USB 連線。"
        lines = [f"影像 {info['size'][0]}x{info['size'][1]}，平均亮度 {info['brightness']:.0f}/255。"]
        if info["ids"] and want not in info["ids"]:
            lines.append(f"畫面中有 Tag ID {info['ids']}，但 config.py 的 tag_id 是 {want}：請改 TargetObjectConfig.tag_id。")
        elif not info["ids"]:
            lines.append("畫面中沒有偵測到任何 tag36h11。常見原因：Tag 不在畫面內或被夾爪擋住、距離太近而模糊、"
                         "Tag 黑框外沒有淺色邊（滿版貼在深色背景上）、畫面過暗或過曝、--serial 填到別台相機。")
        if info["path"]:
            lines.append(f"已存下當下畫面：{info['path']}（請打開確認相機實際看到什麼）")
        return "\n  ".join(lines)

    def teach(self, retreat_m: float, debug_dir: Optional[str] = None,
              ask_more: Optional[Callable[[float], Optional[float]]] = None) -> Target:
        """手臂目前在「閉合就能夾到」的位置：後退 retreat_m 後記錄 s*。

        看不到 Tag 時會存下畫面並說明原因；若提供 ask_more（回傳要再後退幾公尺，None 表示放棄），
        可以原地加大後退距離重試，不必重新擺位。
        """
        r0, p0 = self.robot.tcp_pose()
        total = retreat_m
        if not self.move_along_approach(-retreat_m):
            raise RuntimeError(f"示教後退失敗: {self.robot.last_block_reason}")
        while True:
            obs = self.observe_average(15)
            if obs is not None:
                return Target(obs.s, total, "taught")

            _, p1 = self.robot.tcp_pose()
            moved = p1 - p0
            along = -float((r0.T @ moved) @ self.handeye.approach)
            print(f"\n[Teach] 後退 {total * 100:.0f} cm 後看不到 Tag (ID {self.cfg.target.tag_id})。")
            print(f"  TCP 實際位移（基座座標）: dx {moved[0] * 100:+.1f}  dy {moved[1] * 100:+.1f}  "
                  f"dz {moved[2] * 100:+.1f} cm；沿接近軸後退 {along * 100:.1f} cm。")
            print("  夾爪朝下時，後退應該是 dz 為正（往上）。若方向不對，請修正 HandEyeConfig.approach_axis_tcp。")
            path = os.path.join(debug_dir, "teach_no_tag.png") if debug_dir else None
            print("  " + self.explain_no_tag(self.snapshot(path)))

            extra = ask_more(total) if ask_more is not None else None
            if not extra:
                raise RuntimeError("後退後看不到 Tag，示教未完成（原因見上方說明）")
            if total + extra > self.cfg.grasp.max_descent_m:
                print(f"  [注意] 總後退距離超過 grasp.max_descent_m ({self.cfg.grasp.max_descent_m * 100:.0f} cm)，"
                      "之後下夾會被拒絕，需一併調大該設定。")
            if not self.move_along_approach(-extra):
                raise RuntimeError(f"示教後退失敗: {self.robot.last_block_reason}")
            total += extra

    # ------------------------------------------------------------------ 伺服
    def servo(self, target: Target, mode: str = "guarded", verbose: bool = False,
              on_step: Optional[Callable[[int, Observation, np.ndarray], None]] = None) -> ServoResult:
        cfg = self.cfg.ibvs
        curve: List[float] = []
        used = decreases = moves = lost = settled = 0
        latency: List[float] = []
        last_e = np.full(4, np.nan)
        prev_norm: Optional[float] = None
        diverge_limit: Optional[float] = None
        self.controller.reset_episode()

        def finish(success: bool, reason: str) -> ServoResult:
            return ServoResult(
                method=mode, success=success, reason=reason, steps=len(curve),
                final_error=last_e, final_error_norm=curve[-1] if curve else float("nan"),
                error_curve=curve,
                residual_used_pct=100.0 * used / max(moves, 1),
                decrease_pct=100.0 * decreases / max(moves, 1),
                mean_latency_ms=float(np.mean(latency)) if latency else 0.0,
            )

        for step in range(cfg.max_steps_per_episode):
            t0 = time.perf_counter()
            obs = self.observe()
            if obs is None:
                lost += 1
                if lost > cfg.max_lost_frames:
                    return finish(False, f"連續 {lost} 幀看不到 Tag")
                self.robot.wait_step()      # 不送新命令，手臂保持原位
                continue
            lost = 0

            out = self.controller.compute_action(obs.s, target.s_star, obs.intr, mode, obs.corners)
            last_e = out.error
            curve.append(out.error_norm)
            if prev_norm is not None:
                decreases += int(out.error_norm < prev_norm)
            prev_norm = out.error_norm
            if diverge_limit is None:
                diverge_limit = out.error_norm * cfg.diverge_ratio + cfg.diverge_offset_px
            elif out.error_norm > diverge_limit:
                return finish(False, f"誤差發散（{out.error_norm:.0f} > {diverge_limit:.0f}），已停止")
            if on_step is not None:
                on_step(step, obs, out.error)
            if verbose and step % 10 == 0:
                e = out.error
                print(f"  [{mode}] step {step:3d}: 中心 {np.hypot(e[0], e[1]):6.1f}px  大小 {e[2]:+7.1f}px  "
                      f"偏轉 {e[3]:+6.1f}deg  深度 {out.depth_m * 100:5.1f}cm")

            if self.controller.is_converged(out.error):
                settled += 1
                if settled >= cfg.settle_frames:
                    return finish(True, "收斂")
            else:
                settled = 0

            ok = self.move_camera(out.action)
            latency.append((time.perf_counter() - t0) * 1000.0)
            if not ok:
                return finish(False, f"安全檢查擋下: {self.robot.last_block_reason}")
            moves += 1
            used += int(out.used_residual)
            self.robot.wait_step()

        return finish(False, f"{cfg.max_steps_per_episode} 步內未收斂")

    # ------------------------------------------------------------------ 下降/夾取/提起
    def descend_grasp_lift(self, target: Target, confirm: Optional[Callable[[str], bool]] = None) -> GraspResult:
        rcfg, gcfg = self.cfg.robot, self.cfg.grasp
        sim = self.robot.sim if self.robot.dry_run else None

        if target.descent_m <= 0 or target.descent_m > gcfg.max_descent_m:
            return GraspResult(False, False, False, f"下降距離 {target.descent_m:.3f} m 不合理，已取消")
        if confirm is not None and not confirm(f"即將沿接近軸下降 {target.descent_m * 100:.1f} cm 並閉合夾爪"):
            return GraspResult(False, False, False, "使用者取消")

        self.robot.settle()
        if not self.move_along_approach(target.descent_m):
            reason = f"下降被擋下: {self.robot.last_block_reason}"
            self.move_along_approach(-gcfg.lift_height_m)
            return GraspResult(False, False, False, reason)

        sim_error = sim.grasp_error() if sim is not None else {}
        final_deg = self.robot.set_gripper(rcfg.gripper_closed_deg)
        if sim is not None:
            holding = sim.try_grasp()
        else:
            holding = abs(final_deg - rcfg.gripper_closed_deg) > rcfg.gripper_hold_threshold_deg

        lifted = self.move_along_approach(-gcfg.lift_height_m)
        if sim is not None:
            height = sim.lifted_height_m()
            lifted = lifted and height is not None and height > 0.5 * gcfg.lift_height_m
        else:
            # 提起後夾爪仍停在半路，才算真的把方塊帶起來
            still = abs(self.robot.get_gripper_deg() - rcfg.gripper_closed_deg) > rcfg.gripper_hold_threshold_deg
            lifted = lifted and holding and still
        reason = "" if (holding and lifted) else ("夾爪完全閉合，沒有夾到方塊" if not holding else "提起時方塊掉落或動作被擋下")
        return GraspResult(True, holding, lifted, reason, sim_error)


def servo_result_row(res: ServoResult) -> Dict[str, float]:
    row = asdict(res)
    row.pop("error_curve")
    e = row.pop("final_error")
    row.update({"center_err_px": float(np.hypot(e[0], e[1])), "size_err_px": float(e[2]), "yaw_err_deg": float(e[3])})
    return row
