"""
單臂「看到瓶子就接手」的 IBVS 夾取狀態機。

    IDLE ──(瓶子連續 N 次出現在觸發距離內)──> SERVO ──(3D 誤差收斂)──> DESCEND
      ^                                                                  │
      └──── 交回 ACT（DONE / ABORTED，每次 rollout 只接手一次）<── LIFT <── CLOSE

特徵：OWLv2 框中心 + D405 框內深度，經手眼外參與 FK 換成瓶子在基座座標的 3D 點 B。
目標：示教時「瓶子相對夾爪」的位置 b*（TCP 座標，夾爪在可夾位置沿接近軸後退 descent_m 時量到）。
控制：水平方向用影像 e_xy = 水平[(B - p_tcp) - R_tcp b*]；高度用運動學 e_z = z*_tcp - z_tcp
      （瓶子都站在同一張桌上，示教時的 TCP 高度就是正確的夾取高度）。TCP 速度 v = k e。
  - 直立瓶子在近距離幾乎一定超出畫面，框被切到時中心會沿瓶身軸線偏移，所以高度不靠影像。
  - 收斂時高度與水平相對位置都和示教相同，視角相同、裁切相同，偏差抵銷。
姿態：伺服時 pitch / roll 慢慢轉回示教姿態，yaw 維持接手當下（桌面、直立瓶子、重力繞垂直軸對稱，yaw 不影響夾取）。
"""
from __future__ import annotations

import enum
import json
import logging
from dataclasses import dataclass

import numpy as np

from .detector import Box, Detection, box_depth_m
from .kinematics import ARM_JOINTS, ArmModel, rot_axis, rotation_angle_deg

logger = logging.getLogger(__name__)


class Phase(enum.Enum):
    IDLE = "idle"
    RETREAT = "retreat"
    SERVO = "servo"
    DESCEND = "descend"
    CLOSE = "close"
    LIFT = "lift"
    DONE = "done"
    ABORTED = "aborted"


ACTIVE = {Phase.RETREAT, Phase.SERVO, Phase.DESCEND, Phase.CLOSE, Phase.LIFT}


@dataclass
class ServoConfig:
    trigger_score: float = 0.25
    trigger_frames: int = 3
    trigger_max_depth_m: float = 0.35
    trigger_min_margin_m: float = 0.02         # 瓶子比示教距離 Z* 近超過這麼多 = ACT 已經貼上去：接手後先後退
    retreat_speed_m_s: float = 0.03
    retreat_max_m: float = 0.12                # 後退最多這麼遠（量不到深度時的上限）
    retreat_up_deg: float = 35.0               # 後退方向：夾爪水平方向往後，再往上抬這個角度（基座座標）
    retreat_min_depth_m: float = 0.07          # 看得到瓶子且深度 >= 這個值就停止後退（D405 最近約 7 cm）
    retreat_seen_frames: int = 2               # 連續幾次看得到瓶子就停止後退、開始伺服
    trigger_max_error_m: float = 0.08          # 接手當下誤差超過這麼大就不接手（多半是太近、框被切到，量測不可信）
    trigger_max_px_from_target: float = 120.0  # 觸發時框中心離 s* 最多幾 px（避免選到別支手臂的瓶子）
    track_score: float = 0.10                  # 追蹤（有位置門檻）時的分數門檻；近距離看不到標籤時分數常只有 0.1 多
    track_gate_frac: float = 0.15              # 追蹤時新框中心離上一個最多 畫面寬 * frac（避免跳到另一個瓶子）
    gain: float = 1.5                          # 1/s
    max_speed_m_s: float = 0.04
    tol_m: float = 0.003
    settle_detections: int = 3
    hold_if_stale_s: float = 0.3               # 這麼久沒有新偵測就停住
    lost_timeout_s: float = 1.5
    servo_timeout_s: float = 15.0
    max_lag_deg: float = 6.0                   # 命令角與實際角差距上限
    descend_speed_m_s: float = 0.02
    lift_m: float = 0.06
    lift_speed_m_s: float = 0.02
    extra_depth_m: float = 0.015               # 沿下降方向比示教位置再深入多少（夾得更深、更穩）
    reach_tol_deg: float = 1.0                 # 下降後等實際關節角追到命令這麼近才閉合
    reach_timeout_s: float = 2.0
    gripper_open_deg: float = -55.0
    gripper_closed_deg: float = 0.0
    gripper_speed_deg_s: float = 60.0
    close_hold_s: float = 1.0
    center_u: bool = True                      # 水平目標用畫面中心（相機在兩指中間上方）；False = 用示教時的 u
    # 伺服時把夾爪 pitch / roll 轉回示教姿態（yaw 維持接手當下）。ACT 交出的姿態 roll 偏 8–15° 時，
    # 只平移會讓夾爪中線偏離瓶軸 1–2.5 cm；pitch 偏會讓下降路徑高度偏掉。需要示教檔有 q_teach_deg。
    align_orientation: bool = True
    orient_speed_deg_s: float = 20.0
    orient_tol_deg: float = 2.0                # 姿態離目標超過這麼多就不下降


@dataclass
class TaughtTarget:
    """bottle_teach.py 產生：期望特徵與手眼旋轉。"""
    u: float
    v: float
    depth_m: float
    image_wh: tuple[int, int]
    r_tcp_cam: np.ndarray
    descent_m: float
    approach_tcp: np.ndarray
    query: str
    bottle_tcp: np.ndarray       # 示教位置量到的瓶子點，TCP 座標 (m)
    cam_p_tcp: np.ndarray        # 相機在 TCP 座標的位置（名義值；示教與執行用同一個，誤差大多抵銷）
    tcp_z: float                 # 示教位置 TCP 在基座座標的高度 (m)
    descent_tcp: np.ndarray | None = None   # 示教位置 -> 夾取位姿的實際位移（示教位置的 TCP 座標）；None 則沿接近軸
    q_teach_deg: np.ndarray | None = None   # 示教位置的關節角 (LeRobot deg)：伺服時把 pitch / roll 轉回這個姿態

    @property
    def descent_vector(self) -> np.ndarray:
        return self.descent_tcp if self.descent_tcp is not None else self.approach_tcp * self.descent_m

    @staticmethod
    def load(path: str) -> "TaughtTarget":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return TaughtTarget(float(d["u"]), float(d["v"]), float(d["depth_m"]), tuple(d["image_wh"]),
                            np.array(d["r_tcp_cam"], dtype=float), float(d["descent_m"]),
                            np.array(d.get("approach_tcp", [0.0, 0.0, 1.0]), dtype=float), d.get("query", "bottle"),
                            np.array(d["bottle_tcp"], dtype=float), np.array(d["cam_p_tcp"], dtype=float),
                            float(d["tcp_z"]),
                            np.array(d["descent_tcp"], dtype=float) if "descent_tcp" in d else None,
                            np.array(d["q_teach_deg"], dtype=float) if "q_teach_deg" in d else None)

    def save(self, path: str, extra: dict | None = None) -> None:
        d = {"u": self.u, "v": self.v, "depth_m": self.depth_m, "image_wh": list(self.image_wh),
             "r_tcp_cam": self.r_tcp_cam.tolist(), "descent_m": self.descent_m,
             "approach_tcp": self.approach_tcp.tolist(), "query": self.query,
             "bottle_tcp": self.bottle_tcp.tolist(), "cam_p_tcp": self.cam_p_tcp.tolist(), "tcp_z": self.tcp_z,
             **({"descent_tcp": self.descent_tcp.tolist()} if self.descent_tcp is not None else {}),
             **({"q_teach_deg": np.asarray(self.q_teach_deg).tolist()} if self.q_teach_deg is not None else {}),
             **(extra or {})}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2, ensure_ascii=False)


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    def backproject(self, u: float, v: float, z: float) -> np.ndarray:
        return np.array([(u - self.cx) / self.fx * z, (v - self.cy) / self.fy * z, z])


EDGE_PX = 4
NOMINAL_CAM_P_TCP = (0.0, -0.04, -0.08)


def bottle_in_tcp(intr: Intrinsics, r_tcp_cam: np.ndarray, cam_p_tcp: np.ndarray, u: float, v: float,
                  z: float) -> np.ndarray:
    """影像點 + 深度 -> TCP 座標的 3D 點。"""
    return r_tcp_cam @ intr.backproject(u, v, z) + cam_p_tcp


class ArmGraspServo:
    def __init__(self, side: str, model: ArmModel, target: TaughtTarget, intr: Intrinsics,
                 image_wh: tuple[int, int], cfg: ServoConfig, observe_only: bool = False):
        self.side = side
        self.model = model
        self.target = target
        self.intr = intr
        self.cfg = cfg
        self.observe_only = observe_only
        # 示教與執行時解析度不同：s* 依比例換算
        sx, sy = image_wh[0] / target.image_wh[0], image_wh[1] / target.image_wh[1]
        self.u_star, self.v_star = target.u * sx, target.v * sy
        if cfg.center_u:
            self.u_star = intr.cx              # 瓶子水平置中 = 在兩指正中間
        self._q_obs: np.ndarray | None = None
        self._r_teach = (model.fk(model.deg_to_q(target.q_teach_deg))[0]
                         if cfg.align_orientation and target.q_teach_deg is not None else None)
        if cfg.align_orientation and self._r_teach is None:
            logger.warning("[IBVS] %s 示教檔沒有 q_teach_deg：不修正夾爪姿態（重新示教即可）", side)
        self._r_goal: np.ndarray | None = None
        self.image_w, self.image_h = image_wh
        self.phase = Phase.IDLE
        self.reason = ""
        self._streak = 0
        self._last_det_id = -1
        self._last_box_center: np.ndarray | None = None
        self._last_good_t = 0.0
        self._v_tcp = np.zeros(3)
        self._v_tcp_t = 0.0
        self._settled = 0
        self._q_cmd: np.ndarray | None = None
        self._grip_cmd = 0.0
        self._phase_t0 = 0.0
        self._moved = 0.0
        self._last_err: float | None = None
        self._clock = 0.0          # 狀態機時間：每次 update 前進 dt（控制週期），與牆鐘無關
        self.trace: list[dict] = []   # 接手期間每次偵測的紀錄（診斷用，由策略寫成 CSV）
        self._last_log = -1e9

    # ------------------------------------------------------------------ 外部介面
    @property
    def active(self) -> bool:
        return self.phase in ACTIVE

    def status(self) -> str:
        err = f" 誤差 {self._last_err * 1000:.0f}mm" if self._last_err is not None else ""
        return f"[{self.side}] {self.phase.value}{err} {self.reason}".rstrip()

    def update(self, obs_deg: np.ndarray, grip_deg: float, det: Detection | None, dt: float) -> dict | None:
        """每個控制週期呼叫。回傳此手臂要覆寫的命令 {joint_1..7, gripper: deg}；None 表示交給 ACT。"""
        self._clock += dt
        now = self._clock
        new_det = det is not None and det.frame_id != self._last_det_id
        if new_det:
            self._last_det_id = det.frame_id

        if self.phase == Phase.IDLE:
            self._q_obs = self.model.deg_to_q(obs_deg)
            nxt = self._check_trigger(det) if new_det else None
            if nxt is not None:
                if self.observe_only:
                    logger.info("[IBVS] %s 觀察模式：達到接手條件（%s，不接手）", self.side, nxt.value)
                    self._streak = 0
                    return None
                self._enter(nxt, now, obs_deg, grip_deg)
                if nxt == Phase.RETREAT:
                    r_t, _ = self.model.fk(self._q_cmd)
                    d_w = r_t @ self._retreat_direction_tcp()
                    logger.info("[IBVS] %s 看到瓶子但太近（%.1f cm），深度 < %.0f cm 或看不到瓶子時往後上方斜退｜"
                                "方向（基座）%s，仰角 %.0f°",
                                self.side, self._trigger_depth * 100, self.cfg.retreat_min_depth_m * 100,
                                np.round(d_w, 2).tolist(), np.degrees(np.arcsin(np.clip(d_w[2], -1, 1))))
                else:
                    logger.info("[IBVS] %s 看到瓶子，接手伺服", self.side)
            return None
        if not self.active:
            return None

        q_obs = self.model.deg_to_q(obs_deg)
        self._q_obs = q_obs
        if np.max(np.abs(np.degrees(self._q_cmd - q_obs))) > self.cfg.max_lag_deg:
            # 手臂沒跟上（碰撞、卡住）：命令拉回實際位置，暫停一拍
            self._q_cmd = q_obs.copy()
            return self._command()

        if self.phase == Phase.RETREAT:
            self._retreat_step(det if new_det else None, now, dt)
        elif self.phase == Phase.SERVO:
            self._servo_step(det if new_det else None, now, dt)
        elif self.phase == Phase.DESCEND:
            self._open_gripper(dt)
            vec = self.target.descent_vector          # 照示教時的實際路徑走回夾取位姿，再多深入 extra_depth_m
            dist = float(np.linalg.norm(vec)) + self.cfg.extra_depth_m
            self._move_tcp(vec, self.cfg.descend_speed_m_s, dist, dt, Phase.CLOSE, now)
        elif self.phase == Phase.CLOSE:
            # 先等手臂真的走到底（有重力時實際位置會落後命令），再閉合夾爪
            if not self._reached:
                lag = float(np.max(np.abs(np.degrees(self._q_cmd - q_obs))))
                if lag <= self.cfg.reach_tol_deg or now - self._phase_t0 > self.cfg.reach_timeout_s:
                    self._reached = True
                    self._close_t0 = now
                    if lag > self.cfg.reach_tol_deg:
                        logger.warning("[IBVS] %s 下降後手臂仍落後命令 %.1f 度，照樣閉合", self.side, lag)
                return self._command()
            self._grip_cmd = min(self._grip_cmd + self.cfg.gripper_speed_deg_s * dt, self.cfg.gripper_closed_deg)
            if self._grip_cmd >= self.cfg.gripper_closed_deg and now - self._close_t0 > self.cfg.close_hold_s + 1.0:
                self._enter(Phase.LIFT, now)
        elif self.phase == Phase.LIFT:
            r_tcp, _ = self.model.fk(self._q_cmd)
            up_tcp = r_tcp.T @ np.array([0.0, 0.0, 1.0])       # 基座 +z 換到 TCP 座標
            self._move_tcp(up_tcp, self.cfg.lift_speed_m_s, self.cfg.lift_m, dt, Phase.DONE, now)
            if self.phase == Phase.DONE:
                logger.info("[IBVS] %s 夾取並提起完成，交回 ACT", self.side)
                return None
        return self._command() if self.active else None

    # ------------------------------------------------------------------ 內部
    def _enter(self, phase: Phase, now: float, obs_deg: np.ndarray | None = None, grip_deg: float | None = None):
        self.phase = phase
        self._reached = False
        self._seen = 0
        self._phase_t0 = now
        self._moved = 0.0
        if obs_deg is not None:
            self._q_cmd = self.model.deg_to_q(obs_deg)
            self._r_ref = self.model.fk(self._q_cmd)[0]     # 接手當下的夾爪姿態，全程保持
            self._grip_cmd = float(grip_deg)
            self._last_good_t = now
            self._settled = 0
            self._v_tcp[:] = 0.0
        if phase == Phase.SERVO and self._r_teach is not None:
            self._r_goal = self._goal_orientation()
            logger.info("[IBVS] %s 伺服時把夾爪 pitch / roll 轉回示教姿態（差 %.1f°）", self.side,
                        rotation_angle_deg(self._r_ref.T @ self._r_goal))

    def _goal_orientation(self) -> np.ndarray:
        """示教姿態繞基座垂直軸轉到「目前的 yaw」：pitch / roll 同示教，yaw 不動（少轉、少跑出畫面）。"""
        a_tcp = self.target.descent_vector / np.linalg.norm(self.target.descent_vector)
        a_now, a_teach = self._r_ref @ a_tcp, self._r_teach @ a_tcp
        if np.hypot(*a_now[:2]) < 0.2 or np.hypot(*a_teach[:2]) < 0.2:   # 夾爪幾乎垂直：yaw 改用 TCP x 軸
            a_now, a_teach = self._r_ref[:, 0], self._r_teach[:, 0]
        dyaw = np.arctan2(a_now[1], a_now[0]) - np.arctan2(a_teach[1], a_teach[0])
        return rot_axis(np.array([0.0, 0.0, 1.0]), dyaw) @ self._r_teach

    def _orient_step(self, dt: float) -> float:
        """參考姿態以 orient_speed_deg_s 轉向目標；回傳實際姿態離目標幾度。"""
        if self._r_goal is None:
            return 0.0
        rel = self._r_ref.T @ self._r_goal
        ang = np.radians(rotation_angle_deg(rel))
        if ang > 1e-6:
            w = np.array([rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]])
            if np.linalg.norm(w) > 1e-9:
                step = min(np.radians(self.cfg.orient_speed_deg_s) * dt, ang)
                self._r_ref = self._r_ref @ rot_axis(w / np.linalg.norm(w), step)
            else:                                              # 剛好差 180°：直接設成目標
                self._r_ref = self._r_goal.copy()
        return rotation_angle_deg(self.model.fk(self._q_cmd)[0].T @ self._r_goal)

    def _abort(self, reason: str) -> None:
        self.phase = Phase.ABORTED
        self.reason = reason
        logger.warning("[IBVS] %s 中止：%s，交回 ACT", self.side, reason)

    def _candidates(self, det: Detection, min_score: float, specific_only: bool = False) -> list[tuple[Box, float]]:
        """有深度的框。specific_only：只要符合「指定瓶子」查詢的框（觸發時用，避免選到別支手臂的瓶子）。"""
        out = []
        for b in det.boxes:
            if b.score < min_score or (specific_only and b.label != 0):
                continue
            z = box_depth_m(det, b)
            if z is not None:
                out.append((b, z))
        return out

    def _feature(self, b: Box) -> np.ndarray:
        return b.center

    def _center_inside(self, b: Box) -> bool:
        c = b.center
        return EDGE_PX < c[0] < self.image_w - 1 - EDGE_PX and EDGE_PX < c[1] < self.image_h - 1 - EDGE_PX

    def position_error(self, c: np.ndarray, z: float, q: np.ndarray) -> np.ndarray:
        """TCP 要移動的量（基座座標，m），只用框中心的 u 與深度 Z：

        - 左右：瓶子水平置中到 u*（u 誤差換成公尺，沿相機 x 軸的水平方向移動）
        - 前後：深度到 Z*（沿夾爪接近方向的水平分量移動）
        - 高度：維持示教時的 TCP 高度（運動學）
        不用框的 v（近距離時瓶子一定被畫面上下切到），也不需要手眼傾斜角（相機 x 軸不受傾斜影響）。
        """
        r_t, p_t = self.model.fk(q)
        up = np.array([0.0, 0.0, 1.0])
        x_w = r_t @ self.target.r_tcp_cam[:, 0]                 # 相機 x 軸（影像往右）在基座座標
        x_h = x_w - (x_w @ up) * up
        a_w = r_t @ (self.target.descent_vector / np.linalg.norm(self.target.descent_vector))
        a_h = a_w - (a_w @ up) * up
        x_h = x_h / max(np.linalg.norm(x_h), 1e-6)
        a_h = a_h / max(np.linalg.norm(a_h), 1e-6)
        lateral = (c[0] - self.u_star) / self.intr.fx * z     # 瓶子偏右 -> 正值 -> 往右移
        forward = z - self.target.depth_m                       # 瓶子比示教遠 -> 正值 -> 往前移
        return lateral * x_h + forward * a_h + (self.target.tcp_z - p_t[2]) * up

    def _check_trigger(self, det: Detection) -> Phase | None:
        """回傳要進入的階段：SERVO（距離正常）、RETREAT（太近，先後退），或 None（不接手）。"""
        cfg = self.cfg
        star = np.array([self.u_star, self.v_star])
        z_min = self.target.depth_m - cfg.trigger_min_margin_m
        ok, close, reasons = [], [], []
        for b, z in self._candidates(det, cfg.trigger_score, specific_only=True):
            px = float(np.linalg.norm(self._feature(b) - star))
            if z > cfg.trigger_max_depth_m:
                reasons.append(f"太遠 {z * 100:.0f}cm")
            elif not self._center_inside(b) or px > cfg.trigger_max_px_from_target:
                reasons.append(f"離 s* {px:.0f}px")
            else:
                err = float(np.linalg.norm(self.position_error(self._feature(b), z, self._q_obs)))
                if z < z_min:
                    close.append((b, z))      # 太近：量測不可信，不看誤差，接手後先後退
                elif err > cfg.trigger_max_error_m:
                    reasons.append(f"誤差 {err * 1000:.0f}mm 太大")
                else:
                    ok.append((b, z))
        if reasons and not ok and not close and det.frame_id % 10 == 0:
            logger.info("[IBVS] %s 看到瓶子但不接手：%s", self.side, "；".join(reasons))
        cands = ok or close
        if not cands:
            self._streak = 0
            return None
        best = min(cands, key=lambda bz: np.linalg.norm(self._feature(bz[0]) - star))
        self._last_box_center = self._feature(best[0])
        self._streak += 1
        if self._streak < cfg.trigger_frames:
            return None
        if ok:
            return Phase.SERVO
        self._trigger_depth = best[1]
        return Phase.RETREAT

    def _retreat_direction_tcp(self) -> np.ndarray:
        """後退方向（TCP 座標）：基座座標下夾爪接近方向的水平分量反向，再往上抬 retreat_up_deg。

        相機相對夾爪往下傾，沿夾爪軸直線後退會把瓶子推出畫面上緣；往後上方退才能讓瓶子留在畫面裡。
        不依賴手眼外參，所以外參量得不準也不影響。
        """
        r_t, _ = self.model.fk(self._q_cmd)
        a_w = r_t @ (self.target.descent_vector / np.linalg.norm(self.target.descent_vector))
        h = np.array([a_w[0], a_w[1], 0.0])
        if np.linalg.norm(h) < 0.2:                   # 夾爪幾乎垂直：直接往上
            d_w = np.array([0.0, 0.0, 1.0])
        else:
            ang = np.radians(self.cfg.retreat_up_deg)
            d_w = -np.cos(ang) * h / np.linalg.norm(h) + np.sin(ang) * np.array([0.0, 0.0, 1.0])
        return r_t.T @ d_w

    def _retreat_step(self, det: Detection | None, now: float, dt: float) -> None:
        """太近時：張開夾爪；只有看不到瓶子、或深度 < retreat_min_depth_m 時才往後上方斜退，
        連續 retreat_seen_frames 次看得到（深度 >= retreat_min_depth_m）就交給伺服；最多退 retreat_max_m。
        """
        cfg = self.cfg
        self._open_gripper(dt)
        if det is not None:
            cands = self._candidates(det, cfg.track_score)
            self.trace.append({"t": round(now, 3), "phase": "retreat", "frame": det.frame_id,
                               "n_boxes": len(det.boxes),
                               "best_score": round(max((b.score for b in det.boxes), default=0.0), 3),
                               "retreat_m": round(self._moved, 4),
                               "depths": ";".join(f"{z:.3f}" for _, z in cands),
                               **dict(zip(("tcp_x", "tcp_y", "tcp_z"),
                                          np.round(self.model.fk(self._q_obs if self._q_obs is not None
                                                                 else self._q_cmd)[1], 4).tolist()))})
            seen = [(self._feature(b), z) for b, z in cands
                    if z >= cfg.retreat_min_depth_m and self._center_inside(b)]
            if seen:
                c, z = min(seen, key=lambda cz: np.linalg.norm(cz[0] - self._last_box_center)
                           if self._last_box_center is not None else 0.0)
                self._last_box_center = c
                self._seen += 1
                if self._seen >= cfg.retreat_seen_frames:
                    logger.info("[IBVS] %s 後退 %.1f cm 後看得到瓶子（深度 %.1f cm），開始伺服",
                                self.side, self._moved * 100, z * 100)
                    self._enter(Phase.SERVO, now)
                    self._last_good_t = now
                    return
            else:
                self._seen = 0
        if self._moved >= cfg.retreat_max_m:
            logger.info("[IBVS] %s 已後退 %.0f cm，開始伺服", self.side, self._moved * 100)
            self._enter(Phase.SERVO, now)
            self._last_good_t = now
            return
        step = min(cfg.retreat_speed_m_s * dt, cfg.retreat_max_m - self._moved)
        self._step_tcp(self._retreat_direction_tcp() * step)
        self._moved += step

    def _servo_step(self, det: Detection | None, now: float, dt: float) -> None:
        cfg = self.cfg
        if now - self._phase_t0 > cfg.servo_timeout_s:
            return self._abort(f"{cfg.servo_timeout_s:.0f} 秒內未收斂")
        self._open_gripper(dt)
        orient_err = self._orient_step(dt)
        if det is not None:
            gate = cfg.track_gate_frac * self.image_w
            scored = [b for b in det.boxes if b.score >= cfg.track_score]
            near = [(self._feature(b), z) for b, z in self._candidates(det, cfg.track_score)]
            n_depth = len(near)
            near = [(c, z) for c, z in near
                    if self._last_box_center is None or np.linalg.norm(c - self._last_box_center) <= gate]
            rec = {"t": round(now, 3), "frame": det.frame_id, "n_boxes": len(det.boxes),
                   "best_score": round(max((b.score for b in det.boxes), default=0.0), 3),
                   "n_scored": len(scored), "n_with_depth": n_depth, "n_in_gate": len(near),
                   "depth_valid_frac": round(float((det.depth_m > 0.03).mean()), 3) if det.depth_m is not None else -1,
                   "tcp_z": round(float(self.model.fk(self._q_obs if self._q_obs is not None else self._q_cmd)[1][2]), 4),
                   "last_u": None if self._last_box_center is None else round(float(self._last_box_center[0]), 1),
                   "last_v": None if self._last_box_center is None else round(float(self._last_box_center[1]), 1)}
            self.trace.append(rec)
            if near:
                c, z = min(near, key=lambda cz: np.linalg.norm(cz[0] - self._last_box_center)
                           if self._last_box_center is not None else 0.0)
                self._last_box_center = c
                self._last_good_t = now
                q_img = self._q_obs if self._q_obs is not None else self._q_cmd
                err = self.position_error(c, z, q_img)
                self._last_err = float(np.linalg.norm(err))
                rec.update(u=round(float(c[0]), 1), v=round(float(c[1]), 1), z=round(z, 4),
                           ex=round(float(err[0]), 4), ey=round(float(err[1]), 4), ez=round(float(err[2]), 4))
                if now - self._last_log >= 0.5:
                    self._last_log = now
                    logger.info("[IBVS] %s 伺服：框 (%.0f, %.0f) 深度 %.1f cm｜水平誤差 %.0f mm、高度誤差 %+.0f mm、"
                                "姿態差 %.1f°", self.side, c[0], c[1], z * 100, np.hypot(err[0], err[1]) * 1000,
                                err[2] * 1000, orient_err)
                rec["orient_err_deg"] = round(orient_err, 2)
                if self._last_err < cfg.tol_m and orient_err <= cfg.orient_tol_deg:
                    self._settled += 1
                    if self._settled >= cfg.settle_detections:
                        self._v_tcp[:] = 0.0
                        logger.info("[IBVS] %s 對準完成（誤差 %.1f mm），下降 %.1f cm", self.side,
                                    self._last_err * 1000, self.target.descent_m * 100)
                        return self._enter(Phase.DESCEND, now)
                else:
                    self._settled = 0
                v_w = cfg.gain * err
                speed = np.linalg.norm(v_w)
                if speed > cfg.max_speed_m_s:
                    v_w *= cfg.max_speed_m_s / speed
                r_t, _ = self.model.fk(self._q_cmd)
                self._v_tcp = r_t.T @ v_w
                self._v_tcp_t = now
            elif now - self._last_log >= 0.5:
                self._last_log = now
                logger.info("[IBVS] %s 伺服：這幀沒有可用的瓶子（框 %d 個、分數夠 %d、有深度 %d、在追蹤範圍內 0）",
                            self.side, len(det.boxes), len(scored), n_depth)
        if now - self._last_good_t > cfg.lost_timeout_s:
            return self._abort("瓶子跟丟")
        if now - self._v_tcp_t > cfg.hold_if_stale_s:
            self._v_tcp[:] = 0.0
        if np.any(self._v_tcp) or orient_err > 0.1:      # 還在轉姿態時，沒有平移也要送命令
            self._step_tcp(self._v_tcp * dt)

    def _move_tcp(self, direction_tcp: np.ndarray, speed: float, distance: float, dt: float,
                  next_phase: Phase, now: float) -> None:
        step = min(speed * dt, distance - self._moved)
        if step > 0:
            self._step_tcp(direction_tcp / np.linalg.norm(direction_tcp) * step)
            self._moved += step
        if self._moved >= distance - 1e-6:
            self._enter(next_phase, now)

    def _step_tcp(self, v_tcp: np.ndarray) -> None:
        rot = 0.3 * self.model.orientation_error(self._q_cmd, self._r_ref)   # 修正累積的姿態漂移
        dq = self.model.tcp_twist_to_dq(self._q_cmd, np.concatenate([v_tcp, rot]))
        deg = self.model.q_to_deg(self._q_cmd + dq)
        self._q_cmd = self.model.deg_to_q(self.model.clip_deg(deg))

    def _open_gripper(self, dt: float) -> None:
        target = self.cfg.gripper_open_deg
        step = self.cfg.gripper_speed_deg_s * dt
        self._grip_cmd += float(np.clip(target - self._grip_cmd, -step, step))

    def _command(self) -> dict:
        deg = self.model.clip_deg(self.model.q_to_deg(self._q_cmd))
        cmd = {f"{self.side}_{m}.pos": float(deg[i]) for i, m in enumerate(ARM_JOINTS)}
        cmd[f"{self.side}_gripper.pos"] = float(self._grip_cmd)
        return cmd
