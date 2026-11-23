"""
ibvs_assist：ACT 推論中，某支手臂的腕部相機看到瓶子就由 IBVS 接手夾取，夾完交回 ACT 放置。

以 lerobot-rollout 的第三方策略註冊（--strategy.type=ibvs_assist），不修改 LeRobot 原始碼。
import 本模組即完成註冊；rollout_home.py 會自動 import。

每支手臂各自觸發、互不打斷：
- 另一支手臂照常吃 ACT 的命令；接手的手臂命令被 IBVS 覆寫。
- 每次 rollout 每支手臂只接手一次（夾完或中止都交回 ACT）。
- 交回時重置 ACT 的動作佇列，讓它從目前姿態重新規劃。

需要先用 bottle_teach.py 為每支手臂產生 teach_<side>.json（期望特徵 + 手眼旋轉）。
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

import numpy as np

from lerobot.rollout.configs import RolloutStrategyConfig
from lerobot.rollout.context import RolloutContext
from lerobot.rollout.strategies.base import BaseStrategy
from lerobot.rollout.strategies.core import send_next_action
from lerobot.utils.cycle_timer import CycleTimer

logger = logging.getLogger(__name__)

GRASP_DIR = os.path.dirname(os.path.abspath(__file__))


@RolloutStrategyConfig.register_subclass("ibvs_assist")
@dataclass
class IbvsAssistStrategyConfig(RolloutStrategyConfig):
    """--strategy.type=ibvs_assist 的參數（皆為 --strategy.<name>）。"""

    arms: str = "left,right"
    teach_dir: str = GRASP_DIR
    observe_only: bool = False          # 只偵測、印出何時會接手，不覆寫任何命令（第一次上機先用這個）
    detector: str = "owlv2"             # 偵測器：owlv2（預設）或 yoloe（YOLOE-26-seg，偵測 + 分割，深度只取瓶子像素）
    detector_model: str = "google/owlv2-base-patch16-ensemble"
    yoloe_model: str = "~/models/yoloe/yoloe-26s-seg.pt"   # detector=yoloe 時的權重（n / s / m / l 可換）
    detector_threshold: float = 0.10
    detector_hz: float = 30.0           # 偵測頻率上限；實際受推論時間限制（1 台相機約 28 Hz、2 台約 19 Hz）
    idle_detect_every: int = 3          # 有手臂在接手時，其他閒置手臂的相機每幾拍才送一次偵測（讓接手的手臂偵測更快）
    trigger_score: float = 0.25
    trigger_frames: int = 3
    trigger_max_depth_m: float = 0.35
    min_policy_time_s: float = 1.0      # rollout 開始後至少這麼久才允許接手
    gain: float = 1.5
    max_speed_m_s: float = 0.04
    lift_m: float = 0.05                # 交回 ACT 前往上抬多少（夾到後、中止時都會抬）
    center_u: bool = True               # 瓶子水平置中到畫面中心；false = 對準示教時瓶子所在的 u
    extra_depth_m: float = 0.015        # 比示教位置再往前深入多少才閉合（夾太淺會滑掉）
    extra_depth_left_m: float | None = None   # 只給左臂的 extra_depth_m（空白 = 用 extra_depth_m）
    extra_depth_right_m: float | None = None  # 只給右臂的 extra_depth_m
    retreat_up_deg: float = 35.0        # 太近時往後上方退的仰角
    retreat_min_depth_m: float = 0.07   # 太近時：深度 < 這個值或看不到瓶子才繼續後退
    trigger_max_px: float = 120.0       # 觸發時瓶子框中心離 s* 最多幾 px（畫面裡有別支手臂的瓶子時很重要）
    track_gate_m: float = 0.04          # 接手後只追鎖定的那支瓶子：框的瓶子點離鎖定位置（水平）最多幾 m
    query_left: str = ""                # 左臂要找的瓶子（空白 = 用示教檔的 query），例如 "coca cola bottle"
    query_right: str = ""               # 右臂要找的瓶子，例如 "blue bottle"
    orient_axes: str = "roll,pitch,yaw"  # 伺服時修正哪些姿態軸；空字串 = 只平移（舊行為）
    axis_source: str = "gravity"        # roll / pitch 的瓶軸來源：gravity（瓶子直立，不需相機校正）或 depth（深度擬合）
    orient_tol_deg: float = 1.0         # 姿態誤差都小於這個值才下降
    max_rot_speed_deg_s: float = 15.0
    # 交回 ACT 時 ACT 第一拍常和目前姿態差 10° 以上：該手臂的 step_guard 每拍上限從 handback_start_step_deg
    # 漸增回原本的上限（OPENARM_MAX_STEP_DEG，預設 4°），歷時 handback_ramp_s 秒。0 = 不額外限速
    handback_ramp_s: float = 2.0
    handback_start_step_deg: float = 0.3


class IbvsAssistStrategy(BaseStrategy):
    def setup(self, ctx: RolloutContext) -> None:
        super().setup(ctx)
        from .detector import AsyncDetector
        from .kinematics import ArmModel
        from .servo import ArmGraspServo, ServoConfig, TaughtTarget

        cfg: IbvsAssistStrategyConfig = self.config
        self._attach_log_file()
        robot = ctx.hardware.robot_wrapper.inner
        self._robot = robot
        self._servos: dict[str, ArmGraspServo] = {}
        self._abort_saved: set[str] = set()   # 已存過中止影像的手臂
        self._cams = {}
        self._depth_scale = {}
        queries = {}
        for side in [s.strip() for s in cfg.arms.split(",") if s.strip()]:
            cam_name = f"wrist_{side}"
            path = os.path.join(cfg.teach_dir, f"teach_{side}.json")
            if cam_name not in robot.cameras:
                logger.warning("[IBVS] 沒有相機 %s，%s 手臂只用 ACT", cam_name, side)
                continue
            if not os.path.isfile(path):
                logger.warning("[IBVS] 找不到 %s（先執行 bottle_teach.py --side %s），%s 手臂只用 ACT", path, side, side)
                continue
            cam = robot.cameras[cam_name]
            self._enable_depth(cam)
            intr, scale = self._intrinsics(cam)
            target = TaughtTarget.load(path)
            override = getattr(cfg, f"query_{side}", "")
            if override:
                target.query = override
            servo_cfg = ServoConfig(trigger_score=cfg.trigger_score, trigger_frames=cfg.trigger_frames,
                                    trigger_max_depth_m=cfg.trigger_max_depth_m, gain=cfg.gain,
                                    max_speed_m_s=cfg.max_speed_m_s, lift_m=cfg.lift_m,
                                    extra_depth_m=(cfg.extra_depth_m if getattr(cfg, f"extra_depth_{side}_m") is None
                                                   else getattr(cfg, f"extra_depth_{side}_m")),
                                    trigger_max_px_from_target=cfg.trigger_max_px,
                                    track_gate_m=cfg.track_gate_m,
                                    retreat_up_deg=cfg.retreat_up_deg, center_u=cfg.center_u,
                                    retreat_min_depth_m=cfg.retreat_min_depth_m,
                                    orient_axes=cfg.orient_axes, axis_source=cfg.axis_source,
                                    orient_tol_deg=cfg.orient_tol_deg,
                                    max_rot_speed_deg_s=cfg.max_rot_speed_deg_s)
            self._servos[side] = ArmGraspServo(side, ArmModel(side), target, intr, (cam.width, cam.height),
                                               servo_cfg, observe_only=cfg.observe_only)
            logger.info("[IBVS] %s 比示教位置再往前 %.1f cm 才閉合", side, servo_cfg.extra_depth_m * 100)
            self._cams[side] = (cam_name, cam)
            self._depth_scale[side] = scale
            # 觸發用指定的瓶子；追蹤時也接受一般的 bottle（近距離看不到標籤時，指定查詢常常認不出來）
            queries[cam_name] = [target.query] if target.query.strip() == "bottle" else [target.query, "bottle"]
            logger.info("[IBVS] %s 手臂啟用：查詢「%s」，s*=(%.0f, %.0f) px、%.1f cm", side, target.query,
                        target.u, target.v, target.depth_m * 100)
        model_name = cfg.yoloe_model if cfg.detector == "yoloe" else cfg.detector_model
        if queries:
            logger.info("[IBVS] 偵測器：%s（%s）", cfg.detector, model_name)
        self._detector = AsyncDetector(queries, cfg.detector_threshold, model_name, cfg.detector_hz,
                                       backend=cfg.detector) if queries else None
        self._overrides: dict[str, float] = {}

        # 攔截送往機器人的命令：接手中的手臂改用 IBVS 命令
        wrapper = ctx.hardware.robot_wrapper
        orig_send = wrapper.send_action

        def send_with_overrides(action):
            if self._overrides:
                action = {**action, **self._overrides}
            return orig_send(action)

        wrapper.send_action = send_with_overrides
        self._orig_send = orig_send

    @staticmethod
    def _enable_depth(cam) -> None:
        """腕部 D405 重新以「彩色 + 深度」連線（只影響 IBVS 讀深度，ACT 的輸入特徵不變）。"""
        if getattr(cam, "use_depth", False):
            return
        cam.disconnect()
        cam.use_depth = True
        cam.config.use_depth = True
        cam.connect()

    @staticmethod
    def _intrinsics(cam):
        import pyrealsense2 as rs

        from .servo import Intrinsics

        prof = cam.rs_profile
        i = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        scale = prof.get_device().first_depth_sensor().get_depth_scale()
        return Intrinsics(i.fx, i.fy, i.ppx, i.ppy), float(scale)

    def run(self, ctx: RolloutContext) -> None:
        engine = self._require_engine()
        cfg = ctx.runtime.cfg
        robot = ctx.hardware.robot_wrapper
        interpolator = self._require_interpolator()
        timer = CycleTimer(cfg.fps, interpolator.multiplier, records_data=False, report=ctx.runtime.cadence_report)
        dt = 1.0 / cfg.fps

        start = time.perf_counter()
        last_log = 0.0
        engine.resume()
        logger.info("[IBVS] ibvs_assist 控制迴圈開始（%s）", "觀察模式" if self.config.observe_only else "會接手")
        try:
            while not ctx.runtime.shutdown_event.is_set():
                timer.tick(new_cycle=interpolator.needs_new_action())
                now = time.perf_counter()
                if cfg.duration > 0 and now - start >= cfg.duration:
                    logger.info("Duration limit reached (%.0fs)", cfg.duration)
                    break

                with timer.section("observe"):
                    obs = robot.get_observation()
                with timer.section("ibvs"):
                    self._overrides = self._update_servos(obs, now - start, dt, engine, interpolator)
                with timer.section("process_obs"):
                    obs_processed = self._process_observation_and_notify(ctx.processors, obs)
                if self._handle_warmup(cfg.use_torch_compile, timer):
                    continue

                action_dict = send_next_action(obs_processed, obs, ctx, interpolator, timer)
                if action_dict is None and self._overrides:
                    self._orig_send(dict(self._overrides))     # ACT 還沒有動作時，接手的手臂照樣前進
                with timer.section("telemetry"):
                    self._log_telemetry(obs_processed, action_dict, ctx.runtime)
                with timer.section("query"):
                    engine.pump_query(obs_processed)

                if now - last_log > 1.0 and any(s.active for s in self._servos.values()):
                    last_log = now
                    logger.info("[IBVS] " + "  ".join(s.status() for s in self._servos.values()))
                timer.wait()
        finally:
            logger.info("[IBVS] 結束：" + "  ".join(s.status() for s in self._servos.values()))
            timer.log_run_summary()

    def _update_servos(self, obs: dict, elapsed: float, dt: float, engine, interpolator) -> dict[str, float]:
        if self._detector is None:
            return {}
        overrides: dict[str, float] = {}
        handback = False
        self._tick = getattr(self, "_tick", 0) + 1
        any_active = any(s.active for s in self._servos.values())
        for side, servo in self._servos.items():
            cam_name, cam = self._cams[side]
            rgb = obs.get(cam_name)
            # 偵測排程：已完成 / 中止的手臂不再偵測；有手臂接手時，閒置手臂降頻，讓接手的手臂偵測更快
            if servo.phase.value in ("done", "aborted"):
                want = False
            elif any_active and not servo.active:
                want = self._tick % max(self.config.idle_detect_every, 1) == 0
            else:
                want = True
            if rgb is not None and want:
                try:
                    depth = cam.read_latest_depth().squeeze(-1).astype(np.float32) * self._depth_scale[side]
                except Exception:  # noqa: BLE001
                    depth = None
                self._detector.submit(cam_name, rgb, depth)
            if elapsed < self.config.min_policy_time_s:
                continue
            was_active = servo.active
            joints = np.array([float(obs[f"{side}_joint_{i}.pos"]) for i in range(1, 8)])
            det = self._detector.latest(cam_name)
            cmd = servo.update(joints, float(obs[f"{side}_gripper.pos"]), det, dt)
            if cmd:
                overrides.update(cmd)
            if not was_active and servo.active:
                self._save_snapshot(side, rgb, det, "takeover")
            if servo.aborting and side not in self._abort_saved:
                self._abort_saved.add(side)              # 中止當下的影像（抬高之前），看得出為什麼跟丟
                self._save_snapshot(side, rgb, det, "aborted")
            if was_active and not servo.active:
                handback = True
                self._soften_handback(side)
                if not (servo.phase.value == "aborted" and side in self._abort_saved):
                    self._save_snapshot(side, rgb, det, servo.phase.value)
                self._save_trace(side, servo)
        if handback:
            # 交回 ACT：清掉接手前規劃的動作，讓 ACT 從目前姿態重新推論
            engine.reset()
            interpolator.reset()
        return overrides

    def _soften_handback(self, side: str) -> None:
        """交回 ACT：這支手臂的 step_guard 暫時更嚴格，避免 ACT 第一拍大跳動。"""
        if self.config.handback_ramp_s <= 0:
            return
        guard = getattr(getattr(self, "_robot", None), "_step_guard", None)
        if guard is None:
            logger.warning("[IBVS] %s 交回 ACT：沒有 step_guard（不是用 rollout_home.py 啟動，或 OPENARM_STEP_GUARD=0），"
                           "無法限速", side)
            return
        guard.soften(f"{side}_", self.config.handback_ramp_s, self.config.handback_start_step_deg)

    def _attach_log_file(self) -> None:
        """整次 rollout 的 log（含所有 [IBVS] 訊息）另存一份到 runs/<時間>/ibvs.log。"""
        self._log_handler = logging.FileHandler(os.path.join(self._run_dir(), "ibvs.log"), encoding="utf-8")
        self._log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logging.getLogger().addHandler(self._log_handler)
        if logging.getLogger().level > logging.INFO or logging.getLogger().level == logging.NOTSET:
            logging.getLogger().setLevel(logging.INFO)
        logger.info("[IBVS] 本次 log：%s", os.path.join(self._run_dir(), "ibvs.log"))

    def _run_dir(self) -> str:
        if not hasattr(self, "_run_path"):
            self._run_path = os.path.join(GRASP_DIR, "runs", time.strftime("%Y%m%d_%H%M%S"))
            os.makedirs(self._run_path, exist_ok=True)
        return self._run_path

    def _save_snapshot(self, side: str, rgb, det, tag: str) -> None:
        """接手、中止、完成時存下腕部影像，只畫夾爪在追蹤的那一個框（鎖定的瓶子）。"""
        if rgb is None:
            return
        try:
            import cv2

            img = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
            servo = self._servos[side]
            tracked = servo.tracked
            if det is not None and tracked is not None and tracked[0] == det.frame_id:
                b = tracked[1]
                if b.mask is not None and b.mask.shape == img.shape[:2]:      # YOLOE-seg：瓶子像素塗綠
                    img[b.mask] = (0.5 * img[b.mask] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
                cv2.rectangle(img, (int(b.x0), int(b.y0)), (int(b.x1), int(b.y1)), (0, 255, 0), 2)
                cv2.putText(img, f"{b.score:.2f}", (int(b.x0) + 2, int(b.y0) + 14), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (0, 255, 0), 1)
            else:
                cv2.putText(img, "locked bottle not detected in this frame", (6, 16), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (0, 0, 255), 1)
            cv2.drawMarker(img, (int(servo.u_star), int(servo.v_star)), (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
            cv2.putText(img, f"{side} {tag} {servo.reason}", (6, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 0, 255), 1)
            path = os.path.join(self._run_dir(), f"{side}_{tag}.png")
            cv2.imwrite(path, img)
            logger.info("[IBVS] 影像已存：%s", path)
        except Exception:  # noqa: BLE001
            logger.exception("[IBVS] 存影像失敗")

    def _save_trace(self, side: str, servo) -> None:
        if not servo.trace:
            return
        import csv

        keys = []
        for r in servo.trace:                        # 依出現順序收集欄位（後退與伺服的欄位不同）
            keys += [k for k in r if k not in keys]
        path = os.path.join(self._run_dir(), f"{side}_trace.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, restval="")
            w.writeheader()
            w.writerows(servo.trace)
        logger.info("[IBVS] 接手紀錄已存：%s（%d 筆偵測）", path, len(servo.trace))

    def teardown(self, ctx: RolloutContext) -> None:
        if getattr(self, "_detector", None) is not None:
            self._detector.stop()
        for side, servo in getattr(self, "_servos", {}).items():
            if servo.active:                       # 被中斷時也把接手紀錄存下來
                self._save_trace(side, servo)
        try:
            super().teardown(ctx)
        finally:
            if getattr(self, "_log_handler", None) is not None:
                logging.getLogger().removeHandler(self._log_handler)
                self._log_handler.close()
