# openarm_src/test/main_experiment.py
"""
OpenArm 最後一哩實驗主程式：手臂靠近並夾起 5x5x5 cm AprilTag 方塊。

三種模式：
  compare  （預設）先做微幅試探訓練殘差，再用相同初始狀態比較 IBVS / Learned / Guarded
  approach  執行一次完整流程：伺服靠近 ->（--grasp）下降、夾取、提起
  teach     示教期望特徵 s*：手臂先擺在「閉合就能夾到」的位置，再執行本模式
  view      不移動手臂，檢查腕部相機目前看到什麼（畫面存到 <out-dir>/view.png）

離線模擬（不需任何硬體）：
  python main_experiment.py
  python main_experiment.py --learner torch --episodes 20
  python main_experiment.py --mode approach --method guarded

真機（需 pinocchio + OPENARM_URDF；先讀 README 的安全守則）：
  python main_experiment.py --real --side left --port can1 --leader-port can2 --mode teach   # 會先開主從控制
  python main_experiment.py --real --side left --port can1 --mode approach            # 只對準，不下夾
  python main_experiment.py --real --side left --port can1 --mode approach --grasp
  python main_experiment.py --real --side left --port can1 --mode compare --episodes 5
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from approach_pipeline import (
    GraspResult,
    LastMileApproach,
    Target,
    servo_result_row,
    target_from_handeye,
)
from arm_kinematics import ApproxArmModel, HandEye, load_arm_model
from camera_stream import RealSenseMultiCamera, SimCameraStream
from config import SystemConfig
from data_collector import collect_probe_dataset
from ibvs_pinn_controller import IBVSPINNController, Intrinsics
from openarm_robot import OpenArmInterface
from sim_world import SimWorld
from tag_detector import AprilTagDetector

METHODS = ["ibvs", "learned", "guarded"]
LABELS = {"ibvs": "Baseline (Pure IBVS)", "learned": "IBVS + Learned Residual",
          "guarded": "IBVS + Residual + Convergence Guard"}
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))


# ---------------------------------------------------------------------- 系統組裝
@dataclass
class System:
    cfg: SystemConfig
    args: argparse.Namespace
    world: Optional[SimWorld]
    robot: OpenArmInterface
    camera: object
    approach: LastMileApproach

    @property
    def dry_run(self) -> bool:
        return self.world is not None

    def close(self) -> None:
        self.camera.stop()
        self.robot.stop()


def add_hardware_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--real", action="store_true", help="連接真實手臂與相機（預設為離線模擬）")
    parser.add_argument("--side", choices=["left", "right"], default="left", help="使用哪一支手臂")
    parser.add_argument("--port", default=None, help="真機從臂 CAN 埠（預設依 --side 取 openarm_src/utils/config.py 的設定）")
    parser.add_argument("--leader-port", default=None, help="主臂 CAN 埠，主從擺位用（預設同上）")
    parser.add_argument("--leader-id", default=None, help="主臂校正檔 id（預設 openarm_<side>_leader）")
    parser.add_argument("--teleop", dest="teleop", action="store_true", default=None,
                        help="開始前先開主從控制，用主臂把從臂擺到位（teach 模式預設開啟）")
    parser.add_argument("--no-teleop", dest="teleop", action="store_false", help="不開主從控制，從臂留在目前姿態")
    parser.add_argument("--camera", default=None, choices=["wrist_left", "wrist_right"],
                        help="伺服用的腕部相機（預設為 wrist_<side>）")
    parser.add_argument("--serial", default=None, help="該腕部相機的 RealSense 序號（多台相機時必填）")
    parser.add_argument("--learner", choices=["ridge", "torch"], default="ridge", help="殘差學習器")
    parser.add_argument("--seed", type=int, default=2026)


def build_system(args: argparse.Namespace) -> System:
    cfg = SystemConfig()
    cfg.robot.side = args.side
    camera_name = args.camera or f"wrist_{args.side}"
    dry_run = not args.real

    handeye = HandEye(cfg.handeye.translation_m, cfg.handeye.rpy_deg, cfg.handeye.approach_axis_tcp)
    detector = AprilTagDetector(tag_size_m=cfg.target.tag_size_meters)
    controller = IBVSPINNController(cfg.ibvs, cfg.target.tag_size_meters, learner=args.learner)

    if dry_run:
        world = SimWorld(cfg, camera_name, seed=args.seed)
        model = ApproxArmModel()   # 模擬一律用近似模型（名義尺寸；SimWorld 內的真實手臂略有不同）
        print("[Kinematics] 離線模擬：使用近似 7 軸模型（非 OpenArm 真實幾何）")
        robot = OpenArmInterface(cfg.robot, model, dry_run=True, sim=world)
        camera = SimCameraStream(world)
    else:
        world = None
        if _REPO_ROOT not in sys.path:
            sys.path.insert(0, _REPO_ROOT)
        from teleop_position import default_ports

        follower_default, leader_default = default_ports(args.side)
        args.port = args.port or follower_default
        args.leader_port = getattr(args, "leader_port", None) or leader_default
        print(f"[Ports] 從臂 {args.side}: {args.port} | 主臂 {args.side}: {args.leader_port}")
        try:
            model = load_arm_model(args.side, allow_approx=False)
        except RuntimeError as e:
            sys.exit(f"[錯誤] {e}")
        if args.serial:
            cfg.cameras[camera_name].serial_number = args.serial
        camera = RealSenseMultiCamera(cfg.cameras, only=[camera_name])
        if not camera.is_connected:
            sys.exit(f"相機 [{camera_name}] 啟動失敗：請用 check_cameras.py 確認序號與 USB 連線")
        robot = OpenArmInterface(cfg.robot, model, dry_run=False, port=args.port,
                                 table_z_m=cfg.table_z_m, table_margin_m=cfg.table_margin_m)
        if robot.table_z_m is None:
            print("[警告] 未設定桌面高度 (OPENARM_TABLE_Z)，下降時不會做桌面防撞檢查。")

    approach = LastMileApproach(cfg, robot, camera, detector, controller, handeye, camera_name)
    return System(cfg, args, world, robot, camera, approach)


def confirm(system: System, message: str) -> bool:
    """真機動作前請使用者確認；模擬或 --yes 則直接通過。"""
    if system.dry_run or getattr(system.args, "yes", False):
        return True
    answer = input(f"\n[確認] {message}。手放在急停旁，按 Enter 繼續，輸入 n 取消: ")
    return answer.strip().lower() not in ("n", "no")


def maybe_teleop(system: System, default: bool, purpose: str) -> None:
    """真機：先開主從控制，讓使用者用主臂把從臂擺到位。"""
    if system.dry_run:
        return
    enabled = default if getattr(system.args, "teleop", None) is None else system.args.teleop
    if not enabled:
        print("[Teleop] 未開主從控制，從臂留在目前姿態。")
        return
    from teleop_position import teleop_until_enter

    teleop_until_enter(system.robot, system.args.leader_port, getattr(system.args, "leader_id", None), purpose)


# ---------------------------------------------------------------------- 期望特徵 s*
def sim_teach(system: System) -> Target:
    """模擬中的示教：把夾爪擺到理想夾取位姿，再走與真機完全相同的示教流程。"""
    world, robot = system.world, system.robot
    world.reset(np.zeros(2), 0.0)
    world.place_tcp_at_grasp_pose()
    robot.resync()
    robot.p_start = None
    target = system.approach.teach(system.args.retreat)
    world.reset(np.zeros(2), 0.0)
    robot.resync()
    return target


def resolve_target(system: System) -> Target:
    args = system.args
    choice = args.target
    if choice == "auto":
        if system.dry_run:
            choice = "taught"
        else:
            choice = "taught" if os.path.isfile(args.target_file) else "handeye"

    if choice == "taught":
        if system.dry_run:
            target = sim_teach(system)
        elif os.path.isfile(args.target_file):
            target = Target.load(args.target_file)
        else:
            sys.exit(f"找不到示教檔 {args.target_file}：請先執行 --mode teach")
    else:
        obs = system.approach.observe_average(5)
        if obs is None:
            sys.exit("看不到 Tag，無法取得相機內參：請先把手臂移到腕部相機看得到方塊的位置")
        target = target_from_handeye(system.cfg, system.approach.handeye, obs.intr)
        if not system.dry_run:
            print("[警告] s* 由名義手眼外參推算；外參若不準，夾取位置會偏。建議改用 --mode teach。")
    s = target.s_star
    print(f"[Target] 來源={target.source}  s*=[u {s[0]:.1f}, v {s[1]:.1f}, w {s[2]:.1f}px, psi {s[3]:.1f}deg]  "
          f"下降距離={target.descent_m * 100:.1f}cm")
    return target


def goto_start(system: System, target: Target) -> None:
    """真機：把手臂帶到伺服起點 = 示教 s* 的位置再沿接近軸多後退 --start-offset。

    路徑：目前姿態 -> reset.json（依序經過所有 waypoint）-> 示教時的關節姿態 -> 多後退一段。
    已用 --teleop 手動擺位、或加了 --no-goto 時不執行。
    """
    args = system.args
    if system.dry_run or args.teleop or args.no_goto:
        return
    if target.q_standoff is None:
        print("[Start] 示教檔沒有記錄關節姿態（舊版示教檔或 s* 來自手眼外參），手臂留在目前姿態。"
              "請重新執行 --mode teach，或加 --teleop 手動擺位。")
        return
    if not confirm(system, f"即將經 reset.json 移到示教 s* 的位置，再後退 {args.start_offset * 100:.0f} cm 作為起點"):
        sys.exit("已取消")
    robot = system.robot
    robot.go_home_via_waypoints()
    if not robot.go_to_joints_checked(target.q_standoff, 4.0, "示教 s* 的位置"):
        sys.exit("無法安全移動到示教位置，已停止")
    if args.start_offset > 0 and not system.approach.move_along_approach(-args.start_offset):
        print(f"[Start] 多後退時被擋下（{robot.last_block_reason}），改從示教位置開始。")
    robot.settle()
    if system.approach.observe_average(5) is None:
        info = system.approach.snapshot(os.path.join(args.out_dir, "start_no_tag.png"))
        print("[Start] 到達起點後看不到 Tag。\n  " + system.approach.explain_no_tag(info))


# ---------------------------------------------------------------------- 殘差訓練
def train_residual(system: System, out_dir: str) -> None:
    if not confirm(system, "即將在目前姿態附近做微幅試探（每步數 mm）"):
        sys.exit("已取消")
    if system.dry_run:
        system.world.reset(np.zeros(2), 0.0)
        system.robot.resync()
    system.robot.mark_start()
    q_before = system.robot.q_cmd.copy()
    data = collect_probe_dataset(system.approach, system.args.samples, seed=system.args.seed,
                                 out_path=os.path.join(out_dir, "probe_data.npz"))
    intr = Intrinsics(*data["intr"][:3], int(data["intr"][3]), int(data["intr"][4]))
    system.approach.controller.fit_residual(data["s"], data["a"], data["ds"], intr,
                                            val_fraction=system.cfg.probe.val_fraction, seed=system.args.seed)
    system.robot.move_to_joints(q_before)


# ---------------------------------------------------------------------- 模式：比較
def run_compare(system: System, out_dir: str) -> None:
    args, cfg = system.args, system.cfg
    maybe_teleop(system, False, "把手臂帶到腕部相機看得到方塊的位置")
    target = resolve_target(system)
    goto_start(system, target)
    print("\n步驟 1/3：微幅試探並訓練 Jacobian 殘差")
    train_residual(system, out_dir)

    rng = np.random.default_rng(args.seed)
    n = args.episodes
    if system.dry_run:
        # 模擬：每回合把方塊放在不同位置與角度
        offsets = rng.uniform(-cfg.sim.xy_range_m, cfg.sim.xy_range_m, size=(n, 2))
        yaws = rng.uniform(-cfg.sim.yaw_range_deg, cfg.sim.yaw_range_deg, size=n)
    else:
        # 真機：方塊不動，改讓手臂從起點先偏移一段，製造不同初始狀態
        offsets = np.column_stack([rng.uniform(-25, 25, n), rng.uniform(-25, 25, n),
                                   rng.uniform(-15, 5, n), rng.uniform(-12, 12, n)])
        yaws = np.zeros(n)
        if not confirm(system, f"即將進行 {len(METHODS)} x {n} 回合無接觸對準（不下夾）"):
            sys.exit("已取消")
    q_start = system.robot.q_cmd.copy()

    print(f"\n步驟 2/3：三種方法各 {n} 回合（相同初始狀態）")
    episodes: List[Dict] = []
    curves: Dict[str, List[float]] = {}
    for method in METHODS:
        print(f"  執行 [{method.upper()}] ...")
        for ep in range(n):
            if system.dry_run:
                system.world.reset(offsets[ep], yaws[ep])
                system.robot.resync()
            else:
                system.robot.move_to_joints(q_start)
                system.robot.mark_start()
                if not system.approach.move_camera_total(offsets[ep]):
                    print(f"    回合 {ep}: 初始偏移被安全檢查擋下，略過")
                    continue
            system.robot.mark_start()
            res = system.approach.servo(target, method)
            row = {"episode": ep, **servo_result_row(res)}
            if system.dry_run:
                grasp = system.approach.descend_grasp_lift(target) if res.success else GraspResult(False, False, False)
                row.update({"grasp_success": bool(grasp.holding and grasp.lifted),
                            **{f"grasp_{k}": v for k, v in grasp.sim_error.items()}})
            episodes.append(row)
            if ep == 0:
                curves[method] = res.error_curve
    if not system.dry_run:
        system.robot.move_to_joints(q_start)

    print("\n步驟 3/3：彙整指標")
    summary = summarize(episodes, system.dry_run)
    print_summary(summary, system.dry_run, cfg)
    save_outputs(out_dir, summary, episodes, curves, cfg)
    report_failures(episodes)
    print("\n注意：模擬成功率只代表這組合成條件下的結果，不代表真實手臂的成功率或收斂保證。"
          if system.dry_run else "\n注意：以上為真機無接觸對準結果，未包含夾取。")


def summarize(episodes: List[Dict], with_grasp: bool) -> List[Dict]:
    rows = []
    for method in METHODS:
        eps = [e for e in episodes if e["method"] == method]
        if not eps:
            continue
        errs = np.array([e["final_error_norm"] for e in eps], dtype=float)
        row = {
            "method": method,
            "episodes": len(eps),
            "servo_success_pct": 100.0 * np.mean([e["success"] for e in eps]),
            "mean_final_err_px": float(np.nanmean(errs)),
            "p95_final_err_px": float(np.nanpercentile(errs, 95)),
            "mean_steps": float(np.mean([e["steps"] for e in eps])),
            "decrease_pct": float(np.mean([e["decrease_pct"] for e in eps])),
            "residual_used_pct": float(np.mean([e["residual_used_pct"] for e in eps])),
            "latency_ms": float(np.mean([e["mean_latency_ms"] for e in eps])),
        }
        if with_grasp:
            row["grasp_success_pct"] = 100.0 * np.mean([e["grasp_success"] for e in eps])
            xy = [e["grasp_xy_mm"] for e in eps if "grasp_xy_mm" in e]
            row["mean_grasp_xy_mm"] = float(np.mean(xy)) if xy else float("nan")
        rows.append(row)
    return rows


def print_summary(summary: List[Dict], with_grasp: bool, cfg: SystemConfig) -> None:
    line = "=" * 96
    print("\n" + line)
    header = f"{'方法':<9}|{'伺服成功':>9} |{'平均誤差':>9} |{'P95誤差':>9} |{'平均步數':>8} |{'誤差下降步':>10} |{'殘差採納':>9}"
    if with_grasp:
        header += f" |{'夾取成功':>9} |{'夾取XY偏差':>10}"
    print(header)
    print("-" * 96)
    for r in summary:
        text = (f"{r['method'].upper():<10}|{r['servo_success_pct']:9.1f}% |{r['mean_final_err_px']:10.2f}px |"
                f"{r['p95_final_err_px']:9.2f}px |{r['mean_steps']:11.1f} |{r['decrease_pct']:13.1f}% |"
                f"{r['residual_used_pct']:11.1f}%")
        if with_grasp:
            text += f" |{r['grasp_success_pct']:11.1f}% |{r['mean_grasp_xy_mm']:11.1f}mm"
        print(text)
    print(line)
    i = cfg.ibvs
    print(f"伺服成功 = 中心 <= {i.center_tol_px}px、視邊長 <= {i.size_tol_px}px、偏轉 <= {i.yaw_tol_deg}deg，"
          f"連續 {i.settle_frames} 幀。誤差為像素等效加權範數。")


def report_failures(episodes: List[Dict]) -> None:
    failures = [e for e in episodes if not e["success"] or e.get("grasp_success") is False]
    if not failures:
        print("本次沒有失敗回合。")
        return
    print(f"失敗回合 {len(failures)} 個（前 5 個）：")
    for e in failures[:5]:
        detail = e["reason"] if not e["success"] else (
            f"伺服收斂但沒夾到：XY 偏差 {e.get('grasp_xy_mm', float('nan')):.1f}mm、"
            f"Z {e.get('grasp_z_mm', float('nan')):+.1f}mm、偏轉 {e.get('grasp_yaw_deg', float('nan')):+.1f}deg")
        print(f"  [{e['method']}] 回合 {e['episode']}: {detail}")


def save_outputs(out_dir: str, summary: List[Dict], episodes: List[Dict],
                 curves: Dict[str, List[float]], cfg: SystemConfig) -> None:
    def write(name: str, rows: List[Dict]) -> None:
        if not rows:
            return
        fields = list(dict.fromkeys(k for r in rows for k in r))
        path = os.path.join(out_dir, name)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[輸出] {path}")

    write("experiment_results.csv", summary)
    write("experiment_episodes.csv", episodes)
    length = max((len(c) for c in curves.values()), default=0)
    write("experiment_error_curves.csv", [
        {"step": t, **{m: (curves[m][t] if t < len(curves[m]) else "") for m in curves}} for t in range(length)
    ])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[輸出] 未安裝 matplotlib，略過誤差曲線圖 (pip install matplotlib)")
        return
    colors = {"ibvs": "#d9534f", "learned": "#f0ad4e", "guarded": "#5cb85c"}
    plt.figure(figsize=(8, 5))
    for m, curve in curves.items():
        plt.plot(range(len(curve)), curve, marker="o", markersize=3, label=LABELS[m], color=colors[m], linewidth=1.8)
    plt.yscale("log")
    plt.xlabel("Control step")
    plt.ylabel("Weighted feature error ||s - s*|| (pixel-equivalent)")
    plt.title("OpenArm last-mile approach, episode 0 (same initial state)")
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "experiment_error_curves.png")
    plt.savefig(path, dpi=180)
    plt.close()
    print(f"[輸出] {path}")


# ---------------------------------------------------------------------- 模式：單次靠近
def run_approach(system: System, out_dir: str) -> None:
    args = system.args
    maybe_teleop(system, False, "把手臂帶到腕部相機看得到方塊的位置")
    target = resolve_target(system)
    goto_start(system, target)
    if args.method != "ibvs":
        train_residual(system, out_dir)
    if system.dry_run:
        rng = np.random.default_rng(args.seed)
        s = system.cfg.sim
        system.world.reset(rng.uniform(-s.xy_range_m, s.xy_range_m, 2), rng.uniform(-s.yaw_range_deg, s.yaw_range_deg))
        system.robot.resync()
    if not confirm(system, f"即將以 [{args.method}] 伺服靠近方塊"):
        sys.exit("已取消")

    system.robot.mark_start()
    system.robot.set_gripper(system.cfg.robot.gripper_open_deg)
    print(f"\n[SERVO] 方法 = {args.method}")
    res = system.approach.servo(target, args.method, verbose=True)
    e = res.final_error
    print(f"[SERVO] {'成功' if res.success else '失敗'}（{res.reason}），{res.steps} 步；"
          f"中心 {np.hypot(e[0], e[1]):.1f}px、視邊長 {e[2]:+.1f}px、偏轉 {e[3]:+.1f}deg；"
          f"殘差採納 {res.residual_used_pct:.0f}%，每步運算 {res.mean_latency_ms:.1f}ms")
    if not res.success:
        return
    if not (args.grasp or system.dry_run):
        print("[完成] 已停在預備抓取位置（未下夾）。確認對準無誤後，加上 --grasp 才會下降夾取。")
        return

    print("[DESCEND / GRASP / LIFT]")
    grasp = system.approach.descend_grasp_lift(target, confirm=lambda msg: confirm(system, msg))
    if grasp.sim_error:
        g = grasp.sim_error
        print(f"  下降後 TCP 相對理想夾取位姿：XY {g['xy_mm']:.1f}mm、Z {g['z_mm']:+.1f}mm、"
              f"偏轉 {g['yaw_deg']:+.1f}deg、傾斜 {g['tilt_deg']:.1f}deg")
    if grasp.holding and grasp.lifted:
        print("[完成] 已夾住並提起方塊。")
    else:
        print(f"[失敗] {grasp.reason}")


# ---------------------------------------------------------------------- 模式：示教 / 檢視
def ask_more_retreat(total_m: float) -> Optional[float]:
    """示教看不到 Tag 時，問使用者要不要原地再後退一點。"""
    answer = input(f"\n[Teach] 目前已後退 {total_m * 100:.0f} cm。"
                   "輸入要「再後退」幾公分後重試（例如 3），直接按 Enter 放棄: ")
    try:
        extra_cm = float(answer.strip())
    except ValueError:
        return None
    if not 0.0 < extra_cm <= 10.0:
        print("  只接受 0~10 公分。")
        return None
    return extra_cm / 100.0


def run_view(system: System, out_dir: str) -> None:
    """不移動手臂（除非加 --teleop）：看腕部相機目前看到什麼，並把畫面存檔。"""
    maybe_teleop(system, False, "把手臂帶到想檢查的位置")
    approach = system.approach
    info = approach.snapshot(os.path.join(out_dir, "view.png"))
    obs = approach.observe_average(10)
    if obs is None:
        print("[View] 看不到目標 Tag。\n  " + approach.explain_no_tag(info))
        return
    s = obs.s
    print(f"[View] 看到 Tag：u {s[0]:.1f}, v {s[1]:.1f}, 視邊長 {s[2]:.1f}px, 旋轉 {s[3]:+.1f}deg；"
          f"估計距離 {obs.depth_pnp_m * 100:.1f} cm；畫面中的 Tag ID: {info['ids']}")
    if info["path"]:
        print(f"[View] 畫面已存至 {info['path']}")


def run_teach(system: System) -> None:
    args = system.args
    if system.dry_run:
        target = sim_teach(system)
    else:
        maybe_teleop(system, True, "把張開的夾爪擺到「閉合就能夾到方塊」的位置")
        print("示教前提：夾爪已張開、位於「閉合就能夾到方塊」的位置。")
        if not confirm(system, f"即將沿接近軸後退 {args.retreat * 100:.0f} cm 並記錄影像特徵"):
            sys.exit("已取消")
        system.robot.mark_start()
        target = system.approach.teach(args.retreat, debug_dir=args.out_dir, ask_more=ask_more_retreat)
    target.save(args.target_file, system.approach.camera_name)
    s = target.s_star
    print(f"[Teach] s*=[u {s[0]:.1f}, v {s[1]:.1f}, w {s[2]:.1f}px, psi {s[3]:.1f}deg]，"
          f"下降距離 {target.descent_m * 100:.1f}cm，已儲存至 {args.target_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenArm 最後一哩：IBVS + 殘差 + 守門，靠近並夾取 AprilTag 方塊")
    add_hardware_args(parser)
    parser.add_argument("--mode", choices=["compare", "approach", "teach", "view"], default="compare")
    parser.add_argument("--method", choices=METHODS, default="guarded", help="approach 模式使用的控制方法")
    parser.add_argument("--episodes", type=int, default=15, help="compare 模式每種方法的回合數")
    parser.add_argument("--samples", type=int, default=None, help="試探樣本數")
    parser.add_argument("--target", choices=["auto", "taught", "handeye"], default="auto",
                        help="s* 來源：示教檔或名義手眼外參（auto：模擬用示教；真機有示教檔就用）")
    parser.add_argument("--target-file", default=os.path.join(_HERE, "target_feature.json"))
    parser.add_argument("--retreat", type=float, default=None,
                        help="示教時沿接近軸後退的距離 (m)；預設真機 0.05、模擬 0.10")
    parser.add_argument("--start-offset", type=float, default=0.05,
                        help="真機 approach/compare：起點比示教 s* 的位置多後退多少 (m)")
    parser.add_argument("--no-goto", action="store_true",
                        help="真機 approach/compare：不自動移到示教位置，從手臂目前姿態開始")
    parser.add_argument("--grasp", action="store_true", help="真機 approach 模式：對準後繼續下降、夾取、提起")
    parser.add_argument("--yes", action="store_true", help="真機：略過每個動作前的確認提示")
    parser.add_argument("--out-dir", default="experiment_outputs")
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes 必須為正整數")
    if args.retreat is None:
        # 真機先保守退 5 cm（看不到 Tag 時可在提示中原地再加）；模擬的相機幾何需要 10 cm 才看得到整個 Tag
        args.retreat = 0.05 if args.real else 0.10
    if not 0.0 < args.retreat <= 0.20:
        parser.error("--retreat 需介於 0 與 0.20 m 之間")

    os.makedirs(args.out_dir, exist_ok=True)
    print("=" * 70)
    print(" OpenArm Last-Mile：IBVS + Physics-Informed Residual + Convergence Guard")
    print(f" 模式: {args.mode} | {'真實硬體' if args.real else '離線模擬（不輸出馬達命令）'} | "
          f"手臂: {args.side} | 學習器: {args.learner}")
    print("=" * 70)

    system = build_system(args)
    try:
        if args.mode == "compare":
            run_compare(system, args.out_dir)
        elif args.mode == "approach":
            run_approach(system, args.out_dir)
        elif args.mode == "view":
            run_view(system, args.out_dir)
        else:
            run_teach(system)
    except RuntimeError as e:
        print(f"\n[錯誤] {e}")
    except KeyboardInterrupt:
        print("\n[中斷] 使用者按下 Ctrl+C，停止動作。")
    finally:
        system.close()


if __name__ == "__main__":
    main()
