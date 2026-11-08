from __future__ import annotations

import argparse
import json
import logging
import sys
import time

import openarm_src.utils.config as config_mod
from openarm_src.utils.config import (
    ARM_JOINTS,
    DEFAULT_FOLLOWER_LEFT_PORT,
    DEFAULT_FOLLOWER_RIGHT_PORT,
    DEFAULT_HOME_PATH,
    DEFAULT_HOME_TIME_S,
    DEFAULT_LEADER_LEFT_PORT,
    DEFAULT_LEADER_RIGHT_PORT,
    GRIPPER,
)
from openarm_src.utils.homing import go_home, load_waypoints
from openarm_src.utils.patches import (
    install_patches,
    patch_follower_park,
    patch_record_loop,
    patch_rollout,
    patch_teleop_loop,
)

logger = logging.getLogger("openarm_gc.commands")


def run_check(argv):
    from lerobot.teleoperators.bi_openarm_leader import BiOpenArmLeader
    from lerobot.teleoperators.bi_openarm_leader.config_bi_openarm_leader import BiOpenArmLeaderConfig
    from lerobot.teleoperators.openarm_leader.config_openarm_leader import OpenArmLeaderConfigBase

    p = argparse.ArgumentParser()
    p.add_argument("--id", required=True, help="與 lerobot-record 的 --teleop.id 相同 (讀同一份 calibration)")
    p.add_argument("--left-port", default=DEFAULT_LEADER_LEFT_PORT)
    p.add_argument("--right-port", default=DEFAULT_LEADER_RIGHT_PORT)
    p.add_argument("--save", default=None, help="按 Enter 時把目前 leader 姿態存成 json")
    p.add_argument("--waypoints", action="store_true",
                   help="每按一次 Enter 追加一個 waypoint (依序經過，最後一個 = 起始姿態)")
    a = p.parse_args(argv)

    install_patches()
    cfg = BiOpenArmLeaderConfig(
        id=a.id,
        left_arm_config=OpenArmLeaderConfigBase(port=a.left_port),
        right_arm_config=OpenArmLeaderConfigBase(port=a.right_port),
    )
    teleop = BiOpenArmLeader(cfg)
    print("\n[注意] 啟動前請讓兩支 leader 自然下垂 (零點姿態)！")
    teleop.connect()
    print("重力補償已開啟。慢慢把手臂抬到水平：應該感覺『變輕、放手會停住』。")
    print("若某關節放手後往上飄 → 該關節 JOINT_SIGN 可能要改 -1，或 JOINT_GC_SCALE 調小。")
    if a.save:
        print(f"擺好你要的起始姿態後按 Enter 存到 {a.save}；Ctrl+C 離開。\n")

    import select

    wps: list[dict] = []

    try:
        while True:
            lines = []
            for side, arm in (("left", teleop.left_arm), ("right", teleop.right_arm)):
                pos = arm._gc.positions()
                tau = arm._gc.last_tau
                lines.append(
                    f"{side:5s} q=" + " ".join(f"{pos[m]:7.1f}" for m in ARM_JOINTS)
                    + f" | g={pos.get(GRIPPER, 0):6.1f} | tau=" + " ".join(f"{t:5.2f}" for t in tau)
                )
            print("\r" + "   ".join(lines), end="", flush=True)
            if a.save and select.select([sys.stdin], [], [], 0)[0]:
                sys.stdin.readline()
                home = {}
                for prefix, arm in (("left_", teleop.left_arm), ("right_", teleop.right_arm)):
                    for m, v in arm._gc.positions().items():
                        home[f"{prefix}{m}.pos"] = round(float(v), 2)
                if a.waypoints:
                    wps.append(home)
                    out = {"waypoints": wps}
                else:
                    out = home
                with open(a.save, "w") as f:
                    json.dump(out, f, indent=2)
                print(f"\n[OK] saved pose #{len(wps) if a.waypoints else 1} -> {a.save}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print()
    finally:
        teleop.disconnect()


def run_home(argv):
    """只控制 follower：移到 reset.json 的起始姿態並保持，按 Enter 後慢慢回下垂再關力矩。"""
    from lerobot.robots.bi_openarm_follower import BiOpenArmFollower, BiOpenArmFollowerConfig
    from lerobot.robots.openarm_follower import OpenArmFollowerConfigBase

    p = argparse.ArgumentParser()
    p.add_argument("--id", required=True, help="與 lerobot-record 的 --robot.id 相同")
    p.add_argument("--left-port", default=DEFAULT_FOLLOWER_LEFT_PORT)
    p.add_argument("--right-port", default=DEFAULT_FOLLOWER_RIGHT_PORT)
    p.add_argument("--home", default=DEFAULT_HOME_PATH, help="reset.json 路徑")
    p.add_argument("--time", type=float, default=DEFAULT_HOME_TIME_S)
    a = p.parse_args(argv)
    if not a.home:
        sys.exit("請用 --home reset.json 或 export OPENARM_HOME=reset.json")
    config_mod.DEFAULT_HOME_PATH = a.home

    patch_follower_park()
    cfg = BiOpenArmFollowerConfig(
        id=a.id,
        left_arm_config=OpenArmFollowerConfigBase(port=a.left_port, side="left"),
        right_arm_config=OpenArmFollowerConfigBase(port=a.right_port, side="right"),
    )
    robot = BiOpenArmFollower(cfg)
    print("\n[注意] follower 將開啟力矩並移動，請淨空桌面、手放在急停旁。")
    robot.connect()
    try:
        go_home(robot, None, load_waypoints(a.home), a.time)
        input("\n[完成] follower 已在起始姿態。按 Enter 讓它慢慢回到下垂並關閉力矩 ...")
    except KeyboardInterrupt:
        print()
    finally:
        robot.disconnect()


def parse_bilateral_flag(args: list[str]) -> tuple[bool, list[str]]:
    """從命令列參數中解析並抽離 --bilateral=true/false，並設定 OPENARM_FF_SCALE。"""
    cleaned = []
    bilateral = None
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("--bilateral="):
            val = arg.split("=", 1)[1].strip().lower()
            bilateral = val in ("1", "true", "yes", "on")
        elif arg == "--bilateral":
            if i + 1 < len(args) and not args[i + 1].startswith("-"):
                val = args[i + 1].strip().lower()
                bilateral = val in ("1", "true", "yes", "on")
                i += 1
            else:
                bilateral = True
        else:
            cleaned.append(arg)
        i += 1

    if bilateral is not None:
        if bilateral:
            cur_scale = config_mod.DEFAULT_FORCE_FEEDBACK_SCALE
            target_scale = 0.05 if cur_scale <= 0 else cur_scale
            config_mod.DEFAULT_FORCE_FEEDBACK_SCALE = target_scale
            logger.info("[gc] 雙向力回饋模式已啟用 (--bilateral=true)，FF_SCALE=%s", config_mod.DEFAULT_FORCE_FEEDBACK_SCALE)
        else:
            config_mod.DEFAULT_FORCE_FEEDBACK_SCALE = 0.0
            logger.info("[gc] 雙向力回饋模式已停用 (--bilateral=false)")

    return bool(bilateral), cleaned


def parse_webrtc_flag(args: list[str]) -> tuple[bool, int, list[str]]:
    """從命令列參數中解析並抽離 -webrtc=true/false / --webrtc=true/false 及 port 設定。

    設定 config 中的 DEFAULT_WEBRTC_ENABLE 與 DEFAULT_WEBRTC_PORT，並抽離該參數避免 LeRobot 報錯。
    """
    cleaned = []
    webrtc_enabled = None
    webrtc_port = config_mod.DEFAULT_WEBRTC_PORT
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith(("--webrtc=", "-webrtc=")):
            val = arg.split("=", 1)[1].strip().lower()
            webrtc_enabled = val in ("1", "true", "yes", "on")
        elif arg in ("--webrtc", "-webrtc"):
            if i + 1 < len(args) and not args[i + 1].startswith("-"):
                val = args[i + 1].strip().lower()
                webrtc_enabled = val in ("1", "true", "yes", "on")
                i += 1
            else:
                webrtc_enabled = True
        elif arg.startswith(("--webrtc-port=", "-webrtc-port=")):
            try:
                webrtc_port = int(arg.split("=", 1)[1].strip())
            except ValueError:
                pass
        elif arg in ("--webrtc-port", "-webrtc-port"):
            if i + 1 < len(args) and not args[i + 1].startswith("-"):
                try:
                    webrtc_port = int(args[i + 1].strip())
                except ValueError:
                    pass
                i += 1
        else:
            cleaned.append(arg)
        i += 1

    if webrtc_enabled is not None:
        if webrtc_enabled:
            config_mod.DEFAULT_WEBRTC_ENABLE = True
            config_mod.DEFAULT_WEBRTC_PORT = webrtc_port
            logger.info("[gc] WebRTC 實時串流伺服器已啟用 (-webrtc=true)，通訊埠: %d", webrtc_port)
        else:
            config_mod.DEFAULT_WEBRTC_ENABLE = False
            logger.info("[gc] WebRTC 實時串流已停用 (-webrtc=false)")

    return bool(webrtc_enabled), webrtc_port, cleaned


def parse_force_monitor_flag(args: list[str]) -> tuple[bool, str | None, list[str]]:
    """解析並過濾 --force-monitor 或 --arduino-port=<port> 參數。"""
    cleaned = []
    monitor_force = False
    arduino_port = config_mod.DEFAULT_ARDUINO_PORT
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith(("--arduino-port=", "-arduino-port=")):
            arduino_port = arg.split("=", 1)[1].strip()
            monitor_force = True
        elif arg in ("--arduino-port", "-arduino-port"):
            if i + 1 < len(args) and not args[i + 1].startswith("-"):
                arduino_port = args[i + 1].strip()
                i += 1
            monitor_force = True
        elif arg in ("--force-monitor", "-force-monitor", "--show-force"):
            monitor_force = True
        else:
            cleaned.append(arg)
        i += 1

    if monitor_force:
        config_mod.DEFAULT_FORCE_MONITOR_ENABLE = True
        if arduino_port:
            config_mod.DEFAULT_ARDUINO_PORT = arduino_port
            logger.info("[force-monitor] 已啟用力矩與 Arduino 壓力實時印出 (Arduino Port=%s)", arduino_port)
        else:
            logger.info("[force-monitor] 已啟用夾爪力矩實時印出")
    return monitor_force, arduino_port, cleaned


def parse_task_slot_flags(args: list[str]) -> tuple[str | None, list[str]]:
    """從命令列參數中解析並抽離 --task 與 --slots 參數。"""
    cleaned = []
    task_or_path = None
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("--task=") or arg.startswith("-task="):
            task_or_path = arg.split("=", 1)[1].strip()
        elif arg.startswith("--slots=") or arg.startswith("-slots="):
            task_or_path = arg.split("=", 1)[1].strip()
        elif arg in ("--task", "-task", "--slots", "-slots"):
            if i + 1 < len(args) and not args[i + 1].startswith("-"):
                task_or_path = args[i + 1].strip()
                i += 1
        else:
            cleaned.append(arg)
        i += 1
    return task_or_path, cleaned


def run_cli_mode(mode: str, rest: list[str]):
    if mode == "check":
        run_check(rest)
        return
    if mode == "home":
        run_home(rest)
        return
    if mode == "teach-slots":
        from openarm_src.utils.slot_teacher import run_teach_slots
        run_teach_slots(rest)
        return

    # 解析並抽離 --task / --slots
    task_or_path, rest = parse_task_slot_flags(rest)

    # 解析並過濾 --bilateral 參數，避免傳給 LeRobot 引發不認識參數的錯誤
    _, cleaned_rest = parse_bilateral_flag(rest)
    # 解析並過濾 -webrtc 參數
    _, _, cleaned_rest = parse_webrtc_flag(cleaned_rest)
    # 解析並過濾 --arduino-port / --force-monitor 參數
    parse_force_monitor_flag(cleaned_rest)

    install_patches()
    if mode == "record-hybrid":
        sys.argv = ["lerobot-record"] + cleaned_rest
        from lerobot.scripts import lerobot_record
        from openarm_src.utils.hybrid_recorder import patch_hybrid_record

        patch_record_loop(lerobot_record)
        patch_hybrid_record(lerobot_record, task_or_path=task_or_path)
        lerobot_record.main()
    elif mode == "record":
        sys.argv = [f"lerobot-{mode}"] + cleaned_rest
        from lerobot.scripts import lerobot_record

        patch_record_loop(lerobot_record)
        lerobot_record.main()
    elif mode == "rollout":
        sys.argv = [f"lerobot-{mode}"] + cleaned_rest
        from lerobot.scripts import lerobot_rollout

        patch_rollout(lerobot_rollout)
        lerobot_rollout.main()
    else:
        sys.argv = [f"lerobot-{mode}"] + cleaned_rest
        from lerobot.scripts import lerobot_teleoperate

        patch_teleop_loop(lerobot_teleoperate)
        lerobot_teleoperate.main()

