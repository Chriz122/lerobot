from __future__ import annotations

import argparse
import datetime
import json
import logging
from pathlib import Path
import time

from openarm_src.utils.config import (
    ARM_JOINTS,
    DEFAULT_APPROACH_OFFSET_DEG,
    DEFAULT_GRIPPER_MIN_DEG,
    DEFAULT_LEADER_ID,
    DEFAULT_LEADER_LEFT_PORT,
    DEFAULT_LEADER_RIGHT_PORT,
    DEFAULT_SLOT_FILENAME,
    DEFAULT_SLOTS_COUNT,
    DEFAULT_TASK_NAME,
    GRIPPER,
)
from openarm_src.utils.patches import install_patches

logger = logging.getLogger("openarm_gc.slot_teacher")

DEFAULT_TASKS_ROOT = Path(__file__).resolve().parent.parent / "tasks"


def get_task_slot_path(task_name: str, filename: str = DEFAULT_SLOT_FILENAME) -> Path:
    """取得特定任務的槽位配置檔案路徑: openarm_src/tasks/<task_name>/tray_slots.json"""
    task_dir = DEFAULT_TASKS_ROOT / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    return task_dir / filename


def run_teach_slots(argv: list[str]) -> None:
    """互動式托盤槽位示教工具。

    引導操作員透過 Leader 重力補償將夾爪移至各槽位擺放位置，
    自動計算並生成Approach、Place、Release、Retract路徑點，
    並依任務名稱存放至 openarm_src/tasks/<task_name>/tray_slots.json。
    """
    from lerobot.teleoperators.bi_openarm_leader import BiOpenArmLeader
    from lerobot.teleoperators.bi_openarm_leader.config_bi_openarm_leader import BiOpenArmLeaderConfig
    from lerobot.teleoperators.openarm_leader.config_openarm_leader import OpenArmLeaderConfigBase

    parser = argparse.ArgumentParser(description="托盤槽位 Waypoint 示教工具 (Teach Slots)")
    parser.add_argument("--task", default=DEFAULT_TASK_NAME, help=f"任務名稱 (預設: {DEFAULT_TASK_NAME}，將建立專屬任務資料夾)")
    parser.add_argument("--slots", type=int, default=DEFAULT_SLOTS_COUNT, help=f"槽位數量 (預設: {DEFAULT_SLOTS_COUNT})")
    parser.add_argument("--save", default=None, help="自訂儲存路徑 (若未指定，預設存至 openarm_src/tasks/<task>/tray_slots.json)")
    parser.add_argument("--id", default=DEFAULT_LEADER_ID, help=f"Leader 手臂 ID (預設: {DEFAULT_LEADER_ID}，讀取校正參數)")
    parser.add_argument("--left-port", default=DEFAULT_LEADER_LEFT_PORT, help=f"左臂 CAN 埠號 (預設: {DEFAULT_LEADER_LEFT_PORT})")
    parser.add_argument("--right-port", default=DEFAULT_LEADER_RIGHT_PORT, help=f"右臂 CAN 埠號 (預設: {DEFAULT_LEADER_RIGHT_PORT})")
    parser.add_argument("--approach-offset-deg", type=float, default=DEFAULT_APPROACH_OFFSET_DEG, help=f"自動生成 Approach 上方點時的關節抬升補償度數 (預設: {DEFAULT_APPROACH_OFFSET_DEG}°)")
    parser.add_argument("--gripper-open-deg", type=float, default=DEFAULT_GRIPPER_MIN_DEG, help=f"Release 階段張開夾爪的目標角度 (預設: {DEFAULT_GRIPPER_MIN_DEG}°)")

    args = parser.parse_args(argv)

    save_path = Path(args.save) if args.save else get_task_slot_path(args.task)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    install_patches()
    cfg = BiOpenArmLeaderConfig(
        id=args.id,
        left_arm_config=OpenArmLeaderConfigBase(port=args.left_port),
        right_arm_config=OpenArmLeaderConfigBase(port=args.right_port),
    )
    teleop = BiOpenArmLeader(cfg)

    print("=" * 75)
    print(f"  [托盤槽位示教工具] 任務名稱: {args.task}")
    print(f"  槽位數量: {args.slots} | 儲存目標: {save_path}")
    print("=" * 75)
    print("\n[注意] 啟動前請讓兩支 Leader 自然下垂 (零點姿態)！")

    teleop.connect()
    print("[OK] 重力補償已啟用，手臂已進入輕盈浮動狀態。\n")

    slots_data: dict[str, dict] = {}

    try:
        for slot_idx in range(1, args.slots + 1):
            print("-" * 60)
            print(f"【示教第 {slot_idx}/{args.slots} 個槽位】")
            print(f"  請拉動手臂，將夾爪移動到托盤「第 {slot_idx} 格」的【工整貼盤放置位置】...")
            print("  調整完畢後，按 [Enter] 確認記錄該點...")

            input()

            # 讀取當前姿態 (放置點)
            place_pose: dict[str, float] = {}
            for prefix, arm in (("left_", teleop.left_arm), ("right_", teleop.right_arm)):
                for m, v in arm._gc.positions().items():
                    place_pose[f"{prefix}{m}.pos"] = round(float(v), 2)

            # 自動生成上方 Approach 接近點 (肩膀/手肘 Joint 2 向上抬起)
            approach_pose = dict(place_pose)
            for side in ("left_", "right_"):
                j2_key = f"{side}joint_2.pos"
                if j2_key in approach_pose:
                    # joint_2 抬高安全餘裕
                    approach_pose[j2_key] = round(approach_pose[j2_key] - args.approach_offset_deg, 2)

            # 生成張開夾爪動作 (Release)
            release_pose = dict(place_pose)
            for side in ("left_", "right_"):
                g_key = f"{side}gripper.pos"
                if g_key in release_pose:
                    release_pose[g_key] = float(args.gripper_open_deg)

            # 生成垂直退回點 (Retract): 與 Approach 姿態相同但保持夾爪張開
            retract_pose = dict(approach_pose)
            for side in ("left_", "right_"):
                g_key = f"{side}gripper.pos"
                if g_key in retract_pose:
                    retract_pose[g_key] = float(args.gripper_open_deg)

            slots_data[str(slot_idx)] = {
                "slot_id": slot_idx,
                "name": f"Slot_{slot_idx}",
                "approach": approach_pose,
                "place": place_pose,
                "release": release_pose,
                "retract": retract_pose,
            }

            print(f"  [OK] 已成功記錄 Slot {slot_idx}！")
            time.sleep(0.3)

        # 封裝為完整任務 JSON
        result = {
            "task": args.task,
            "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "num_slots": args.slots,
            "approach_offset_deg": args.approach_offset_deg,
            "gripper_open_deg": args.gripper_open_deg,
            "slots": slots_data,
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print("\n" + "=" * 75)
        print(f"[OK] 全部 {args.slots} 個槽位示教完成！")
        print(f"檔案已儲存至: {save_path.resolve()}")
        print("=" * 75)

    except KeyboardInterrupt:
        print("\n[注意] 示教中斷，未儲存檔案。")
    finally:
        teleop.disconnect()
