#!/usr/bin/env python
"""
OpenArm (v1) leader 重力補償 + 起始姿態 (home) — LeRobot 外掛包裝
Gravity compensation for OpenArm leader arms + home pose before every episode.

不修改 LeRobot 原始碼：本腳本在執行時 monkey-patch
  - OpenArmLeader.connect / get_action / disconnect  -> 背景執行緒 (200 Hz) 送 MIT 指令
        kp=0, kd=小阻尼, tau_ff = Pinocchio 算出的重力力矩
  - lerobot_record.record_loop -> 每個「錄製」episode 開始前，leader + follower
        一起用 min-jerk 軌跡平滑移到 home.json，然後 leader 放開回到重力補償模式
  - lerobot_rollout (DAgger) -> 啟動前移到 home.json；推論時 leader 即時鏡像跟隨 follower，介入時自動放開

用法 / Usage
------------
  # 0) 產生 URDF (在有 ROS 2 + openarm_description 的機器上)，並設定環境變數
  export OPENARM_URDF=~/openarm_bimanual.urdf

  # 1) 檢查重力補償方向 + 擺出起始姿態並存檔
  python openarm_gc.py check --id my_bimanual_leader --left-port can2 --right-port can3 --save home.json

  # 2) 錄資料 (參數與 lerobot-record 完全相同，只是把 lerobot-record 換成 python openarm_gc.py record)
  export OPENARM_HOME=home.json
  python openarm_gc.py record --robot.type=bi_openarm_follower ... --teleop.type=bi_openarm_leader ...

  # 2b) 只讓 follower 移到起始姿態 (不需要 leader)，按 Enter 後慢慢回下垂
  python openarm_gc.py home --id bi_follower --left-port can1 --right-port can0 --home reset.json

  # 3) 只做遙操作 (無 home)
  python openarm_gc.py teleoperate <lerobot-teleoperate 參數>

  # 4) HIL / DAgger (參數與 lerobot-rollout 相同；設了 OPENARM_HOME 會在 policy 開始前先 home 一次)
  python openarm_gc.py rollout --strategy.type=dagger --policy.path=... --teleop.type=bi_openarm_leader ...

  # 力回饋雙向控制開關 (Bilateral teleoperation)
  --bilateral=true / --bilateral=false (在 record 或 teleoperate 時可用，預設依 OPENARM_FF_SCALE 決定)

環境變數 / Env vars
  OPENARM_URDF        (必填) bimanual URDF 路徑，關節名 openarm_{left,right}_joint1..7
  OPENARM_HOME        home.json 路徑；不設則不做 homing
  OPENARM_HOME_TIME   移到 home 的時間 (秒)，預設 4
  OPENARM_TABLE_Z     (選填) 桌面高度 (URDF 世界座標, m)，設了會在 homing 前檢查碰桌
  OPENARM_TABLE_MARGIN 碰桌檢查的安全餘裕 (m)，預設 0.05
  OPENARM_PARK        1 (預設) = follower 結束時先慢慢回下垂再關力矩；0 = 原本行為 (直接關力矩掉下來)
  OPENARM_PARK_TIME   回下垂每段的秒數，預設 3
  OPENARM_HOME_AT     record (預設，開錄前回起始姿態) / reset (episode 結束就回)
  OPENARM_KD_SCALE    阻尼倍率，預設 1.0 (覺得黏就調小，0 = 無阻尼)
  OPENARM_FRICTION_SCALE 摩擦補償倍率，預設 0 (關閉)；建議從 0.5 開始試
  OPENARM_FRICTION_EPS 摩擦補償的速度平滑區 (rad/s)，預設 0.15；靜止時會抖就調大
  OPENARM_CONTROL_HZ  重力補償迴圈頻率，預設 200
  OPENARM_GC_SCALE    重力補償整體倍率，預設 1.0 (第一次測試建議 0.5)
  # --- 力回饋 (Bilateral) ---
  OPENARM_FF_SCALE    follower 夾爪力回饋倍率，預設 0.05；設 0 關閉
  OPENARM_FF_SIGN     力回饋方向，預設 -1.0；若手感方向相反設 1.0
  OPENARM_FF_LIMIT    每關節力回饋上限 (Nm)，預設 0.2
  OPENARM_FF_ALPHA    力回饋低通係數，預設 0.6；越小越穩定
  OPENARM_FF_DEADBAND 力回饋死區 (Nm)，預設 0.02
  # -------------------------
  OPENARM_LEADER_SIDE 單臂 openarm_leader 時指定 left / right
  OPENARM_GC_STATS    1 (預設) = 每 5 秒印出重力補償迴圈實際頻率 / CAN 通訊時間；0 = 關閉
  OPENARM_PROFILE     1 (預設) = record 時每 5 秒印出主迴圈各步驟耗時；0 = 關閉
  OPENARM_LEADER_TRACK 1 (預設) = DAgger 推論時 leader 鏡像跟隨 follower，介入時自動釋放；0 = 關閉跟隨

  --- CAN 管理 / 安全關閉 ---
  OPENARM_CAN_SETUP   1 (預設) = 開始前自動 lerobot-setup-can 打開介面，並測試到所有馬達都回應為止；0 = 不管
  OPENARM_CAN_IFACES  要管理的介面，預設 can0,can1,can2,can3
  OPENARM_CAN_DOWN    0 (預設) = 結束後保持介面 UP；1 = 所有手臂都關完力矩後才把介面關掉
  OPENARM_DISABLE_RETRY  關力矩時每支手臂重送 disable 的次數，預設 3
  OPENARM_ENABLE_RETRY   開力矩時每顆馬達重送 enable 的次數，預設 3；無回覆會中止啟動
"""

from __future__ import annotations

import atexit
import logging
from pathlib import Path
import signal
import sys

# 確保專案根目錄在 sys.path 中（無論在 ~/lerobot 或 ~/lerobot/openarm_src 執行都能正確找到 openarm_src）
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# 僅匯入 openarm_gc CLI 入口所需函式
from openarm_src.utils import (
    can_setup_and_check,
    can_teardown,
    final_safety_net,
    run_cli_mode,
)

logger = logging.getLogger("openarm_gc")


def main():
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2 or sys.argv[1] not in ("record", "teleoperate", "rollout", "check", "home"):
        print(__doc__)
        sys.exit(1)
    mode, rest = sys.argv[1], sys.argv[2:]

    # kill / 關終端機 (SIGTERM, SIGHUP) 也走正常的關閉流程
    def _on_signal(signum, _frame):
        raise KeyboardInterrupt(f"signal {signum}")

    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _on_signal)
    atexit.register(final_safety_net)  # 萬一 finally 沒跑到的最後保險

    can_setup_and_check()
    try:
        run_cli_mode(mode, rest)
    finally:
        final_safety_net()  # 還連著的手臂一律關力矩
        can_teardown()  # 只在 OPENARM_CAN_DOWN=1 時關介面，且一定在關力矩之後


if __name__ == "__main__":
    main()