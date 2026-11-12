"""
OpenArm LeRobot 外掛入口腳本 (openarm_gc)
完整功能說明、環境變數表格與 CLI 命令範例，請參閱 openarm_src/README.md。
"""
from __future__ import annotations

import atexit
import logging
from pathlib import Path
import signal
import sys

from openarm_src.utils.log_config import configure_logging

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
    configure_logging()
    valid_modes = ("record", "teleoperate", "rollout", "check", "home", "teach-slots", "record-hybrid")
    if len(sys.argv) < 2 or sys.argv[1] not in valid_modes:
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