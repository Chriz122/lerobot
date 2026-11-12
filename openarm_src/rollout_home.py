#!/usr/bin/env python3
"""rollout_local.py — 在本機 GPU（1080 Ti）用 lerobot-rollout 推論，條件與 client_home.py 相同：
    歸位（reset.json）→ Enter 開始推論 → 結束回下垂 → 夾爪濾波 → 印出每次推論時間

用來跟 5090 非同步推論（client_home.py）做公平比較：同一個模型、同樣的安全限制與夾爪濾波，
只差在推論在哪台 GPU 上跑。

用法（在 ~/lerobot/openarm_src）：
    export PYTHONPATH=$HOME/lerobot
    export OPENARM_HOME=$HOME/lerobot/openarm_src/reset.json
    python rollout_local.py --strategy.type=base --policy.path=... --duration=60 ...（與 lerobot-rollout 相同的參數）

環境變數：
    OPENARM_GRIPPER_DEADBAND / OPENARM_GRIPPER_SMOOTH  同 client_home.py
    OPENARM_TIMING  1（預設）= 印出每次 SmolVLA 推論時間；0 = 不印
    DEFAULT_RESET_FILE      起始姿態檔，預設 reset.json
    DEFAULT_HOME_TIME_S     每一段 waypoint 的移動秒數，預設 4
    DEFAULT_PARK_ENABLE     1（預設）= 結束時回下垂再關力矩
"""
import sys
import time
from importlib.metadata import entry_points

import client_home  # noqa: F401  載入即套用：歸位、結束回下垂、夾爪濾波（不會啟動 robot_client）

if os.environ.get("OPENARM_TIMING", "1") == "1":
    try:
        import torch
        from lerobot.policies.smolvla import modeling_smolvla as m

        _orig = m.VLAFlowMatching.sample_actions
        _n = [0]

        def _timed(self, *a, **k):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t = time.perf_counter()
            out = _orig(self, *a, **k)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            ms = (time.perf_counter() - t) * 1000
            _n[0] += 1
            print(f"[timing] #{_n[0]:3d}  推論 {ms:6.0f} ms = {ms / 33.3:4.1f} 步 (30Hz)", flush=True)
            return out

        m.VLAFlowMatching.sample_actions = _timed
    except Exception as e:  # 不是 SmolVLA 或版本不同就略過計時
        print(f"[timing] 略過計時：{e}")


def main():
    ep = next(e for e in entry_points(group="console_scripts") if e.name == "lerobot-rollout")
    sys.argv = ["lerobot-rollout"] + sys.argv[1:]
    sys.exit(ep.load()())


if __name__ == "__main__":
    main()