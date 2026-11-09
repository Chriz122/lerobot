#!/usr/bin/env python3
"""robot_client + 歸位 + 夾爪濾波

流程：連線 → Enter 歸位到 reset.json → Enter 開始推論 → Ctrl+C 結束回下垂
夾爪濾波（只在推論期間、只作用在 *gripper.pos）：
    OPENARM_GRIPPER_DEADBAND  變化小於幾度就忽略，預設 3；設 0 關閉死區
    OPENARM_GRIPPER_SMOOTH    每步往新目標靠近的比例 0~1，預設 0.3；設 1 關閉平滑

用法（在 ~/lerobot/openarm_src）：
    export PYTHONPATH=$HOME/lerobot
    export OPENARM_HOME=$HOME/lerobot/openarm_src/reset.json
    python client_home.py --server_address=... --robot.type=bi_openarm_follower ...
"""
import os
import runpy
import sys
import rollout_home                      # 先載入 openarm_src（避免 segfault）

DEADBAND = float(os.environ.get("OPENARM_GRIPPER_DEADBAND", "3"))
ALPHA = float(os.environ.get("OPENARM_GRIPPER_SMOOTH", "0.3"))

rollout_home.install_patches()           # 歸位 + 結束回下垂；也會註冊 bi_openarm_follower
cls = rollout_home._find_bi_follower_cls()

_active = [False]
_last = {}

_home_connect = cls.connect


def connect(self, *a, **k):
    _active[0] = False
    _last.clear()
    _home_connect(self, *a, **k)         # 歸位、等第二次 Enter
    _active[0] = True                    # 之後才開始濾波
    print(f"[gripper] 濾波啟用：deadband={DEADBAND}°  smooth={ALPHA}")


cls.connect = connect

_orig_send = cls.send_action


def send_action(self, action, *a, **k):
    if _active[0]:
        action = dict(action)
        for key, v in action.items():
            if not key.endswith("gripper.pos"):
                continue
            v = float(v)
            prev = _last.get(key)
            if prev is not None:
                d = v - prev
                v = prev if abs(d) < DEADBAND else prev + ALPHA * d
            _last[key] = v
            action[key] = v
    return _orig_send(self, action, *a, **k)


cls.send_action = send_action

if __name__ == "__main__":
    sys.argv = ["robot_client"] + sys.argv[1:]
    runpy.run_module("lerobot.async_inference.robot_client", run_name="__main__")