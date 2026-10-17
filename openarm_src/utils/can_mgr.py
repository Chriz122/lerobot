from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import time

logger = logging.getLogger("openarm_gc.can_mgr")


def get_can_ifaces() -> list[str]:
    return [x.strip() for x in os.environ.get("OPENARM_CAN_IFACES", "can0,can1,can2,can3").split(",") if x.strip()]


def can_setup_and_check(max_tries: int = 3):
    """打開 CAN 介面，並反覆測試直到每條 bus 的 8 顆馬達都回應。
    介面剛打開時第一個訊息常會掉 (看起來像 joint_1 沒回應)，所以第一次 test 當暖機。"""
    if os.environ.get("OPENARM_CAN_SETUP", "1") == "0":
        return
    ifaces = get_can_ifaces()
    arg = "--interfaces=" + ",".join(ifaces)
    r = subprocess.run(["lerobot-setup-can", "--mode=setup", arg], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout + r.stderr)
        sys.exit("[can] lerobot-setup-can setup 失敗，請檢查 USB-CAN 轉接器")
    logger.info(f"[can] {','.join(ifaces)} UP")
    expected = 8 * len(ifaces)
    for t in range(1, max_tries + 1):
        out = subprocess.run(["lerobot-setup-can", "--mode=test", arg], capture_output=True, text=True).stdout
        m = re.search(r"Total motors found:\s*(\d+)", out)
        found = int(m.group(1)) if m else 0
        if found >= expected:
            logger.info(f"[can] 馬達測試通過：{found}/{expected}（第 {t} 次）")
            return
        missing, cur = [], "?"
        for line in out.splitlines():
            mi = re.match(r"\s*(can\d+):", line)
            if mi:
                cur = mi.group(1)
            if "No response" in line:
                mj = re.search(r"\((\w+)\)", line)
                missing.append(f"{cur}/{mj.group(1) if mj else '?'}")
        logger.info(f"[can] 第 {t} 次測試 {found}/{expected}，沒回應：{', '.join(missing)}")
        time.sleep(0.3)
    print(f"\n⚠ 測了 {max_tries} 次仍有馬達沒回應：{', '.join(missing)}")
    print("  建議：檢查急停、24V 電源；或關掉 24V 等 10 秒再開。")
    input("  按 Enter 仍要繼續，Ctrl+C 取消 ...")


def can_teardown():
    if os.environ.get("OPENARM_CAN_DOWN", "0") != "1":
        return
    for iface in get_can_ifaces():
        r = subprocess.run(["ip", "link", "set", iface, "down"], capture_output=True, text=True)
        if r.returncode != 0:
            r = subprocess.run(["sudo", "-n", "ip", "link", "set", iface, "down"], capture_output=True, text=True)
        logger.info(f"[can] {iface} down" + ("" if r.returncode == 0 else f" 失敗：{r.stderr.strip()}"))

