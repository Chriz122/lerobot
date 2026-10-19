from __future__ import annotations

import concurrent.futures
import logging
import os
import subprocess
import sys
import time

logger = logging.getLogger("openarm_gc.can_mgr")

MOTOR_NAMES = {
    0x01: "joint_1",
    0x02: "joint_2",
    0x03: "joint_3",
    0x04: "joint_4",
    0x05: "joint_5",
    0x06: "joint_6",
    0x07: "joint_7",
    0x08: "gripper",
}


def get_can_ifaces() -> list[str]:
    return [x.strip() for x in os.environ.get("OPENARM_CAN_IFACES", "can0,can1,can2,can3").split(",") if x.strip()]


def is_iface_up(iface: str) -> bool:
    """檢查介面是否已經處於 UP 狀態且具備 CAN FD"""
    try:
        r = subprocess.run(["ip", "link", "show", iface], capture_output=True, text=True)
        if r.returncode == 0:
            out = r.stdout
            is_up = "UP" in out
            is_fd = "fd on" in out.lower() or "canfd" in out.lower() or "dlen" in out.lower()
            return is_up and is_fd
    except Exception:
        pass
    return False


def probe_bus(iface: str, target_motors: list[int] | None = None) -> tuple[str, list[int], list[int]]:
    """在單一 CAN bus 上快速探測馬達。回傳 (iface, found_list, missing_list)"""
    import can
    from lerobot.scripts.lerobot_setup_can import test_motor

    if target_motors is None:
        target_motors = list(range(0x01, 0x09))

    found = []
    missing = []
    try:
        bus = can.interface.Bus(
            channel=iface,
            interface="socketcan",
            bitrate=1000000,
            data_bitrate=5000000,
            fd=True,
        )
    except Exception as e:
        logger.error(f"[can] 無法開啟 {iface}: {e}")
        return iface, [], target_motors

    try:
        # 清除快取中的殘留訊息
        while bus.recv(timeout=0.002):
            pass

        for mid in target_motors:
            responses, err = test_motor(bus, mid, timeout=0.08, use_fd=True)
            if responses and not err:
                found.append(mid)
            else:
                missing.append(mid)
    finally:
        bus.shutdown()

    return iface, found, missing


def can_setup_and_check(max_tries: int = 3):
    """打開 CAN 介面並測試每條 bus 的 8 顆馬達。
    已在 UP 狀態的介面會直接沿用，不需重複 down/up 造成硬體重置與掉包。
    4 條 bus 採多執行緒平行檢測，收到回應即刻返回，將啟動時間從 70+ 秒降為毫秒級。"""
    if os.environ.get("OPENARM_CAN_SETUP", "1") == "0":
        return

    ifaces = get_can_ifaces()

    # 1. 檢查哪些介面需要 setup（若已在 UP + CAN FD 狀態則略過）
    down_ifaces = [i for i in ifaces if not is_iface_up(i)]
    if down_ifaces:
        logger.info(f"[can] 設定並開啟介面: {','.join(down_ifaces)}")
        arg = "--interfaces=" + ",".join(down_ifaces)
        r = subprocess.run(["lerobot-setup-can", "--mode=setup", arg], capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout + r.stderr)
            sys.exit("[can] lerobot-setup-can setup 失敗，請檢查 USB-CAN 轉接器")
    else:
        logger.info(f"[can] {','.join(ifaces)} 已在 UP 狀態，略過介面重設")

    expected = 8 * len(ifaces)

    # 2. 並行快速探測馬達（針對尚未回應的馬達進行 probe）
    all_found: dict[str, set[int]] = {i: set() for i in ifaces}
    missing_desc: list[str] = []

    for t in range(1, max_tries + 1):
        pending_targets = {
            i: [m for m in range(0x01, 0x09) if m not in all_found[i]]
            for i in ifaces
        }
        active_ifaces = [i for i, mids in pending_targets.items() if mids]
        if not active_ifaces:
            break

        start_t = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(active_ifaces)) as executor:
            futures = {
                executor.submit(probe_bus, iface, pending_targets[iface]): iface
                for iface in active_ifaces
            }
            for f in concurrent.futures.as_completed(futures):
                iface, found_ids, _ = f.result()
                all_found[iface].update(found_ids)

        elapsed = time.perf_counter() - start_t
        total_found = sum(len(s) for s in all_found.values())

        if total_found >= expected:
            logger.info(f"[can] 馬達測試通過：{total_found}/{expected}（第 {t} 次，耗時 {elapsed*1000:.1f}ms）")
            return

        # 整理缺少的馬達
        missing_desc = []
        for iface in ifaces:
            for m in range(0x01, 0x09):
                if m not in all_found[iface]:
                    name = MOTOR_NAMES.get(m, f"0x{m:02X}")
                    missing_desc.append(f"{iface}/{name}")

        logger.info(f"[can] 第 {t} 次測試 {total_found}/{expected}，沒回應：{', '.join(missing_desc)}（耗時 {elapsed*1000:.1f}ms）")
        time.sleep(0.05)

    print(f"\n⚠ 測了 {max_tries} 次仍有馬達沒回應：{', '.join(missing_desc)}")
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

