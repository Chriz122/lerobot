#!/usr/bin/env python
"""OpenArm 夾爪 (Gripper) 力道驗證與校正工具 (Gripper Force Validation & Calibration Tool)

功能：
1. 透過 Serial 同步讀取 Arduino 壓力感測器數值 (FSR/稱重模組)。
2. 連線 OpenArm Follower 讀取達妙馬達回傳的即時力矩 (Torque, Nm) 與角度 (Position, deg)。
3. 【模式 1: 實時監測 (Monitor)】
   - 同步顯示馬達回傳力矩與 Arduino 感測器數值，方便手動按壓比對。
4. 【模式 2: 自動階梯夾取測試 (Step Test)】
   - 夾爪由開至合，以指定步進角度逐步夾緊壓力感測器。
   - 記錄每個角度下的「馬達回報力矩」與「外部實測壓力」，繪製/輸出資料點。
5. 【模式 3: 自動擬合校正 (Calibrate)】
   - 透過最小平方法擬合：Force (N 或 g) = a * Torque (Nm) + b
   - 輸出並保存校正檔 `gripper_force_calib.json`，後續可直接將馬達力矩換算為真實物理力。

用法：
------
# 1. 搜尋可用的 COM Port
uv run python openarm_src/test_gripper_force.py --list-ports

# 2. 實時手動測試 (只讀取不主動下發動作)
uv run python openarm_src/test_gripper_force.py --arduino-port COM3 --arm-port can0 --side right --mode monitor

# 3. 自動夾取並執行校正 (步進夾緊)
uv run python openarm_src/test_gripper_force.py --arduino-port COM3 --arm-port can0 --side right --mode calibrate
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import threading
import time

import numpy as np
import serial
import serial.tools.list_ports

# 確保專案根目錄在 sys.path 中 (相對於 openarm_src/test/ 向上兩層)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from openarm_src.utils.can_mgr import can_setup_and_check
from openarm_src.utils.config import GRIPPER
from openarm_src.utils.safety import final_safety_net, register_arm, safe_disable, unregister_arm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("gripper_test")


class ArduinoReader:
    """背景執行緒負責讀取 Arduino 壓力感測器資料。"""

    def __init__(self, port: str, baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self.ser: serial.Serial | None = None
        self._stop = threading.Event()
        self.thread: threading.Thread | None = None

        self.lock = threading.Lock()
        self.latest_raw = 0.0
        self.latest_volt = 0.0
        self.latest_force = 0.0
        self.connected = False

    def start(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1.0)
            time.sleep(2.0)  # 等待 Arduino 重啟
            self.connected = True
            logger.info("[Arduino] 成功連線至 %s (Baud: %d)", self.port, self.baudrate)
        except Exception as e:
            logger.error("[Arduino] 無法開啟序列埠 %s: %s", self.port, e)
            raise

        self._stop.clear()
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self._stop.set()
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.ser and self.ser.is_open:
            self.ser.close()
            logger.info("[Arduino] 已中斷序列埠連線")
        self.connected = False

    def _read_loop(self):
        while not self._stop.is_set():
            if not self.ser or not self.ser.is_open:
                break
            try:
                line = self.ser.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                # 解析格式: RAW:<val>,VOLT:<val> 或單一數值
                raw_val = None
                volt_val = None
                force_val = None

                parts = line.split(",")
                for p in parts:
                    if ":" in p:
                        k, v = p.split(":", 1)
                        k = k.strip().upper()
                        try:
                            val = float(v.strip())
                            if k == "RAW":
                                raw_val = val
                            elif k == "VOLT":
                                volt_val = val
                            elif k in ("FORCE", "F", "WEIGHT"):
                                force_val = val
                        except ValueError:
                            pass
                    else:
                        # 純數值直接當 RAW
                        try:
                            raw_val = float(p.strip())
                        except ValueError:
                            pass

                with self.lock:
                    if raw_val is not None:
                        self.latest_raw = raw_val
                    if volt_val is not None:
                        self.latest_volt = volt_val
                    if force_val is not None:
                        self.latest_force = force_val
                    else:
                        # 預設簡易電壓換算估算 (若未指定)
                        self.latest_force = self.latest_raw
            except Exception as e:
                logger.debug("[Arduino] 讀取例外: %s", e)
                time.sleep(0.01)

    def get_latest(self) -> tuple[float, float, float]:
        """傳回 (raw, volt, force)。"""
        with self.lock:
            return self.latest_raw, self.latest_volt, self.latest_force


def list_serial_ports():
    """列出電腦上所有可用的序列埠。"""
    ports = serial.tools.list_ports.comports()
    print("\n可用的序列埠清單 (Available COM Ports):")
    if not ports:
        print("  (未偵測到任何序列埠設備)")
        return
    for p in ports:
        print(f"  - {p.device}: {p.description} ({p.hwid})")
    print()


def create_follower_arm(arm_port: str, side: str = "right"):
    """連線 OpenArm 單臂 Follower。"""
    from lerobot.robots.openarm_follower import OpenArmFollower, OpenArmFollowerConfig

    cfg = OpenArmFollowerConfig(
        port=arm_port,
        side=side,
        id=f"openarm_{side}_follower",
    )
    arm = OpenArmFollower(cfg)
    logger.info("[Robot] 正在連線至機械手臂 (%s, port=%s)...", side, arm_port)
    arm.connect()
    register_arm(arm)
    logger.info("[Robot] 手臂連線成功！")
    return arm


def run_monitor_mode(
    arm=None,
    arduino: ArduinoReader | None = None,
    source: str = "both",
):
    """實時監測模式：支援同時顯示、只看機械手臂馬達力矩、或只看 Arduino 壓力數值。"""
    print("\n" + "=" * 70)
    if source == "both":
        print("  【實時監控模式：同步對照 (Motor + Arduino)】")
        print("  同時顯示馬達回傳力矩與 Arduino 壓力值，方便手動按壓比對。")
    elif source == "motor":
        print("  【實時監控模式：僅監看馬達 (Motor Only)】")
        print("  只讀取並顯示夾爪角度與馬達回傳力矩 (無需連接 Arduino)。")
    elif source == "arduino":
        print("  【實時監控模式：僅監看壓力感測器 (Arduino Only)】")
        print("  只讀取並顯示 SY40/Arduino 感測器電壓與 RAW 數值 (無需連接手臂)。")
    print("  按 Ctrl+C 結束測試")
    print("=" * 70 + "\n")

    try:
        while True:
            parts = []

            # 1. 讀取機械手臂夾爪馬達數值
            if source in ("both", "motor") and arm is not None:
                try:
                    states = arm.bus.sync_read_all_states([GRIPPER])
                    g_state = states.get(GRIPPER, {})
                    pos = g_state.get("position", 0.0)
                    torque = g_state.get("torque", 0.0)
                    parts.append(f"夾爪位置: {pos:6.1f}° | 馬達回傳力矩: {torque:6.3f} Nm")
                except Exception as e:
                    parts.append(f"[馬達讀取錯誤: {e}]")

            # 2. 讀取 Arduino 壓力數值
            if source in ("both", "arduino") and arduino is not None:
                raw, volt, force = arduino.get_latest()
                parts.append(f"Arduino RAW: {raw:6.1f} | 電壓: {volt:5.2f} V | 感測力估算: {force:6.1f}")

            print("\r" + " | ".join(parts), end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n\n已結束監控模式。")


def run_step_and_calibrate(
    arm,
    arduino: ArduinoReader,
    start_deg: float = 40.0,
    end_deg: float = 0.0,
    steps: int = 15,
    hold_sec: float = 1.0,
    calib_out_file: str = "gripper_force_calib.json",
):
    """自動階梯夾緊測試並擬合校正公式。"""
    print("\n" + "=" * 75)
    print("  【自動階梯夾取與校正測試】")
    print(f"  夾爪移動範圍: {start_deg}° -> {end_deg}°，測試階數: {steps} 階")
    print(f"  每階停留採樣: {hold_sec} 秒")
    print("  ⚠ 請確保 Arduino 壓力感測器已放置於夾爪兩指之間！")
    print("=" * 75)

    input("\n請確認感測器已就定位後，按 [Enter] 鍵開始測試 (隨時可按 Ctrl+C 緊急停止)...")

    # 1. 先張開到起始位置
    logger.info("[Test] 正在將夾爪移至起始角度: %.1f°...", start_deg)
    arm.send_action({"gripper.pos": start_deg})
    time.sleep(2.0)

    # 採集歸零基準值 (Tare Offset)
    tare_raws = []
    for _ in range(20):
        r, _, _ = arduino.get_latest()
        tare_raws.append(r)
        time.sleep(0.05)
    offset_raw = np.mean(tare_raws)
    logger.info("[Test] 感測器無受力初始基準值 (Offset): %.2f", offset_raw)

    angles = np.linspace(start_deg, end_deg, steps)
    recorded_torques = []
    recorded_sensor_forces = []
    recorded_angles = []

    try:
        for idx, target_angle in enumerate(angles):
            logger.info("[Test] 階梯 [%d/%d] -> 目標角度: %.1f°", idx + 1, steps, target_angle)
            arm.send_action({"gripper.pos": target_angle})

            # 等待動作就位並穩定接觸
            time.sleep(0.5)

            # 在保持受力期間收集多個樣本平均
            sample_torques = []
            sample_forces = []
            start_t = time.perf_counter()
            while time.perf_counter() - start_t < hold_sec:
                states = arm.bus.sync_read_all_states([GRIPPER])
                g_state = states.get(GRIPPER, {})
                t_val = g_state.get("torque", 0.0)
                r_val, _, _ = arduino.get_latest()

                sample_torques.append(abs(t_val))
                sample_forces.append(max(r_val - offset_raw, 0.0))
                time.sleep(0.03)

            avg_t = float(np.mean(sample_torques))
            avg_f = float(np.mean(sample_forces))

            recorded_torques.append(avg_t)
            recorded_sensor_forces.append(avg_f)
            recorded_angles.append(target_angle)

            print(
                f"       -> 角度={target_angle:5.1f}° | 馬達回報力矩={avg_t:6.3f} Nm | 實測接觸壓力增量={avg_f:6.2f}"
            )

    except KeyboardInterrupt:
        logger.warning("\n使用者手動中斷測試！立即放開夾爪...")
    finally:
        # 測試完成或中斷時，張開夾爪以策安全
        try:
            arm.send_action({"gripper.pos": start_deg})
            time.sleep(1.0)
        except Exception:
            pass

    if len(recorded_torques) < 3:
        logger.warning("資料點過少 (<3 點)，取消擬合校正。")
        return

    # 執行線性回歸校正擬合: Force = a * Torque + b
    x = np.array(recorded_torques)
    y = np.array(recorded_sensor_forces)

    # 排除無接觸（壓力近乎為 0）的空夾階段，只擬合實際接觸受力區段
    mask = y > (0.05 * np.max(y))
    if np.count_nonzero(mask) >= 3:
        x_fit = x[mask]
        y_fit = y[mask]
    else:
        x_fit = x
        y_fit = y

    poly = np.polyfit(x_fit, y_fit, deg=1)
    slope_a = float(poly[0])
    intercept_b = float(poly[1])

    # 計算相關係數 R^2
    y_pred = slope_a * x_fit + intercept_b
    ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
    ss_res = np.sum((y_fit - y_pred) ** 2)
    r_squared = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    print("\n" + "=" * 75)
    print("  【校正分析結果 (Calibration Results)】")
    print(f"  線性擬合公式: 實際力量 (F) = {slope_a:.4f} * 馬達力矩(Nm) + ({intercept_b:.4f})")
    print(f"  擬合判定係數 (R^2): {r_squared:.4f}  (越接近 1.0 代表回報力矩與真實受力線性度越高)")
    if r_squared > 0.85:
        print("  ✔ 驗證結論：OpenArm 馬達力矩回報與真實夾取力道具備高度線性正相關！")
    else:
        print("  ⚠ 提示：R^2 偏低，可能是夾爪齒隙、感測器滑動或彈性形變引起，建議複測或更換平整夾持點。")
    print("=" * 75)

    # 儲存校正參數
    calib_data = {
        "calibrated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "slope_a": slope_a,
        "intercept_b": intercept_b,
        "r_squared": r_squared,
        "offset_zero": offset_raw,
        "formula": "force = slope_a * motor_torque_nm + intercept_b",
        "raw_points": [
            {"angle": a, "torque_nm": t, "sensor_val": f}
            for a, t, f in zip(recorded_angles, recorded_torques, recorded_sensor_forces)
        ],
    }

    out_path = Path(calib_out_file)
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(calib_data, fp, indent=2, ensure_ascii=False)
    logger.info("[校正檔存檔] 已成功保存校正參數至: %s", out_path.resolve())


def main():
    parser = argparse.ArgumentParser(description="OpenArm Gripper 力道回傳測試與 Arduino 壓力校正工具")
    parser.add_argument("--list-ports", action="store_true", help="列出目前電腦上所有的 COM 序列埠")
    parser.add_argument("--arduino-port", default="COM3", help="Arduino 序列埠 (如 COM3 或 /dev/ttyUSB0)")
    parser.add_argument("--baudrate", type=int, default=115200, help="Arduino Serial Baudrate，預設 115200")
    parser.add_argument("--arm-port", default="can0", help="機械手臂 CAN 通訊埠 (如 can0, can1)")
    parser.add_argument("--side", default="right", choices=["left", "right"], help="測試左臂或右臂，預設 right")
    parser.add_argument(
        "--mode",
        default="monitor",
        choices=["monitor", "calibrate"],
        help="執行模式: monitor (即時監看比對) 或 calibrate (階梯夾取自動擬合校正)",
    )
    parser.add_argument(
        "--source",
        default="both",
        choices=["both", "motor", "arduino"],
        help="實時監控模式的資料來源: both (同步對照馬達與Arduino), motor (只看手臂馬達力矩), arduino (只看壓力感測器)",
    )
    parser.add_argument("--start-deg", type=float, default=45.0, help="校正起始開合角度 (開)，預設 45°")
    parser.add_argument("--end-deg", type=float, default=0.0, help="校正終點閉合角度 (夾)，預設 0°")
    parser.add_argument("--steps", type=int, default=12, help="校正夾取階數，預設 12")
    parser.add_argument("--output", default="gripper_force_calib.json", help="校正結果保存檔名")
    args = parser.parse_args()

    if args.list_ports:
        list_serial_ports()
        return

    # 若為 calibrate 模式，必須兩者都有
    need_arduino = args.mode == "calibrate" or args.source in ("both", "arduino")
    need_arm = args.mode == "calibrate" or args.source in ("both", "motor")

    # 1. 啟動 Arduino 讀取執行緒 (若需要)
    arduino = None
    if need_arduino:
        arduino = ArduinoReader(port=args.arduino_port, baudrate=args.baudrate)
        arduino.start()

    # 2. 連線機械手臂 (若需要)
    arm = None
    try:
        if need_arm:
            # 確保 CAN 介面已啟動 (自動執行 lerobot-setup-can 邏輯)
            try:
                can_setup_and_check()
            except Exception as e:
                logger.debug("can_setup_and_check: %s", e)
            arm = create_follower_arm(arm_port=args.arm_port, side=args.side)

        if args.mode == "monitor":
            run_monitor_mode(arm=arm, arduino=arduino, source=args.source)
        else:
            if not arm or not arduino:
                logger.error("[Calibrate] 校正模式需要同時連線手臂與 Arduino！")
                return
            run_step_and_calibrate(
                arm,
                arduino,
                start_deg=args.start_deg,
                end_deg=args.end_deg,
                steps=args.steps,
                calib_out_file=args.output,
            )
    except Exception as e:
        logger.error("[Main] 執行發生錯誤: %s", e, exc_info=True)
    finally:
        if arduino:
            arduino.stop()
        if arm:
            try:
                safe_disable(arm)
                arm.disconnect()
            except Exception:
                pass
            unregister_arm(arm)
        final_safety_net()


if __name__ == "__main__":
    main()

