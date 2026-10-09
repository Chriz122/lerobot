#!/usr/bin/env python3
"""
eval_trials.py — OpenArm 實機成功率測試（多 checkpoint 交錯、可盲測、可中斷續測）

每一回合：
  1. 呼叫 rollout_home.py：follower 平滑移到 reset.json 起始姿態 → 等你擺好物體按 Enter
     → 模型推論 duration 秒（可 Ctrl+C 提早結束）→ 手臂慢慢回下垂
  2. 你輸入結果：成功 / 失敗(原因) / 作廢重做，選填放上托盤的物體數與備註
  3. 寫入 CSV，隨時可中斷，下次用同一個 --log 會自動接續剩下的回合

用法（在 ~/lerobot，與 rollout_home.py、openarm_gc.py、reset.json 同一資料夾）：

  python eval_trials.py \
      --ckpts 200k=$HOME/models/act_openarm_200k/200000/pretrained_model \
              400k=$HOME/models/act_openarm_200k/400000/pretrained_model \
      --trials 10 --duration 45 --n-objects 2 --blind \
      --log eval_bottle_tray.csv \
      -- \
      --robot.type=bi_openarm_follower ...（其餘 robot / cameras / task 參數，與 lerobot-rollout 相同）

只看統計結果：
  python eval_trials.py --summary --log eval_bottle_tray.csv
"""
import argparse
import csv
import math
import os
import random
import signal
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime

FAIL_MODES = {
    "1": "沒抓到",
    "2": "抓到後掉落",
    "3": "放置失敗/放歪",
    "4": "動作異常/碰撞",
    "5": "卡住/超時",
    "6": "其他",
}
FIELDS = ["trial", "time", "ckpt", "success", "placed", "failure_mode", "elapsed_s", "notes"]


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def read_log(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_log(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def summarize(path):
    rows = read_log(path)
    if not rows:
        print(f"{path} 沒有資料")
        return
    by = defaultdict(list)
    for r in rows:
        by[r["ckpt"]].append(r)

    print(f"\n{'=' * 64}\n成功率統計（{path}，共 {len(rows)} 回合）\n{'=' * 64}")
    print(f"{'ckpt':10s} {'回合':>4s} {'成功':>4s} {'成功率':>7s}   95% 信賴區間   {'平均放置數':>8s}")
    for ck, rs in by.items():
        n = len(rs)
        k = sum(r["success"] == "1" for r in rs)
        lo, hi = wilson(k, n)
        placed = [int(r["placed"]) for r in rs if r["placed"] not in ("", None)]
        avg_p = f"{sum(placed) / len(placed):.2f}" if placed else "-"
        print(f"{ck:10s} {n:4d} {k:4d} {k / n:7.0%}   [{lo:4.0%}, {hi:4.0%}]   {avg_p:>8s}")

    print("\n失敗原因：")
    for ck, rs in by.items():
        c = Counter(r["failure_mode"] for r in rs if r["success"] == "0")
        if c:
            print(f"  {ck}: " + "、".join(f"{m} ×{v}" for m, v in c.most_common()))
    print("\n註：每組 10 回合的信賴區間約 ±25～30%，要分辨兩個 checkpoint 的差異，建議每組 20 回合以上。")


def ask_result(n_objects):
    while True:
        r = input("\n結果？  s = 成功   f = 失敗   x = 作廢重做（設定/硬體問題，不計入）: ").strip().lower()
        if r in ("s", "f", "x"):
            break
    if r == "x":
        return None
    mode = ""
    if r == "f":
        print("  失敗原因：" + "  ".join(f"{k}={v}" for k, v in FAIL_MODES.items()))
        while mode not in FAIL_MODES:
            mode = input("  選擇: ").strip()
        mode = FAIL_MODES[mode]
    placed = ""
    if n_objects:
        while True:
            p = input(f"  放上托盤的物體數 (0-{n_objects}): ").strip()
            if p.isdigit() and 0 <= int(p) <= n_objects:
                placed = p
                break
    notes = input("  備註（可留空）: ").strip()
    return {"success": "1" if r == "s" else "0", "failure_mode": mode, "placed": placed, "notes": notes}


def run_trial(policy_path, duration, rollout_args):
    cmd = [sys.executable, "rollout_home.py",
           "--strategy.type=base",
           f"--policy.path={policy_path}",
           f"--duration={duration}",
           *rollout_args]
    # 父行程忽略 Ctrl+C（用 Python handler 而非 SIG_IGN，子行程 exec 後會恢復預設，
    # 所以 Ctrl+C 只會讓推論提早結束、手臂回下垂，不會中斷整個測試）
    old = signal.signal(signal.SIGINT, lambda s, f: None)
    t0 = time.time()
    try:
        rc = subprocess.call(cmd)
    finally:
        signal.signal(signal.SIGINT, old)
    return rc, time.time() - t0


def main():
    argv = sys.argv[1:]
    rollout_args = []
    if "--" in argv:
        i = argv.index("--")
        argv, rollout_args = argv[:i], argv[i + 1:]

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", help="名稱=路徑，例如 200k=/path/to/pretrained_model")
    ap.add_argument("--trials", type=int, default=10, help="每個 checkpoint 的回合數")
    ap.add_argument("--duration", type=float, default=45)
    ap.add_argument("--n-objects", type=int, default=0, help="每回合物體數（填了會詢問放上幾個）")
    ap.add_argument("--blind", action="store_true", help="測試中不顯示是哪個 checkpoint")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", default="eval_results.csv")
    ap.add_argument("--summary", action="store_true", help="只顯示統計")
    a = ap.parse_args(argv)

    if a.summary:
        summarize(a.log)
        return
    if not a.ckpts or not rollout_args:
        ap.error("需要 --ckpts 以及 -- 之後的 robot/cameras/task 參數")

    ckpts = dict(c.split("=", 1) for c in a.ckpts)
    for name, p in ckpts.items():
        if not os.path.exists(os.path.join(os.path.expanduser(p), "model.safetensors")):
            sys.exit(f"找不到模型：{name} → {p}")

    # 續測：扣掉 log 裡已完成的回合
    done = Counter(r["ckpt"] for r in read_log(a.log))
    queue = []
    for name in ckpts:
        queue += [name] * max(0, a.trials - done[name])
    random.Random(a.seed + sum(done.values())).shuffle(queue)
    total = len(queue)
    if not total:
        print("全部回合都已完成。")
        summarize(a.log)
        return
    print(f"待測 {total} 回合（已完成 {sum(done.values())}），順序已隨機交錯"
          + ("，盲測模式" if a.blind else ""))

    trial_no = sum(done.values())
    i = 0
    while i < len(queue):
        name = queue[i]
        shown = "???" if a.blind else name
        print(f"\n{'#' * 64}\n# 回合 {i + 1}/{total}   checkpoint: {shown}\n{'#' * 64}")
        rc, elapsed = run_trial(ckpts[name], a.duration, rollout_args)
        if rc not in (0, None):
            print(f"\n(推論程式結束碼 {rc}，若是硬體/連線錯誤請選 x 作廢重做)")
        res = ask_result(a.n_objects)
        if res is None:
            print("→ 作廢，這回合重做")
            continue
        trial_no += 1
        append_log(a.log, {"trial": trial_no, "time": datetime.now().isoformat(timespec="seconds"),
                           "ckpt": name, "elapsed_s": f"{elapsed:.1f}", **res})
        i += 1
        if input("\nEnter 繼續下一回合，q 先暫停（下次會接續）: ").strip().lower() == "q":
            break

    summarize(a.log)


if __name__ == "__main__":
    main()
