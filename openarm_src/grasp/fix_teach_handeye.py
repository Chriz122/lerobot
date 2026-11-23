#!/usr/bin/env python3
"""把舊示教檔的手眼旋轉換成正確的名義值（不用重新示教）。

舊的名義值 --handeye-rpy -35 0 0 把「影像往右」當成 TCP x，但真機側向夾取時 TCP x 朝上，
左右對準會變成前後移動（20261123_194426 右臂往旁邊跑、跟丟）。正確的是 -35 0 90（影像往右 = TCP +y）。

示教檔裡存的是相機量到的 (u, v, 深度)；這裡用舊的手眼換回相機座標的瓶子點，再用新的手眼換成 TCP 座標，
所以示教內容（瓶子位置、下降路徑）都不變。原檔備份成 teach_<side>.json.bak。

用法：
  python -m openarm_src.grasp.fix_teach_handeye            # 修正 teach_left.json、teach_right.json
  python -m openarm_src.grasp.fix_teach_handeye path.json  # 指定檔案
"""
from __future__ import annotations

import json
import os
import shutil
import sys

import numpy as np

from openarm_src.grasp.kinematics import matrix_to_rpy, rpy_to_matrix
from openarm_src.grasp.servo import NOMINAL_CAM_P_TCP, NOMINAL_HANDEYE_RPY

HERE = os.path.dirname(os.path.abspath(__file__))


def fix(path: str) -> None:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    r_old, p_old = np.array(d["r_tcp_cam"]), np.array(d["cam_p_tcp"])
    r_new, p_new = rpy_to_matrix(NOMINAL_HANDEYE_RPY), np.array(NOMINAL_CAM_P_TCP)
    if d.get("handeye_source") == "measured":
        print(f"[略過] {path}：手眼是實際量測的（{np.round(matrix_to_rpy(r_old), 1)}），不改")
        return
    if np.allclose(r_old, r_new, atol=1e-6) and np.allclose(p_old, p_new, atol=1e-6):
        print(f"[略過] {path}：已經是正確的名義值")
        return
    pt_cam = r_old.T @ (np.array(d["bottle_tcp"]) - p_old)          # 相機座標的瓶子點（量測值，不變）
    d["r_tcp_cam"] = r_new.tolist()
    d["cam_p_tcp"] = p_new.tolist()
    d["bottle_tcp"] = (r_new @ pt_cam + p_new).tolist()
    d["handeye_rpy"] = list(NOMINAL_HANDEYE_RPY)
    if not os.path.exists(path + ".bak"):
        shutil.copy2(path, path + ".bak")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)
    print(f"[已修正] {path}：手眼 {np.round(matrix_to_rpy(r_old), 1).tolist()} -> {list(NOMINAL_HANDEYE_RPY)}"
          f"（備份 {path}.bak）")


def main() -> None:
    paths = sys.argv[1:] or [os.path.join(HERE, f"teach_{s}.json") for s in ("left", "right")]
    for p in paths:
        if os.path.exists(p):
            fix(p)
        else:
            print(f"[略過] 找不到 {p}")


if __name__ == "__main__":
    main()
