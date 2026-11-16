# OpenArm + RealSense + AprilTag Cube：最後一哩靠近與夾取

本目錄實作 **OpenArm 手臂如何靠近並夾起 5×5×5 cm AprilTag 方塊**，做法依循《ACT／VLA Last-Mile Manipulation：學生快速實作指南》：

> 先跑傳統 IBVS，再學習「傳統物理模型算錯多少」（Physics-Informed Residual），最後加入保守的收斂檢查（Convergence Guard）。

套件的示範程式是 2 維影像誤差 (u, v) 的離線模擬。只對準 (u, v) 手臂不會往方塊前進，所以這裡把同一套方法擴成 4 維，並接上運動學、下降與夾取。

---

## 1. 手臂如何靠近方塊

```
 Slow Policy（ACT/VLA、遙操作或 homing）把手臂帶到「腕部相機看得到 Tag」的位置
                               │
                               ▼
 ┌─ 前置 ──────────────────────────────────────────────────────────────────┐
 │ [示教 s*]  夾爪擺在「閉合就能夾到」的位置 → 沿接近軸後退 d → 記錄 s*    │
 │ [微幅試探] 在原地附近送小命令 a，量 Tag 特徵變化 Δs → 訓練 Jacobian 殘差 │
 └─────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
 ┌─ SERVO：閉迴路 IBVS（20 Hz）────────────────────────────────────────────┐
 │ 腕部 D405 取像 → 偵測 Tag → s = [u, v, w, ψ]                            │
 │ e = s − s*                                                              │
 │ 命令 a = [dx, dy, dz, dyaw]（相機座標）                                  │
 │    ibvs    : u_IBVS = −λ·J_phys⁺·e                                      │
 │    learned : u_nom  = −λ·(J_phys + ΔJθ)⁺·e                              │
 │    guarded : 預測 u_nom 比 u_IBVS 好才採用，否則退回 u_IBVS              │
 │ a → 手臂雅可比（阻尼最小二乘）→ 7 軸關節增量 → 限幅與安全檢查 → 送出     │
 │ 中心 ≤ 3px、視邊長 ≤ 3px、偏轉 ≤ 2° 連續 3 幀 → 收斂                    │
 └─────────────────────────────────────────────────────────────────────────┘
                               │  此時夾爪在方塊正上方固定高度，且已轉正
                               ▼
   DESCEND：沿夾爪接近軸開迴路直線下降 d
                               ▼
   GRASP  ：閉合夾爪，夾爪停在半路 = 夾到東西
                               ▼
   LIFT   ：沿接近軸後退提起
```

四個特徵各自負責一件事：

| 特徵 | 意義 | 由哪個命令修正 |
|---|---|---|
| `u, v` | Tag 中心像素 | `dx, dy`：水平對準 |
| `w` | Tag 視邊長（√像素面積），`w = f·L/Z` | `dz`：**靠近到預備距離** |
| `ψ` | Tag 在影像中的旋轉角（對 90° 取模） | `dyaw`：夾爪轉正對齊方塊邊 |

幾個設計重點：

- **先對準再靠近**：中心誤差大於 130 px 時不前進，40 px 以內才全速前進；Tag 貼近畫面邊緣時也停止前進，避免 Tag 跑出畫面。
- **最後一段開迴路**：太近時 Tag 會超出視野或失焦（D405 最近約 7 cm），所以伺服只做到預備距離，剩下的固定距離直線下降。
- **s\* 用示教取得**：示教時後退多少，之後就下降多少，不需要手眼標定。若改用 `config.py` 的名義手眼外參推算 s\*，外參誤差會直接變成夾取位置誤差（見第 4 節）。
- **殘差寫成乘法** `ΔJθ = J_phys·Mθ`：J_phys 隨深度約以 1/Z 變化，而手眼、運動學誤差大致是「命令被扭曲了多少」，與深度無關。這樣在遠處試探學到的修正才能用到近處。
- **守門之外再加實測監看**：指南的守門條件只是模型預測。這裡另外監看實際量到的誤差，採用殘差後連續 3 步沒下降，該回合就停用殘差；任何方法只要誤差明顯發散就中止。

---

## 2. 檔案

| 檔案 | 內容 |
|---|---|
| `config.py` | 所有參數與單位說明（相機、方塊、手眼外參、限幅、收斂門檻、模擬誤差） |
| `tag_detector.py` | AprilTag 偵測，輸出特徵 `[u, v, w, ψ]` |
| `ibvs_pinn_controller.py` | 解析 interaction matrix、殘差模型（ridge / torch）、守門、實測監看 |
| `arm_kinematics.py` | 相機座標命令 → 7 軸關節增量（Pinocchio + URDF；模擬用近似模型） |
| `openarm_robot.py` | 手臂介面與安全檢查（模擬 / lerobot `OpenArmFollower`） |
| `camera_stream.py` | RealSense 串流 / 模擬相機 |
| `sim_world.py` | 離線模擬世界：依 3D 幾何算出影像，並判定是否夾得到 |
| `approach_pipeline.py` | 示教、伺服、下降、夾取、提起 |
| `data_collector.py` | 微幅安全試探（有邊界的隨機漫步） |
| `main_experiment.py` | 主程式：`compare` / `approach` / `teach` 三種模式 |
| `test_ibvs_approach.py` | 離線測試 |
| `teleop_position.py` | 真機主從擺位：用主臂把從臂帶到位，按 Enter 結束 |
| `check_cameras.py` | 列出 RealSense 序號、預覽 Tag 偵測 |

---

## 3. 離線模擬（不需任何硬體）

```bash
cd openarm_src/test
pip install numpy opencv-python          # 選用：matplotlib（曲線圖）、torch（--learner torch）

python main_experiment.py                                  # 三方法比較，各 15 回合
python main_experiment.py --learner torch
python main_experiment.py --mode approach --method guarded # 單次完整流程，逐步印出誤差
python main_experiment.py --target handeye                 # 改用名義手眼外參推算 s*
pytest test_ibvs_approach.py
```

輸出在 `experiment_outputs/`：`experiment_results.csv`（彙整）、`experiment_episodes.csv`（每回合，含失敗原因）、`experiment_error_curves.csv`、`probe_data.npz`，有裝 matplotlib 時另有 `experiment_error_curves.png`。

模擬中的「真實世界」刻意與控制器的名義模型不同（手眼外參轉了 3°/−2°/15°、偏移數 mm，連桿長 4%，關節只追到命令的 90%，另有像素與關節雜訊），控制器只能透過影像得知結果。

---

## 4. 模擬參考結果

固定種子 2026、15 回合、方塊隨機偏移 ±4.5 cm 與 ±30°、起始距離約 30 cm：

| 方法 | 伺服成功 | 平均最終誤差 | 平均步數 | 夾取成功 | 夾取 XY 偏差 |
|---|---|---|---|---|---|
| IBVS | 100% | 0.62 px | 40.4 | 100% | 0.9 mm |
| Learned (ridge) | 100% | 0.51 px | 40.6 | 100% | 0.8 mm |
| Guarded (ridge) | 100% | 0.61 px | 40.4 | 100% | 0.8 mm |

- 三種方法差異很小。這組條件下名義模型已經夠準，而且步數主要被單步上限（4 mm）決定，殘差能改善的空間有限。Learned／Guarded 不會永遠優於 IBVS。
- 改用 `--target handeye` 時，三種方法伺服都收斂，但夾取全部失敗（XY 偏差約 16 mm、偏轉約 15°）。影像誤差收斂只代表到達 s\*，s\* 本身錯了就夾不到，殘差模型修不了這件事。
- 以上只是這組合成條件下的結果，**不代表真實手臂的成功率，也不構成收斂保證**。

---

## 5. 真機流程

真機部分**尚未在實體手臂上執行過**，第一次上機請逐步確認。

需求：Linux + SocketCAN、`pyrealsense2`、Pinocchio，並設定 `OPENARM_URDF`（與 `openarm_gc.py` 相同）；建議設定 `OPENARM_TABLE_Z` 啟用桌面防撞。

```bash
export PYTHONPATH=$HOME/lerobot
export OPENARM_URDF=/path/to/openarm.urdf
export OPENARM_TABLE_Z=...                    # 桌面在 URDF 世界座標的高度 (m)

# 1) 確認相機序號與 Tag 偵測
python check_cameras.py

# 2) 示教 s*：程式會先開主從控制，用主臂把張開的夾爪擺到「閉合就能夾到方塊」的位置，按 Enter 後記錄
python main_experiment.py --real --side left --port can1 --leader-port can2 --serial 353322271542 --mode teach

# 3) 無接觸對準：先用純 IBVS，確認方向正確
python main_experiment.py --real --side left --port can1 --serial <D405序號> --mode approach --method ibvs

# 4) 加入殘差與守門（會先做微幅試探）
python main_experiment.py --real --side left --port can1 --serial <D405序號> --mode approach

# 5) 對準無誤後才下夾
python main_experiment.py --real --side left --port can1 --serial <D405序號> --mode approach --grasp

# 6) 三方法比較（只對準、不下夾；每回合從相同起點加上相同的初始偏移）
python main_experiment.py --real --side left --port can1 --serial <D405序號> --mode compare --episodes 5
```

伺服起點：`approach` / `compare` 開始時，手臂會自動經 `reset.json` 移到**示教 s\* 時記錄的關節姿態**，再沿接近軸多後退 `--start-offset`（預設 5 cm）作為起點，然後才開始伺服。舊版示教檔沒有記錄關節姿態，要重新示教一次。加 `--no-goto` 則從手臂目前姿態開始；加 `--teleop` 則改用主臂手動擺位。

主從控制（`teleop_position.py`）：

- `teach` 模式預設開啟；`approach` / `compare` 要加 `--teleop` 才會開（用來把手臂帶到看得到 Tag 的位置）。`--no-teleop` 可關閉，從臂會留在目前姿態。
- 連線前主臂要自然下垂（LeRobot 會把連線當下的姿態設為零點）。連線後主臂會先自動走到從臂目前的姿態，再放開讓你帶動。
- `--port` / `--leader-port` 省略時，依 `--side` 取 `openarm_src/utils/config.py` 的預設（左：從臂 can1、主臂 can2；右：從臂 can0、主臂 can3）。
- 主臂校正檔 id 預設為 `openarm_<side>_leader`，不同時用 `--leader-id` 指定。

每個動作前會要求按 Enter 確認（`--yes` 可略過）。結束或 Ctrl+C 時，從臂會先回到 `reset.json`（`OPENARM_HOME`）的起始姿態，再回下垂姿態並關力矩；回起始姿態只動 7 個關節，夾爪維持原狀。

上機前要確認的設定（`config.py`）：

| 項目 | 說明 |
|---|---|
| `HandEyeConfig` | 腕部相機相對 TCP 的位置與朝向。只需大致正確，但若軸向差很多（例如 90°），純 IBVS 會往錯的方向走 |
| `approach_axis_tcp` | TCP 座標中夾爪前進的方向，預設 +z，請對照 URDF 的 `hand_tcp` |
| `gripper_open_deg` / `gripper_closed_deg` | 預設 −55 / 0，請先確認實機正負號 |
| `tag_size_meters` | Tag **黑框**邊長。滿版 5 cm 的 Tag 需要周圍有淺色背景才偵測得到；若列印時留白邊，這裡要填黑框的實際邊長 |
| `JOINT_SIGN`（`openarm_src/utils/config.py`） | LeRobot 角度與 URDF 角度的正負號 |

安全機制（`openarm_robot.py`、`approach_pipeline.py`）：

1. 單步命令上限 4 mm / 1.5°，單步單關節上限 2°。
2. 關節軟限位與 lerobot `OpenArmFollower` 相同，再內縮 1°。
3. TCP 低於桌面 + 餘裕、或離起點超過 45 cm 時，該步取消。
4. 連續 15 幀看不到 Tag、或誤差發散時中止伺服。
5. 預設只做無接觸對準，加 `--grasp` 才會下降夾取。

---

## 6. 目前的限制

- **粗定位不在範圍內**：腕部相機要先看得到 Tag。把手臂帶到附近是 Slow Policy（ACT／VLA、遙操作、homing）的工作；`ego` (D455) 相機目前沒有用到。
- **只控制 4 個自由度**：夾爪傾斜角（roll／pitch）維持起始姿態，不會自動修正。起始時夾爪需大致垂直於方塊頂面。
- **下降是開迴路**：下降途中不再看 Tag，方塊若在這段時間被碰動就會夾偏。
- **夾取判定很粗略**：真機只看夾爪是否停在半路，沒有使用力矩或壓力感測器。
- **近似運動學只供模擬**：`ApproxArmModel` 不是 OpenArm 的真實幾何，真機一律使用 URDF。

---

## 7. 常見問題

| 現象 | 優先檢查 |
|---|---|
| 動作方向反了 | `HandEyeConfig.rpy_deg`、`JOINT_SIGN`、`approach_axis_tcp` |
| 殘差反而變差 | 訓練訊息中的驗證誤差（learned 應小於 physics）、試探幅度是否太小而被雜訊蓋過、取像時手臂是否已靜止 |
| Guard 一直退回 IBVS | 殘差未通過驗證（訊息會註明）、`guard_margin_px` 太大、名義模型本來就夠準 |
| 影像已對準但夾不到 | s\* 是否用示教取得、下降距離、夾爪開度、方塊是否被碰動、夾爪是否傾斜 |
| 示教後看不到 Tag | 真機預設只後退 5 cm；在提示中輸入要再後退幾公分，或用 `--retreat` 指定。也可先用 `--mode view` 看相機畫面 |
