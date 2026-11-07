# OpenArm HIL / DAgger 資料收集與即時鏡像跟隨指南

本指南說明如何透過 `openarm_gc.py rollout` 運行 LeRobot 的 DAgger (Human-in-the-Loop) 策略，搭配雙臂 OpenArm Leader 重力補償、自動 Home 姿態以及 **Leader 即時鏡像跟隨 Follower** 機制進行資料收集與示範修正。


## 環境變數設定

在執行前，請先於終端機設定必要的環境變數：

```bash
# 1) URDF 路徑 (重力補償必備)
export OPENARM_URDF=/path/to/openarm_bimanual.urdf

# 2) 起始姿態 waypoint 檔案路徑 (自動 home 必備)
export OPENARM_HOME=/path/to/home.json

# (選填) 移動到 home 的秒數 (預設 4 秒)
export OPENARM_HOME_TIME=4

# (選填) DAgger 推論時是否讓 Leader 跟隨 Follower (1=開啟[預設], 0=關閉)
export OPENARM_LEADER_TRACK=1

# (選填) 桌面高度保護 (m)，可防止移動到 home 過程中撞桌
# export OPENARM_TABLE_Z=0.15
```

---

## 完整 CLI 執行指令

> **注意**：LeRobot 的 `rollout` 策略要求 Dataset 儲存庫名稱必須以 `rollout_` 開頭（例如 `--dataset.repo_id=JJC489/rollout_openarm_hil`）。

```bash
python openarm_gc.py rollout \
  --strategy.type=dagger \
  --strategy.num_episodes=50 \
  --policy.path=/home/chris/models/act_openarm_200k/400000/pretrained_model \
  --robot.type=bi_openarm_follower \
  --robot.left_arm_config.port=can1 \
  --robot.left_arm_config.side=left \
  --robot.right_arm_config.port=can0 \
  --robot.right_arm_config.side=right \
  --robot.id=bi_follower \
  --robot.cameras='{"ego": {"type": "intelrealsense", "serial_number_or_name": "239222300027", "width": 640, "height": 480, "fps": 30}, "wrist_left": {"type": "intelrealsense", "serial_number_or_name": "353322271140", "width": 424, "height": 240, "fps": 30}, "wrist_right": {"type": "intelrealsense", "serial_number_or_name": "353322271542", "width": 424, "height": 240, "fps": 30}}' \
  --teleop.type=bi_openarm_leader \
  --teleop.left_arm_config.port=can2 \
  --teleop.right_arm_config.port=can3 \
  --teleop.id=bi_leader \
  --dataset.repo_id=JJC489/rollout_openarm_hil \
  --dataset.single_task="Put the tea bag on the plate." \
  --dataset.fps=30 \
  --dataset.episode_time_s=300 \
  --dataset.reset_time_s=15 \
  --dataset.encoder_threads=2 \
  --dataset.rgb_encoder.vcodec=h264 \
  --dataset.push_to_hub=false \
  --display_data=false \
  --fps=30 \
  --interpolation_multiplier=2
```

---

## 鍵盤操作與狀態切換流程

| 步驟 | 按鍵 | 狀態 | 說明 |
|---|---|---|---|
| **0. 啟動準備** | - | `STARTUP` | 請讓 Leader 雙臂自然垂下（初始化以當前姿態為零點）。載入完成後會自動執行 `go_home` 移動到起始姿態。 |
| **1. 自主推論** | - | `AUTONOMOUS` | Policy 開始自主控制 Follower 執行任務。此時 **Leader 會鏡像同步跟隨 Follower 移動**。 |
| **2. 暫停 Policy** | `Space` | `PAUSED` | 當發現機器人動作歪掉或即將失敗時，按一下 `Space` 暫停。Follower 與 Leader 停在原處。 |
| **3. 開始修正錄製** | `Tab` | `CORRECTING` | 按下 `Tab`，**Leader 立即釋放目標進入純重力補償**，控制權交由人類，**開始錄製 Dataset**！手動操作 Leader 完成正確動作。 |
| **4. 結束修正儲存** | `Tab` | `PAUSED` | 修正示範完畢後再按一次 `Tab`，系統會**將剛才的修正片段儲存為 1 個 Episode**。 |
| **5. 恢復自主運行** | `Space` | `AUTONOMOUS` | 按 `Space` 讓 Policy 從當前位置繼續自主執行任務，Leader 重新恢復鏡像跟隨。 |

---

## 進階選項與常見問題

### 1. 連同 Policy 自主運行的過程一起錄製？
預設為 `corrections-only`（僅記錄人類修正）。若希望自主與修正過程均被完整錄製下來（附帶 `intervention=True/False` 標籤），可在指令加入：
```bash
--strategy.record_autonomous=true
```

### 2. 雙向力回饋開關 (`--bilateral=true/false`)
在 `record` 或 `teleoperate` 時，可直接透過 CLI 旗標開啟或關閉夾爪力回饋：
```bash
python openarm_gc.py record --bilateral=true ...
```
- `--bilateral=true`：開啟力回饋（若未設 `OPENARM_FF_SCALE`，會預設以 `0.05` 啟動）。
- `--bilateral=false`：關閉力回饋（將 `OPENARM_FF_SCALE=0`）。

### 3. DAgger 動作防跳躍保護 (Step Jump Guard)
系統預設自動啟用 Step Jump Guard，全面防止 DAgger 狀態切換與推論過程中的動作階躍爆衝：
- **交接平滑化（PAUSED -> AUTONOMOUS）**：人類手動介入結束切回 Policy 時，以當前手臂實際姿態為基準，限制 Policy 首拍最大移動量，平滑 Ramp 過渡，避免機器人瞬間甩動。
- **單步速率限制（Rate Limiting）**：限制每拍（1/30 秒）關節最大變動量，防止 Policy 輸出分佈外跳變或遙操作瞬變。

可透過環境變數微調：
```bash
# 是否啟用防跳躍保護 (1=開啟[預設], 0=關閉)
export OPENARM_STEP_GUARD=1

# 關節每拍最大位移限制 (度/tick，預設 4.0°，30 FPS 下相當於 120°/s)
export OPENARM_MAX_STEP_DEG=4.0

# 夾爪每拍最大位移限制 (度/tick，預設 15.0°)
export OPENARM_MAX_STEP_GRIPPER=15.0
```

---

## 任務多槽位管理與切換式混合錄製 (Hybrid Teleop + Script)

針對需要「手動遙操作夾取目標，並精準自動放至托盤指定槽位」的任務（例如 4 個茶包整齊擺放至托盤 4 個格子），系統提供了任務目錄管理、拖曳式 Waypoint 示教器以及切換式混合錄製工具。

### 1. 任務資料夾結構 (Task Directory Structure)

所有任務的 Waypoint 與槽位配置統一存放於 `openarm_src/tasks/` 資料夾下，並依不同任務名稱建立專屬子目錄：

```text
openarm_src/tasks/
├── teabag_4slots/               # 任務名稱 A：4 茶包分類擺放
│   ├── tray_slots.json          # 4 個槽位專屬的路徑點 (Approach, Place, Release, Retract)
│   └── home.json                # (選填) 該任務專屬起始 Home 姿態
└── bottle_sorting/              # 任務名稱 B：瓶罐分類
    └── tray_slots.json
```

---

### 2. 托盤槽位 Waypoint 示教工具 (`teach-slots`)

在開始錄製前，請先使用示教工具標定托盤的各個放置槽位（支援 4 槽位或任意 N 槽位）。  
透過 Leader 手臂的重力補償直接拖曳手臂至各槽位貼盤位置，按下 Enter 即可自動計算並生成上方接近點（Approach）、貼盤放置點（Place）、鬆爪點（Release）與垂直退回點（Retract）：

```bash
# 基本用法：指定任務名稱與槽位數量 (預設自動存至 openarm_src/tasks/<task_name>/tray_slots.json)
python openarm_gc.py teach-slots --task teabag_4slots --slots 4

# 選填參數：指定 Leader 連線埠號與手臂 ID
python openarm_gc.py teach-slots \
  --task teabag_4slots \
  --slots 4 \
  --id bi_leader \
  --left-port can2 \
  --right-port can3 \
  --approach-offset-deg 12.0 \
  --gripper-open-deg 65.0
```

**互動流程**：
1. 啟動後雙臂進入重力補償模式。
2. 終端機提示：`【示教第 1/4 個槽位】請將夾爪移動到托盤第 1 格貼盤位置...`。
3. 手動移到位後，按一下 `Enter` 鍵。
4. 依序完成第 2、3、4 個槽位，全部完成後自動輸出 JSON 檔案至對應任務目錄。

---

### 3. 切換式混合錄製工具 (`record-hybrid`)

錄製時，**人類操作員專注於手動 Teleop 夾取茶包**；夾取完成後，**按下對應槽位數字鍵**，機器人便會平滑接管並以極度工整的幾何軌跡將茶包放至托盤指定槽位。  
放置完畢後夾爪張開並垂直抬升，控制權自動無縫交回 Teleop，繼續夾取下一個茶包。**相機與關節數據全程以 30 FPS 持續寫入 Dataset，確保時序無中斷**。

```bash
# 啟動切換式混合錄製 (只需傳入 --task 即會自動載入該任務資料夾下的 tray_slots.json)
python openarm_gc.py record-hybrid \
  --task teabag_4slots \
  --robot.type=bi_openarm_follower \
  --robot.left_arm_config.port=can1 \
  --robot.left_arm_config.side=left \
  --robot.right_arm_config.port=can0 \
  --robot.right_arm_config.side=right \
  --robot.id=bi_follower \
  --robot.cameras='{"ego": {"type": "intelrealsense", "serial_number_or_name": "239222300027", "width": 640, "height": 480, "fps": 30}, "wrist_left": {"type": "intelrealsense", "serial_number_or_name": "353322271140", "width": 424, "height": 240, "fps": 30}, "wrist_right": {"type": "intelrealsense", "serial_number_or_name": "353322271542", "width": 424, "height": 240, "fps": 30}}' \
  --teleop.type=bi_openarm_leader \
  --teleop.left_arm_config.port=can2 \
  --teleop.right_arm_config.port=can3 \
  --teleop.id=bi_leader \
  --dataset.repo_id=JJC489/openarm_teabag_hybrid \
  --dataset.single_task="Pick up the tea bag and place it into the designated tray slot." \
  --dataset.fps=30 \
  --dataset.episode_time_s=180 \
  --dataset.reset_time_s=10 \
  --dataset.push_to_hub=false \
  --display_data=false \
  --fps=30
```

#### 鍵盤操作說明

| 按鍵 | 功能 | 說明 |
|---|---|---|
| **數字鍵 `1` ~ `4`** | **指定槽位放置 (Script)** | 夾穩茶包後按 `1`，機器人自動 Min-Jerk 平滑移動並將茶包放至 Slot 1；按 `2` 放至 Slot 2，以此類推。 |
| **`Space`** | **依序自動輪流放置** | 夾穩後按空白鍵，系統依序自動循環：Slot 1 -> Slot 2 -> Slot 3 -> Slot 4。 |
| **`Right` / `n`** | **完成並保存當前集** | 結束目前 Episode 錄製並儲存，重置計時器準備下一集。 |
| **`Left` / `r`** | **放棄重錄 (Re-record)** | 放棄目前正在錄製的 Episode 並重錄。 |
| **`Esc` / `q`** | **結束程式** | 安全停止並中斷錄製迴圈。 |

---

### 4. 末端笛卡爾空間受控擾動機制 (Cartesian Perturbation)

為了豐富資料集的多樣性並防止 Policy 死背單一固定軌跡，系統在執行 Script 放置時會自動施加**微小隨機擾動**。

> **防累加誤差核心設計**：  
> 若直接對 7 個關節各加隨機角度，由於關節鏈（Kinematic Chain）幾何槓桿效應，各關節角度誤差會在末端急劇放大（可能導致夾爪位移暴增 20~30mm 而偏出托盤格子）。  
> 因此系統直接載入 OpenArm URDF，透過 **Pinocchio 6D 雅可比矩陣阻尼逆運動學（Jacobian DLS）**，嚴格在末端夾爪笛卡爾空間中施加受控擾動：

- **水平位置（X/Y 平面）**：$\sqrt{\Delta x^2 + \Delta y^2} \le 4.0\text{ mm}$（嚴格控制在托盤格子容許裕度內）。
- **垂直高度（Z 軸）**：$\Delta z \in [0.0, +1.0\text{ mm}]$（**向下嚴格為 0**，絕不向下多切，防止夾爪下壓撞擊托盤）。
- **末端自旋（Yaw 偏角）**：$|\Delta \text{yaw}| \le 2.5^\circ$（Roll 與 Pitch 保持水平平行於托盤）。
- **通道一致性**：Approach（接近）、Place（放置）、Retract（退回）三點共享相同的水平偏移量，保證下刀與垂直抬升通道筆直，絕不刮碰托盤邊框。

可透過環境變數控制：
```bash
# 是否啟用托盤槽位放置擾動 (1=開啟[預設], 0=關閉)
export OPENARM_SLOT_NOISE=1

# 末端最大水平隨機位移 (mm，預設 4.0)
export OPENARM_SLOT_MAX_XY_MM=4.0

# 末端最大高度向上餘裕 (mm，預設 1.0)
export OPENARM_SLOT_MAX_Z_MM=1.0

# 夾爪最大繞法線偏轉角度 (度，預設 2.5)
export OPENARM_SLOT_MAX_YAW_DEG=2.5
```

