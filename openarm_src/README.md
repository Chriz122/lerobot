# OpenArm LeRobot 外掛工具庫 (openarm_src)

本模組為 OpenArm 雙臂/單臂機器人整合於 Hugging Face [LeRobot](https://github.com/huggingface/lerobot) 的擴充工具庫。  
- **主臂重力補償 (Gravity Compensation)**：背景 200 Hz 執行緒計算 Pinocchio 動態模型與摩擦前饋補償，手臂呈現零重力自然懸浮手感。
- **自動起始姿態導引 (Homing)**：在錄製開始前，Leader 與 Follower 以 Min-Jerk 平滑五次多項式自動走至指定姿態，並具備桌面防撞保護 (Table Guard)。
- **安全回位 (Safe Park)**：程式退出或中斷時，自動先經由反向中繼點慢慢垂回下垂姿態再關閉力矩，杜絕手臂直接重摔在桌面上。
- **雙向力回饋 (Bilateral Teleoperation)**：從臂夾爪受力即時反射回傳至主臂握把，提供觸覺力回饋感知。
- **DAgger 即時鏡像跟隨 (Leader Tracking)**：自主推論期間主臂鏡像跟隨從臂，操作員隨時接管修正無落差。
- **動作跳變防護器 (Step Jump Guard)**：單步速率限幅器 (Rate Limiter)，防止接管瞬間、斷訊或模型幻覺造成的高速甩動爆衝。
- **托盤槽位示教與切換式混合錄製 (Hybrid Teleop + Script)**：人類專注手動抓取，鍵盤一鍵觸發工整自動放置，搭配笛卡爾空間微擾動機制，顯著提升資料集品質。
- **WebRTC 影像串流與感測器監控**：支援瀏覽器遠端實時觀看相機畫面與 Arduino 壓力感測器力矩印出。

---

## 目錄 (Table of Contents)

1. [環境變數與功能開關設定表格](#環境變數與功能開關設定表格)
2. [CLI 命令完整列表與範例](#cli-命令完整列表與範例)
   - [0. 產生 URDF 與環境初始化](#0-產生-urdf-與環境初始化)
   - [1. 檢查主臂重力補償與儲存起始姿態 (check)](#1-檢查主臂重力補償與儲存起始姿態-check)
   - [2. 從臂獨立回起始姿態測試 (home)](#2-從臂獨立回起始姿態測試-home)
   - [3. 純遙操作模式 (teleoperate)](#3-純遙操作模式-teleoperate)
   - [4. 一般資料集錄製 (record)](#4-一般資料集錄製-record)
   - [5. 人機協同 DAgger / HIL 評估與示範 (rollout)](#5-人機協同-dagger--hil-評估與示範-rollout)
   - [6. 托盤槽位 Waypoint 示教器 (teach-slots)](#6-托盤槽位-waypoint-示教器-teach-slots)
   - [7. 切換式混合錄製 (record-hybrid)](#7-切換式混合錄製-record-hybrid)
3. [進階機制深入解析](#進階機制深入解析)
   - [DAgger 狀態機切換流程](#dagger-狀態機切換流程)
   - [動作防跳變保護機制 (Step Jump Guard)](#動作防跳變保護機制-step-jump-guard)
   - [末端笛卡爾空間受控擾動機制 (Cartesian Perturbation)](#末端笛卡爾空間受控擾動機制-cartesian-perturbation)

---

## 環境變數與功能開關設定表格

所有環境變數皆已統一彙整於 `openarm_src/utils/config.py` 中，提供完整的型別轉換與預設值備援。使用者可在終端機透過 `export` 覆蓋設定：

### 1. 核心模型與 Homing / Park 姿態

| 環境變數 | 型別 | 預設值 | 說明 |
| :--- | :---: | :---: | :--- |
| `OPENARM_URDF` | 字串 | `None` | **(必填)** 雙臂 URDF 模型絕對路徑 (關節名 `openarm_{left,right}_joint1..7`) |
| `OPENARM_HOME` | 字串 | `"reset.json"` | 起始姿態 waypoint 檔案路徑；若設為空則略過自動 Homing |
| `OPENARM_HOME_TIME` | 浮點數 | `4.0` | 移動至起始姿態的總時間 (秒) |
| `OPENARM_HOME_AT` | 字串 | `"record"` | 觸發 Homing 時機：`"record"` (每集開錄前) 或 `"reset"` (每集結束重置時) |
| `OPENARM_TABLE_Z` | 浮點數 | `None` | (選填) 桌面在 URDF 世界座標系之高度 (m)，設定後會自動啟用桌面防撞檢查 |
| `OPENARM_TABLE_MARGIN` | 浮點數 | `0.05` | 桌面防撞檢查之安全餘裕高度 (m) |
| `OPENARM_PARK` | 布林值 | `1` (開啟) | 結束退出時是否先平滑回垂下姿態再關閉力矩 (1=開啟, 0=關閉直落) |
| `OPENARM_PARK_TIME` | 浮點數 | `3.0` | 回垂下姿態每段路徑的移動秒數 |
| `OPENARM_HOMING_RATE_HZ` | 浮點數 | `50.0` | Homing 軌跡插值發送頻率 (Hz) |

### 2. 主臂重力補償與雙向力回饋 (Gravity Comp & Force Feedback)

| 環境變數 | 型別 | 預設值 | 說明 |
| :--- | :---: | :---: | :--- |
| `OPENARM_CONTROL_HZ` | 浮點數 | `200.0` | 主臂重力補償控制迴圈頻率 (Hz) |
| `OPENARM_GC_SCALE` | 浮點數 | `1.0` | 重力補償力矩整體放大倍率 (首次初測建議設 `0.5`) |
| `OPENARM_KD_SCALE` | 浮點數 | `1.0` | 阻尼增益放大倍率 (手感偏黏時調小，0=完全無阻尼) |
| `OPENARM_FRICTION_SCALE` | 浮點數 | `0.0` | 摩擦前饋補償倍率 (預設關閉，建議從 `0.3~0.5` 開始微調) |
| `OPENARM_FRICTION_EPS` | 浮點數 | `0.15` | 摩擦補償平滑臨界速度 (rad/s)，靜止若微震可微幅調大 |
| `OPENARM_GC_STATS` | 布林值 | `1` (開啟) | 是否每 5 秒印出主臂 GC 實際頻率、超時次數與最大輸出矩 |
| `OPENARM_LEADER_SIDE` | 字串 | `None` | 單臂 Leader 時手動指定側別 (`"left"` 或 `"right"`) |
| `OPENARM_FF_SCALE` | 浮點數 | `0.05` | 從臂夾爪對主臂之雙向力回饋倍率 (設 `0` 關閉) |
| `OPENARM_FF_SIGN` | 浮點數 | `-1.0` | 力回饋正負方向符號 (若手感阻力方向相反時設 `1.0`) |
| `OPENARM_FF_LIMIT` | 浮點數 | `0.2` | 每關節力回饋輸出保護上限 (Nm) |
| `OPENARM_FF_ALPHA` | 浮點數 | `0.6` | 力回饋一階低通濾波平滑係數 (越小濾波越重越平穩) |
| `OPENARM_FF_DEADBAND` | 浮點數 | `0.02` | 力回饋感應死區 (Nm)，過濾感測器微小雜訊 |

### 3. DAgger 動作安全防護 (Safety Guard)

| 環境變數 | 型別 | 預設值 | 說明 |
| :--- | :---: | :---: | :--- |
| `OPENARM_STEP_GUARD` | 布林值 | `1` (開啟) | 是否啟用 DAgger 單步動作防跳變限幅保護器 |
| `OPENARM_MAX_STEP_DEG` | 浮點數 | `4.0` | 每拍 (1/30s) 最大容許關節移動度數 (30 FPS 下相當於 120°/s) |
| `OPENARM_MAX_STEP_GRIPPER` | 浮點數 | `15.0` | 每拍最大容許夾爪移動度數 |
| `OPENARM_WARN_THROTTLE_S` | 浮點數 | `2.0` | 防跳變告警日誌節流間隔 (秒) |
| `OPENARM_LEADER_TRACK` | 布林值 | `1` (開啟) | DAgger 推論時 Leader 是否即時鏡像跟隨 Follower |
| `OPENARM_PROFILE` | 布林值 | `1` (開啟) | 是否每 5 秒印出主控制迴圈各步驟耗時剖析 |

### 4. 托盤槽位示教與末端受控擾動 (Hybrid Task & Perturbation)

| 環境變數 | 型別 | 預設值 | 說明 |
| :--- | :---: | :---: | :--- |
| `OPENARM_TASK_NAME` | 字串 | `"teabag_4slots"` | 預設任務名稱，對應 `openarm_src/tasks/<task>/` |
| `OPENARM_SLOTS_COUNT` | 整數 | `4` | 預設示教槽位數量 |
| `OPENARM_APPROACH_OFFSET_DEG` | 浮點數 | `12.0` | Approach 上方點相對於 Place 點之關節抬升度數 |
| `OPENARM_TRAJ_FPS` | 整數 | `30` | 混合錄製軌跡控制與記錄頻率 (FPS) |
| `OPENARM_SLOT_NOISE` | 布林值 | `1` (開啟) | 混合錄製執行 Script 放置時是否施加笛卡爾空間受控微擾 |
| `OPENARM_SLOT_MAX_XY_MM` | 浮點數 | `4.0` | 末端水平平面 (XY) 最大隨機位移擾動半徑 (mm) |
| `OPENARM_SLOT_MAX_Z_MM` | 浮點數 | `1.0` | 末端垂直高度 (Z) 最大向上隨機位移 (mm，向下嚴格為 0) |
| `OPENARM_SLOT_MAX_YAW_DEG` | 浮點數 | `2.5` | 夾爪末端最大隨機 Yaw 偏擺旋轉角度 (度) |
| `OPENARM_SLOT_GRIPPER_MIN_DEG` | 浮點數 | `30.0` | 夾爪張開基準角度與高斯截斷下限 (度) |
| `OPENARM_SLOT_GRIPPER_MAX_DEG` | 浮點數 | `60.0` | 夾爪張開高斯截斷上限 (度) |
| `OPENARM_SLOT_GRIPPER_SIGMA_DEG`| 浮點數 | `10.0` | 夾爪張開半常態分佈標準差 (度) |

### 5. CAN 匯流排管理與通訊硬體

| 環境變數 | 型別 | 預設值 | 說明 |
| :--- | :---: | :---: | :--- |
| `OPENARM_CAN_SETUP` | 布林值 | `1` (開啟) | 啟動前是否自動執行並並行快速探測所有馬達連線 |
| `OPENARM_CAN_IFACES` | 字串 | `"can0,can1,can2,can3"` | 納入管理的 CAN 介面清單 (逗號分隔) |
| `OPENARM_CAN_DOWN` | 布林值 | `0` (關閉) | 程式退出時是否強制將 SocketCAN 介面 `ip link set down` |
| `OPENARM_DISABLE_RETRY` | 整數 | `3` | 安全關閉力矩時指令重複發送重試次數 |
| `OPENARM_ENABLE_RETRY` | 整數 | `3` | 開啟力矩時馬達確認連線重試次數 |
| `OPENARM_WEBRTC` | 布林值 | `0` (關閉) | 是否開啟 WebRTC 實時影像串流伺服器 |
| `OPENARM_WEBRTC_PORT` | 整數 | `8080` | WebRTC 服務通訊埠 |
| `OPENARM_FORCE_MONITOR` | 布林值 | `0` (關閉) | 是否在終端機實時印出夾爪力矩與壓力感測器數值 |
| `OPENARM_ARDUINO_PORT` | 字串 | `None` | (選填) Arduino 壓力感測器序列埠 (如 `/dev/ttyACM0`) |

---

## CLI 命令完整列表與範例

所有功能均以 `python openarm_gc.py <subcommand>` 作為統一入口：

### 0. 產生 URDF 與環境初始化

在有 ROS 2 + `openarm_description` 的環境下產生專用 URDF，並於終端機設定環境變數：

```bash
export OPENARM_URDF=~/ros2_ws/src/openarm_description/assets/robot/openarm_v1.0/urdf/example/v1.urdf
export OPENARM_HOME=home.json
```

---

### 1. 檢查主臂重力補償與儲存起始姿態 (`check`)

校正主臂重力補償方向，手動將手臂抬起確認是否自然浮起；擺好起始姿態後可直接存為 JSON 檔案：

```bash
# 基本用法：指定 Leader 手臂 ID 與 CAN 埠號
python openarm_gc.py check \
  --robot.type=bi_openarm_follower \
  --robot.left_arm_config.port=can1 \
  --robot.left_arm_config.side=left \
  --robot.right_arm_config.port=can0 \
  --robot.right_arm_config.side=right \
  --robot.id=bi_follower
```
- **檢查重點**：抬起手臂放手時應感覺「變輕、放手即停」。若某關節越補越重或往上飄，請至 `config.py` 將對應關節 `JOINT_SIGN` 設為 `-1`，或微調 `JOINT_GC_SCALE`。

---

### 2. 從臂獨立回起始姿態測試 (`home`)

在不啟動 Leader 的情況下，單獨驅動 Follower 從臂平滑走至指定的 `reset.json` 起始姿態，確認路徑無干涉碰撞：

```bash
python openarm_gc.py home \
  --robot.type=bi_openarm_follower \
  --robot.left_arm_config.port=can1 \
  --robot.left_arm_config.side=left \
  --robot.right_arm_config.port=can0 \
  --robot.right_arm_config.side=right \
  --robot.id=bi_follower \
  --time 4.0
```
- 執行到位後會在終端機提示，按下 `Enter` 鍵後會平滑垂回下垂姿態再安全關閉力矩。

---

### 3. 純遙操作模式 (`teleoperate`)

參數與 `lerobot-teleoperate` 完全相容，自動掛載重力補償與安全關力矩保護：

```bash
python openarm_gc.py teleoperate \
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
  --fps=30 \
  --interpolation_multiplier=2 \
```

- **力回饋雙向控制**：附加 `--bilateral=true` 即可啟動夾爪阻力反饋。
- **壓力感測器即時監控**：附加 `--force-monitor --arduino-port=/dev/ttyACM0` 即可在終端機實時印出 SY40 電壓與力矩。

---

### 4. 一般資料集錄製 (`record`)

參數與 `lerobot-record` 完全相容。在每個 Episode 開錄前，自動控制 Leader + Follower 同步移至 `OPENARM_HOME` 姿態，到位後釋放 Leader 供手動示範：

```bash
export OPENARM_HOME=home.json

python openarm_gc.py record \
  --robot.type=bi_openarm_follower \
  --robot.left_arm_config.port=can1 \
  --robot.right_arm_config.port=can0 \
  --robot.id=bi_follower \
  --robot.cameras='{"ego": {"type": "intelrealsense", "serial_number_or_name": "239222300027", "width": 640, "height": 480, "fps": 30}, "wrist_left": {"type": "intelrealsense", "serial_number_or_name": "353322271140", "width": 424, "height": 240, "fps": 30}, "wrist_right": {"type": "intelrealsense", "serial_number_or_name": "353322271542", "width": 424, "height": 240, "fps": 30}}' \
  --teleop.type=bi_openarm_leader \
  --teleop.left_arm_config.port=can2 \
  --teleop.right_arm_config.port=can3 \
  --teleop.id=bi_leader \
  --dataset.repo_id=my_org/openarm_demo \
  --dataset.single_task="Pick up the cube." \
  --dataset.fps=30 \
  --dataset.episode_time_s=60 \
  --dataset.reset_time_s=10 \
  --dataset.push_to_hub=false \
  --display_data=false \
  --fps=30 \
  --interpolation_multiplier=2 \
```

- **WebRTC 串流伺服器**：附加 `-webrtc=true --webrtc-port=8080`，可於其他裝置瀏覽器輸入 `http://<機器人IP>:8080` 實時監看鏡頭畫面與通訊延遲。

---

### 5. 人機協同 DAgger / HIL 評估與示範 (`rollout`)

在 Policy 推論時，主臂即時鏡像跟隨從臂；當操作員介入示範時，主臂自動釋放目標回歸零力重力補償：

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
  --dataset.repo_id=my_org/openarm_hil \
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

### 6. 托盤槽位 Waypoint 示教器 (`teach-slots`)

透過 Leader 重力補償直接拖曳手臂至各槽位貼盤位置，按下 `Enter` 自動計算 Approach、Place、Release、Retract 點並存成任務設定檔：

```bash
# 示教 teabag_4slots 任務的 4 個槽位 (自動存至 openarm_src/tasks/teabag_4slots/tray_slots.json)
python openarm_gc.py teach-slots \
  --task teabag_4slots \
  --slots 4 \
  --id bi_leader \
  --left-port can2 \
  --right-port can3 \
  --approach-offset-deg 12.0
```

---

### 7. 切換式混合錄製 (`record-hybrid`)

操作員專注於手動 Teleop 抓取物件，抓穩後按下槽位鍵（`1`~`4` 或 `Space`），系統自動以工整平滑軌跡放至指定槽位，隨後無縫交回控制權：

```bash
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
  --dataset.repo_id=my_org/openarm_hybrid \
  --dataset.single_task="Pick up the tea bag and place it into the designated tray slot." \
  --dataset.fps=30 \
  --dataset.episode_time_s=180 \
  --dataset.reset_time_s=10 \
  --dataset.push_to_hub=false \
  --display_data=false \
  --fps=30 \
  --interpolation_multiplier=2
```

---

## 進階機制深入解析

### DAgger 狀態機切換流程

| 步驟 | 按鍵 | 狀態 | 行為說明 |
| :---: | :---: | :---: | :--- |
| **0. 啟動準備** | - | `STARTUP` | Leader 雙臂自然垂下作為零點，啟動完成後自動以 Min-Jerk 移至起始姿態。 |
| **1. 自主推論** | - | `AUTONOMOUS` | Policy 控制從臂執行任務，Leader 即時鏡像同步跟隨從臂動作。 |
| **2. 暫停推論** | `Space` | `PAUSED` | 發現動作歪掉時按空白鍵暫停，從臂與主臂鎖定在當前位置。 |
| **3. 開始修正** | `Tab` | `CORRECTING` | 按 `Tab`，Leader 立即釋放 target 進入純重力補償，控制權交由人類，開始向 Dataset 寫入修正資料。 |
| **4. 結束修正** | `Tab` | `PAUSED` | 示範完成後再按 `Tab`，將該段修正儲存為 1 個 Episode。 |
| **5. 恢復推論** | `Space` | `AUTONOMOUS` | 按空白鍵讓 Policy 繼續運行，自動透過 Step Jump Guard 進行平滑交接。 |

---

### 動作防跳變保護機制 (Step Jump Guard)

防止人類介入放開時 Policy 首拍輸出落差、斷訊封包遺失或 OOD 模型幻覺造成的高速爆衝：
1. **交接姿態基準對齊**：在 `PAUSED -> AUTONOMOUS` 或 `PAUSED -> CORRECTING` 狀態轉移時，自動抓取當前實際姿態作為平滑限速的起始基準。
2. **單步速率限幅 (Rate Limiting)**：
   $$\Delta q_i = \text{clamp}(q_i^{\text{target}} - q_i^{\text{last}}, -\text{limit}, +\text{limit})$$
   關節預設限制 $4.0^\circ/\text{tick}$ (相當於 $120^\circ/\text{s}$)，夾爪限制 $15.0^\circ/\text{tick}$。超限時進行截斷並啟用警告節流。

---

### 末端笛卡爾空間受控擾動機制 (Cartesian Perturbation)

若直接對 7 個關節施加隨機角度，幾何槓桿效應會使末端位移放大至數公分而偏出格子。因此系統透過 Pinocchio 逆運動學直接在末端笛卡爾空間進行受控擾動：
- **水平位移 (XY 平面)**：$\sqrt{\Delta x^2 + \Delta y^2} \le 4.0\text{ mm}$。
- **垂直高度 (Z 軸)**：$\Delta z \in [0.0, +1.0\text{ mm}]$ (**嚴格禁止向下**，防止下壓撞擊托盤)。
- **末端偏角 (Yaw)**：$|\Delta \text{yaw}| \le 2.5^\circ$ (Roll 與 Pitch 嚴格保持水平)。
- **夾爪開度 (Gripper)**：以 $30.0^\circ$ 為基準進行半常態分佈抽樣，嚴格不低於 $30.0^\circ$，上限截斷至 $60.0^\circ$。
- **通道一致性**：Approach、Place、Retract 共享相同的水平位移量，確保垂直下刀與抬升通道筆直，絕不刮碰托盤壁。
