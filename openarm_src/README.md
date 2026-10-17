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

> ⚠️ **注意**：LeRobot 的 `rollout` 策略要求 Dataset 儲存庫名稱必須以 `rollout_` 開頭（例如 `--dataset.repo_id=JJC489/rollout_openarm_hil`）。

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