# ACT + IBVS 瓶子夾取（`ibvs_assist`）

ACT 照常推論；某支手臂的腕部相機看到瓶子、而且夠近時，該手臂改由 IBVS 對準、下降、夾取、提起，再交回 ACT 放到托盤上。兩支手臂各自觸發，互不干擾。

```
ACT ──(腕部相機連續 3 次看到瓶子)──> [太近就先後退] ──> IBVS 對準 ──> 沿示教路徑下降 ──> 閉合夾爪 ──> 往上提 ──> 交回 ACT
```

- 偵測：OWLv2（`google/owlv2-base-patch16-ensemble`），兩台腕部相機一起推論，RTX 3090 Ti 上約 70 ms。
- 特徵：只用瓶子框中心的 **u** 和 D405 框內**深度 Z**。
  - 左右：預設把瓶子水平置中到**畫面中心** cx（相機在兩指正中間上方，`center_u=true`）；`center_u=false` 才用示教時的 u\*。沿相機 x 軸的水平方向移動。
  - 前後：讓深度等於示教的 Z\*，沿夾爪接近方向的水平分量移動。
  - 高度：維持示教時的 TCP 高度（運動學）。
  - 姿態：目前**不修正**，全程保持 ACT 交出時的姿態，只做平移。已知限制（模擬）：
    - roll（繞接近軸）偏 8–15° 時，夾爪中線會偏離瓶軸 1–2.5 cm，看起來就是斜著進去。原因是相機在夾爪上方、往下傾 35°，roll 會把相機甩到側邊。
    - pitch 偏 8° 時，下降路徑高度偏約 1.6 cm。
    - yaw 偏 15° 只差 2 mm：桌面、直立瓶子、重力繞垂直軸都對稱，所以 yaw 不需要修。
    - 之後改成靠影像修正 roll / pitch，不靠示教姿態。
  - 不用框的 v：近距離時直立瓶子一定被畫面上下切到。也不需要手眼傾斜角：相機 x 軸不受傾斜影響。模擬中實際傾斜在 20–35° 時誤差約 2–4 mm，偏到 55° 時誤差變大到約 1 cm。
- 偵測：OWLv2 對同一支瓶子常給好幾個框（整支、只有標籤、只有瓶蓋），後處理又沒有 NMS。所以輸出前先合併重複框（`detector.merge_duplicate_boxes`）：重疊的框保留分數高的；小框被大框包住時保留大框，不再挑到只框住標籤的小框。大框同時包住左右並排的兩個框時，當成框到兩支瓶子，直接丟掉。
- 偵測：觸發時只認指定的瓶子（例如 `coca cola bottle`）；追蹤時也接受一般的 `bottle`，因為近距離看不到標籤時，指定查詢常常認不出來。
- 對準後照「示教時實際後退的路徑」走回夾取位姿，所以後退時偏離直線（例如關節碰到限位）也不會夾偏。
- 每次 rollout，每支手臂只接手一次。伺服中跟丟 1.5 秒、或 15 秒內未收斂就中止，交回 ACT。
- 手臂跟不上命令（命令角與實際角差 > 6°，例如碰撞、卡住）**不會中止**：命令拉回目前實際位置、暫停一拍後繼續。
- 下降到底後，會先等實際關節角追到命令 1° 以內（最多等 2 秒）才閉合夾爪。
- 交回 ACT 時會清空它的動作佇列，讓它從目前姿態重新推論。

## 檔案

| 檔案 | 內容 |
|---|---|
| `bottle_teach.py` | 每支手臂示教一次：記錄期望特徵，並量測腕部相機的手眼旋轉，產生 `teach_<side>.json` |
| `ibvs_assist.py` | lerobot-rollout 策略 `--strategy.type=ibvs_assist`；`rollout_home.py` 會自動註冊 |
| `servo.py` | 單臂狀態機：等待觸發 → 伺服 → 下降 → 夾取 → 提起 → 交回 |
| `detector.py` | OWLv2 背景偵測執行緒 |
| `kinematics.py` | Pinocchio 單臂運動學（`OPENARM_URDF`） |
| `test_grasp_servo.py` | 離線閉迴路測試：真實 URDF 運動學 + 合成相機 |
| `lgd_server.py` | （選用，未接上）Grasp-Anything LGD 推論服務，跑在 `lgd` conda 環境 |

## 使用步驟

```bash
conda activate lerobot
cd ~/lerobot/openarm_src
export PYTHONPATH=$HOME/lerobot
export OPENARM_HOME=$HOME/lerobot/openarm_src/reset.json
# OPENARM_URDF 已在 ~/.bashrc 設定
```

### 1. 每支手臂示教一次

換相機位置或夾爪後要重做。

```bash
python -m openarm_src.grasp.bottle_teach --side left  --query "bottle"
python -m openarm_src.grasp.bottle_teach --side right --query "bottle"
```

1. 主臂先自然下垂，按 Enter。
2. 用主臂把**張開的**夾爪擺到「一閉合就能夾到瓶子」的位置，按 Enter。
3. 手臂沿接近軸後退 10 cm（`--retreat`），記錄瓶子位置與 TCP 高度。
   - 下降距離用 FK 量到的**實際**後退量。手臂有重力、追蹤落後時，實際走的距離會比指令少。
4. 量測手眼旋轉：TCP 沿 x / y / z 各前後移動 1.5 cm。
   - 量測需要看到**完整的**瓶子，看不到時會自動再往後退，最多 15 cm，量完再回到示教位置。

看輸出的數字：

- **方向殘差**：應該 < 5°。
- **深度位移 / 手臂位移**：應接近 1。
- **相機光軸與接近軸夾角**：實測約 35°。

量測一直失敗時，可以直接指定旋轉，跳過量測：`--handeye-rpy -35 0 0`（相機繞自身 x 軸往夾爪傾斜約 35°）。正負號不確定時，先用量測結果確認。

看不到瓶子時，標註圖會存在 `openarm_src/grasp/teach_debug_<side>_*.png`：綠框可用，紅框表示被畫面切到或太大。

每支手臂的瓶子不同時，可以用 `--query` 區分，例如 `"blue bottle"`、`"cola bottle"`。

### 2. 第一次：只觀察，不接手

```bash
python rollout_home.py --strategy.type=ibvs_assist --strategy.observe_only=true \
  --policy.path=$HOME/models/act_openarm_200k/400000/pretrained_model --duration=45 \
  --robot.type=bi_openarm_follower \
  --robot.left_arm_config.port=can1 --robot.left_arm_config.side=left \
  --robot.right_arm_config.port=can0 --robot.right_arm_config.side=right \
  --robot.id=bi_follower \
  --robot.cameras='{"ego": {"type": "intelrealsense", "serial_number_or_name": "239222300027", "width": 640, "height": 480, "fps": 30}, "wrist_left": {"type": "intelrealsense", "serial_number_or_name": "353322271140", "width": 424, "height": 240, "fps": 30}, "wrist_right": {"type": "intelrealsense", "serial_number_or_name": "353322271542", "width": 424, "height": 240, "fps": 30}}' \
  --fps=30
```

這一步完全由 ACT 控制。終端機會印出 `觀察模式：達到接手條件`，確認接手時機合理後再進行下一步。

### 3. 實際接手

把上面的 `--strategy.observe_only=true` 拿掉。只想先測一支手臂時加 `--strategy.arms=left`。

### 4. eval：ACT 與 ACT+IBVS 交錯盲測

```bash
python eval_trials.py \
  --ckpts act=$HOME/models/act_openarm_200k/400000/pretrained_model \
          act_ibvs=$HOME/models/act_openarm_200k/400000/pretrained_model \
  --ibvs act_ibvs --trials 10 --duration 45 --n-objects 2 --blind --log eval_ibvs.csv \
  -- --robot.type=bi_openarm_follower ...（與上面相同的 robot / cameras 參數）
```

- 只傳給 IBVS 回合的參數用 `--ibvs-arg`，例如 `--ibvs-arg=--strategy.arms=left`。
- 看統計：`python eval_trials.py --summary --log eval_ibvs.csv`。

## 每次 rollout 的紀錄（`openarm_src/grasp/runs/<時間>/`）

| 檔案 | 內容 |
|---|---|
| `ibvs.log` | 整次 rollout 的 log，包括「看到瓶子但不接手」的原因（太近、太遠、離 s* 太遠、誤差太大） |
| `<side>_takeover.png` | 接手當下的腕部影像與偵測框（紅十字 = s*） |
| `<side>_done.png` / `<side>_aborted.png` | 結束當下的影像 |
| `<side>_trace.csv` | 接手期間每次偵測的框、分數、深度、誤差、TCP 高度 |

接手的條件：指定瓶子的框（分數 ≥ 0.25、框內有深度）在 35 cm 以內、框中心沒貼著畫面邊緣、離 s* 120 px 以內，而且連續 3 次偵測都符合。

- **一般情況**：還要求誤差小於 8 cm，接手後直接伺服。
- **太近**：比示教距離近超過 2 cm，代表 ACT 已經貼到瓶子上了。接手後先張開夾爪；只有深度 < 7 cm（D405 量不到）或看不到瓶子時，才往後上方斜退（最多 12 cm），連續 2 次看得到瓶子就開始伺服。
  - 後退方向 = 夾爪接近方向的水平分量反向，再往上抬 `retreat_up_deg`（預設 35°）。這個仰角是**相對基座水平面**，不是相對夾爪；log 會印出實際方向與仰角。
  - 相機光軸相對夾爪接近軸往下傾約 35°，所以只有夾爪接近軸是水平時，35° 才剛好是沿光軸往後退（瓶子只會縮小、留在畫面裡）。夾爪本身往下傾 θ 時，應設成 θ + 35°（見下方「後退仰角」）。

收斂：位置誤差 < 3 mm 連續 3 次偵測，就沿示教路徑下降。

## 可調參數（`--strategy.<name>`）

| 參數 | 預設 | 說明 |
|---|---|---|
| `arms` | `left,right` | 哪些手臂可以接手 |
| `observe_only` | `false` | 只偵測、不接手 |
| `detector_threshold` | 0.10 | OWLv2 輸出框的最低分數（追蹤時也用 0.10） |
| `trigger_score` / `trigger_frames` | 0.25 / 3 | 觸發用的分數門檻與連續次數 |
| `trigger_max_depth_m` | 0.35 | 瓶子比這個近才接手 |
| `min_policy_time_s` | 1.0 | rollout 開始後多久才允許接手 |
| `gain` / `max_speed_m_s` | 1.5 / 0.04 | 伺服增益與最大速度 |
| `lift_m` | 0.06 | 夾到後往上提多少 |
| `detector_hz` | 30 | 偵測頻率上限。實際受推論時間限制：1 台相機約 28 次/s，2 台約 19 次/s |
| `idle_detect_every` | 3 | 有手臂在接手時，閒置手臂的相機每幾拍才偵測一次，讓接手的手臂更快 |
| `trigger_max_px` | 120 | 瓶子框中心離 s* 最多幾 px 才接手，避免選到另一支手臂的瓶子 |
| `query_left` / `query_right` | 空白 | 每支手臂要找的瓶子，例如 `"coca cola bottle"`、`"blue bottle"`；空白就用示教檔的設定 |
| `center_u` | `true` | 左右對準到畫面中心；`false` = 對準示教時的 u\* || `retreat_up_deg` | 35 | 太近時往後上方退的仰角（相對基座水平面） |
| `retreat_min_depth_m` | 0.07 | 太近時，深度小於這個值或看不到瓶子才繼續後退 |
| `extra_depth_m` | 0.015 | 比示教位置再往前深入多少才閉合；夾太淺、會滑掉就加大，例如 0.025 |

夾爪開合角度（`-55` / `0`）、收斂門檻（3 mm × 3 次）、後退速度與上限（3 cm/s、12 cm）、下降速度等在 `servo.py` 的 `ServoConfig`。
