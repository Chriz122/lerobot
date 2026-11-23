# ACT + IBVS 瓶子夾取（`ibvs_assist`）

ACT 照常推論；某支手臂的腕部相機看到瓶子、而且夠近時，該手臂改由 IBVS 對準、下降、夾取、提起，再交回 ACT 放到托盤上。兩支手臂各自觸發，互不干擾。

```
ACT ──(腕部相機連續 3 次看到瓶子)──> [太近就先後退] ──> IBVS 對準 ──> 沿示教路徑下降 ──> 閉合夾爪 ──> 抬高 5 cm ──> 交回 ACT
```

- 偵測：OWLv2（`google/owlv2-base-patch16-ensemble`），兩台腕部相機一起推論，RTX 3090 Ti 上約 70 ms。
- 特徵：只用瓶子框中心的 **u** 和 D405 框內**深度 Z**。
  - 左右：預設把瓶子水平置中到**畫面中心** cx（相機在兩指正中間上方，`center_u=true`）；`center_u=false` 才用示教時的 u\*。沿相機 x 軸的水平方向移動。
  - 前後：讓深度等於示教的 Z\*，沿夾爪接近方向的水平分量移動。
  - 高度：維持示教時的 TCP 高度（運動學）。
  - 姿態（`orient_axes`，預設 `roll,pitch,yaw`）：伺服時同時修正三軸，各軸誤差都 < 1° 才下降。
    - **roll / pitch**：讓夾爪相對重力的姿態等於示教時的姿態（瓶子直立，`axis_source=gravity`，預設）。只用運動學，不需要相機校正。
      - `axis_source=depth`：改由框內 D405 深度點雲**擬合圓柱**量瓶軸，瓶子歪了也能對齊。但會用到手眼旋轉：角度差 δ 會讓夾爪多轉 δ。真機上每次 rollout 只有 1–5 幀擬合成功，成功時 pitch 誤差 5–14°，所以預設不用。
    - **yaw**：讓「夾爪接近方向」與「肩部 → 瓶子」水平連線的夾角，等於示教時的夾角。瓶子位置用影像加深度量出，所以瓶子放在不同位置時，也會用和示教時相同的方向接近。
    - 旋轉都繞**量到的瓶子點**，瓶子在畫面中的位置不變，不會被轉出畫面，也不會干擾平移控制。
    - 姿態目標來自示教檔的 `q_teach_deg`（`bottle_teach.py` 一直有存，舊示教檔可直接用）。沒有這個欄位時只做平移，log 會出現警告。
    - `orient_axes=""` = 舊行為：全程保持 ACT 交出的姿態，只做平移。
    - 為什麼要修（模擬）：ACT 交出的姿態 roll 偏 8–15° 時，相機（在夾爪上方、往下傾 35°）會被甩到側邊，只平移的話夾爪中線會偏離瓶軸 1–2.5 cm。pitch 偏 8° 時，下降路徑高度偏約 1.6 cm。
    - 已知限制：yaw 目標的參考點是 joint1 原點。瓶子在很側邊時，對到示教夾角可能讓腕部關節碰到限位，15 秒內沒收斂就會中止。這時可以改用 `orient_axes=roll,pitch`。
  - 不用框的 v：近距離時直立瓶子一定被畫面上下切到。也不需要準確的手眼傾斜角：相機 x 軸不受傾斜影響。但相機 x 軸（影像往右）是哪個方向要對，見「1. 示教」。
- 偵測：OWLv2 對同一支瓶子常給好幾個框（整支、只有標籤、只有瓶蓋），後處理又沒有 NMS。所以輸出前先合併重複框（`detector.merge_duplicate_boxes`）：重疊的框保留分數高的；小框被大框包住時保留大框，不再挑到只框住標籤的小框。大框同時包住左右並排的兩個框時，當成框到兩支瓶子，直接丟掉。
- 偵測：觸發時只認指定的瓶子（例如 `coca cola bottle`）；追蹤時也接受一般的 `bottle`，因為近距離看不到標籤時，指定查詢常常認不出來。
- **只追一個框**：接手時用框中心加深度算出瓶子在基座座標的位置，並鎖定它。瓶子不會動，所以之後每一幀只接受「瓶子點離鎖定位置水平 4 cm 以內」的那一個框（`track_gate_m`），其他框一律不看。
  - 以前是「離上一個框 64 px 以內」：目標漏偵測的那幾幀，旁邊另一支瓶子的框會被當成目標，一幀一幀跳過去（20261123_193345 右臂就是這樣跟丟）。
  - 直立瓶子表面任一點的水平位置都差不多，所以框被畫面上下切到、框中心上下跳，都不影響比對。
  - 觸發時，連續 N 次看到的也必須是同一支瓶子，否則重新計數。
  - 太近而先後退時，接手當下量到的位置較不準，門檻放寬為 8 cm。
- 對準後照「示教時實際後退的路徑」走回夾取位姿，所以後退時偏離直線（例如關節碰到限位）也不會夾偏。
- 每次 rollout，每支手臂只接手一次。伺服中跟丟 1.5 秒、或 15 秒內未收斂就中止。
- **交回 ACT 前一律先抬高 `lift_m`（預設 5 cm）**：夾到後抬（夾著瓶子），中止時也抬（夾爪維持張開），不讓 ACT 從貼著瓶子的位置接手。抬高時 log 顯示 `lift`，中止的影像 `<side>_aborted.png` 是中止當下（抬高之前）存的。
- 手臂跟不上命令（命令角與實際角差 > 6°，例如碰撞、卡住）**不會中止**：命令拉回目前實際位置、暫停一拍後繼續。
- 下降到底後，會先等實際關節角追到命令 1° 以內（最多等 2 秒）才閉合夾爪。
- 交回 ACT 時會清空它的動作佇列，讓它從目前姿態重新推論。
- 交回 ACT 後的 2 秒，該手臂的 step_guard 更嚴格：每拍上限從 0.3°（9°/s）依時間平方漸增回原本的 4°（120°/s）。ACT 第一拍常和目前姿態差 10° 以上（IBVS 轉過手腕，姿態和 ACT 訓練時不同），原本會以 120°/s 衝過去；現在約 0.9 s 平順轉過去。另一支手臂不受影響。

## 檔案

| 檔案 | 內容 |
|---|---|
| `bottle_teach.py` | 每支手臂示教一次：記錄期望特徵，產生 `teach_<side>.json` |
| `ibvs_assist.py` | lerobot-rollout 策略 `--strategy.type=ibvs_assist`；`rollout_home.py` 會自動註冊 |
| `servo.py` | 單臂狀態機：等待觸發 → 伺服 → 下降 → 夾取 → 提起 → 交回 |
| `detector.py` | OWLv2 背景偵測執行緒 |
| `kinematics.py` | Pinocchio 單臂運動學（`OPENARM_URDF`） |
| `test_grasp_servo.py` | 離線閉迴路測試：真實 URDF 運動學 + 合成相機 |
| `test_orientation.py` | 姿態控制的數學測試（瓶軸擬合、對準旋轉），不需 URDF |
| `fix_teach_handeye.py` | 把舊示教檔的手眼旋轉改成正確的名義值 `-35 0 90`，不用重新示教 |
| `lgd_server.py` | （選用，未接上）Grasp-Anything LGD 推論服務，跑在 `lgd` conda 環境 |

## 使用步驟

```bash
conda activate lerobot
cd ~/lerobot/openarm_src
export PYTHONPATH=$HOME/lerobot
export OPENARM_HOME=$HOME/lerobot/openarm_src/reset.json
# OPENARM_URDF 已在 ~/.bashrc 設定
```

### 1. 每支手臂示教一次（不需要校正）

換相機位置或夾爪後要重做。

**不需要相機校正**：伺服的目標是「示教時腕部相機看到瓶子的樣子」（瓶子在畫面中的位置與深度），執行時用同一台相機比對，所以相機裝得歪一點、手眼角度不準，都會在示教與執行之間抵銷。
- 手眼旋轉直接用名義值 `--handeye-rpy -35 0 90`（預設）：相機在夾爪上方、往下傾約 35°，**影像往右 = TCP +y**。
  - 真機側向夾取時 TCP x 朝上、y 是兩指張開方向、z 是接近軸，所以第三個值是 90°。
  - 傾斜角不用準：模擬中名義值寫成 0°、-15°、-25°、-45°（真實 -35°），夾取點誤差都是 2–4.5 mm。
  - **左右方向要對**：舊的名義值 `-35 0 0` 把影像往右當成 TCP x（朝上），左右對準會變成前後移動，瓶子一直偏在畫面一邊、夾爪往旁邊跑（20261123_194426 右臂）。啟動時若偵測到這種示教檔，log 會出現「手眼旋轉不對」。
  - 2026-11-23 之前的示教檔不用重新示教，執行一次 `python -m openarm_src.grasp.fix_teach_handeye` 即可修正（原檔備份成 `.json.bak`）。
- 只有「示教」這一步要做：用主臂擺一次夾取位置。

```bash
python -m openarm_src.grasp.bottle_teach --side left  --query "bottle"
python -m openarm_src.grasp.bottle_teach --side right --query "bottle"
```

1. 主臂先自然下垂，按 Enter。
2. 用主臂把**張開的**夾爪擺到「一閉合就能夾到瓶子」的位置，按 Enter。
3. 手臂沿接近軸後退 10 cm（`--retreat`），記錄瓶子位置與 TCP 高度。
   - 下降距離用 FK 量到的**實際**後退量。手臂有重力、追蹤落後時，實際走的距離會比指令少。
4. 存檔，完成。

（選用）想實際量測手眼旋轉時加 `--measure-handeye`：TCP 沿 x / y / z 各前後移動 1.5 cm，需要看到完整的瓶子。一般不需要。

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
| `<side>_takeover.png` | 接手當下的腕部影像，只畫鎖定的那個框（紅十字 = s*） |
| `<side>_done.png` / `<side>_aborted.png` | 結束當下的影像 |
| `<side>_trace.csv` | 接手期間每次偵測的框、分數、深度、誤差、TCP 高度；`locked` = 這幀有對上鎖定的瓶子，`lock_d_mm` = 最近的框離鎖定位置多遠，`lock_x/y/z` = 鎖定位置 |

接手的條件：指定瓶子的框（分數 ≥ 0.25、框內有深度）在 35 cm 以內、框中心沒貼著畫面邊緣、離 s* 120 px 以內，而且連續 3 次偵測都符合。

- **一般情況**：還要求誤差小於 8 cm，接手後直接伺服。
- **太近**：比示教距離近超過 2 cm，代表 ACT 已經貼到瓶子上了。接手後先張開夾爪；只有深度 < 7 cm（D405 量不到）或看不到瓶子時，才往後上方斜退（最多 12 cm），連續 2 次看得到瓶子就開始伺服。
  - 後退方向 = 夾爪接近方向的水平分量反向，再往上抬 `retreat_up_deg`（預設 35°）。這個仰角是**相對基座水平面**，不是相對夾爪；log 會印出實際方向與仰角。
  - 相機光軸相對夾爪接近軸往下傾約 35°，所以只有夾爪接近軸是水平時，35° 才剛好是沿光軸往後退（瓶子只會縮小、留在畫面裡）。夾爪本身往下傾 θ 時，應設成 θ + 35°（見下方「後退仰角」）。

收斂：位置誤差 < 3 mm、姿態各軸 < 1°，連續 3 次偵測都符合，就沿示教路徑下降。`<side>_trace.csv` 有每次偵測的 `e_roll_deg` / `e_pitch_deg` / `e_yaw_deg` 與 `axis_src`（`depth` / `depth_hold` / `gravity`）。

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
| `lift_m` | 0.05 | 交回 ACT 前往上抬多少（夾到後、中止時都會抬）；0 = 不抬 |
| `detector_hz` | 30 | 偵測頻率上限。實際受推論時間限制：1 台相機約 28 次/s，2 台約 19 次/s |
| `idle_detect_every` | 3 | 有手臂在接手時，閒置手臂的相機每幾拍才偵測一次，讓接手的手臂更快 |
| `trigger_max_px` | 120 | 瓶子框中心離 s* 最多幾 px 才接手，避免選到另一支手臂的瓶子 |
| `track_gate_m` | 0.04 | 接手後只追鎖定的瓶子：框的瓶子點離鎖定位置（水平）最多幾 m。常出現「沒有鎖定的瓶子，最近的框離鎖定的瓶子 4.x cm」時可放寬到 0.06 |
| `query_left` / `query_right` | 空白 | 每支手臂要找的瓶子，例如 `"coca cola bottle"`、`"blue bottle"`；空白就用示教檔的設定 |
| `center_u` | `true` | 左右對準到畫面中心；`false` = 對準示教時的 u\* |
| `orient_axes` | `roll,pitch,yaw` | 伺服時修正哪些姿態軸；空字串 = 只平移（舊行為） |
| `axis_source` | `gravity` | roll / pitch 的瓶軸來源：`gravity`（瓶子直立，不需校正）或 `depth`（深度擬合圓柱，瓶子歪了也能對，但需要準確的手眼角度） |
| `orient_tol_deg` | 1.0 | 各軸姿態誤差都小於這個值才下降 |
| `max_rot_speed_deg_s` | 15 | 姿態修正的最大角速度 |
| `retreat_up_deg` | 35 | 太近時往後上方退的仰角（相對基座水平面） |
| `retreat_min_depth_m` | 0.07 | 太近時，深度小於這個值或看不到瓶子才繼續後退 |
| `extra_depth_m` | 0.015 | 比示教位置再往前深入多少才閉合；夾太淺、會滑掉就加大，例如 0.025 |
| `handback_ramp_s` / `handback_start_step_deg` | 2.0 / 0.3 | 交回 ACT 後多久內加嚴 step_guard、起始每拍上限（度）；還是太快就把 0.3 調小或 2.0 調大，0 = 不加嚴 |
| `extra_depth_left_m` / `extra_depth_right_m` | 空白 | 只給某支手臂的 `extra_depth_m`，例如左臂夾太淺：`--strategy.extra_depth_left_m=0.03`。下降到底時 log 會印「實際比命令少往前 X mm」，夾太淺時先加這個數 |

夾爪開合角度（`-55` / `0`）、收斂門檻（3 mm × 3 次）、後退速度與上限（3 cm/s、12 cm）、下降速度等在 `servo.py` 的 `ServoConfig`。
