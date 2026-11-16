# OpenArm + RealSense (D455 + D405*2) + AprilTag Cube 視覺伺服實驗套件

本目錄針對 **OpenArm 機械手臂**、**Intel RealSense D455 (Ego)**、**Intel RealSense D405 x 2 (雙腕相機)** 與 **5×5×5 cm AprilTag Cube** 部署最後一哩操作（Last-Mile Manipulation）實驗。

完全落實《學生快速實作指南》規範：
> **核心原則：** 先跑傳統 IBVS，再學習「傳統物理模型算錯多少」（Physics-Informed Residual），最後加入保守收斂檢查（Convergence Guard）。

---

## 1. 目錄結構與模組職責

```
openarm_src/test/
├── config.py                 # 硬體與控制參數（D455/D405、Cube尺寸、關節限位、目標特徵 s*）
├── camera_stream.py          # 多相機管理器（pyrealsense2 實機驅動 + Mock 離線合成串流）
├── tag_detector.py           # AprilTag (tag36h11) 偵測器與 3D 姿態估計 (cv2.aruco)
├── openarm_robot.py          # OpenArm 機械手臂介面封裝（安全限幅、速度限制、Dry-Run 模擬）
├── ibvs_pinn_controller.py   # IBVS + Physics-Informed Residual + Convergence Guard 控制器
├── data_collector.py         # 安全微幅試探 (Dithering) 資料收集腳本
├── main_experiment.py        # 完整三方法 (IBVS / Learned / Guarded) 閉迴路實驗與評估主程式
├── requirements.txt          # 相依套件清單
└── README.md                 # 本說明文件
```

---

## 2. 硬體配置與標籤設定

- **機械手臂**：OpenArm (6-DOF)，本實驗主要以末端/前兩軸進行局部視覺伺服對準。
- **視角配置**：
  - `ego` (**D455**)：架設於手臂基座上方或全局俯視視角，負責全局監控與宏觀定位。
  - `wrist_left` (**D405**)：左腕部微距深度相機，最近對焦距離可達 7cm，作為最後一哩 IBVS 主伺服相機。
  - `wrist_right` (**D405**)：右腕部相機，提供輔助視角或雙臂協同觀測。
- **待夾物**：$5\text{ cm} \times 5\text{ cm} \times 5\text{ cm}$ 立方體。
  - 正上方中心貼有 **AprilTag 36h11 (ID: 0)**，標籤邊長設定為 $5\text{ cm} \times 5\text{ cm}$（滿版貼合）。

---

## 3. 快速上手步驟

### 步驟 1：安裝相依套件
在 Python 3.8+ 環境下執行：
```bash
cd openarm_src/test
pip install numpy matplotlib opencv-python torch scipy
# 若要在真實機台上連接 RealSense 相機：
pip install pyrealsense2
```

### 步驟 2：相機序號確認與 AprilTag 檢驗
執行檢查工具查看已連接之 D455 / D405 序號：
```bash
python check_cameras.py
```
將輸出的序號填入 `config.py` 中的 `cameras` 區段。

### 步驟 3：執行離線 Dry-Run 模擬實驗（不需馬達或實體相機）
```bash
python main_experiment.py --learner ridge --episodes 15
```
或者使用 PyTorch MLP 學習器：
```bash
python main_experiment.py --learner torch --episodes 15
```
程式會自動完成：
1. 執行安全小步長試探（收集 120 筆 $\Delta q \to \Delta s^{\text{obs}}$ 資料）。
2. 訓練殘差雅可比 $\Delta J_\theta$。
3. 對同一個初始狀態分組執行 **IBVS**、**Learned**、**Guarded** 三種方法。
4. 輸出終端統計表，並產生 `experiment_outputs/experiment_results.csv` 與 `experiment_error_curves.png`。

### 步驟 4：真機閉迴路工作流 (Real Robot Workflow)
確認硬體與方向無誤後，使用 `--real` 旗標啟動：
```bash
python main_experiment.py --real --camera wrist_left --episodes 10
```

---

## 4. 真機標準作業工作流 (Real Robot SOP)

```
[階段 1: 硬體巡檢與頻寬配置] 
   └── 3 台相機獨立 USB 3.0 Hub / 埠確認 -> E-Stop 急停按鈕就緒
[階段 2: 標籤視覺驗證與方向確認] 
   └── 5x5 cm Tag36h11 辨識率確認 -> 無接觸手動微動驗證 +/- 方向
[階段 3: 原地微幅試探 (Dithering)] 
   └── 執行 data_collector.py (關節增量 < 0.015 rad) -> 擬合 Delta J_theta
[階段 4: 三方法真機對比實驗] 
   └── 相同初始偏移測試: 純 IBVS vs 學習殘差 vs 守門機制 (Guarded)
[階段 5: 銜接下夾 (Last-Mile Grasp)] 
   └── 誤差 <= 3px -> 下降 Z 軸至預備高度 -> 閉合夾爪 -> 提起驗證
```

---

## 4. 實機安全守則（依指南第 7 條）

1. **先做無接觸對準**：確認相機座標系、影像特徵方向與關節增量正負號一致，切勿一開始就直接測試下夾動作。
2. **單步增量嚴格限幅**：`config.py` 中已預設 `max_delta_q_rad = 0.035 rad`（約 2 度），防止因模型預測偏差產生劇烈運動。
3. **退回機制保證**：Convergence Guard 會在每一步檢查候選命令的預測誤差，若未達最低改善門檻 $\delta$，立即退回名義 IBVS 解析指令。
