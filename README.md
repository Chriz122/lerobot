# git rule
## 正確的解決步驟：使用 upstream 機制
  請不要刪除原專案的 .git，並在終端機依序執行以下指令：
  保留原版連結：將原作者的儲存庫重新命名為 upstream（上游）。
  - 指令：
    - git remote add upstream https://github.com/huggingface/lerobot
    - git remote -v
  加入自己連結：將你在 GitHub 新建的儲存庫設為新的 origin。
  - 指令：
    - git remote add origin <你自己的新儲存庫網址>
    - git remote -v
  推送至你自己的 repo：將目前的程式碼推送到你自己的儲存庫。
  - 指令：
    - git push -u origin main（若預設分支為 master 則改為 master）

## 日後如何同步原作者的更新？
  當原作者更新了程式碼，你只需要在終端機輸入以下指令，就能將更新合併到你的進度中：
  - 抓取原版更新：git fetch upstream
  - 切換至主分支：git checkout main
  - 合併原版進度：git merge upstream/main
  - 同步至你的 repo：git push origin main

# OPENARM
## 啟用虛擬環境 
- source bin/activate

## 配置 CAN 介面 (針對標準 CAN FD／OpenArms 推薦)
<!-- - sudo ip link set can0 down
  sudo ip link set can0 type can bitrate 1000000 dbitrate 5000000 fd on
  sudo ip link set can0 up -->

- lerobot-setup-can --mode=setup --interfaces=can0,can1

## 測試電機通訊
- lerobot-setup-can --mode=test --interfaces=can0,can1

## openarm 操作
### 零位校正
lerobot-calibrate \
--robot.type=openarm_follower \
--robot.port=can0 \
--robot.side=right \
--robot.id=my_openarm_follower

### 遠端操作
1. lerobot-teleoperate \
    --robot.type=openarm_follower \
    --robot.port=can0 \
    --robot.side=right \
    --robot.id=my_follower \
    --teleop.type=openarm_leader \
    --teleop.port=can1 \
    --teleop.id=my_leader
2. lerobot-teleoperate \
    --robot.type=bi_openarm_follower \
    --robot.left_arm_config.port=can0 \
    --robot.left_arm_config.side=left \
    --robot.right_arm_config.port=can1 \
    --robot.right_arm_config.side=right \
    --robot.id=my_bimanual_follower \
    --teleop.type=bi_openarm_leader \
    --teleop.left_arm_config.port=can2 \
    --teleop.right_arm_config.port=can3 \
    --teleop.id=my_bimanual_leader

### 資料錄製
lerobot-record \
    --robot.type=openarm_follower \
    --robot.port=can0 \
    --robot.side=right \
    --robot.id=my_follower \
    --teleop.type=openarm_leader \
    --teleop.port=can1 \
    --teleop.id=my_leader \
    --repo-id=my_hf_username/my_openarm_dataset \
    --fps=30 \
    --num-episodes=10

# 安裝
## uv:
  0. curl -LsSf https://astral.sh/uv/install.sh | sh  # 安裝uv
  1. git clone https://github.com/Chriz122/lerobot
  2. uv venv --python 3.12.14
  3. source bin/activate
  4. cd lerobot
  5. uv pip install -e .
  6. uv pip install 
  7. uv pip install 'lerobot[all]'
  > [!NOTE]
  > 如果遇到建置錯誤，您可能需要安裝額外的依賴項目：cmake、build-essential 和 ffmpeg libs。在 Linux 上安裝這些項目，請執行:
  > sudo apt-get install cmake build-essential python3-dev pkg-config libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libswresample-dev libavfilter-dev
