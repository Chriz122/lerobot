import time
import cv2
import numpy as np
import pyrealsense2 as rs

from .lerobot.src.lerobot.robots.bi_openarm_follower import BiOpenArmFollower, BiOpenArmFollowerConfig
from .lerobot.src.lerobot.teleoperators.bi_openarm_leader import BiOpenArmLeader, BiOpenArmLeaderConfig
from .lerobot.src.lerobot.robots.openarm_follower import OpenArmFollowerConfig
from .lerobot.src.lerobot.teleoperators.openarm_leader import OpenArmLeaderConfig

# ==========================================
# 1. 初始化 Intel RealSense D435i 相機管線
# ==========================================
pipeline = rs.pipeline()
rs_config = rs.config()

# 啟用彩色串流 (解析度 640x480, 30 FPS, BGR8 格式)
rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 若需要深度畫面，可取消註解下行：
# rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

print("正在啟動 D435i 相機...")
pipeline.start(rs_config)

# ==========================================
# 2. 設定雙臂系統配置 (Follower & Leader)
# ==========================================
follower_config = BiOpenArmFollowerConfig(
    id="my_bimanual_follower",
    left_arm_config=OpenArmFollowerConfig(port="can0", side="left", use_can_fd=True),
    right_arm_config=OpenArmFollowerConfig(port="can1", side="right", use_can_fd=True),
)

leader_config = BiOpenArmLeaderConfig(
    id="my_bimanual_leader",
    left_arm_config=OpenArmLeaderConfig(port="can2", side="left", manual_control=True, use_can_fd=True),
    right_arm_config=OpenArmLeaderConfig(port="can3", side="right", manual_control=True, use_can_fd=True),
)

follower = BiOpenArmFollower(follower_config)
leader = BiOpenArmLeader(leader_config)

print("正在連線雙臂系統...")
follower.connect()
leader.connect()
print("系統就緒！按視窗中的 'q' 鍵或終端機 Ctrl+C 退出。")

# ==========================================
# 3. 主從控制 + 影像串流迴圈 (30 FPS)
# ==========================================
fps = 30
dt = 1.0 / fps

try:
    while True:
        start_time = time.perf_counter()

        # (1) 雙臂主從同步
        action = leader.get_action()
        follower.send_action(action)

        # (2) 擷取 D435i 相機畫面
        # 使用 wait_for_frames 等待最新一幀
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if color_frame:
            color_image = np.asanyarray(color_frame.get_data())

            # 在畫面上繪製即時 FPS 資訊
            cv2.putText(
                color_image,
                "Bi-OpenArm Teleop Running",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            # 顯示即時視窗
            cv2.imshow("D435i Real-time Stream", color_image)

        # 監聽鍵盤事件 (設定 1ms 避免阻塞迴圈)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("使用者按下 'q'，退出中...")
            break

        # (3) 維持 30 FPS 週期
        elapsed = time.perf_counter() - start_time
        if elapsed < dt:
            time.sleep(dt - elapsed)

except KeyboardInterrupt:
    print("\n偵測到中斷訊號，正在關閉...")

finally:
    # 4. 資源釋放
    cv2.destroyAllWindows()
    pipeline.stop()
    leader.disconnect()
    follower.disconnect()
    print("雙臂與相機已安全中斷並釋放資源。")