import gymnasium as gym
import time

# 1. 建立 CartPole 環境，並設定 render_mode 為 "human" 以顯示視窗
env = gym.make("CartPole-v1", render_mode="human")

# 2. 重設環境，獲取初始狀態 (Observation)
observation, info = env.reset()

print("--- 開始執行 CartPole 本地測試 ---")
for step in range(200):
    # 從動作空間中「隨機」抽取一個動作 (0: 往左推, 1: 往右推)
    action = env.action_space.sample()
    
    # 執行動作，獲得下一個狀態、獎勵、以及是否結束的標記
    observation, reward, terminated, truncated, info = env.step(action)
    
    # 控制一下更新速度（每秒約 60 幀），方便肉眼觀察
    time.sleep(1 / 60)
    
    # 如果桿子倒了 (terminated) 或達到單局步數上限 (truncated)，就重設環境
    if terminated or truncated:
        print(f"第 {step} 步時回合結束，重設環境。")
        observation, info = env.reset()

# 3. 關閉環境視窗
env.close()
print("🎉 CartPole 本地測試順利完成！物理引擎與畫面渲染正常。")