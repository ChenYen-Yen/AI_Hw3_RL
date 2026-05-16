import gymnasium as gym
import ale_py

# 【關鍵步驟】強迫 Gymnasium 去讀取並註冊 ale-py 裡面的所有遊戲環境
gym.register_envs(ale_py)

print("--- 正在嘗試建立新版 Atari Pong 環境 ---")
try:
    # 新版 Gymnasium 與 ale-py 的標準命名格式為 "ALE/遊戲名稱-v5"
    env = gym.make("ALE/Pong-v5", render_mode="human")
    print("🎉 成功使用 ALE/Pong-v5 建立環境！")
except Exception as e:
    print(f"❌ 建立失敗，錯誤訊息: {e}")
    print("💡 提示：如果看到 'ArgumentError' 或 'ROM not found'，請看下方的手動修復步驟。")
    exit()

# 測試運行
observation, info = env.reset()
for _ in range(300):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()

env.close()
print("🎉 測試順利完成！")