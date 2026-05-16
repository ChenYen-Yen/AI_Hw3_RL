建議執行 : python3 codes/atari/ppo_atari.py --track --device mps
ppo_atari.py : 主要訓練(on breakout)
- --track : track using mandb
- --device cpu : use cpu instead of gpu

*模型儲存在runs/{run_name}/checkpoints

inferance.py : 測試模型
- --model-path : 指定model
- --episodes : 進行局數
- --device : 指定裝置
- --render : 是否顯示
