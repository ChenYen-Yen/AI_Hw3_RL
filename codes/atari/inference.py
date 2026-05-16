"""
Inference script for Atari Breakout PPO model.
Loads a trained model and runs inference with optional rendering and result saving.
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from gym import Env
import gymnasium as gym
import ale_py

# Register ALE environments with Gymnasium
gym.register_envs(ale_py)

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_model(model_path, device="mps", render=False):
    """
    Load a trained agent model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint (final_model.pt)
        device: 'mps' for M1/M2 Mac, 'cpu' for CPU
        render: Whether to render the environment
        
    Returns:
        agent: The loaded agent
        env: The environment
        device: The device being used
    """
    from ppo_atari import Agent
    from cleanrl_utils.atari_wrappers import (
        NoopResetEnv,
        MaxAndSkipEnv,
        EpisodicLifeEnv,
        FireResetEnv,
        ClipRewardEnv,
    )
    
    # Device selection
    if device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✓ Using MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        print("✓ Using CPU")
    
    # Create environment with same wrappers as training
    env = gym.make("ALE/Breakout-v5", render_mode="human" if render else None)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ClipRewardEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayscaleObservation(env)
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    
    # Create agent
    agent = Agent(env).to(device)
    
    # Load model state
    if os.path.exists(model_path):
        print(f"✓ Loading model from: {model_path}")
        agent.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"✗ Model not found at: {model_path}")
        return None, None, None
    
    agent.eval()
    return agent, env, device


def run_episode(agent, env, device, num_episodes=5):
    """
    Run inference episodes with the agent.
    
    Args:
        agent: The trained agent
        env: The environment
        device: Device to use
        num_episodes: Number of episodes to run
        
    Returns:
        dict: Contains episodes list and statistics
    """
    results = {
        "episodes": [],
        "total_reward": 0,
        "average_reward": 0,
        "max_reward": float('-inf'),
        "min_reward": float('inf'),
    }
    
    with torch.no_grad():
        for episode_num in range(num_episodes):
            obs, _ = env.reset()
            obs = torch.Tensor(obs).unsqueeze(0).to(device).float()
            
            episode_reward = 0
            steps = 0
            done = False
            
            while not done and steps < 10000:  # Max 10000 steps per episode
                # Forward pass
                action, _, _, _ = agent.get_action_and_value(obs)
                
                # Environment step
                next_obs, reward, terminated, truncated, info = env.step(action.cpu().numpy()[0])
                done = terminated or truncated
                
                episode_reward += reward
                obs = torch.Tensor(next_obs).unsqueeze(0).to(device).float()
                steps += 1
            
            results["episodes"].append({
                "episode": episode_num + 1,
                "reward": float(episode_reward),
                "steps": steps,
            })
            results["total_reward"] += episode_reward
            results["max_reward"] = max(results["max_reward"], episode_reward)
            results["min_reward"] = min(results["min_reward"], episode_reward)
            
            print(f"Episode {episode_num + 1}/{num_episodes}: Reward = {episode_reward:.1f}, Steps = {steps}")
    
    results["average_reward"] = results["total_reward"] / num_episodes
    return results


def find_latest_model(runs_dir="runs"):
    """
    Find the most recent trained model checkpoint.
    
    Args:
        runs_dir: Directory containing run folders
        
    Returns:
        str: Path to the latest model, or None if not found
    """
    if not os.path.exists(runs_dir):
        print(f"✗ Runs directory not found: {runs_dir}")
        return None
    
    # Find all checkpoints
    checkpoints = []
    for root, dirs, files in os.walk(runs_dir):
        if "checkpoints" in dirs:
            checkpoint_dir = os.path.join(root, "checkpoints")
            model_path = os.path.join(checkpoint_dir, "final_model.pt")
            if os.path.exists(model_path):
                # Get modification time
                mtime = os.path.getmtime(model_path)
                checkpoints.append((mtime, model_path))
    
    if not checkpoints:
        print(f"✗ No models found in {runs_dir}")
        return None
    
    # Sort by modification time and return the latest
    latest_path = sorted(checkpoints, reverse=True)[0][1]
    print(f"✓ Found latest model: {latest_path}")
    return latest_path


def save_results(results, model_path, device, render, output_dir="codes/atari/inference_result"):
    """
    Save inference results to timestamped folder.
    
    Args:
        results: Dictionary with inference results
        model_path: Path to the model used
        device: Device used
        render: Whether rendering was used
        output_dir: Base output directory
    """
    # Create timestamped subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(output_dir, timestamp)
    os.makedirs(result_dir, exist_ok=True)
    
    # Prepare data for JSON
    json_data = {
        "timestamp": timestamp,
        "model_path": model_path,
        "device": str(device),
        "render": render,
        "episodes": results["episodes"],
        "statistics": {
            "total_episodes": len(results["episodes"]),
            "average_reward": results["average_reward"],
            "max_reward": results["max_reward"],
            "min_reward": results["min_reward"],
            "total_reward": results["total_reward"],
        }
    }
    
    # Save JSON
    json_path = os.path.join(result_dir, "result.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"✓ Saved JSON results to: {json_path}")
    
    # Save text summary
    txt_path = os.path.join(result_dir, "result.txt")
    with open(txt_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("ATARI BREAKOUT - INFERENCE RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Render: {render}\n\n")
        
        f.write("EPISODE RESULTS:\n")
        f.write("-" * 60 + "\n")
        for ep in results["episodes"]:
            f.write(f"Episode {ep['episode']:3d}: Reward = {ep['reward']:7.1f}, Steps = {ep['steps']:5d}\n")
        
        f.write("\nSTATISTICS:\n")
        f.write("-" * 60 + "\n")
        f.write(f"Total Episodes:  {len(results['episodes'])}\n")
        f.write(f"Average Reward:  {results['average_reward']:.2f}\n")
        f.write(f"Max Reward:      {results['max_reward']:.1f}\n")
        f.write(f"Min Reward:      {results['min_reward']:.1f}\n")
        f.write(f"Total Reward:    {results['total_reward']:.1f}\n")
    
    print(f"✓ Saved text results to: {txt_path}")
    print(f"\n✓ Results saved to: {result_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Run inference on trained Atari Breakout model")
    parser.add_argument("--model-path", type=str, default=None, help="Path to model checkpoint (default: auto-detect latest)")
    parser.add_argument("--episodes", type=int, default=5, help="Number of inference episodes (default: 5)")
    parser.add_argument("--device", type=str, default="mps", choices=["mps", "cpu"], help="Device: 'mps' for M1/M2 Mac, 'cpu' for CPU")
    parser.add_argument("--render", action="store_true", help="Render the environment during inference")
    parser.add_argument("--save-results", action="store_true", help="Save inference results to file")
    parser.add_argument("--output-dir", type=str, default="codes/atari/inference_result", help="Output directory for results")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("ATARI BREAKOUT - INFERENCE")
    print("=" * 60 + "\n")
    
    # Find or load model
    model_path = args.model_path or find_latest_model()
    if not model_path:
        print("✗ No model specified and none found. Use --model-path to specify.")
        return
    
    # Load model
    agent, env, device = load_model(model_path, device=args.device, render=args.render)
    if agent is None:
        return
    
    print(f"✓ Model loaded successfully\n")
    
    # Run inference
    print(f"Running {args.episodes} inference episodes...")
    results = run_episode(agent, env, device, num_episodes=args.episodes)
    
    # Display results
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Average Reward: {results['average_reward']:.2f}")
    print(f"Max Reward:     {results['max_reward']:.1f}")
    print(f"Min Reward:     {results['min_reward']:.1f}")
    print()
    
    # Save results if requested
    if args.save_results:
        save_results(results, model_path, device, args.render, args.output_dir)
    
    # Cleanup
    env.close()
    print("✓ Done!")


if __name__ == "__main__":
    main()
