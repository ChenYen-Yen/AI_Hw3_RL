"""
Inference script to use the trained PPO model
"""
import os
import json
import torch
import gymnasium as gym
from pathlib import Path
from datetime import datetime
from ppo import Agent, Args


def load_model(model_path: str, device: str = "mps", render: bool = False) -> tuple:
    """Load trained model from checkpoint
    
    Args:
        model_path: Path to the saved model checkpoint
        device: Device to load model on ('mps' or 'cpu')
        render: Whether to render the environment
    
    Returns:
        Tuple of (agent, device, env)
    """
    # Verify model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    # Setup device
    if device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✓ Using MPS device")
    else:
        device = torch.device("cpu")
        print("✓ Using CPU device")
    
    # Create environment with render mode if needed
    render_mode = "human" if render else None
    try:
        env = gym.make("CartPole-v1", render_mode=render_mode)
        print(f"✓ Environment created with render_mode={render_mode}")
    except Exception as e:
        print(f"⚠ Could not create environment with render_mode: {e}")
        env = gym.make("CartPole-v1")
        print("✓ Environment created without rendering")
    
    # Wrap to get the required interface
    class DummyEnv:
        def __init__(self, env):
            self.env = env
            self.single_observation_space = env.observation_space
            self.single_action_space = env.action_space
    
    envs = DummyEnv(env)
    
    # Create agent
    agent = Agent(envs).to(device)
    
    # Load weights
    state_dict = torch.load(model_path, map_location=device)
    agent.load_state_dict(state_dict)
    agent.eval()
    
    print(f"✓ Model loaded from {model_path}")
    return agent, device, env


def run_episode(agent, env, device, num_episodes: int = 1) -> dict:
    """Run trained agent in environment
    
    Args:
        agent: Trained agent model
        env: Gymnasium environment
        device: Device to run on
        num_episodes: Number of episodes to run
    
    Returns:
        Dictionary containing episode results
    """
    episode_details = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done:
            # Convert observation to tensor
            obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device).float()
            
            # Get action from agent
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs_tensor)
            
            action = action.item()
            
            # Take action in environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1
        
        episode_details.append({
            "episode": episode + 1,
            "reward": float(episode_reward),
            "steps": steps
        })
        print(f"Episode {episode + 1}: Reward = {episode_reward}, Steps = {steps}")
    
    env.close()
    
    avg_reward = sum([e["reward"] for e in episode_details]) / len(episode_details)
    print(f"\nAverage reward over {num_episodes} episodes: {avg_reward:.2f}")
    
    return {
        "episodes": episode_details,
        "average_reward": avg_reward,
        "total_episodes": num_episodes
    }


def find_latest_model(runs_dir: str = "runs") -> str:
    """Find the latest trained model checkpoint
    
    Args:
        runs_dir: Directory containing training runs
    
    Returns:
        Path to the latest model checkpoint
    """
    runs_path = Path(runs_dir)
    
    if not runs_path.exists():
        raise FileNotFoundError(f"Runs directory not found at {runs_dir}")
    
    # Find latest run directory
    run_dirs = sorted([d for d in runs_path.iterdir() if d.is_dir()])
    if not run_dirs:
        raise FileNotFoundError(f"No training runs found in {runs_dir}")
    
    latest_run = run_dirs[-1]
    model_path = latest_run / "checkpoints" / "final_model.pt"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    return str(model_path)


def save_results(results: dict, model_path: str, device: str, render: bool, 
                output_dir: str = "codes/cartpole/inference_result") -> str:
    """Save inference results to files
    
    Args:
        results: Dictionary containing inference results
        model_path: Path to the model used
        device: Device used for inference
        render: Whether rendering was enabled
        output_dir: Directory to save results
    
    Returns:
        Path to the saved results directory
    """
    # Create timestamp for unique folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create timestamped subfolder
    base_output_path = Path(output_dir)
    base_output_path.mkdir(parents=True, exist_ok=True)
    
    timestamped_folder = base_output_path / timestamp
    timestamped_folder.mkdir(parents=True, exist_ok=True)
    
    # Prepare results data
    results_data = {
        "timestamp": timestamp,
        "model_path": model_path,
        "device": device,
        "render": render,
        "results": results
    }
    
    # Save as JSON
    json_path = timestamped_folder / "result.json"
    with open(json_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\n✓ Results saved to JSON: {json_path}")
    
    # Save as human-readable text
    txt_path = timestamped_folder / "result.txt"
    with open(txt_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("PPO CartPole Inference Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Render: {render}\n\n")
        
        f.write("Episode Results:\n")
        f.write("-" * 60 + "\n")
        for ep in results["episodes"]:
            f.write(f"Episode {ep['episode']}: Reward = {ep['reward']:.1f}, Steps = {ep['steps']}\n")
        
        f.write("\n" + "-" * 60 + "\n")
        f.write(f"Average Reward: {results['average_reward']:.2f}\n")
        f.write(f"Total Episodes: {results['total_episodes']}\n")
        f.write("=" * 60 + "\n")
    
    print(f"✓ Results saved to TXT: {txt_path}")
    
    return str(timestamped_folder)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run inference with trained PPO model")
    parser.add_argument("--model-path", type=str, default=None, 
                       help="Path to saved model checkpoint. If None, uses latest model.")
    parser.add_argument("--episodes", type=int, default=5, 
                       help="Number of episodes to run")
    parser.add_argument("--device", type=str, default="mps", 
                       choices=["mps", "cpu"], help="Device to use")
    parser.add_argument("--render", action="store_true", 
                       help="Whether to render the environment")
    parser.add_argument("--save-results", action="store_true", 
                       help="Save results to file")
    parser.add_argument("--output-dir", type=str, default="codes/cartpole/inference_result",
                       help="Directory to save results")
    
    args = parser.parse_args()
    
    # Find or load model
    if args.model_path is None:
        print("Finding latest model...")
        model_path = find_latest_model()
        print(f"Using model: {model_path}")
    else:
        model_path = args.model_path
    
    # Load model with render setting
    agent, device, env = load_model(model_path, device=args.device, render=args.render)
    
    # Run episodes
    print(f"\nRunning {args.episodes} episode(s)...\n")
    results = run_episode(agent, env, device, num_episodes=args.episodes)
    
    # Save results if requested
    if args.save_results:
        save_results(results, model_path, str(device), args.render, args.output_dir)
