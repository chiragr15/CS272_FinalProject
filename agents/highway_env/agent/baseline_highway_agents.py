"""
baseline_highway_agents.py

Optimized PPO agents for Highway-v0 with two observation types:
  - LidarObservation -> MLP policy
  - GrayscaleObservation -> CNN policy

Features:
  - Train OR load a saved PPO model per observation type.
  - Record per-episode training rewards and plot a learning curve.
  - Evaluate the trained model over many episodes with deterministic actions
    (no exploration) and plot a violin plot of returns.

Saved artifacts:
  Models:
    models/highway/ppo_highway_lidar.zip
    models/highway/ppo_highway_gray.zip
  Training rewards:
    models/highway/ppo_highway_lidar_train_rewards.npy
    models/highway/ppo_highway_gray_train_rewards.npy
  Plots:
    results/highway/highway_LidarObservation_learning.png
    results/highway/highway_LidarObservation_violin.png
    results/highway/highway_GrayscaleObservation_learning.png
    results/highway/highway_GrayscaleObservation_violin.png

Requirements:
  pip install highway-env gymnasium "stable-baselines3[extra]" matplotlib
"""

import os
import numpy as np
import gymnasium as gym
import highway_env  # noqa: F401

import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.preprocessing import is_image_space_channels_first


# --------------------
# Config
# --------------------
ENV_ID = "highway-v0"

RESULTS_DIR = "results/highway"
MODELS_DIR = "models/highway"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

RANDOM_SEED = 42

# NOTE: tuned total timesteps for better performance
TOTAL_TIMESTEPS_LIDAR = 300_000   
TOTAL_TIMESTEPS_GRAY = 300_000    

# Evaluation episodes (for violin plots)
EVAL_EPISODES = 500

# Force retraining even if model file exists
# Set to True at least once after changing env config/hyperparams
FORCE_RETRAIN_LIDAR = False
FORCE_RETRAIN_GRAY = True


# --------------------
# Utility: Reward Tracking Callback
# --------------------
class EpisodeRewardCallback(BaseCallback):
    """
    Custom callback to record episode rewards during training.

    Uses Monitor wrapper (env must be wrapped) which puts 'episode' info
    in infos when an episode terminates.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
        return True


# --------------------
# Environment builder
# --------------------
def make_highway_env(env_id: str, obs_type: str, seed: int = 0):
    """
    Returns a thunk for DummyVecEnv that creates a single Monitor-wrapped env.

    obs_type: "LidarObservation" or "GrayscaleObservation"

    IMPORTANT: Configs are fixed to match the professor's baseline.
    """

    def _init():
        # Match professor's configs exactly
        if obs_type == "GrayscaleObservation":
            config = {
                "observation": {
                    "type": "GrayscaleObservation",
                    "observation_shape": (128, 64),
                    "stack_size": 4,
                    "weights": [0.2989, 0.5870, 0.1140],  # RGB->Gray
                    "scaling": 1.75,
                }
            }
        elif obs_type == "LidarObservation":
            config = {
                "observation": {
                    "type": "LidarObservation",
                    "cells": 128,
                }
            }
        else:
            raise ValueError(f"Unsupported observation type: {obs_type}")

        env = gym.make(env_id, config=config)
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _init


# --------------------
# Train-or-load PPO helper
# --------------------
def train_or_load_ppo(
    model_path: str,
    rewards_path: str,
    policy_type: str,
    env_id: str,
    obs_type: str,
    total_timesteps: int,
    algo_name: str,
    seed: int,
    force_retrain: bool = False,
):
    """
    If model_path exists and not force_retrain:
        load model + training rewards from disk.
    Else:
        create env, train PPO, save model and rewards to disk.
    """
    if os.path.exists(model_path) and not force_retrain:
        print(f"[INFO] Loading existing model from {model_path}")
        model = PPO.load(model_path)
        if os.path.exists(rewards_path):
            rewards = np.load(rewards_path).tolist()
            print(f"[INFO] Loaded training rewards from {rewards_path}")
        else:
            rewards = []
            print(f"[WARN] No saved rewards found at {rewards_path}, skipping learning curve history.")
        return model, rewards

    # ----- Training path -----
    print(f"[INFO] No saved model or force_retrain=True for {algo_name} ({obs_type}). Training from scratch.")

    # Vec env
    if obs_type == "GrayscaleObservation":
        vec_env = DummyVecEnv(
            [make_highway_env(env_id, obs_type, seed=seed)]
        )

        # Only transpose if the env is channels-last (H, W, C or H, W, stack)
        if not is_image_space_channels_first(vec_env.observation_space):
            print("[INFO] Grayscale obs is channels-last; applying VecTransposeImage.")
            vec_env = VecTransposeImage(vec_env)
        else:
            print("[INFO] Grayscale obs is already channels-first; no transposition needed.")
    else:
        vec_env = DummyVecEnv(
            [make_highway_env(env_id, obs_type, seed=seed)]
        )

    # Hyperparameters tuned separately for Lidar vs Grayscale
    if obs_type == "LidarObservation":
        # Slightly larger batch + more epochs for more stable updates
        model = PPO(
            policy_type,
            vec_env,
            n_steps=1024,
            batch_size=256,      # was 128
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            learning_rate=3e-4,
            n_epochs=15,         # was 10
            clip_range=0.2,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            seed=seed,
        )
    else:  # GrayscaleObservation
        # More steps + larger batch to exploit CNN and stacked frames
        model = PPO(
            policy_type,
            vec_env,
            n_steps=2048,
            batch_size=512,      # was 256
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            learning_rate=2.5e-4,
            n_epochs=4,
            clip_range=0.2,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            seed=seed,
        )

    callback = EpisodeRewardCallback()
    print(f"\n=== Training {algo_name} ({obs_type}) for {total_timesteps} steps ===")
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Close vec env to free resources
    vec_env.close()

    # Save model and rewards
    model.save(model_path)
    np.save(rewards_path, np.array(callback.episode_rewards, dtype=np.float32))
    print(f"[INFO] Saved model to {model_path}")
    print(f"[INFO] Saved training rewards to {rewards_path}")

    return model, callback.episode_rewards


# --------------------
# Plotting helpers
# --------------------
def plot_learning_curve(
    rewards,
    algo_name: str,
    obs_type: str,
    save_dir: str,
    smoothing_window: int = 20,
):
    """
    rewards: [ep_reward1, ep_reward2, ...]
    Saves one learning curve figure per obs_type.
    """

    if len(rewards) == 0:
        print(f"[WARN] No rewards to plot for {obs_type} (maybe loaded old model without rewards).")
        return

    plt.figure(figsize=(10, 6))

    rewards = np.array(rewards, dtype=float)
    episodes = np.arange(1, len(rewards) + 1)

    if len(rewards) >= smoothing_window:
        kernel = np.ones(smoothing_window) / smoothing_window
        smoothed = np.convolve(rewards, kernel, mode="valid")
        smoothed_episodes = np.arange(1, len(smoothed) + 1)
        plt.plot(
            smoothed_episodes,
            smoothed,
            label=f"{algo_name} (smoothed)",
        )
    else:
        plt.plot(episodes, rewards, label=algo_name)

    plt.xlabel("Episode")
    plt.ylabel("Episode return")
    plt.title(f"Highway-v0 Learning Curve ({obs_type})")
    plt.grid(True, alpha=0.3)
    plt.legend()

    filename = os.path.join(save_dir, f"highway_{obs_type}_learning.png")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"[INFO] Saved learning curve to {filename}")


def evaluate_agent(
    model,
    env_id: str,
    obs_type: str,
    n_episodes: int = 500,
    seed: int = 0,
):
    """
    Evaluate a trained model using deterministic actions (no exploration).

    Returns:
        list of episode returns (length n_episodes)
    """
    env = make_highway_env(env_id, obs_type, seed=seed)()

    returns = []

    for ep in range(n_episodes):
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"[EVAL {obs_type}] Episode {ep + 1}/{n_episodes}")

        obs, info = env.reset(seed=seed + ep)
        done = False
        truncated = False
        ep_return = 0.0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_return += reward

        returns.append(ep_return)

    env.close()
    return returns


def plot_violin(
    returns,
    obs_type: str,
    save_dir: str,
    algo_name: str = "PPO",
):
    """
    Plot a violin plot of episodic returns for evaluation.
    """

    plt.figure(figsize=(6, 6))
    plt.violinplot(returns, showmeans=True, showextrema=True, showmedians=True)

    plt.xticks([1], [algo_name])
    plt.ylabel("Episode return")
    plt.title(f"Highway-v0 Test Performance ({obs_type})\n{algo_name}, {len(returns)} episodes")
    plt.grid(True, axis="y", alpha=0.3)

    filename = os.path.join(save_dir, f"highway_{obs_type}_violin.png")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"[INFO] Saved violin plot to {filename}")


# --------------------
# Main
# --------------------
def main():
    np.random.seed(RANDOM_SEED)

    # ---------- LIDAR ----------
    obs_type = "LidarObservation"
    lidar_model_path = os.path.join(MODELS_DIR, "ppo_highway_lidar.zip")
    lidar_rewards_path = os.path.join(MODELS_DIR, "ppo_highway_lidar_train_rewards.npy")

    ppo_lidar_model, ppo_lidar_rewards = train_or_load_ppo(
        model_path=lidar_model_path,
        rewards_path=lidar_rewards_path,
        policy_type="MlpPolicy",
        env_id=ENV_ID,
        obs_type=obs_type,
        total_timesteps=TOTAL_TIMESTEPS_LIDAR,
        algo_name="PPO-MLP",
        seed=RANDOM_SEED,
        force_retrain=FORCE_RETRAIN_LIDAR,
    )

    plot_learning_curve(
        rewards=ppo_lidar_rewards,
        algo_name="PPO-MLP",
        obs_type=obs_type,
        save_dir=RESULTS_DIR,
    )

    lidar_returns = evaluate_agent(
        ppo_lidar_model,
        env_id=ENV_ID,
        obs_type=obs_type,
        n_episodes=EVAL_EPISODES,
        seed=RANDOM_SEED + 100,
    )
    plot_violin(
        returns=lidar_returns,
        obs_type=obs_type,
        save_dir=RESULTS_DIR,
        algo_name="PPO-MLP",
    )

    # ---------- GRAYSCALE ----------
    obs_type = "GrayscaleObservation"
    gray_model_path = os.path.join(MODELS_DIR, "ppo_highway_gray.zip")
    gray_rewards_path = os.path.join(MODELS_DIR, "ppo_highway_gray_train_rewards.npy")

    ppo_gray_model, ppo_gray_rewards = train_or_load_ppo(
        model_path=gray_model_path,
        rewards_path=gray_rewards_path,
        policy_type="CnnPolicy",
        env_id=ENV_ID,
        obs_type=obs_type,
        total_timesteps=TOTAL_TIMESTEPS_GRAY,
        algo_name="PPO-CNN",
        seed=RANDOM_SEED + 1,
        force_retrain=FORCE_RETRAIN_GRAY,
    )

    plot_learning_curve(
        rewards=ppo_gray_rewards,
        algo_name="PPO-CNN",
        obs_type=obs_type,
        save_dir=RESULTS_DIR,
    )

    gray_returns = evaluate_agent(
        ppo_gray_model,
        env_id=ENV_ID,
        obs_type=obs_type,
        n_episodes=EVAL_EPISODES,
        seed=RANDOM_SEED + 200,
    )
    plot_violin(
        returns=gray_returns,
        obs_type=obs_type,
        save_dir=RESULTS_DIR,
        algo_name="PPO-CNN",
    )

    print("\n[INFO] PPO training/loading + evaluation finished.")
    print("[INFO] Check results/highway/ for saved models and plots.")


if __name__ == "__main__":
    main()
