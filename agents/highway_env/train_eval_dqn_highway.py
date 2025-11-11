# train_eval_dqn_highway.py  (fixed)
import os
import time
import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
import highway_env  # noqa: F401

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.preprocessing import is_image_space_channels_first

# ---------------------------
# Utilities
# ---------------------------
def timestamp():
    return time.strftime("%Y%m%d-%H%M%S")


def moving_average(x, w):
    if len(x) == 0:
        return np.array([])
    w = min(w, len(x))
    return np.convolve(x, np.ones(w)/w, mode="valid")


class EpisodeReturnCallback(BaseCallback):
    """Collect episodic *training* returns emitted by Monitor in info['episode']."""
    def __init__(self):
        super().__init__()
        self.episode_returns = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if isinstance(info, dict) and "episode" in info:
                ep = info["episode"]
                self.episode_returns.append(ep["r"])
                self.episode_lengths.append(ep["l"])
        return True


# ---------------------------
# Env factories (using config=... at make)
# ---------------------------
def make_highway_grayscale_env(seed=0):
    """
    Highway with GrayscaleObservation and 4-frame stack provided by the env itself.
    We'll transpose HWC->CHW via VecTransposeImage for SB3 CnnPolicy.
    """
    grayscale_cfg = {
        "observation": {
            "type": "GrayscaleObservation",
            "weights": [0.2989, 0.5870, 0.1140],
            "stack_size": 4,
            "observation_shape": (84, 84),
        },
        "policy_frequency": 15,
        "lanes_count": 4,
    }

    def _init():
        env = gym.make("highway-v0", render_mode=None, config=grayscale_cfg)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def make_highway_lidar_env(seed=0):
    """Highway with LidarObservation (vector). Use MLP policy."""
    lidar_cfg = {
        "observation": {
            "type": "LidarObservation",
            "cells": 32,
            "maximum_range": 50.0,
            "normalize": True,
        },
        "policy_frequency": 15,
        "lanes_count": 4,
    }

    def _init():
        env = gym.make("highway-v0", render_mode=None, config=lidar_cfg)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


# ---------------------------
# Training / eval helpers
# ---------------------------
def train_dqn(env_fn,
              policy: str,
              total_timesteps: int,
              learning_rate: float,
              buffer_size: int,
              batch_size: int,
              seed: int,
              transpose_for_cnn: bool = None):  # None -> auto
    set_random_seed(seed)

    vec_env = DummyVecEnv([env_fn])

    # Auto-detect image layout and transpose only if channels-last.
    if policy == "CnnPolicy":
        obs_space = vec_env.observation_space
        # If caller forced a choice, obey it; else infer.
        if transpose_for_cnn is None:
            need_transpose = not is_image_space_channels_first(obs_space)
        else:
            need_transpose = transpose_for_cnn
        if need_transpose:
            vec_env = VecTransposeImage(vec_env)

    model = DQN(
        policy,
        vec_env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        verbose=1,
        seed=seed,
    )

    cb = EpisodeReturnCallback()
    model.learn(total_timesteps=total_timesteps, callback=cb)
    return model, cb.episode_returns


def evaluate(model, make_env_fn, n_episodes=500, deterministic=True, seed=123):
    env = make_env_fn()()
    ep_returns = []
    rng = np.random.default_rng(seed)

    for _ in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 10_000_000)))
        done = False
        truncated = False
        ep_ret = 0.0
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, truncated, _ = env.step(action)
            ep_ret += reward
        ep_returns.append(ep_ret)
    env.close()
    return ep_returns


def plot_learning_curve(episode_returns, label, out_path, ma_window=50):
    plt.figure(figsize=(8, 5))
    xs = np.arange(1, len(episode_returns)+1)
    plt.plot(xs, episode_returns, alpha=0.35, label=f"{label} (per-ep)")
    ma = moving_average(episode_returns, ma_window)
    if len(ma) > 0:
        plt.plot(np.arange(ma_window, ma_window + len(ma)), ma, linewidth=2, label=f"{label} (MA{ma_window})")
    plt.xlabel("Training Episodes")
    plt.ylabel("Return")
    plt.title(f"Learning Curve – {label}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_violin(returns_dict, out_path):
    names = list(returns_dict.keys())
    data = [returns_dict[k] for k in names]
    plt.figure(figsize=(8, 5))
    plt.violinplot(data, showmeans=True, showextrema=True, showmedians=True)
    plt.xticks(np.arange(1, len(names)+1), names)
    plt.ylabel("Return (500 deterministic episodes)")
    plt.title("Performance Distribution – DQN on Highway")
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    run_id = timestamp()
    out_dir = os.path.join("runs", f"highway_dqn_{run_id}")
    os.makedirs(out_dir, exist_ok=True)

    seed = 42
    STEPS_GRAY = 400_000
    STEPS_LIDAR = 300_000

    # 1) GrayscaleObservation (CnnPolicy)
    print("\n=== Training DQN (GrayscaleObservation) ===")
    model_gray, train_returns_gray = train_dqn(
        env_fn=make_highway_grayscale_env(seed=seed),
        policy="CnnPolicy",
        total_timesteps=STEPS_GRAY,
        learning_rate=1e-4,
        buffer_size=200_000,
        batch_size=64,
        seed=seed,
        transpose_for_cnn=False,
    )

    gray_model_path = os.path.join(out_dir, "dqn_highway_gray.zip")
    model_gray.save(gray_model_path)

    lc_gray_path = os.path.join(out_dir, "learning_curve_grayscale.png")
    plot_learning_curve(train_returns_gray, "DQN-Grayscale", lc_gray_path, ma_window=50)

    print("\n=== Evaluating DQN (GrayscaleObservation) — 500 episodes, deterministic ===")
    eval_returns_gray = evaluate(model_gray, make_highway_grayscale_env, n_episodes=500, deterministic=True, seed=seed)
    print(f"Grayscale: mean={np.mean(eval_returns_gray):.2f}, std={np.std(eval_returns_gray):.2f}")

    violin_gray_path = os.path.join(out_dir, "violin_grayscale.png")
    plot_violin({"Grayscale": eval_returns_gray}, violin_gray_path)

    # 2) LidarObservation (MlpPolicy)
    print("\n=== Training DQN (LidarObservation) ===")
    model_lidar, train_returns_lidar = train_dqn(
        env_fn=make_highway_lidar_env(seed=seed),
        policy="MlpPolicy",
        total_timesteps=STEPS_LIDAR,
        learning_rate=2.5e-4,
        buffer_size=200_000,
        batch_size=128,
        seed=seed,
        transpose_for_cnn=False,
    )

    lidar_model_path = os.path.join(out_dir, "dqn_highway_lidar.zip")
    model_lidar.save(lidar_model_path)

    lc_lidar_path = os.path.join(out_dir, "learning_curve_lidar.png")
    plot_learning_curve(train_returns_lidar, "DQN-Lidar", lc_lidar_path, ma_window=50)

    print("\n=== Evaluating DQN (LidarObservation) — 500 episodes, deterministic ===")
    eval_returns_lidar = evaluate(model_lidar, make_highway_lidar_env, n_episodes=500, deterministic=True, seed=seed)
    print(f"Lidar: mean={np.mean(eval_returns_lidar):.2f}, std={np.std(eval_returns_lidar):.2f}")

    violin_lidar_path = os.path.join(out_dir, "violin_lidar.png")
    plot_violin({"Lidar": eval_returns_lidar}, violin_lidar_path)

    # Combined comparison violin
    combined_violin_path = os.path.join(out_dir, "violin_grayscale_vs_lidar.png")
    plot_violin({"Grayscale": eval_returns_gray, "Lidar": eval_returns_lidar}, combined_violin_path)

    print("\n=== Done ===")
    print(f"Saved models to:\n  {gray_model_path}\n  {lidar_model_path}")
    print("Saved figures to:")
    for p in [lc_gray_path, lc_lidar_path, violin_gray_path, violin_lidar_path, combined_violin_path]:
        print(" ", p)
