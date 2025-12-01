import os
import time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # OpenMP workaround on Windows

import glob
import numpy as np
import pandas as pd
import gymnasium as gym 
import highway_env       

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from intersection_envs import make_intersection_env

LOG_ROOT = "logs_intersection"
os.makedirs(LOG_ROOT, exist_ok=True)


# -------------------------------------------------------------------
# Small helpers for ETA + quick reward sanity checks
# -------------------------------------------------------------------
def _format_eta(start_time: float, current_steps: int, total_timesteps: int) -> str:
    if current_steps <= 0:
        return "ETA: ?"
    elapsed = time.time() - start_time
    sec_per_step = elapsed / current_steps
    remaining_steps = max(total_timesteps - current_steps, 0)
    eta_sec = sec_per_step * remaining_steps
    mins = int(eta_sec // 60)
    secs = int(eta_sec % 60)
    return f"Elapsed: {elapsed:.1f}s | ETA: {mins}m {secs}s @ {sec_per_step:.4f}s/step"


def _print_recent_rewards(monitor_prefix: str, label: str, last_n: int = 10):
    """Read the last monitor CSV and print recent episode returns."""
    files = glob.glob(monitor_prefix + "*.monitor.csv")
    if not files:
        print(f"[{label}] No monitor files yet.")
        return

    files.sort()
    df = pd.read_csv(files[-1], comment="#")
    if df.empty or "r" not in df.columns:
        print(f"[{label}] Monitor file has no rewards yet.")
        return

    recent = df["r"].tail(last_n)
    print(
        f"[{label}] Last {len(recent)} episode returns: "
        f"mean={recent.mean():.2f}, last={recent.iloc[-1]:.2f}"
    )


def _make_single_env(
    obs_type: str,
    subdir: str,
    monitor_basename: str,
):
    """Create a single monitored Intersection env.

    The monitor file will be:
        logs_intersection/{subdir}/{monitor_basename}.monitor.csv

    This matches what intersection_eval_violin.py and intersection_plots.py
    expect when they glob for monitor files.
    """
    log_dir = os.path.join(LOG_ROOT, subdir)
    os.makedirs(log_dir, exist_ok=True)

    # Factory from intersection_envs.py (already configured with required obs config)
    env = make_intersection_env(obs_type)()
    monitor_prefix = os.path.join(log_dir, monitor_basename)
    env = Monitor(env, filename=monitor_prefix)

    return env, log_dir, monitor_prefix


# -------------------------------------------------------------------
# Lidar training (MlpPolicy)
# -------------------------------------------------------------------
def train_intersection_lidar(
    total_timesteps: int = 200_000,
):
    print("\n========== LIDAR TRAINING START (single env) ==========")
    env, log_dir, monitor_prefix = _make_single_env(
        obs_type="LidarObservation",
        subdir="lidar",
        monitor_basename="monitor_lidar",
    )

    # PPO hyperparameters chosen for stable learning on Intersection.
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,      # longer rollouts → more stable updates
        batch_size=256,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log=log_dir,
        # If you have a CUDA GPU, uncomment the next line:
        # device="cuda",
    )

    print(f"[LIDAR] Training for {total_timesteps} timesteps...")
    start = time.time()
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    print(f"[LIDAR] {_format_eta(start, total_timesteps, total_timesteps)}")
    print(f"[LIDAR] Training complete in {(time.time() - start)/60:.1f} min.")
    _print_recent_rewards(monitor_prefix, "LIDAR", last_n=20)

    # Save with .zip so intersection_eval_violin.py can load it directly.
    model_path = os.path.join(log_dir, "ppo_intersection_lidar.zip")
    model.save(model_path)
    print(f"[LIDAR] Model saved to: {model_path}")

    env.close()
    print("========== LIDAR TRAINING FINISHED ==========")


# -------------------------------------------------------------------
# Grayscale training (CnnPolicy)
# -------------------------------------------------------------------
def train_intersection_gray(
    total_timesteps: int = 200_000,
):

    print("\n========== GRAYSCALE TRAINING START (single env) ==========")
    env, log_dir, monitor_prefix = _make_single_env(
        obs_type="GrayscaleObservation",
        subdir="gray",
        monitor_basename="monitor_gray",
    )

    model = PPO(
        policy="CnnPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=128,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log=log_dir,
        # device="cuda",
    )

    print(f"[GRAY] Training for {total_timesteps} timesteps...")
    start = time.time()
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    print(f"[GRAY] {_format_eta(start, total_timesteps, total_timesteps)}")
    print(f"[GRAY] Training complete in {(time.time() - start)/60:.1f} min.")
    _print_recent_rewards(monitor_prefix, "GRAY", last_n=20)

    model_path = os.path.join(log_dir, "ppo_intersection_gray.zip")
    model.save(model_path)
    print(f"[GRAY] Model saved to: {model_path}")

    env.close()
    print("========== GRAYSCALE TRAINING FINISHED ==========")


if __name__ == "__main__":
    print(">>> About to train LIDAR model...")
    train_intersection_lidar()
    print(">>> LIDAR training done.")

    print(">>> About to train GRAYSCALE model...")
    train_intersection_gray()
    print(">>> GRAYSCALE training done.")

    print(">>> All training finished.")
