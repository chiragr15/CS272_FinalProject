# intersection_train.py
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

print(">>> intersection_train.py starting...")
print(f">>> LOG_ROOT = {LOG_ROOT}")


def _print_eta(start_time, current_steps, total_timesteps, prefix: str):
    """Print elapsed time and rough ETA."""
    if current_steps <= 0:
        return
    elapsed = time.time() - start_time
    sec_per_step = elapsed / current_steps
    remaining_steps = max(total_timesteps - current_steps, 0)
    eta_sec = sec_per_step * remaining_steps
    mins = int(eta_sec // 60)
    secs = int(eta_sec % 60)
    print(
        f"[{prefix}] Elapsed: {elapsed:.1f}s | "
        f"Estimated remaining: {mins}m {secs}s "
        f"@ {sec_per_step:.4f}s/step"
    )


def _print_recent_rewards(monitor_prefix: str, label: str, last_n: int = 10):
    """Read the monitor CSV(s) and print mean/last reward for recent episodes."""
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


def train_intersection_lidar(
    total_timesteps: int = 20000,
    log_dir: str = os.path.join(LOG_ROOT, "lidar"),
):
    print("\n========== LIDAR TRAINING START ==========\n")
    os.makedirs(log_dir, exist_ok=True)

    print("[LIDAR] Creating environment...")
    env = make_intersection_env("LidarObservation")()
    monitor_prefix = os.path.join(log_dir, "monitor_lidar")
    env = Monitor(env, filename=monitor_prefix)

    print("[LIDAR] Creating evaluation environment (unused, just for parity)...")
    eval_env = make_intersection_env("LidarObservation")()
    eval_env = Monitor(eval_env)

    print("[LIDAR] Initializing PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log=log_dir,
    )

    print(f"[LIDAR] Model initialized. Beginning training for {total_timesteps} timesteps...")
    # Use small chunk size so we print often
    step_chunk = 1024
    current_steps = 0
    start_time = time.time()

    while current_steps < total_timesteps:
        this_chunk = min(step_chunk, total_timesteps - current_steps)
        print(f"\n[LIDAR] Training chunk: {current_steps} → {current_steps + this_chunk} timesteps...")

        model.learn(total_timesteps=this_chunk, reset_num_timesteps=False)

        current_steps += this_chunk
        percent = int((current_steps / total_timesteps) * 100)
        print(f"[LIDAR] Progress: {percent}% complete ({current_steps}/{total_timesteps})")

        _print_eta(start_time, current_steps, total_timesteps, "LIDAR")
        _print_recent_rewards(monitor_prefix, "LIDAR")

    print("[LIDAR] Training complete. Saving model...")
    model_path = os.path.join(log_dir, "ppo_intersection_lidar")
    model.save(model_path)
    print(f"[LIDAR] Model saved to: {model_path}")

    env.close()
    eval_env.close()
    print("\n========== LIDAR TRAINING FINISHED ==========\n")



def train_intersection_gray(
    total_timesteps: int = 20000,
    log_dir: str = os.path.join(LOG_ROOT, "gray"),
):
    print("\n========== GRAYSCALE TRAINING START ==========\n")
    os.makedirs(log_dir, exist_ok=True)

    print("[GRAY] Creating environment...")
    env = make_intersection_env("GrayscaleObservation")()
    monitor_prefix = os.path.join(log_dir, "monitor_gray")
    env = Monitor(env, filename=monitor_prefix)

    print("[GRAY] Creating evaluation environment (unused, just for parity)...")
    eval_env = make_intersection_env("GrayscaleObservation")()
    eval_env = Monitor(eval_env)

    print("[GRAY] Initializing PPO model (CNN)...")
    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=32,
        n_epochs=3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log=log_dir,
    )

    print(f"[GRAY] Model initialized. Beginning training for {total_timesteps} timesteps...")
    step_chunk = 512
    current_steps = 0
    start_time = time.time()

    while current_steps < total_timesteps:
        this_chunk = min(step_chunk, total_timesteps - current_steps)
        print(f"\n[GRAY] Training chunk: {current_steps} → {current_steps + this_chunk} timesteps...")

        model.learn(total_timesteps=this_chunk, reset_num_timesteps=False)

        current_steps += this_chunk
        percent = int((current_steps / total_timesteps) * 100)
        print(f"[GRAY] Progress: {percent}% complete ({current_steps}/{total_timesteps})")

        _print_eta(start_time, current_steps, total_timesteps, "GRAY")
        _print_recent_rewards(monitor_prefix, "GRAY")

    print("[GRAY] Training complete. Saving model...")
    model_path = os.path.join(log_dir, "ppo_intersection_gray")
    model.save(model_path)
    print(f"[GRAY] Model saved to: {model_path}")

    env.close()
    eval_env.close()
    print("\n========== GRAYSCALE TRAINING FINISHED ==========\n")



if __name__ == "__main__":
    print(">>> About to train LIDAR model...")
    train_intersection_lidar()
    print(">>> LIDAR training done.")

    # If you want to skip grayscale for speed, comment these lines:
    print(">>> About to train GRAYSCALE model...")
    train_intersection_gray()
    print(">>> GRAYSCALE training done.")

    print(">>> All training finished.")
