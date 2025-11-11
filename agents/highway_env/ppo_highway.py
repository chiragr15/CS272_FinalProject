# single_agent_highway_multiobs.py
import os, time, numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import highway_env  # noqa: F401

from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback

# ---------- Shared config ----------
IMG_SHAPE = (84, 84, 4)   # H, W, stacked frames (channels-last for SB3 CNN)
LIDAR_LEN = 64            # unified lidar vector length
SEED = 42

def timestamp(): 
    return time.strftime("%Y%m%d-%H%M%S")

# ---------- Dict wrappers ----------
class GrayAsDictWrapper(gym.ObservationWrapper):
    """
    Accepts grayscale stacks in either CHW (4,84,84) or HWC (84,84,4) and
    always returns HWC for MultiInputPolicy.
    """
    def __init__(self, env):
        super().__init__(env)
        self.image_space = spaces.Box(low=0.0, high=1.0, shape=IMG_SHAPE, dtype=np.float32)  # (84,84,4)
        self.lidar_space = spaces.Box(low=0.0, high=1.0, shape=(LIDAR_LEN,), dtype=np.float32)
        self.observation_space = spaces.Dict({"image": self.image_space, "lidar": self.lidar_space})

    def observation(self, obs):
        arr = np.asarray(obs, dtype=np.float32)
        # If frames are channels-first (C,H,W), transpose to (H,W,C)
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            # e.g., (4,84,84) -> (84,84,4)
            arr = np.transpose(arr, (1, 2, 0))
        # If already channels-last (H,W,C), leave as-is
        image = arr
        # Safety: ensure final shape is exactly IMG_SHAPE
        if image.shape != IMG_SHAPE:
            # try a final-resize-free fallback by reordering if possible
            if image.ndim == 3 and image.shape[::-1] == IMG_SHAPE:
                image = np.transpose(image, (2, 1, 0))
            else:
                raise ValueError(f"Unexpected grayscale shape {image.shape}, expected {IMG_SHAPE} or (4,84,84)")
        lidar = np.zeros((LIDAR_LEN,), dtype=np.float32)
        return {"image": image, "lidar": lidar}



class LidarAsDictWrapper(gym.ObservationWrapper):
    """
    LidarObservation(cells=LIDAR_LEN, normalize=True)
    Returns Dict: {"image": dummy_zeros, "lidar": vector}
    Pads or trims lidar automatically to LIDAR_LEN.
    """
    def __init__(self, env):
        super().__init__(env)
        self.image_space = spaces.Box(low=0.0, high=1.0, shape=IMG_SHAPE, dtype=np.float32)
        self.lidar_space = spaces.Box(low=0.0, high=1.0, shape=(LIDAR_LEN,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "image": self.image_space,
            "lidar": self.lidar_space
        })

    def observation(self, obs):
        vec = np.asarray(obs, dtype=np.float32).ravel()
        if vec.size < LIDAR_LEN:
            vec = np.pad(vec, (0, LIDAR_LEN - vec.size))
        elif vec.size > LIDAR_LEN:
            vec = vec[:LIDAR_LEN]
        image = np.zeros(IMG_SHAPE, dtype=np.float32)
        return {"image": image, "lidar": vec}


# ---------- Env factories ----------
def make_gray_env(seed=SEED):
    gray_cfg = {
        "observation": {
            "type": "GrayscaleObservation",
            "weights": [0.2989, 0.5870, 0.1140],
            "stack_size": 4,
            "observation_shape": (84, 84)
        },
        "policy_frequency": 15,
        "lanes_count": 4
    }

    def _thunk():
        env = gym.make("highway-v0", render_mode=None, config=gray_cfg)
        env = GrayAsDictWrapper(env)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _thunk


def make_lidar_env(seed=SEED):
    lidar_cfg = {
        "observation": {
            "type": "LidarObservation",
            "cells": LIDAR_LEN,
            "maximum_range": 50.0,
            "normalize": True
        },
        "policy_frequency": 15,
        "lanes_count": 4
    }

    def _thunk():
        env = gym.make("highway-v0", render_mode=None, config=lidar_cfg)
        env = LidarAsDictWrapper(env)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _thunk


# ---------- Logging callback ----------
class EpReturnCB(BaseCallback):
    """Collect episode returns during training."""
    def __init__(self):
        super().__init__()
        self.rets = []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if isinstance(info, dict) and "episode" in info:
                self.rets.append(info["episode"]["r"])
        return True


# ---------- Plot helpers ----------
def moving_average(x, w=50):
    if len(x) == 0:
        return np.array([])
    w = min(w, len(x))
    return np.convolve(x, np.ones(w)/w, mode="valid")


def plot_curve(ep_returns, label, path):
    xs = np.arange(1, len(ep_returns)+1)
    plt.figure(figsize=(8,5))
    plt.plot(xs, ep_returns, alpha=.25, label=f"{label} (per-ep)")
    ma = moving_average(ep_returns, 50)
    if len(ma):
        plt.plot(np.arange(50, 50+len(ma)), ma, lw=2, label=f"{label} (MA50)")
    plt.xlabel("Training Episodes")
    plt.ylabel("Return")
    plt.title(f"Learning Curve — {label}")
    plt.grid(alpha=.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_violin(name2rets, path):
    names = list(name2rets.keys())
    data = [name2rets[k] for k in names]
    plt.figure(figsize=(8,5))
    plt.violinplot(data, showmeans=True, showmedians=True, showextrema=True)
    plt.xticks(np.arange(1, len(names)+1), names)
    plt.ylabel("Return (eval episodes)")
    plt.title("Performance Distribution — Single Agent on Highway")
    plt.grid(axis='y', alpha=.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ---------- Train & Evaluate ----------
if __name__ == "__main__":
    out_dir = os.path.join("runs", f"highway_single_agent_{timestamp()}")
    os.makedirs(out_dir, exist_ok=True)

    # Quick test settings (adjust later)
    TOTAL_STEPS = 60_000          # small run (≈10 min on CPU)
    EVAL_EPISODES = 50            # quick evaluation
    n_gray, n_lidar = 2, 2        # fewer envs for quick test

    env_fns = [make_gray_env(SEED+i) for i in range(n_gray)] + \
              [make_lidar_env(SEED+100+i) for i in range(n_lidar)]

    # Use DummyVecEnv for macOS stability (switch to SubprocVecEnv later)
    vec_env = DummyVecEnv(env_fns)

    model = PPO(
        policy="MultiInputPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        seed=SEED,
        verbose=1
    )

    cb = EpReturnCB()
    model.learn(total_timesteps=TOTAL_STEPS, callback=cb)
    model.save(os.path.join(out_dir, "ppo_highway_multiobs.zip"))
    vec_env.close()

    plot_curve(cb.rets, "PPO-MultiObs", os.path.join(out_dir, "learning_curve_multiobs.png"))

    # ---- Evaluation helper ----
    def evaluate(make_env_thunk, n_episodes=EVAL_EPISODES):
        env = make_env_thunk()
        returns = []
        for i in range(n_episodes):
            obs, _ = env.reset(seed=SEED+i)
            done = truncated = False
            total_r = 0.0
            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, r, done, truncated, _ = env.step(action)
                total_r += r
            returns.append(total_r)
        env.close()
        return returns

    # Evaluate on each modality
    gray_eval = evaluate(make_gray_env(SEED+999))
    lidar_eval = evaluate(make_lidar_env(SEED+1999))

    print(f"Grayscale eval: mean={np.mean(gray_eval):.2f}, std={np.std(gray_eval):.2f}")
    print(f"Lidar     eval: mean={np.mean(lidar_eval):.2f}, std={np.std(lidar_eval):.2f}")

    plot_violin(
        {"Grayscale": gray_eval, "Lidar": lidar_eval},
        os.path.join(out_dir, "violin_grayscale_vs_lidar_single_agent.png")
    )

    print("Saved results to:", out_dir)
