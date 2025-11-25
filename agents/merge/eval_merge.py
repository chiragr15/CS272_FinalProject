# eval_merge.py
import os, argparse
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from tqdm import trange
import highway_env


def make_env(obs_type: str, seed: int):
    if obs_type == "GrayscaleObservation":
        obs_cfg = {
            "type": "GrayscaleObservation",
            "observation_shape": (128, 64),    
            "stack_size": 4,
            "weights": [0.2989, 0.5870, 0.1140],
            "scaling": 1.75,
        }
    else:  # LidarObservation
        obs_cfg = {
            "type": "LidarObservation",
            "cells": 128,        
            # maximum_range / normalize use defaults
            "maximum_range": 100.0,
            "normalize": True,
        }

    cfg = {
        "observation": obs_cfg,
        "action": {"type": "DiscreteMetaAction"},
        "duration": 40,
        "vehicles_count": 25,
        "policy_frequency": 2,   
        "simulation_frequency": 15,
        "offscreen_rendering": False,
        "lanes_count": 2,
        "controlled_vehicles": 1,
    }

    env = gym.make("merge-v0", render_mode=None, config=cfg)
    env.reset(seed=seed)
    return env


def evaluate(model_dir: str, obs_type: str, episodes: int = 500, seed: int = 0):
    env = make_env(obs_type, seed)

    best = os.path.join(model_dir, "best_model.zip")
    final = os.path.join(model_dir, "final_model.zip")
    model_path = best if os.path.exists(best) else final
    model = PPO.load(model_path, device="cpu")

    returns = []
    for _ in trange(episodes):
        obs, info = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, terminated, truncated, info = env.step(action)
            ep_ret += float(rew)
            done = terminated or truncated
        returns.append(ep_ret)

    env.close()
    return np.array(returns)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--obs", choices=["LidarObservation", "GrayscaleObservation"], required=True)
    args = p.parse_args()

    run_dir = f"runs/Merge_{args.obs}_PPO"
    R = evaluate(run_dir, args.obs, episodes=500)

    # Save with correct IDs
    out = os.path.join(
        run_dir,
        "ID6_Merge_LidarObs_returns.npy" if args.obs == "LidarObservation"
        else "ID8_Merge_GrayscaleObs_returns.npy"
    )
    np.save(out, R)
    print("Saved returns:", out)
