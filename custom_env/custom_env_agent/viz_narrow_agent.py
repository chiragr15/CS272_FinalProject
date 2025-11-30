"""
Visualize a trained PPO agent on narrow-street-v0.

Usage examples:

    # Visualize the 100k-step quick model
    python viz_narrow_agent.py \
        --model_path models/ppo_quick.zip \
        --vecnorm_path models/vecnorm_quick.pkl \
        --episodes 5 \
        --fps 30
"""
from __future__ import annotations

import argparse
import time
from typing import Optional

import gymnasium as gym
import numpy as np

# Registers 'narrow-street-v0'
import narrow_street  # noqa: F401

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from highway_env.envs.common.action import DiscreteMetaAction

# 🔴 IMPORTANT: reuse the SAME env factory and headway function as training
from ppo_narrow_agent import make_env as make_train_env, compute_headway as compute_headway_fn


def get_raw_env(vec_env):
    """Helper: VecNormalize -> DummyVecEnv -> underlying gym env."""
    return vec_env.venv.envs[0]


def run_viz(
    model_path: str,
    vecnorm_path: str,
    episodes: int = 5,
    deterministic: bool = True,
    fps: int = 30,
) -> None:
    # ---- Build VecEnv + load normalization and model ----
    print(f"[VIZ] Loading VecNormalize from {vecnorm_path}")

    # ⚠️ Critical: use the SAME make_env as in training (includes SafeObs wrapper)
    base_env = DummyVecEnv([lambda: make_train_env(render=True, seed=None)])
    env = VecNormalize.load(vecnorm_path, base_env)

    # Disable further stats updates & keep raw reward scale for viz
    env.training = False
    env.norm_reward = False

    print(f"[VIZ] Loading PPO model from {model_path}")
    model = PPO.load(model_path, env=env, device="auto")

    try:
        print("Action meanings:", DiscreteMetaAction.ACTIONS)
    except Exception:
        print("Action class:", DiscreteMetaAction)

    for ep in range(episodes):
        print(f"\n========== Episode {ep + 1}/{episodes} ==========")
        obs = env.reset()
        done = False
        ep_ret = 0.0
        step = 0

        raw_env = get_raw_env(env)
        cfg = raw_env.unwrapped.config
        road_len = float(cfg.get("road_length", 600.0))

        while not done:
            # Policy action
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, rewards, dones, infos = env.step(action)

            r = float(rewards[0])
            done = bool(dones[0])
            ep_ret += r
            step += 1

            # Grab debug info from underlying highway-env
            raw_env = get_raw_env(env)
            ego = raw_env.unwrapped.vehicle
            x = float(ego.position[0])
            lane_index = None
            try:
                lane_index = ego.target_lane_index[2]
            except Exception:
                pass
            lane_id = lane_index if lane_index is not None else "?"
            v = float(ego.speed)
            crashed = bool(getattr(ego, "crashed", False))

            lead, _ = ego.road.neighbour_vehicles(ego, ego.target_lane_index)
            t_head = compute_headway_fn(ego, lead)

            # Termination heuristic (matches training/eval logic)
            term = False
            try:
                term = raw_env.unwrapped._is_terminal()
            except Exception:
                term = done
            trunc = False  # max_episode_steps handled by wrapper

            if step % 10 == 0 or done:
                print(
                    f"[STEP ep={ep+1} step={step}] "
                    f"x={x:.1f} lane={lane_id} v={v:.1f} "
                    f"action={int(action)} r={r:.3f} crashed={crashed} "
                    f"t_headway={t_head:.2f} term={term} trunc={trunc}"
                )

            # Render one frame
            raw_env.render()
            time.sleep(1.0 / fps)

            if done:
                success = (not crashed) and (x >= road_len - 5.0)
                print(
                    f"[EPISODE {ep+1} DONE] "
                    f"steps={step} return={ep_ret:.2f} "
                    f"crashed={crashed} success={success} final_x={x:.1f}"
                )
                break

    env.close()
    print("\n[VIZ] Finished visualization.")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to PPO .zip model")
    parser.add_argument("--vecnorm_path", type=str, required=True, help="Path to VecNormalize .pkl file")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to visualize")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic (non-deterministic) policy")
    parser.add_argument("--fps", type=int, default=30, help="Render FPS (sleep between steps)")
    args = parser.parse_args()

    run_viz(
        model_path=args.model_path,
        vecnorm_path=args.vecnorm_path,
        episodes=args.episodes,
        deterministic=not args.stochastic,
        fps=args.fps,
    )
