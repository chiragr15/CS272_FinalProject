# train_merge.py
import os, argparse
import gymnasium as gym
import numpy as np
import highway_env

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure

def make_env(obs_type: str, seed: int, rank: int):
    def _f():
        # Merge env id is 'merge-v0'
        cfg = {
            "observation": {"type": "LidarObservation"} if obs_type=="LidarObservation" else
                           {"type": "GrayscaleObservation", "weights": [0.2989, 0.5870, 0.1140],  "stack_size": 4, "observation_shape": (84,84)},
            "action": {"type": "DiscreteMetaAction"},
            "duration": 40,           # shorter episodes help throughput
            "vehicles_count": 25,
            "policy_frequency": 10,
            "offscreen_rendering": False,
            "lanes_count": 2,         # Merge defaults are fine; tweak if needed
            "controlled_vehicles": 1,
        }
        env = gym.make("merge-v0", render_mode=None, config=cfg)
        env.reset(seed=seed + rank)
        return env
    return _f

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--obs", choices=["LidarObservation","GrayscaleObservation"], required=True)
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--n_envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    run_dir = f"runs/Merge_{args.obs}_PPO"
    os.makedirs(run_dir, exist_ok=True)

    env_fns = [make_env(args.obs, args.seed, i) for i in range(args.n_envs)]
    vec = SubprocVecEnv(env_fns)
    vec = VecMonitor(vec, filename=os.path.join(run_dir, "monitor.csv"))

    policy = "MlpPolicy" if args.obs=="LidarObservation" else "CnnPolicy"
    model = PPO(
        policy, vec, verbose=1, seed=args.seed,
        n_steps=2048, batch_size=256, learning_rate=3e-4,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.0, vf_coef=0.5, n_epochs=10
    )

    # Single-env for evaluation during training
    eval_env = make_env(args.obs, args.seed, 10_000)()
    eval_cb = EvalCallback(
        eval_env, best_model_save_path=run_dir, log_path=run_dir,
        eval_freq=25_000, n_eval_episodes=20, deterministic=True, render=False
    )

    logger = configure(run_dir, ["stdout", "tensorboard"])
    model.set_logger(logger)

    model.learn(total_timesteps=args.steps, callback=eval_cb)
    model.save(os.path.join(run_dir, "final_model.zip"))

    vec.close(); eval_env.close()

if __name__ == "__main__":
    main()
