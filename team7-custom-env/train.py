"""
Simplified fast PPO training for Team7-v0 (CPU-friendly).
Uses DummyVecEnv (no subprocesses) and fewer timesteps.
Goal: quick training for testing and generating learning curve + violin plots.

Run:
    pip install -e .
    pip install "stable-baselines3[extra]" gymnasium torch tensorboard highway-env
    python train_sb3_fast_cpu.py
"""
import os
import gymnasium as gym
import custom_env  # registers Team7-v0
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed

LOG_DIR = "runs/ppo_team7_fast"
os.makedirs(LOG_DIR, exist_ok=True)
SEED = 42

def make_env():
    def _init():
        env = gym.make("Team7-v0")  # no extra kwargs
        env = Monitor(env, filename=os.path.join(LOG_DIR, "monitor.csv"))
        env.reset(seed=SEED)
        return env
    return _init

def main():
    set_random_seed(SEED)

    # Single DummyVecEnv so it's fully compatible on CPU
    env = DummyVecEnv([make_env()])

    policy_kwargs = dict(net_arch=[64, 64])
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        n_steps=512,
        batch_size=512,
        n_epochs=10,
        gamma=0.98,
        gae_lambda=0.92,
        learning_rate=2.5e-4,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=LOG_DIR,
        verbose=1,
        seed=SEED,
    )

    # TOTAL_STEPS = int(1e4)   # 10,000 steps
    TOTAL_STEPS = int(5e4)  

    model.learn(total_timesteps=TOTAL_STEPS)
    model.save(os.path.join(LOG_DIR, "ppo_team7_fast_final"))
    print("Fast CPU training complete.")

if __name__ == "__main__":
    main()
