# train_narrow.py
import os
import gymnasium as gym
import narrow_street  # registers 'narrow-street-v0'
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

def make_env():
    def _f():
        # override the observation to be safe & fast
        return gym.make("narrow-street-v0", config={
            "observation": {
                "type": "Kinematics",
                "features": ["x","y","vx","vy","cos_h","sin_h"],
                "normalize": True,
                "vehicles_count": 8
            }
        })
    return _f

logdir = "runs/NarrowStreet_Kin_PPO"
os.makedirs(logdir, exist_ok=True)

env = DummyVecEnv([make_env()])
env = VecMonitor(env, filename=os.path.join(logdir, "monitor.csv"))

model = PPO("MlpPolicy", env, learning_rate=1e-4, n_steps=2048,
            batch_size=256, gamma=0.99, gae_lambda=0.95,
            clip_range=0.2, ent_coef=0.0, vf_coef=0.5, n_epochs=10, verbose=1)

model.learn(total_timesteps=1_000_000)  # quick test; later 1M+
model.save(os.path.join(logdir, "final_model.zip"))
env.close()
