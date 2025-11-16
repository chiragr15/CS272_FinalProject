# eval_narrow.py
import gymnasium as gym
import narrow_street          # registers 'narrow-street-v0'
import numpy as np
from stable_baselines3 import PPO
from tqdm import trange

LOGDIR = "runs/NarrowStreet_Kin_PPO"
MODEL_PATH = f"{LOGDIR}/final_model.zip"

eval_cfg = {
    "observation": {
        "type": "Kinematics",
        "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],  # 6 features
        "normalize": True,
        "vehicles_count": 8,
    }
}

env = gym.make("narrow-street-v0", config=eval_cfg)
obs, info = env.reset()  # no fixed seed for varied episodes
print("Sanity check obs shape:", np.array(obs).shape)  # should be (8, 6)

model = PPO.load(MODEL_PATH, device="cpu")

returns = []
for _ in trange(500):
    obs, info = env.reset()
    done = False
    ep_ret = 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(action)
        ep_ret += float(r)
        done = term or trunc
    returns.append(ep_ret)

env.close()
R = np.array(returns)
np.save(f"{LOGDIR}/ID14_NarrowStreet_returns.npy", R)
print(f"Mean return: {R.mean():.2f} ± {R.std():.2f}")
print(f"Saved to {LOGDIR}/ID14_NarrowStreet_returns.npy")
