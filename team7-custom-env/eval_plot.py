import os, numpy as np, pandas as pd, gymnasium as gym
import custom_env
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

LOG_DIR = "runs/ppo_team7_fast"  
MODEL = "ppo_team7_fast_final.zip"   
MODEL_PATH = os.path.join(LOG_DIR, MODEL)

assert os.path.exists(MODEL_PATH), f"Model not found: {MODEL_PATH}"

# 1) Run deterministic evaluation
EPISODES = 500    
env = gym.make("Team7-v0", render_mode=None)
model = PPO.load(MODEL_PATH)

returns = []
for _ in range(EPISODES):
    obs, info = env.reset()
    done = truncated = False
    G = 0.0
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, truncated, info = env.step(action)
        G += r
    returns.append(G)
env.close()

# 2) Save + print summary
os.makedirs(LOG_DIR, exist_ok=True)
out_csv = os.path.join(LOG_DIR, "eval_returns.csv")
pd.DataFrame({"episode": np.arange(1, EPISODES+1), "return": returns}).to_csv(out_csv, index=False)

print(f"Saved deterministic returns → {out_csv}")
print(f"Episodes: {len(returns)}")
print(f"Mean return: {np.mean(returns):.3f}")
print(f"Std  return: {np.std(returns):.3f}")
print(f"Max  return: {np.max(returns):.3f}")
print(f"Min  return: {np.min(returns):.3f}")

# 3) Violin plot
plt.figure(figsize=(6,5))
plt.violinplot(returns, showmeans=True, showextrema=True, showmedians=True)
plt.title(f"Performance (Deterministic) — {len(returns)} Episodes")
plt.ylabel("Episode return")
plt.xticks([1], ["Team7-v0"])
plt.tight_layout()
plt.savefig(os.path.join(LOG_DIR, "eval_violin.png"), dpi=150)
plt.close()
print(f"Saved violin plot → {os.path.join(LOG_DIR, 'eval_violin.png')}")

MONITOR_CSV = os.path.join(LOG_DIR, "monitor.csv")

# Read the monitor file 
df = pd.read_csv(MONITOR_CSV, comment="#")
df["episode"] = np.arange(1, len(df)+1)

# Simple moving average to smooth the jagged line
window = max(1, len(df)//50)
df["SMA"] = df["r"].rolling(window=window, min_periods=1).mean()

plt.figure(figsize=(8,5))
plt.plot(df["episode"], df["r"], label="Episode return", alpha=0.6)
plt.plot(df["episode"], df["SMA"], label=f"Moving avg (window={window})", linewidth=2)
plt.xlabel("Training episodes")
plt.ylabel("Return")
plt.title("Learning Curve: Mean episodic training reward vs. episodes")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(LOG_DIR, "learning_curve.png"), dpi=150)
plt.close()

print(f"Saved → {os.path.join(LOG_DIR, 'learning_curve.png')}")