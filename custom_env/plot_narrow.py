# plot_narrow.py
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt

LOGDIR = "runs/NarrowStreet_Kin_PPO"
monitor_csv = f"{LOGDIR}/monitor.csv"
returns_npy = f"{LOGDIR}/ID14_NarrowStreet_returns.npy"

# --- Learning curve ---
df = pd.read_csv(monitor_csv, comment="#")
df["ep"] = np.arange(1, len(df)+1)
win = max(10, len(df)//50)
df["r_smooth"] = df["r"].rolling(window=win, min_periods=1).mean()

plt.figure(figsize=(7,4))
plt.plot(df["ep"], df["r_smooth"])
plt.xlabel("Episodes"); plt.ylabel("Mean Episodic Return")
plt.title("ID13 – NarrowStreet – Learning Curve")
plt.tight_layout()
plt.savefig(f"{LOGDIR}/ID13_NarrowStreet_LearningCurve.png", dpi=200)
plt.close()

# --- Violin plot ---
R = np.load(returns_npy)
plt.figure(figsize=(4,4))
plt.violinplot(R, showmeans=True)
plt.ylabel("Episodic Return")
plt.title("ID14 – NarrowStreet – 500-Episode Performance")
plt.tight_layout()
plt.savefig(f"{LOGDIR}/ID14_NarrowStreet_PerformanceViolin.png", dpi=200)
plt.close()

print("Saved ID13 and ID14 plots in", LOGDIR)
