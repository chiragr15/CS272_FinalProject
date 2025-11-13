# plot_merge.py
import os, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import highway_env

def plot_learning_curve(monitor_csv: str, out_png: str, title: str):
    df = pd.read_csv(monitor_csv, comment="#")
    
    df["ep"] = np.arange(1, len(df)+1)
    # smooth with rolling mean (window ~ max(10, 2% of episodes))
    win = max(10, len(df)//50)
    df["r_smooth"] = df["r"].rolling(window=win, min_periods=1).mean()

    plt.figure(figsize=(7,4))
    plt.plot(df["ep"], df["r_smooth"])
    plt.xlabel("Episodes")
    plt.ylabel("Mean Episodic Return (smoothed)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200); plt.close()

def plot_violin(returns_npy: str, out_png: str, title: str):
    R = np.load(returns_npy)
    plt.figure(figsize=(4,4))
    plt.violinplot(R, showmeans=True)
    plt.ylabel("Episodic Return")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200); plt.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--obs", choices=["LidarObservation","GrayscaleObservation"], required=True)
    args = p.parse_args()

    run_dir = f"runs/Merge_{args.obs}_PPO"
    mon_csv = os.path.join(run_dir, "monitor.csv")

    if args.obs == "LidarObservation":
        out_curve = os.path.join(run_dir, "ID5_Merge_LidarObs_LearningCurve.png")
        out_violin = os.path.join(run_dir, "ID6_Merge_LidarObs_PerformanceViolin.png")
        ret_npy = os.path.join(run_dir, "ID6_Merge_LidarObs_returns.npy")
        title_c = "ID5 – Merge – LidarObservation – Learning Curve"
        title_v = "ID6 – Merge – LidarObservation – 500-Episode Performance"
    else:
        out_curve = os.path.join(run_dir, "ID7_Merge_GrayscaleObs_LearningCurve.png")
        out_violin = os.path.join(run_dir, "ID8_Merge_GrayscaleObs_PerformanceViolin.png")
        ret_npy = os.path.join(run_dir, "ID8_Merge_GrayscaleObs_returns.npy")
        title_c = "ID7 – Merge – GrayscaleObservation – Learning Curve"
        title_v = "ID8 – Merge – GrayscaleObservation – 500-Episode Performance"

    plot_learning_curve(mon_csv, out_curve, title_c)
    plot_violin(ret_npy, out_violin, title_v)
    print("Wrote:", out_curve, "and", out_violin)
