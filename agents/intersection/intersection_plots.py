import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

LOG_ROOT = "logs_intersection"


def load_monitor_rewards(monitor_prefix: str):
    files = glob.glob(monitor_prefix + "*.monitor.csv")
    if not files:
        raise FileNotFoundError(f"No monitor files for {monitor_prefix}")

    rewards = []
    episodes = []

    ep_counter = 0
    for f in files:
        data = pd.read_csv(f, comment="#")
        r = data["r"].values
        rewards.extend(r.tolist())
        episodes.extend(range(ep_counter, ep_counter + len(r)))
        ep_counter += len(r)

    return np.array(episodes), np.array(rewards)


def plot_learning_curve(
    episodes: np.ndarray,
    rewards: np.ndarray,
    window: int = 500,   # big window => very smooth
    title: str = "",
    save_path: str | None = None,
):
    """
    Plot a heavily smoothed learning curve.

    - rewards: 1D array of episodic returns
    - window: number of episodes to average over
    """

    def moving_average(x, w):
        if len(x) < w:
            return x
        return np.convolve(x, np.ones(w) / w, mode="valid")

    # Smooth the rewards
    smoothed = moving_average(rewards, window)

    # Make a matching x-axis for the smoothed data
    x = np.arange(len(smoothed))

    plt.figure(figsize=(10, 5))
    plt.plot(x, smoothed, linewidth=2.0)

    plt.xlabel("Episodes", fontsize=12)
    plt.ylabel(f"Mean Episodic Return (window={window})", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)

    if save_path:
        print(f"[PLOT] Saving learning curve to: {save_path}")
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()




def make_all_learning_curves():
    # Lidar
    lidar_prefix = os.path.join(LOG_ROOT, "lidar", "monitor_lidar")
    ep_lidar, rew_lidar = load_monitor_rewards(lidar_prefix)
    plot_learning_curve(
        ep_lidar,
        rew_lidar,
        window=500,
        title="Intersection-v1 LidarObservation – Learning Curve",
        save_path="intersection_lidar_learning_curve.png",
    )

    # Grayscale
    gray_prefix = os.path.join(LOG_ROOT, "gray", "monitor_gray")
    ep_gray, rew_gray = load_monitor_rewards(gray_prefix)
    plot_learning_curve(
        ep_gray,
        rew_gray,
        window=500,
        title="Intersection-v1 GrayscaleObservation – Learning Curve",
        save_path="intersection_gray_learning_curve.png",
    )


if __name__ == "__main__":
    make_all_learning_curves()
