# intersection_eval_violin.py
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # Fix Intel OpenMP issue on Windows
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import highway_env

from stable_baselines3 import PPO

from intersection_envs import make_intersection_env

LOG_ROOT = "logs_intersection"


def evaluate_model(model_path: str, obs_type: str, n_episodes: int = 500):
    # Load model without binding it to an env (we only need predict)
    model = PPO.load(model_path)

    # Create a fresh env for evaluation
    env = make_intersection_env(obs_type)()

    episode_returns = []

    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_ret = 0.0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_ret += float(reward)

        episode_returns.append(ep_ret)

    env.close()
    return np.array(episode_returns)


def plot_violin(data_list, labels, title, save_path):
    plt.figure()
    plt.violinplot(data_list, showmeans=True, showmedians=True)
    plt.xticks(range(1, len(labels) + 1), labels)
    plt.ylabel("Episodic Return")
    plt.title(title)
    plt.grid(True, axis="y")
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()


if __name__ == "__main__":
    # Lidar
    lidar_model = os.path.join(LOG_ROOT, "lidar", "ppo_intersection_lidar.zip")
    lidar_returns = evaluate_model(lidar_model, "LidarObservation")
    np.save("intersection_lidar_returns.npy", lidar_returns)
    plot_violin(
        [lidar_returns],
        ["Lidar"],
        "Intersection-v0 – LidarObservation (500 episodes)",
        "intersection_lidar_violin.png",
    )

    # Grayscale
    gray_model = os.path.join(LOG_ROOT, "gray", "ppo_intersection_gray.zip")
    gray_returns = evaluate_model(gray_model, "GrayscaleObservation")
    np.save("intersection_gray_returns.npy", gray_returns)
    plot_violin(
        [gray_returns],
        ["Grayscale"],
        "Intersection-v0 – GrayscaleObservation (500 episodes)",
        "intersection_gray_violin.png",
    )
