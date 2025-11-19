"""
Train & Evaluate PPO on narrow-safe-v0 (safer variant of narrow-street-v0)

This script:
1) Runs a short PPO training and prints eval metrics (success rate, return, collisions, headway violations).
2) Optionally launches a longer PPO training with stronger hyperparameters.
3) Optionally runs a finetune config (continue from saved model with smaller LR).
4) For long/finetune runs, it also:
   - saves a training learning curve plot
   - saves a violin plot of 500-episode deterministic evaluation returns.

Requirements:
    pip install stable-baselines3 gymnasium numpy torch matplotlib

Usage:
    # quick sanity run (single seed)
    python ppo_narrow_agent.py --quick --quick_steps 100000

    # longer training (single seed)
    python ppo_narrow_agent.py --long --long_steps 500000

    # long training with explicit seed/run name
    python ppo_narrow_agent.py --long --long_steps 500000 --seed 0 --run_name seed0

    # finetune training (from a previous model)
    python ppo_narrow_agent.py --finetune --load models/seed0_ppo_final.zip --long_steps 300000 --seed 0 --run_name seed0_ft
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt

# Registers 'narrow-safe-v0'
import narrow_safe_env  # noqa: F401

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
    BaseCallback,
)


# =========================
# Observation sanitizer
# =========================
class SafeObs(gym.ObservationWrapper):
    """Replace NaNs/Infs and clip extreme values (defensive for custom envs)."""

    def __init__(self, env, clip_value: float = 1e6):
        super().__init__(env)
        self.clip_value = float(clip_value)

        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(
            low=-clip_value,
            high=clip_value,
            shape=old_space.shape,
            dtype=np.float32,
        )

    def observation(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        obs = np.nan_to_num(
            obs,
            nan=0.0,
            posinf=self.clip_value,
            neginf=-self.clip_value,
        )
        return np.clip(obs, -self.clip_value, self.clip_value)


# =========================
# Env factory & utilities
# =========================
def make_env(
    render: bool = False,
    seed: Optional[int] = 42,
    overrides: Dict[str, Any] | None = None,
):
    """Creates and seeds the env, then wraps with SafeObs."""
    kwargs: Dict[str, Any] = {}
    if overrides:
        kwargs.update(overrides)
    render_mode = "human" if render else None
    # use our safer env
    env = gym.make("narrow-safe-v0", render_mode=render_mode, **kwargs)
    env = SafeObs(env)  # sanitize observations
    env.reset(seed=seed)
    return env


def compute_headway(ego, lead) -> float:
    """Time headway: distance / ego_speed. Inf if no lead or ego v ~ 0."""
    if lead is None:
        return np.inf
    v = max(1e-3, float(getattr(ego, "speed", 0.0)))
    try:
        gap = max(0.0, float(lead.distance_to(ego)))
    except Exception:
        gap = float(
            np.linalg.norm(
                np.array(getattr(lead, "position", [0, 0]), dtype=np.float32)
                - np.array(ego.position, dtype=np.float32)
            )
        )
    return gap / v if v > 0 else np.inf


@dataclass
class EvalStats:
    episodes: int
    success_rate: float
    avg_return: float
    avg_steps: float
    collisions: int
    headway_violations: float  # average count per episode


# =========================
# Training return callback
# =========================
class EpisodeReturnCallback(BaseCallback):
    """
    Logs episodic training returns (sum of rewards per episode).
    Assumes a single environment (DummyVecEnv with n_envs=1).
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_returns: List[float] = []
        self._current_return: float = 0.0

    def _on_step(self) -> bool:
        # self.locals["rewards"] and ["dones"] are arrays of shape (n_envs,)
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")
        if rewards is None or dones is None:
            return True

        r = float(rewards[0])
        done = bool(dones[0])

        self._current_return += r
        if done:
            self.episode_returns.append(self._current_return)
            self._current_return = 0.0
        return True


# =========================
# Evaluation (step-accurate, with VecNormalize)
# =========================
def evaluate(model_path: str, vecnorm_path: str, n_episodes: int = 20) -> EvalStats:
    """
    Load model + VecNormalize stats and evaluate on a single raw env.

    - Uses VecNormalize only to normalize observations (no stepping through it).
    - Steps a raw SafeObs(narrow-safe-v0) env directly -> no auto-resets.
    - Prints per-step debug for the first few episodes and per-episode summaries.
    """
    print(f"\n[DEBUG] Evaluating model={model_path}, vecnorm={vecnorm_path}, episodes={n_episodes}")

    # Build a dummy VecEnv just to load VecNormalize stats (for obs normalization)
    dummy_env = DummyVecEnv([lambda: make_env(render=False)])
    vecnorm = VecNormalize.load(vecnorm_path, dummy_env)
    vecnorm.training = False
    vecnorm.norm_reward = False

    # Load model (env only needed for compatibility)
    model = PPO.load(model_path, env=vecnorm, device="auto")

    # Raw env for actual stepping (NO VecEnv / NO VecNormalize auto-reset)
    raw_env = make_env(render=False, seed=123)
    road_len = float(raw_env.unwrapped.config.get("road_length", 600.0))

    returns, steps = [], []
    successes, collisions = 0, 0
    violations_per_ep = []

    verbose_eps = min(3, n_episodes)   # first few episodes with detailed prints
    max_steps_per_ep = 1000            # safety cap

    for ep in range(n_episodes):
        obs, _ = raw_env.reset(seed=123 + ep)
        ep_ret, ep_steps, vios = 0.0, 0, 0

        if ep < verbose_eps:
            print(f"\n[DEBUG] --- Eval episode {ep + 1} ---")

        while True:
            # Normalize obs via VecNormalize stats (shape: (1, obs_dim))
            obs_vec = np.array(obs, dtype=np.float32)[None, ...]
            obs_norm = vecnorm.normalize_obs(obs_vec.copy())
            action, _ = model.predict(obs_norm, deterministic=True)

            # Step raw env
            obs, r, term, trunc, info = raw_env.step(action)
            ep_ret += float(r)
            ep_steps += 1

            ego = raw_env.unwrapped.vehicle
            lead, _ = ego.road.neighbour_vehicles(ego, ego.target_lane_index)
            t_headway = compute_headway(ego, lead)
            if t_headway < 1.0:
                vios += 1

            x = float(ego.position[0])
            lane_idx = getattr(ego, "target_lane_index", (None, None, 0))[2]
            crashed = bool(getattr(ego, "crashed", False))

            if ep < verbose_eps and ep_steps % 10 == 0:
                print(
                    f"[STEP ep={ep + 1} step={ep_steps}] "
                    f"x={x:.1f} lane={lane_idx} v={ego.speed:.1f} "
                    f"action={int(action[0])} r={r:.3f} crashed={crashed} "
                    f"t_headway={t_headway:.2f} term={term} trunc={trunc}"
                )

            done = term or trunc or (ep_steps >= max_steps_per_ep)
            if done:
                break

        ego = raw_env.unwrapped.vehicle
        crashed = bool(getattr(ego, "crashed", False))
        if crashed:
            collisions += 1

        x = float(ego.position[0])
        success = (not crashed) and (x >= road_len - 5.0)
        successes += int(success)

        returns.append(ep_ret)
        steps.append(ep_steps)
        violations_per_ep.append(vios)

        term_reason = "success" if success else ("crash" if crashed else "timeout/other")
        if ep < verbose_eps:
            print(
                f"[Eval ep {ep + 1} DONE] steps={ep_steps} "
                f"return={ep_ret:.2f} crashed={crashed} success={success} "
                f"final_x={x:.1f} headway_violations={vios} reason={term_reason}"
            )

    raw_env.close()

    stats = EvalStats(
        episodes=n_episodes,
        success_rate=successes / n_episodes,
        avg_return=float(np.mean(returns)),
        avg_steps=float(np.mean(steps)),
        collisions=collisions,
        headway_violations=float(np.mean(violations_per_ep)),
    )

    print("\n[Eval Summary]")
    print("Episodes\tSuccess\tAvgReturn\tAvgSteps\tCollisions\tHeadway<1s")
    print(
        f"{stats.episodes}\t\t{stats.success_rate:.2f}\t{stats.avg_return:.2f}\t"
        f"{stats.avg_steps:.0f}\t\t{stats.collisions}\t\t{stats.headway_violations:.1f}"
    )

    return stats


def eval_returns_for_violin(
    model_path: str,
    vecnorm_path: str,
    n_episodes: int = 500,
) -> List[float]:
    """
    Evaluate a trained model deterministically for many episodes and return
    the list of episodic returns (for violin plotting).
    """
    print(f"\n[DEBUG] Collecting returns for violin plot: model={model_path}, episodes={n_episodes}")

    dummy_env = DummyVecEnv([lambda: make_env(render=False)])
    vecnorm = VecNormalize.load(vecnorm_path, dummy_env)
    vecnorm.training = False
    vecnorm.norm_reward = False

    model = PPO.load(model_path, env=vecnorm, device="auto")
    raw_env = make_env(render=False, seed=999)

    returns: List[float] = []
    max_steps_per_ep = 1000

    for ep in range(n_episodes):
        obs, _ = raw_env.reset(seed=999 + ep)
        ep_ret, ep_steps = 0.0, 0

        while True:
            obs_vec = np.array(obs, dtype=np.float32)[None, ...]
            obs_norm = vecnorm.normalize_obs(obs_vec.copy())
            action, _ = model.predict(obs_norm, deterministic=True)

            obs, r, term, trunc, info = raw_env.step(action)
            ep_ret += float(r)
            ep_steps += 1

            done = term or trunc or (ep_steps >= max_steps_per_ep)
            if done:
                break

        returns.append(ep_ret)

    raw_env.close()
    print(f"[DEBUG] Collected {len(returns)} returns for violin plot.")
    return returns


# =========================
# PPO hyperparameters
# =========================
def ppo_defaults() -> Dict[str, Any]:
    """Good quick-run defaults tuned for this env."""
    return dict(
        gamma=0.995,
        n_steps=1024,           # shorter rollouts -> more frequent updates
        batch_size=256,
        n_epochs=15,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,          # slightly more exploration
        learning_rate=3e-4,
        verbose=1,
        max_grad_norm=0.5,
        vf_coef=0.5,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=torch.nn.Tanh,
            ortho_init=True,
        ),
    )


def ppo_long() -> Dict[str, Any]:
    """Stronger config for longer runs."""
    cfg = ppo_defaults().copy()
    cfg.update(
        dict(
            n_steps=2048,
            batch_size=512,
            n_epochs=20,
            ent_coef=0.01,      # reduce exploration a bit
            learning_rate=1e-4, # more conservative updates
        )
    )
    return cfg


def ppo_finetune_cfg() -> Dict[str, Any]:
    """Finetune config: smaller LR, tighter clipping (used when training from scratch)."""
    cfg = ppo_long().copy()
    cfg.update(
        dict(
            learning_rate=8e-5,
            n_epochs=20,
            clip_range=0.15,
            ent_coef=0.01,
        )
    )
    return cfg


# =========================
# Training helpers & plotting
# =========================
def debug_print_env_summary(seed: int = 42) -> None:
    """Print one-time env/space summary for sanity."""
    env = make_env(render=False, seed=seed)
    print("\n[DEBUG] Env summary (narrow-safe-v0)")
    print("Observation space:", env.observation_space)
    print("Action space:     ", env.action_space)
    print(
        "Config snippet: road_length={}, parked_count={}, left_lane_traffic={}".format(
            env.unwrapped.config.get("road_length"),
            env.unwrapped.config.get("parked_count"),
            env.unwrapped.config.get("left_lane_traffic"),
        )
    )
    env.close()


def debug_print_ppo_cfg(name: str, cfg: Dict[str, Any]) -> None:
    print(f"\n[DEBUG] PPO config '{name}':")
    for k, v in cfg.items():
        if k == "policy_kwargs":
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk}: {vv}")
        else:
            print(f"  {k}: {v}")


def plot_training_curve(episode_returns: List[float], save_path: str) -> None:
    """Plot episodic training returns vs episode index."""
    if not episode_returns:
        print("[WARN] No training returns collected; skipping training curve plot.")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure()
    plt.plot(episode_returns)
    plt.xlabel("Training episodes")
    plt.ylabel("Episodic return")
    plt.title("PPO Training Curve - narrow-safe-v0")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Saved training learning curve to {save_path}")


def plot_violin(returns: List[float], save_path: str) -> None:
    """Plot violin of episodic returns."""
    if not returns:
        print("[WARN] No eval returns provided; skipping violin plot.")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure()
    # single violin
    plt.violinplot([returns], showmeans=True, showmedians=False)
    plt.xticks([1], ["PPO Agent"])
    plt.ylabel("Episodic return")
    plt.title("PPO Performance (500 deterministic episodes) - narrow-safe-v0")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Saved evaluation violin plot to {save_path}")


# =========================
# Training runners
# =========================
def train_quick(
    total_timesteps: int = 50_000,
    seed: int = 42,
    run_name: str = "run",
) -> PPO:
    os.makedirs("models", exist_ok=True)

    debug_print_env_summary(seed=seed)

    # Build training VecEnv + normalization
    env = DummyVecEnv([lambda seed=seed: make_env(render=False, seed=seed)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    cfg = ppo_defaults()
    debug_print_ppo_cfg("quick", cfg)

    model = PPO("MlpPolicy", env, seed=seed, **cfg)
    model.learn(total_timesteps=total_timesteps)

    model_path = f"models/{run_name}_ppo_quick.zip"
    vecnorm_path = f"models/{run_name}_vecnorm_quick.pkl"
    model.save(model_path)
    env.save(vecnorm_path)

    print(f"\n[INFO] Saved quick PPO model to {model_path}")
    print(f"[INFO] Saved VecNormalize stats to {vecnorm_path}")

    stats = evaluate(model_path, vecnorm_path, n_episodes=10)
    print("\n=== PPO Quick ({:,} steps) ===".format(total_timesteps))
    print("Success\tAvgReturn\tAvgSteps\tCollisions\tHeadway<1s")
    print(
        f"{stats.success_rate:.2f}\t{stats.avg_return:.2f}\t"
        f"{stats.avg_steps:.0f}\t{stats.collisions}\t{stats.headway_violations:.1f}"
    )
    return model


def train_long(
    total_timesteps: int = 500_000,
    finetune: bool = False,
    load_path: Optional[str] = None,
    seed: int = 42,
    run_name: str = "run",
) -> PPO:
    """
    Long training:
      - finetune=False: uses ppo_long() (stronger base training) from scratch.
      - finetune=True:
          * if load_path is provided, continues from it with smaller LR
          * otherwise, starts from scratch with ppo_finetune_cfg().
    Also:
      - uses EvalCallback to save the best model
      - logs training episode returns and plots a learning curve
      - evaluates best/final model for 500 episodes and plots a violin.
    """
    os.makedirs("models", exist_ok=True)

    debug_print_env_summary(seed=seed)

    # Build training VecEnv + normalization
    env = DummyVecEnv([lambda seed=seed: make_env(render=False, seed=seed)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    if finetune:
        cfg = ppo_finetune_cfg()
        debug_print_ppo_cfg("finetune", cfg)
        if load_path and os.path.isfile(load_path):
            print(f"\n[INFO] Loading PPO model for finetuning from {load_path}")
            model = PPO.load(load_path, env=env, device="auto")
            # Adjust learning rate for all optimizer param groups
            for g in model.policy.optimizer.param_groups:
                g["lr"] = cfg["learning_rate"]
        else:
            print("\n[INFO] Starting finetune training from scratch.")
            model = PPO("MlpPolicy", env, seed=seed, **cfg)
        save_prefix = f"{run_name}_ppo_ft"
        best_dir = f"models/{run_name}_ppo_ft_best"
        ckpt_dir = f"models/{run_name}_ppo_ft_ckpts"
        final_model_path = f"models/{run_name}_ppo_finetune_final.zip"
        vecnorm_path = f"models/{run_name}_vecnorm_finetune.pkl"
    else:
        cfg = ppo_long()
        debug_print_ppo_cfg("long", cfg)
        model = PPO("MlpPolicy", env, seed=seed, **cfg)
        save_prefix = f"{run_name}_ppo"
        best_dir = f"models/{run_name}_ppo_best"
        ckpt_dir = f"models/{run_name}_ppo_ckpts"
        final_model_path = f"models/{run_name}_ppo_final.zip"
        vecnorm_path = f"models/{run_name}_vecnorm_long.pkl"

    # Eval env (for EvalCallback; separate VecNormalize but same settings)
    eval_env = DummyVecEnv([lambda seed=seed: make_env(render=False, seed=seed + 1000)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=best_dir,
        n_eval_episodes=10,
        eval_freq=10_000,
        deterministic=True,
    )

    ckpt = CheckpointCallback(save_freq=50_000, save_path=ckpt_dir, name_prefix=save_prefix)
    train_cb = EpisodeReturnCallback()

    callback = CallbackList([eval_cb, ckpt, train_cb])

    # ============ TRAIN ============
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Save final model + vecnorm stats
    model.save(final_model_path)
    env.save(vecnorm_path)

    print(f"\n[INFO] Saved final PPO model to {final_model_path}")
    print(f"[INFO] Saved VecNormalize stats to {vecnorm_path}")

    # ============ PICK MODEL FOR EVAL ============
    best_model_path = os.path.join(best_dir, "best_model.zip")
    if os.path.isfile(best_model_path):
        print(f"[INFO] Using best model at {best_model_path} for evaluation/plots.")
        model_path_for_eval = best_model_path
    else:
        print("[WARN] No best_model.zip found; using final model for evaluation/plots.")
        model_path_for_eval = final_model_path

    # ============ EVAL SUMMARY ============
    stats = evaluate(model_path_for_eval, vecnorm_path, n_episodes=20)
    print(
        "\n=== PPO {} Training ({:,} steps) ===".format(
            "Finetune" if finetune else "Long", total_timesteps
        )
    )
    print("Success\tAvgReturn\tAvgSteps\tCollisions\tHeadway<1s")
    print(
        f"{stats.success_rate:.2f}\t{stats.avg_return:.2f}\t"
        f"{stats.avg_steps:.0f}\t{stats.collisions}\t{stats.headway_violations:.1f}"
    )

    # ============ PLOTS ============
    results_dir = f"results/narrow_safe/{run_name}"
    os.makedirs(results_dir, exist_ok=True)

    # 1) Training learning curve
    training_curve_path = os.path.join(results_dir, "ppo_training_learning_curve.png")
    plot_training_curve(train_cb.episode_returns, training_curve_path)

    # 2) Violin plot of 500 deterministic eval returns
    eval_returns = eval_returns_for_violin(
        model_path_for_eval,
        vecnorm_path,
        n_episodes=500,
    )
    violin_path = os.path.join(results_dir, "ppo_eval_violin_500eps.png")
    plot_violin(eval_returns, violin_path)

    return model


# =========================
# CLI
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick", action="store_true", help="Run a short PPO training and print eval metrics."
    )
    parser.add_argument(
        "--long", action="store_true", help="Run a longer PPO training (ppo_long)."
    )
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="Run finetune PPO training (ppo_finetune_cfg).",
    )
    parser.add_argument("--quick_steps", type=int, default=50_000)
    parser.add_argument("--long_steps", type=int, default=500_000)
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="Optional model path to continue from (used with --finetune).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for training (PPO + env).",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="run",
        help="Name used to separate model/result files (e.g., seed0, seed1).",
    )
    args = parser.parse_args()

    seed = args.seed
    run_name = args.run_name

    if args.quick:
        train_quick(total_timesteps=args.quick_steps, seed=seed, run_name=run_name)

    if args.long or args.finetune:
        train_long(
            total_timesteps=args.long_steps,
            finetune=args.finetune,
            load_path=args.load,
            seed=seed,
            run_name=run_name,
        )

    if not (args.quick or args.long or args.finetune):
        print("(Tip) Use --quick for a short run, or --long / --finetune for extended training.")
