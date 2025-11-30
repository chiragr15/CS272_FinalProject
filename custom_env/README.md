# CS272 — Custom Narrow Street Environment (`narrow-safe-v0`)

This repository implements a custom Reinforcement Learning environment built on top of Highway-Env to simulate safe navigation through a narrow urban street with parked vehicles, oncoming traffic, constrained maneuverability, and safety-critical decision-making.

It also includes:

- The environment implementation (`narrow_safe_env/`)
- The road network builder (`narrow_street_env.py`)
- A PPO training + evaluation agent (`custom_env_agent/`)
- Saved best PPO model + VecNormalize statistics (`models/`)
- Evaluation plots for the final submission (`results/`)

---

## 1. Repository Structure

```text
custom_env/
│
├── narrow_safe_env/
│   ├── __init__.py
│   ├── narrow_safe_env.py        # Main environment logic (registers narrow-safe-v0)
│   └── narrow_street_env.py      # Road network, parked vehicles, oncoming traffic
│
├── custom_env_agent/
│   ├── ppo_seed2_agent.py        # PPO agent: training + evaluation
│   ├── viz_narrow_agent.py       # Visualization script (load model & render)
│   ├── __init__.py
│
├── models/
│   ├── seed2_ppo_best.zip        # Best performing PPO model (Seed 2)
│   └── seed2_vecnorm_long.pkl    # VecNormalize statistics
│
├── results/
│   ├── training_curve_seed2.png  # Mean reward vs timesteps (Seed 2 PPO)
│   └── violin_seed2_500.png      # 500-episode evaluation violin plot
│
└── README.md                     # This file
```

## 2. Environment Overview — narrow-safe-v0

`narrow-safe-v0` models a two-lane urban road where:

- The right lane is partially blocked by parked vehicles.
- The left lane contains occasional oncoming traffic.
- The agent must overtake safely despite tight road geometry.
- "Safe mode" logic prevents unfair penalties when both lanes are briefly blocked.
- The environment focuses on realistic overtaking, headway safety, and stable control.

This is the primary environment used for the Seed 2 PPO agent.

## 3. Objective

The agent must:

- Drive forward efficiently toward the goal.
- Avoid collisions with parked or moving vehicles.
- Maintain safe headway.
- Perform smooth lane changes when necessary.
- Avoid oscillatory or erratic behavior.
- Reach the scenario end safely and within the time limit.

## 4. Action Space (Discrete)

| Action Index | Description |
|--------------|-------------|
| 0 | Maintain speed |
| 1 | Accelerate |
| 2 | Brake |
| 3 | Lane change left |
| 4 | Lane change right |

Invalid lane changes have no effect unless a collision actually occurs.

## 5. Observation Space

The observation is a structured numerical feature vector that includes:

- Ego speed, longitudinal position, and lane index
- Distances to parked vehicles
- Distances and relative velocities of oncoming/leading vehicles
- Lane-change feasibility indicators
- Distances to lane boundaries
- Scenario progress features

A SafeObs wrapper is applied to:

- Filter out NaN and Inf values
- Clip extreme values
- Ensure numerically stable observations

## 6. Reward Function

The reward function combines progress and safety terms.

### Positive Rewards

- Forward progress along the road
- Maintaining safe headway (> 1 second)

### Penalties

- Unsafe headway (< 1 second)
- Collisions
- Excessive or unnecessary lane changes
- Abrupt or oscillatory behavior

### Safe Mode Behavior

Temporary unavoidable blocking (no free lane available) receives no penalty, preventing unfair punishment.

## 7. Termination Conditions

Episodes terminate under the following conditions:

- Reaching the goal (success)
- Collision with another vehicle or obstacle
- Timeout (maximum episode length exceeded)

## 8. Constraints and Caveats

- Narrow lanes restrict maneuvering space
- Parked vehicles require controlled overtaking
- Oncoming vehicles introduce dynamic hazards
- Off-road driving is not allowed
- Brief "no-free-lane" scenarios are tolerated without penalty

## 9. Usage Instructions

### Register the Environment

```python
import narrow_safe_env  # Registers "narrow-safe-v0"
```

### Create the Environment

```python
import gymnasium as gym
env = gym.make("narrow-safe-v0")
```

### Render the Environment

```python
env = gym.make("narrow-safe-v0", render_mode="human")
```

### Basic Interaction Loop

```python
obs, info = env.reset()
done = False
truncated = False

while not (done or truncated):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
```

## 10. PPO Agent (Seed 2)

The PPO agent implementation is located in:

```
custom_env_agent/ppo_seed2_agent.py
```

Main features:

- Uses Stable-Baselines3 PPO
- Full training and evaluation pipeline for narrow-safe-v0
- Uses VecNormalize for stable feature and reward scaling

Automatically saves:

- `seed2_ppo_best.zip` (best-performing PPO model)
- `seed2_vecnorm_long.pkl` (normalization statistics)

Generates:

- `training_curve_seed2.png` — training curve (mean reward vs timesteps)
- `violin_seed2_500.png` — 500-episode evaluation violin plot

## 11. Visualization Script

Located in:

```
custom_env_agent/viz_narrow_agent.py
```
This script:

- Loads `seed2_ppo_best.zip`
- Applies `seed2_vecnorm_long.pkl`
- Renders the agent in narrow-safe-v0
- Allows qualitative behavior visualization and debugging

## 12. Results

Stored in:

```
results/
```
Includes:

- `training_curve_seed2.png` — Mean episode reward vs timesteps
- `violin_seed2_500.png` — Distribution of returns over 500 evaluation episodes

These plots are used in the final project submission.

## 13. Saved Models

Located in:

```
models/
```

| File | Description |
|------|-------------|
| `seed2_ppo_best.zip` | Best PPO model used for evaluation |
| `seed2_vecnorm_long.pkl` | VecNormalize statistics required for inference |

These files ensure reproducible evaluation and visualization.