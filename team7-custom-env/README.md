# 🛣️ Team7 Custom Environment (`Team7-v0`)

This repository holds **Team 7’s custom reinforcement-learning environment**, built on top of [Highway-Env](https://highway-env.farama.org/) and [Gymnasium](https://gymnasium.farama.org/).  
The environment is packaged as `custom_env`, and the environment ID is `Team7-v0`.

It introduces new vehicle dynamics and road hazards such as sudden braking vehicles, ghost (lidar) vehicles, and potholes

---
## 🚗 Overview

The custom environment is defined in `my_env.py` and extends `HighwayEnv`.  
It modifies the base highway environment by adding:
- **Sudden braking vehicles** - vehicles that can randomly decelerate abruptly.
- **Ghost vehicles** - lidar-only vehicles that exhibit anomalous behavior (e.g., teleporting or flickering).
- **Potholes** - stationary road hazards randomly distributed along lanes. Slows down vehicles upon collision, does not result in termination.

---

## 📋 Environment Specification

### **State**
The agent’s state is obtained from a **LidarObservation**, with the following configuration:
- **Cells:** 36 equally spaced lidar beams  
- **Maximum range:** 100.0 m  
- The observation encodes the distance and relative velocity of nearby vehicles, ghost vehicles, and potholes.

### **Action**
The environment uses a **discrete meta-action space** (`DiscreteMetaAction`), by default.
It can be modified to use other action spaces from Highway-Env as well.

```python
"action": {
    "type": "DiscreteMetaAction",
    "vehicle_class": "custom_env.vehicle.customvehicle.CustomVehicle"
}
```

### **Reward Function**
Reward function is the same as found in Highway-Env. The total reward `R` is a weighted combination of multiple components:
- **Collision penalty:** `-1` on collision  
- **Right-lane incentive:** `+0.1` for driving in rightmost lanes  
- **High-speed reward:** up to `+0.4` when driving near speed limit  
- **Lane-change penalty:** `0` (disabled by default)

All rewards are normalized when `"normalize_reward": True`.

### **Next State**
After each step:
- The environment updates the ego vehicle’s kinematics.
- Potholes and ghost vehicles are updated (ghosts may exhibit anomalies).
- The next lidar observation is computed based on all surrounding entities.

### **Termination Conditions**
The episode terminates when:
- A **collision** occurs with a vehicle.
- The **duration limit** (default `40s`) is reached.

---

## ⚙️ Custom Entities

### 🧩 `MyEnv` Class (`my_env.py`)
The core environment class inheriting from `HighwayEnv`.  
Main responsibilities:
- Constructs the **road**, **vehicles**, and **potholes**.
- Spawns **ghost vehicles** near controlled vehicles.
- Configures custom simulation parameters (e.g., anomaly interval, density, potholes count).

**Key configurable parameters:**
| Parameter | Description | Default |
|------------|-------------|----------|
| `lanes_count` | Number of lanes | 8 |
| `vehicles_count` | Total number of other vehicles | 40 |
| `controlled_vehicles` | Ego vehicles | 1 |
| `duration` | Simulation length (s) | 40 |
| `anomaly_interval` | Steps between ghost anomalies | 3 |
| `potholes.count` | Number of potholes | 20 |
| `speed_limit` | Max vehicle speed (m/s) | 50 |

---

### 👻 `GhostVehicle` (`ghost_vehicle.py`)
A **non-physical**, lidar-visible vehicle that mimics anomalous perception behavior.  
It orbits around a target vehicle and occasionally exhibits one of three anomaly types.

**Anomaly Types:**
| Type | Description |
|------|--------------|
| `TELEPORT` | Randomly jumps ±10–25m along x-axis |
| `FLICKER` | Temporarily becomes invisible to lidar |
| `SPEED` | Displays unrealistic velocity values |

**Behavioral Logic:**
- Ghost follows its target’s position and heading.
- Every `anomaly_interval` steps, a random anomaly occurs. Note: "steps" here refers to Agent steps, NOT simulation steps.
- The ghost does **not** cause physical collisions, but is **visible in lidar**.

---

### 🚨 `SuddenBrakingVehicle` (`SuddenBrakingVehicle.py`)
An extension of the `IDMVehicle` that **randomly performs sudden braking events**:
- With probability `0.0085`, it slows down abruptly.
- If speed < 5 m/s, it accelerates again and resumes normal speed (≈45 m/s).
- Color changes to **red** when braking, **blue** when normal.

It also handles collisions with **potholes**:
- Colliding with potholes reduces speed gradually.
- Regular vehicle-to-vehicle collisions trigger crash flags normally.

Similar pothole-related behavior is exhibited by controlled vehicle (`CustomVehicle.py`)

---

### 🕳️ `Pothole` (`Pothole.py`)
A stationary road **obstacle** representing a pothole:
- Spawned randomly along lanes using `create_random()`.
- Solid and collidable.
- Visible in lidar scans.

---

## 🚦 Constraints and Design Rules

| Constraint | Description |
|-------------|--------------|
| **Lidar visibility** | Only solid objects (`solid=True`) appear in lidar scans. |
| **Ghost collisions** | Disabled (`collidable=False`). Ghosts affect perception but not physics. |
| **Sudden braking** | Randomized; introduces unpredictable slowdown events. |
| **Pothole placement** | Random per lane between `spawn_ahead_min` and `spawn_ahead_max` distances. |
| **Simulation frequency** | Derived from HighwayEnv settings; affects ghost anomaly timing. |

---


## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/adityapatel149/team7-custom-env.git
cd team7-custom-env
```

### 2. (Optional) Set up a virtual environment
**Windows (PowerShell):**
```bash
python -m venv .venv
.venv\Scripts\activate
```
**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the package

Install the environment and its dependencies directly using the project’s `pyproject.toml`:
```bash
pip install -e .
```
This makes the `custom_env` package importable.

---

## ▶️ Running the Example Script

There's a demo script available:
```bash
python scripts/run_env.py
```
This will:
- Create the `Team7-v0` environment (`render_mode="human"` by default or can use `"rgb_array"` for frame capture).  
- Run a few steps (IDLE actions) and render the environment.  
- Print out total reward when the episode ends.

Example usage:
```python
import gymnasium as gym
import custom_env

env = gym.make("Team7-v0", render_mode="human")
obs, info = env.reset()

done = False
while not done:
    action = env.unwrapped.action_type.actions_indexes["IDLE"]
    obs, reward, done, truncated, info = env.step(action)
    env.render()

env.close()
```

---

## 🧩 Project Structure
```
team7-custom-env/
│
├── custom_env/
│   ├── __init__.py
│   ├── register.py                      # Registers the custom Gymnasium environment
│   │
│   ├── envs/
│   │   ├── __init__.py
│   │   └── my_env.py                    # MyEnv class - main environment derived from HighwayEnv
│   │                                    # Adds LidarObservation, GhostVehicle, Potholes, and SuddenBrakingVehicle
│   │
│   ├── vehicle/
│   │   ├── __init__.py
│   │   ├── SuddenBrakingVehicle.py      # Defines a vehicle that occasionally brakes abruptly
│   │   ├── GhostVehicle.py              # Defines ghost (lidar-only) vehicles with anomalies
│   │   └── CustomVehicle.py             # Ego or controlled vehicle with custom logic to handle collision with potholes
│   │
│   ├── objects/
│   │   ├── __init__.py
│   │   └── Pothole.py                   # Defines potholes as stationary obstacles visible to lidar
│
├── scripts/
│   └── run_env.py                       # Demo/test script to instantiate and run the environment
│
├── requirements.txt                     # Python dependencies (Gymnasium, HighwayEnv, NumPy, etc.)
├── pyproject.toml                       # Package metadata for pip installation
└── README.md                            # Project documentation (this file)
```

---

## 🚀 Next Steps

- Integrate training scripts using frameworks such as [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) or RLlib for policy learning.

---

## 🧑‍💻 Authors

**Team 7 — CS271: Reinforcement Learning (San José State University)**   
Aditya Patel  
Karan Jain  
Shareen Rodrigues      
Instructor: Genya Ishigaki

---

## 📜 License

This project is intended for academic/research use.  
The core functionality builds upon Highway-Env © [Farama Foundation](https://highway-env.farama.org/).
