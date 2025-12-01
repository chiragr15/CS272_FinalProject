# CS 272 Final Project – DRL for Intersection-v1
## 🧩 Full Pipeline

You can run all steps (train → evaluate → plot):

```bash
python run_all.py
```

---


## 🚗 Overview
This project implements Deep Reinforcement Learning (DRL) agents to solve the **Intersection-v1** environment from **highway-env** using **PPO (Proximal Policy Optimization)**.  
The goal is to compare performance between two observation modalities:

- **GrayscaleObservation** (CNN policy)  
- **LidarObservation** (MLP policy)

The training, evaluation, and plotting pipelines follow the exact configuration required by our team and professor.

---

## 📂 Project Structure

```
.
├── intersection_envs.py        # Environment factory (Lidar + Grayscale configs)
├── intersection_train.py       # PPO training for both observation types
├── intersection_eval_violin.py # Evaluation over 500 episodes + violin plots
├── intersection_plots.py       # Learning curve generation from monitor logs
├── run_intersection_all.py     # (optional) One-click full pipeline runner
├── logs_intersection/          # Saved models + monitor logs
└── plots/                      # Generated images (learning curves & violins)
```

---

## 🧠 Observation Configurations

### **GrayscaleObservation**
```python
"type": "GrayscaleObservation",
"observation_shape": (128, 64),
"stack_size": 4,
"weights": [0.2989, 0.5870, 0.1140],
"scaling": 1.75
```

### **LidarObservation**
```python
"type": "LidarObservation",
"cells": 128
```

These configs are defined in `intersection_envs.py` inside `make_intersection_env()`.

---

## 🧪 Training

Run PPO training for **both** Lidar and Grayscale agents:

```bash
python intersection_train.py
```

This will train:

- `ppo_intersection_lidar.zip` → `logs_intersection/lidar/`
- `ppo_intersection_gray.zip`  → `logs_intersection/gray/`

---

## 📊 Evaluation

Evaluate each trained agent over **500 episodes**:

```bash
python intersection_eval_violin.py
```

This produces:

- `intersection_lidar_returns.npy`
- `intersection_gray_returns.npy`
- `intersection_lidar_violin.png`
- `intersection_gray_violin.png`

---

## 📈 Learning Curves

To generate smoothed learning curves from monitor logs:

```bash
python intersection_plots.py
```

Creates:

- `intersection_lidar_learning_curve.png`
- `intersection_gray_learning_curve.png`

---

## 📉 Results Summary

### **Learning Behavior**
Both PPO agents successfully learned the Intersection-v1 navigation task.

- **GrayscaleObservation**  
  - Faster early learning  
  - Slightly higher peak reward (~4.4–4.5 smoothed)  
  - Stable convergence

- **LidarObservation**  
  - Slower early learning  
  - Achieves similar long-term performance (~4.1–4.4 smoothed)

### **Episodic Return Distribution (500 episodes)**
Both agents produce episode returns roughly between **-5 and +11**, consistent with the environment's default reward function.

### **Comparison to Baseline**
Professor’s baseline PPO reports ~12–16 cumulative return (different reward aggregation).  
Our agents use the default Intersection-v1 reward and show **strong, stable learning**.

---

## 🧠 PPO Agent Choice Reasoning

We chose PPO because:

- Stable on-policy updates via clipping  
- Works with both MLP (Lidar) and CNN (Grayscale) policies  
- Performs well on autonomous driving benchmarks  
- Easy to reproduce with stable-baselines3

---

## 🧾 Requirements

```
python>=3.9
gymnasium
highway-env
stable-baselines3
numpy
matplotlib
pandas
```

Install with:

```bash
pip install -r requirements.txt
```

---

## 🙌 Authors
Team project for **CS 272: Reinforcement Learning** (Fall 2025).
