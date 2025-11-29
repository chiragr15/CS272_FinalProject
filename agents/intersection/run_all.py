# run_intersection_all.py
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def run(step_name, script_name):
    print(f"\n========== {step_name} ({script_name}) ==========\n")
    result = subprocess.run(
        [sys.executable, str(ROOT / script_name)],
        check=True
    )
    print(f"\n========== {step_name} DONE ==========\n")
    return result

if __name__ == "__main__":
    # 1) Train PPO for Lidar + Grayscale
    run("TRAINING (Lidar + Grayscale)", "intersection_train.py")

    # 2) Evaluate 500 episodes & save .npy + violin plots
    run("EVALUATION & VIOLIN PLOTS", "intersection_eval_violin.py")

    # 3) Build smoothed learning curves from monitor logs
    run("LEARNING CURVES", "intersection_plots.py")

    print("✅ All Intersection experiments finished.")
