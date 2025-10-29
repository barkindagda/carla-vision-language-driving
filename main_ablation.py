#!/usr/bin/env python3
import os
import time
import argparse
import numpy as np
import csv
from datetime import datetime

# ----------------------------------------------------------------------
# 1. VLM controller (pick one – keep the import you actually use)
# ----------------------------------------------------------------------
from Models.vlm_controller import VLMController
# from Models.vlm_qwen import VLMController


# ----------------------------------------------------------------------
# 2. Environment
# ----------------------------------------------------------------------
from environment.carla_env import CarlaEnv

# ----------------------------------------------------------------------
# 3. Helper for plotting (unchanged)
# ----------------------------------------------------------------------
from utils.plot import plot_rewards


# ----------------------------------------------------------------------
# 4. Argument parser – now with weather
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Run VLM-driven episodes in CARLA (zero-shot driver)."
)
parser.add_argument(
    "--episodes", type=int, default=10,
    help="Number of episodes to run (default: 10)"
)
parser.add_argument(
    "--weather", type=str,
    choices=[
        'ClearNoon', 'HardRainNoon'          # <-- only the two you asked for
    ],
    default='ClearNoon',
    help="Weather preset (default: ClearNoon)"
)
parser.add_argument(
    "--vlm-model", type=str,
    default="DAMO-NLP-SG/VideoLLaMA3-2B-Image",
    help="VLM model name on HuggingFace"
)
parser.add_argument(
    "--output-dir", type=str, default="./vlm_outputs",
    help="Directory for CSV, frames, plots"
)
args = parser.parse_args()


# ----------------------------------------------------------------------
# 5. VLM configuration (same as before, but pulled from args)
# ----------------------------------------------------------------------
vlm_config = {
    "model_name": args.vlm_model,
    "update_frequency": 3,
    "frames_needed": 3,
    "output_dir": args.output_dir,
    "max_new_tokens": 512,
    "verbose": True
}
os.makedirs(vlm_config["output_dir"], exist_ok=True)

# ----------------------------------------------------------------------
# 6. CSV setup (per-episode summary)
# ----------------------------------------------------------------------
csv_path = os.path.join(vlm_config["output_dir"], "episode_metrics.csv")
csv_headers = [
    "Episode", "Outcome", "Steps", "Duration (s)", "Total Reward",
    "Avg Speed (km/h)", "Avg Acceleration (m/s²)", "Pedestrian Detections"
]
with open(csv_path, 'w', newline='') as f:
    csv.writer(f).writerow(csv_headers)
print(f"[INFO] CSV initialized: {os.path.abspath(csv_path)}")


# ----------------------------------------------------------------------
# 7. Main training / evaluation loop
# ----------------------------------------------------------------------
def main():
    # ---- 7.1 Initialise environment (weather will be set in reset) ----
    env = CarlaEnv(vlm_frames=vlm_config["frames_needed"])

    # ---- 7.2 Initialise VLM controller (external class) ----
    vlm_controller = VLMController(**vlm_config)
    env.vlm_controller = vlm_controller

    # ---- 7.3 Run episodes ------------------------------------------------
    for episode in range(args.episodes):
        print("\n" + "=" * 60)
        print(f"Episode {episode+1}/{args.episodes} | Weather: {args.weather}")
        print("=" * 60)

        # ---- Reset with explicit weather option -------------------------
        obs, info = env.reset(options={"weather": args.weather})
        done = False
        step = 0
        ep_reward = 0.0
        rewards_hist = []
        speeds = []
        accelerations = []
        start_t = time.time()

        metrics = {
            "steps": 0,
            "collision": False,
            "success": False,
            "stalled": False,
            "ped_detected": 0
        }

        # ---- Episode loop -----------------------------------------------
        while not done:
            # VLM decides every `update_frequency` steps (handled inside)
            vlm_controller.process_if_needed(env)

            action = vlm_controller.current_action_value
            obs, reward, terminated, truncated, info = env.step(action)

            # ---- Logging -------------------------------------------------
            if hasattr(env, "current_reward_components"):
                rewards_hist.append(env.current_reward_components)

            speeds.append(info.get("speed_kmh", 0.0))
            accelerations.append(info.get("acceleration", 0.0))

            if info.get("pedestrian_detected", False):
                metrics["ped_detected"] += 1

            done = terminated or truncated
            ep_reward += reward
            step += 1

            if step % 20 == 0:
                print(
                    f"Step {step:04d} | Action={vlm_controller.current_action_text} "
                    f"({action:+.2f}) | Speed={info['speed_kmh']:.1f} km/h | "
                    f"Ped={info.get('pedestrian_detected')}"
                )

        # ---- Episode summary --------------------------------------------
        duration = time.time() - start_t
        metrics["steps"] = step
        avg_speed = np.mean(speeds) if speeds else 0.0
        avg_acc = np.mean(accelerations) if accelerations else 0.0

        # outcome
        if len(env.collision_hist) > 0:
            outcome = "COLLISION"
            metrics["collision"] = True
        elif env.successful_ep > 0:
            outcome = "SUCCESS"
            metrics["success"] = True
        else:
            outcome = "STALLED"
            metrics["stalled"] = True

        # ---- Write episode row -------------------------------------------
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                episode + 1,
                outcome,
                step,
                f"{duration:.1f}",
                f"{ep_reward:.2f}",
                f"{avg_speed:.2f}",
                f"{avg_acc:.2f}",
                metrics["ped_detected"]
            ])
        print(f"[INFO] Episode {episode+1} saved → {outcome}")

        # ---- Plot reward components --------------------------------------
        plot_path = os.path.join(vlm_config["output_dir"],
                                 f"episode_{episode+1}_rewards.png")
        plot_rewards(rewards_hist, plot_path)

        # ---- Final episode print -----------------------------------------
        print(f"""
Episode {episode+1} finished
  Outcome      : {outcome}
  Steps        : {step}
  Duration     : {duration:.1f}s
  Reward       : {ep_reward:.2f}
  Avg Speed    : {avg_speed:.2f} km/h
  Avg Acc      : {avg_acc:.2f} m/s²
  Ped Dets     : {metrics['ped_detected']}
  VLM Action   : {vlm_controller.current_action_text} ({vlm_controller.current_action_value:.2f})
  Justification: {vlm_controller.current_justification}
        """.strip())

    print("\nAll episodes completed!")


# ----------------------------------------------------------------------
# 8. Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()