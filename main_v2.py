import os
import time
import numpy as np
import csv

# ====== CONFIGURATION ======
MODE = "lane_change"  # Options: "longitudinal" or "lane_change"

# Import the environment and controller you want
from environment.carla_env_v2 import CarlaEnv as ENV_CLASS
# from environment.carla_env import CarlaEnv as ENV_CLASS

from Models.vlm_llama import VLMController
# from Models.vlm_llava import VLMController
# from Models.vlm_qwen import VLMController

MODEL_NAME = 'DAMO-NLP-SG/VideoLLaMA3-2B-Image'  # Full HuggingFace model name
# ============================

from utils.plot import plot_rewards


def main():
    vlm_config = {
        "model_name": MODEL_NAME,  # Use full MODEL_NAME as provided
        "update_frequency": 1,
        "frames_needed": 3,
        "output_dir": "./vlm_outputs",
        "max_new_tokens": 512,
        "verbose": True,
        "mode": MODE
    }

    os.makedirs(vlm_config["output_dir"], exist_ok=True)

    # Use the part of MODEL_NAME after the last '/' for file naming, and sanitize
    safe_model_name = MODEL_NAME.split('/')[-1].replace('/', '_')

    csv_path = os.path.join(vlm_config["output_dir"], f"{MODE}_{safe_model_name}.csv")
    if MODE == "lane_change":
        csv_headers = [
            "Episode", "Outcome", "Steps", "Duration (s)", "Total Reward",
            "Avg Speed (km/h)", "Avg Acceleration (m/s²)", "Pedestrian Detections",
            "Lane Changes", "Avg Lane Change Reward"
        ]
    else:
        csv_headers = [
            "Episode", "Outcome", "Steps", "Duration (s)", "Total Reward",
            "Avg Speed (km/h)", "Avg Acceleration (m/s²)", "Pedestrian Detections"
        ]

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_headers)

    env = ENV_CLASS(vlm_frames=vlm_config["frames_needed"], scenario=2 if MODE == "lane_change" else None)
    vlm_controller = VLMController(**vlm_config)
    env.vlm_controller = vlm_controller

    num_episodes = 1
    for episode in range(num_episodes):
        print(f"\n{'=' * 50}\nStarting Episode {episode + 1}/{num_episodes}\n{'=' * 50}")

        if MODE == "lane_change":
            traj_path = os.path.join(vlm_config["output_dir"], f"trajectory_{safe_model_name}_episode_{episode+1}.csv")
            with open(traj_path, 'w', newline='') as traj_file:
                writer = csv.writer(traj_file)
                writer.writerow([
                    "Timestep", "X", "Y", "Z", "Yaw", "Speed_kmh",
                    "Action_Text", "Throttle_Brake", "Steer", "Distance_to_Obstacle"
                ])

        observation, info = env.reset()
        done = False
        episode_step = 0
        episode_reward = 0
        speeds, accelerations, rewards_history = [], [], []
        lane_changes, lane_change_rewards = 0, []
        start_time = time.time()
        metrics = {
            "steps": 0, "collision": False, "success": False, "stalled": False,
            "pedestrian_detected_count": 0, "lane_changes": 0
        }

        while not done:
            vlm_controller.process_if_needed(env)
            action_value = vlm_controller.current_action_value

            if MODE == "lane_change":
                if not isinstance(action_value, (list, np.ndarray)) or len(action_value) != 2:
                    action_value = np.array([0.5, 0.0])
                else:
                    action_value = np.array(action_value, dtype=np.float32)

                vehicle_state = env.get_current_vehicle_state()
                with open(traj_path, 'a', newline='') as traj_file:
                    writer = csv.writer(traj_file)
                    writer.writerow([
                        episode_step,
                        f"{vehicle_state.get('vehicle_location_x', 0.0):.2f}",
                        f"{vehicle_state.get('vehicle_location_y', 0.0):.2f}",  # Fixed key
                        f"{vehicle_state.get('vehicle_location_z', 0.0):.2f}",  # Fixed key
                        f"{vehicle_state.get('yaw', 0.0):.1f}",  # Fixed default
                        f"{vehicle_state.get('speed_kmh', 0.0):.2f}",
                        vlm_controller.current_action_text,
                        f"{action_value[0]:.2f}", f"{action_value[1]:.2f}",
                        f"{vehicle_state.get('distance_to_obstacle', float('inf')):.1f}"
                    ])

            observation, reward, terminated, truncated, info = env.step(action_value)

            if hasattr(env, 'current_reward_components'):
                rewards_history.append(env.current_reward_components)
                if MODE == "lane_change" and 'lane_change_reward' in env.current_reward_components:
                    lane_change_rewards.append(env.current_reward_components['lane_change_reward'])
                    if env.current_reward_components['lane_change_reward'] != 0:
                        lane_changes += 1

            speeds.append(info.get('speed_kmh', 0.0))
            accelerations.append(info.get('acceleration', 0.0))
            if info.get("pedestrian_detected", False):
                metrics["pedestrian_detected_count"] += 1

            episode_step += 1
            episode_reward += reward
            done = terminated or truncated

        duration = time.time() - start_time
        metrics["steps"] = episode_step
        metrics["lane_changes"] = lane_changes
        avg_speed = np.mean(speeds) if speeds else 0.0
        avg_accel = np.mean(accelerations) if accelerations else 0.0
        avg_lane_change_reward = np.mean(lane_change_rewards) if lane_change_rewards else 0.0

        outcome = "SUCCESS" if env.successful_ep > 0 else "COLLISION" if env.collision_hist else "STALLED"

        row_data = [
            episode + 1, outcome, episode_step,
            f"{duration:.1f}", f"{episode_reward:.2f}",
            f"{avg_speed:.2f}", f"{avg_accel:.2f}",
            metrics["pedestrian_detected_count"]
        ]
        if MODE == "lane_change":
            row_data += [lane_changes, f"{avg_lane_change_reward:.2f}"]

        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row_data)

        print(f"Episode {episode + 1} complete — Outcome: {outcome}, Reward: {episode_reward:.2f}")

        plot_path = os.path.join(vlm_config["output_dir"], f"episode_{episode+1}_rewards.png")
        try:
            plot_rewards(rewards_history, plot_path)
        except Exception as e:
            print(f"[ERROR] Could not save reward plot: {e}")

    print("\nAll episodes completed.")

if __name__ == "__main__":
    print("[DEBUG] Script started.")
    main()