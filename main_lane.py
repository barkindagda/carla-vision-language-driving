import os
import time
import numpy as np
import csv
from environment.carla_env_v2 import CarlaEnv
from Models.vlm_llama_lane_change import VLMController
from utils.plot import plot_rewards

def main():
    """Main function to run CARLA with VLM control, including lane change actions and trajectory logging."""
    vlm_config = {
        "model_name": "DAMO-NLP-SG/VideoLLaMA3-2B-Image",
        "update_frequency": 2,
        "frames_needed": 3,
        "output_dir": "./vlm_outputs",
        "max_new_tokens": 512,
        "verbose": True
    }
    MODEL_NAME = 'VIDEOLLMA3-2B-IMAGE'
    
    os.makedirs(vlm_config["output_dir"], exist_ok=True)
    
    # Initialize metrics CSV
    csv_path = os.path.join(vlm_config["output_dir"], f"lane_change_{MODEL_NAME}.csv")
    csv_headers = [
        "Episode", "Outcome", "Steps", "Duration (s)", "Total Reward",
        "Avg Speed (km/h)", "Avg Acceleration (m/s²)", "Pedestrian Detections",
        "Lane Changes", "Avg Lane Change Reward"
    ]
    
    try:
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_headers)
        print(f"[INFO] CSV file initialized: {os.path.abspath(csv_path)}")
    except Exception as e:
        print(f"[ERROR] Failed to create metrics CSV at {csv_path}: {e}")
        return
    
    try:
        env = CarlaEnv(vlm_frames=vlm_config["frames_needed"], scenario=2)
    except Exception as e:
        print(f"[ERROR] Failed to initialize CarlaEnv: {e}")
        return
    
    try:
        vlm_controller = VLMController(**vlm_config)
        env.vlm_controller = vlm_controller
    except Exception as e:
        print(f"[ERROR] Failed to initialize VLMController: {e}")
        return
    
    num_episodes = 1
    for episode in range(num_episodes):
        print(f"\n{'=' * 50}")
        print(f"Starting Episode {episode + 1}/{num_episodes}")
        print(f"{'=' * 50}")
        
        # Initialize trajectory CSV
        trajectory_csv_path = os.path.join(vlm_config["output_dir"], f"trajectory_{MODEL_NAME}_episode_{episode+1}.csv")
        trajectory_headers = [
            "Timestep", "X", "Y", "Z", "Yaw", "Speed_kmh", "Action_Text",
            "Throttle_Brake", "Steer", "Distance_to_Obstacle"
        ]
        try:
            with open(trajectory_csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(trajectory_headers)
            print(f"[INFO] Trajectory CSV initialized: {os.path.abspath(trajectory_csv_path)}")
        except Exception as e:
            print(f"[ERROR] Failed to create trajectory CSV at {trajectory_csv_path}: {e}")
            continue
        
        try:
            observation, info = env.reset()
        except Exception as e:
            print(f"[ERROR] Failed to reset environment: {e}")
            continue
        done = False
        episode_step = 0
        episode_reward = 0
        rewards_history = []
        speeds = []
        accelerations = []
        lane_changes = 0
        lane_change_rewards = []
        start_time = time.time()
        metrics = {
            "steps": 0,
            "collision": False,
            "success": False,
            "stalled": False,
            "pedestrian_detected_count": 0,
            "lane_changes": 0
        }
        
        while not done:
            try:
                vlm_controller.process_if_needed(env)
                action_value = vlm_controller.current_action_value
                if not isinstance(action_value, (list, np.ndarray)) or len(action_value) != 2:
                    print(f"[WARNING] Invalid action_value: {action_value}, defaulting to [0.5, 0.0]")
                    action_value = np.array([0.5, 0.0], dtype=np.float32)
                else:
                    action_value = np.array(action_value, dtype=np.float32)
                
                # Get vehicle state for trajectory logging
                vehicle_state = env.get_current_vehicle_state()
                print(f"[DEBUG] Step {episode_step}: Action={vlm_controller.current_action_text}, "
                      f"Value={action_value.tolist()}, Distance to obstacle={vehicle_state.get('distance_to_obstacle', float('inf')):.1f} m, "
                      f"X={vehicle_state.get('vehicle_location_x', 0.0):.2f}, Yaw={vehicle_state.get('yaw', 90.0):.1f}°")
                
                # Log trajectory
                try:
                    with open(trajectory_csv_path, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([
                            episode_step,
                            f"{vehicle_state.get('vehicle_location_x', 0.0):.2f}",
                            f"{vehicle_state.get('vehicle_location_y', 0.0):.2f}",
                            f"{vehicle_state.get('vehicle_location_z', 0.0):.2f}",
                            f"{vehicle_state.get('yaw', 90.0):.1f}",
                            f"{vehicle_state.get('speed_kmh', 0.0):.2f}",
                            vlm_controller.current_action_text,
                            f"{action_value[0]:.2f}",
                            f"{action_value[1]:.2f}",
                            f"{vehicle_state.get('distance_to_obstacle', float('inf')):.1f}"
                        ])
                except Exception as e:
                    print(f"[ERROR] Failed to log trajectory at step {episode_step}: {e}")
                
                observation, reward, terminated, truncated, info = env.step(action_value)
                
                if hasattr(env, 'current_reward_components'):
                    rewards_history.append(env.current_reward_components)
                    if 'lane_change_reward' in env.current_reward_components:
                        lane_change_rewards.append(env.current_reward_components['lane_change_reward'])
                        if env.current_reward_components['lane_change_reward'] != 0:
                            lane_changes += 1
                speeds.append(info.get('speed_kmh', 0.0))
                accelerations.append(info.get('acceleration', 0.0))
                if info.get("pedestrian_detected", False):
                    metrics["pedestrian_detected_count"] += 1
                
                done = terminated or truncated
                episode_step += 1
                episode_reward += reward
                metrics["lane_changes"] = lane_changes
                
                if episode_step % 20 == 0:
                    print(f"Step {episode_step}: Action={vlm_controller.current_action_text} "
                          f"(throttle/brake: {action_value[0]:.2f}, steer: {action_value[1]:.2f}), "
                          f"Speed={info['speed_kmh']:.1f} km/h, "
                          f"Ped. detected={info['pedestrian_detected']}, "
                          f"Lane ID={info.get('current_lane_id', 'N/A')}, "
                          f"Lane Changes={lane_changes}")
            except Exception as e:
                print(f"[ERROR] Error during step {episode_step}: {e}")
                done = True
                metrics["stalled"] = True
        
        duration = time.time() - start_time
        metrics["steps"] = episode_step
        avg_speed = np.mean(speeds) if speeds else 0.0
        avg_acceleration = np.mean(accelerations) if accelerations else 0.0
        avg_lane_change_reward = np.mean(lane_change_rewards) if lane_change_rewards else 0.0
        
        if len(env.collision_hist) > 0:
            metrics["collision"] = True
            outcome = "COLLISION"
        elif env.successful_ep > metrics["success"]:
            metrics["success"] = True
            outcome = "SUCCESS"
        else:
            metrics["stalled"] = True
            outcome = "STALLED"
        
        try:
            with open(csv_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    episode + 1,
                    outcome,
                    episode_step,
                    f"{duration:.1f}",
                    f"{episode_reward:.2f}",
                    f"{avg_speed:.2f}",
                    f"{avg_acceleration:.2f}",
                    metrics["pedestrian_detected_count"],
                    metrics["lane_changes"],
                    f"{avg_lane_change_reward:.2f}"
                ])
            print(f"[INFO] Episode {episode + 1} metrics saved to {csv_path}")
        except Exception as e:
            print(f"[ERROR] Failed to write episode metrics to CSV: {e}")
        
        print(f"\nEpisode {episode + 1} completed:")
        print(f"  Outcome: {outcome}")
        print(f"  Steps: {episode_step}")
        print(f"  Duration: {duration:.1f} seconds")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Average Speed: {avg_speed:.2f} km/h")
        print(f"  Average Acceleration: {avg_acceleration:.2f} m/s²")
        print(f"  Pedestrian detections: {metrics['pedestrian_detected_count']}")
        print(f"  Lane Changes: {metrics['lane_changes']}")
        print(f"  Average Lane Change Reward: {avg_lane_change_reward:.2f}")
        print(f"  Final action: {vlm_controller.current_action_text} "
              f"(throttle/brake: {vlm_controller.current_action_value[0]:.2f}, "
              f"steer: {vlm_controller.current_action_value[1]:.2f})")
        print(f"  Justification: {vlm_controller.current_justification}")
        
        reward_plot_path = os.path.join(vlm_config["output_dir"], f"episode_{episode+1}_rewards.png")
        try:
            plot_rewards(rewards_history, reward_plot_path)
            print(f"[INFO] Reward plot saved to {reward_plot_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save reward plot: {e}")
    
    print("\nAll episodes completed!")

if __name__ == "__main__":
    print("[DEBUG] Script started.")
    main()