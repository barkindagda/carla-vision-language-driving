import os
import time
import numpy as np
import csv
from environment.carla_env import CarlaEnv
#from Models.vlm_controller import VLMController
#from Models.vlm_qwen import VLMController
from Models.vlm_llava import VLMController
from utils.plot import plot_rewards

def main():
    """Main function to run CARLA with VLM control."""
    # Configuration
    vlm_config = {
        "model_name": "llava-hf/llava-onevision-qwen2-7b-ov-hf", #Qwen/Qwen2.5-VL-3B-Instruct  DAMO-NLP-SG/VideoLLaMA3-2B-Image  llava-hf/llava-onevision-qwen2-7b-ov-hf
        "update_frequency": 3,
        "frames_needed": 3,
        "output_dir": "./vlm_outputs",
        "max_new_tokens": 512,
        "verbose": True
    }
    
    # Create output directory
    os.makedirs(vlm_config["output_dir"], exist_ok=True)

    # Initialize CSV file path
    csv_path = os.path.join(vlm_config["output_dir"], "episode_metrics.csv")
    csv_headers = [
        "Episode", "Outcome", "Steps", "Duration (s)", "Total Reward",
        "Avg Speed (km/h)", "Avg Acceleration (m/s²)", "Pedestrian Detections"
    ]
    
    # Create CSV file (always overwrite headers to ensure file exists)
    try:
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_headers)
        print(f"[INFO] CSV file initialized: {os.path.abspath(csv_path)}")
    except Exception as e:
        print(f"[ERROR] Failed to create CSV file at {csv_path}: {e}")
        return  # Stop execution if we can't log results
    
    # Initialize environment
    env = CarlaEnv(vlm_frames=vlm_config["frames_needed"])
    
    # Initialize VLM controller
    vlm_controller = VLMController(**vlm_config)
    env.vlm_controller = vlm_controller
    
    # Run episodes
    num_episodes = 1
    for episode in range(num_episodes):
        print(f"\n{'=' * 50}")
        print(f"Starting Episode {episode + 1}/{num_episodes}")
        print(f"{'=' * 50}")
        
        # Reset environment
        observation, info = env.reset()
        done = False
        episode_step = 0
        episode_reward = 0
        rewards_history = []
        speeds = []
        accelerations = []
        
        start_time = time.time()
        metrics = {
            "steps": 0,
            "collision": False,
            "success": False,
            "stalled": False,
            "pedestrian_detected_count": 0
        }
        
        # Episode loop
        while not done:
            vlm_controller.process_if_needed(env)
            action_value = vlm_controller.current_action_value
            observation, reward, terminated, truncated, info = env.step(action_value)
            
            # Collect data
            if hasattr(env, 'current_reward_components'):
                rewards_history.append(env.current_reward_components)
            speeds.append(info.get('speed_kmh', 0.0))
            accelerations.append(info.get('acceleration', 0.0))
            if info.get("pedestrian_detected", False):
                metrics["pedestrian_detected_count"] += 1
            
            done = terminated or truncated
            episode_step += 1
            episode_reward += reward
            
            if episode_step % 20 == 0:
                print(f"Step {episode_step}: Action={vlm_controller.current_action_text} ({action_value:.2f}), "
                      f"Speed={info['speed_kmh']:.1f} km/h, "
                      f"Ped. detected={info['pedestrian_detected']}")
        
        # Episode summary
        duration = time.time() - start_time
        metrics["steps"] = episode_step
        
        avg_speed = np.mean(speeds) if speeds else 0.0
        avg_acceleration = np.mean(accelerations) if accelerations else 0.0
        
        if len(env.collision_hist) > 0:
            metrics["collision"] = True
            outcome = "COLLISION"
        elif env.successful_ep > metrics["success"]:
            metrics["success"] = True
            outcome = "SUCCESS"
        else:
            metrics["stalled"] = True
            outcome = "STALLED"
        
        # Append episode data to CSV
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
                    metrics["pedestrian_detected_count"]
                ])
            print(f"[INFO] Episode {episode + 1} metrics saved to {csv_path}")
        except Exception as e:
            print(f"[ERROR] Failed to write episode metrics to CSV: {e}")
        
        # Print episode results
        print(f"\nEpisode {episode + 1} completed:")
        print(f"  Outcome: {outcome}")
        print(f"  Steps: {episode_step}")
        print(f"  Duration: {duration:.1f} seconds")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Average Speed: {avg_speed:.2f} km/h")
        print(f"  Average Acceleration: {avg_acceleration:.2f} m/s²")
        print(f"  Pedestrian detections: {metrics['pedestrian_detected_count']}")
        print(f"  Final action: {vlm_controller.current_action_text} ({vlm_controller.current_action_value:.2f})")
        print(f"  Justification: {vlm_controller.current_justification}")
        
        reward_plot_path = os.path.join(vlm_config["output_dir"], f"episode_{episode+1}_rewards.png")
        plot_rewards(rewards_history, reward_plot_path)
    
    print("\nAll episodes completed!")

if __name__ == "__main__":
    print("[DEBUG] Script started.")
    main()