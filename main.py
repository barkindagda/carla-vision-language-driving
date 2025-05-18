import os
import time
import numpy as np
from environment.carla_env import CarlaEnv
from Models.vlm_controller import VLMController  # Make sure this matches your actual filename
from utils.plot import plot_rewards

def main():
    """Main function to run CARLA with VLM control."""
    # Configuration
    vlm_config = {
        "model_name": "DAMO-NLP-SG/VideoLLaMA3-2B-Image",
        "update_frequency": 2,  # Update more frequently
        "frames_needed": 3,  # Use 3 frames for each decision
        "output_dir": "./vlm_outputs",
        "max_new_tokens": 512,
        "verbose": True
    }
    
    # Create output directory
    os.makedirs(vlm_config["output_dir"], exist_ok=True)
    
    # Initialize environment with appropriate frame buffer size
    env = CarlaEnv(vlm_frames=vlm_config["frames_needed"])
    
    # Initialize VLM controller and connect to environment
    vlm_controller = VLMController(**vlm_config)
    env.vlm_controller = vlm_controller  # This is crucial
    
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
        rewards_history = []  # Store all reward components for plotting
        
        # Initial logging
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
            # Get VLM decision if enough frames are available
            vlm_controller.process_if_needed(env)
            
            # Take step using VLM-determined action value directly
            action_value = vlm_controller.current_action_value
            observation, reward, terminated, truncated, info = env.step(action_value)
            
            # Capture reward components if available
            if hasattr(env, 'current_reward_components'):
                rewards_history.append(env.current_reward_components)
            
            done = terminated or truncated
            episode_step += 1
            episode_reward += reward
            
            # Track metrics
            if info.get("pedestrian_detected", False):
                metrics["pedestrian_detected_count"] += 1
            
            # Print progress every 20 steps
            if episode_step % 20 == 0:
                print(f"Step {episode_step}: Action={vlm_controller.current_action_text} ({action_value:.2f}), "
                      f"Speed={info['speed_kmh']:.1f} km/h, "
                      f"Ped. detected={info['pedestrian_detected']}")
        
        # Episode summary
        duration = time.time() - start_time
        metrics["steps"] = episode_step
        
        # Determine episode outcome
        if len(env.collision_hist) > 0:
            metrics["collision"] = True
            outcome = "COLLISION"
        elif env.successful_ep > metrics["success"]:
            metrics["success"] = True
            outcome = "SUCCESS"
        else:
            metrics["stalled"] = True
            outcome = "STALLED"
        
        # Log episode results
        print(f"\nEpisode {episode + 1} completed:")
        print(f"  Outcome: {outcome}")
        print(f"  Steps: {episode_step}")
        print(f"  Duration: {duration:.1f} seconds")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Pedestrian detections: {metrics['pedestrian_detected_count']}")
        print(f"  Final action: {vlm_controller.current_action_text} ({vlm_controller.current_action_value:.2f})")
        print(f"  Justification: {vlm_controller.current_justification}")
        
        # Plot rewards
        reward_plot_path = os.path.join(vlm_config["output_dir"], f"episode_{episode+1}_rewards.png")
        plot_rewards(rewards_history, reward_plot_path)
    
    print("\nAll episodes completed!")

if __name__ == "__main__":
    main()