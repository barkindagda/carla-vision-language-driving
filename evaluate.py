import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from environment.carla_env import CarlaEnv
from Models.CLIP import config
import cv2

# Argument parsing
parser = argparse.ArgumentParser(description="Evaluate a trained CARLA agent with PPO")
parser.add_argument("--host", default="localhost", type=str, help="IP of the host server (default: 127.0.0.1)")
parser.add_argument("--port", default=2000, type=int, help="TCP port to listen to (default: 2000)")
parser.add_argument("--model", type=str, required=True, help="Path to the trained model (e.g., tensorboard/PPO_20250926_151823_idcarla_ppo/model_100000_steps.zip)")
parser.add_argument("--no_render", action="store_false", help="If True, render the environment")
parser.add_argument("--fps", type=int, default=10, help="FPS for rendering and video recording")
parser.add_argument("--no_record_video", action="store_false", help="If True, record video of the evaluation")
parser.add_argument("--config", type=str, default="carla_ppo", help="Config to use (default: carla_ppo)")
parser.add_argument("--device", type=str, default="cuda:1", help="cpu, cuda:0, cuda:1, cuda:2")
parser.add_argument("--density", choices=['empty', 'regular', 'dense'], default="regular", help="Traffic density")
parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to evaluate")
args = vars(parser.parse_args())

# Load configuration
CONFIG = config.set_config(args["config"])
CONFIG.algorithm_params.device = args["device"]

# Map density to TRAFFIC and MAX_TRAFFIC settings
if args["density"] == "empty":
    TRAFFIC = False
    MAX_TRAFFIC = 0
elif args["density"] == "regular":
    TRAFFIC = True
    MAX_TRAFFIC = 30
else:  # dense
    TRAFFIC = True
    MAX_TRAFFIC = 50

# Override TRAFFIC and MAX_TRAFFIC in carla_env
import environment.carla_env
environment.carla_env.TRAFFIC = TRAFFIC
environment.carla_env.MAX_TRAFFIC = MAX_TRAFFIC

# Initialize environment
try:
    env = DummyVecEnv([lambda: CarlaEnv(
        render_mode=None if args["no_render"] else "human",
        vlm_frames=0  # No VLM frames needed for evaluation
    )])
except Exception as e:
    print(f"Failed to initialize environment: {e}")
    raise

# Load the trained model
try:
    model = PPO.load(
        args["model"],
        env=env,
        device=args["device"]
    )
except Exception as e:
    print(f"Failed to load as PPO: {e}")
    try:
        from Models.CLIP.vlm_rewarded_ppo import VLMRewardedPPO
        model = VLMRewardedPPO.load(
            args["model"],
            env=env,
            config=CONFIG,
            device=args["device"],
            load_clip=False
        )
        model.inference_only = True
    except Exception as e2:
        print(f"Failed to load as VLMRewardedPPO: {e2}")
        raise
print("Model loaded successfully...")

# Evaluation function
def run_eval(env, model, model_path, record_video=False, episodes=10):
    model_name = os.path.basename(model_path)
    log_path = os.path.join(os.path.dirname(model_path), f'eval_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(log_path, exist_ok=True)
    csv_path = os.path.join(log_path, model_name.replace(".zip", "_eval.csv"))
    video_path = os.path.join(log_path, model_name.replace(".zip", "_eval.avi"))
    model_id = f"{model_path.split('/')[-2]}-{model_name.split('_')[-2]}"

    # Define CSV columns based on CarlaEnv info
    columns = [
        "model_id", "episode", "step", "throttle", "steer", "vehicle_location_x", "vehicle_location_y",
        "reward", "distance_to_goal", "pedestrian_distance", "speed_ms",
        "safety_reward", "progress_reward", "smoothness_reward", "collision_penalty",
        "done_reason", "routes_completed"
    ]
    df = pd.DataFrame(columns=columns)

    # Video recording setup
    if record_video:
        frame_shape = (384, 384, 3)  # Based on IM_WIDTH, IM_HEIGHT from carla_env.py
        print(f"Recording video to {video_path} ({frame_shape[0]}x{frame_shape[1]}x{frame_shape[2]}@{args['fps']}fps)")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(video_path, fourcc, args["fps"], (frame_shape[1], frame_shape[0]))

    episode_idx = 0
    try:
        state = env.reset()[0]  # Get observation from reset
    except Exception as e:
        print(f"Environment reset failed: {e}")
        raise

    routes_completed = 0

    while episode_idx < episodes:
        print(f"Episode {episode_idx + 1}/{episodes}")
        step_count = 0
        done = False
        episode_reward = 0

        while not done:
            try:
                # Predict action (deterministic for evaluation)
                action, _ = model.predict(state, deterministic=True)
                print(f"Step {step_count}: action={action}")  # Debug action
                # Handle 4-element step return (fallback for older Gym API)
                step_result = env.step(action)
                if len(step_result) == 5:
                    next_state, reward, done, truncated, info = step_result
                else:  # len(step_result) == 4
                    next_state, reward, done, info = step_result
                    truncated = False  # Assume no truncation if not provided
                episode_reward += reward[0]  # Single environment

                # Debug navigation
                route_ind = getattr(env.envs[0], 'route_ind', None)
                route = getattr(env.envs[0], 'route', None)
                if route_ind is not None and route is not None:
                    print(f"Step {step_count}: route_ind={route_ind}, current_waypoint={route[route_ind]}")

                # Extract vehicle control and location
                vehicle = env.envs[0].vehicle
                throttle = vehicle.get_control().throttle
                steer = vehicle.get_control().steer
                location = vehicle.get_location()

                # Extract info metrics
                info_dict = info[0]  # Single environment
                done_reason = ""
                if done or truncated:
                    if len(info_dict.get("collision_hist", [])) > 0:
                        done_reason = "collision"
                        routes_completed = 0
                    elif info_dict.get("distance_to_goal", float("inf")) <= 2:
                        done_reason = "success"
                        routes_completed += 1
                    elif step_count >= 1000:  # Match carla_env.py max timesteps
                        done_reason = "timeout"
                        routes_completed = 0

                # Log data
                new_row = pd.DataFrame([[
                    model_id, episode_idx, step_count, throttle, steer,
                    location.x, location.y, reward[0],
                    info_dict.get("distance_to_goal", 0),
                    info_dict.get("pedestrian_distance", 0),
                    info_dict.get("speed_ms", 0),
                    info_dict.get("safety_reward", 0),
                    info_dict.get("progress_reward", 0),
                    info_dict.get("smoothness_reward", 0),
                    info_dict.get("collision_penalty", 0),
                    done_reason, routes_completed
                ]], columns=columns)
                df = pd.concat([df, new_row], ignore_index=True)

                # Video recording
                if record_video:
                    frame = state[0].transpose(1, 2, 0)  # Convert CHW to HWC
                    frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
                    video_writer.write(frame)

                state = next_state
                step_count += 1

                # Timeout check
                if step_count >= 1000:
                    done = True
                    routes_completed = 0

            except Exception as e:
                print(f"Step {step_count} failed: {e}")
                done = True
                done_reason = f"error: {str(e)}"
                break

        print(f"Episode {episode_idx + 1} completed. Total reward: {episode_reward:.2f}, Reason: {done_reason}")
        try:
            state = env.reset()[0]
        except Exception as e:
            print(f"Environment reset failed: {e}")
            break
        episode_idx += 1

    # Save CSV and release video
    df.to_csv(csv_path, index=False)
    print(f"Evaluation results saved to {csv_path}")
    if record_video:
        video_writer.release()
        print(f"Video saved to {video_path}")

    # Summary statistics
    summary = {
        "model_id": model_id,
        "total_episodes": episodes,
        "success_rate": routes_completed / episodes,
        "avg_reward": df["reward"].mean(),
        "avg_distance_to_goal": df["distance_to_goal"].mean(),
        "avg_speed_ms": df["speed_ms"].mean(),
        "collisions": len(df[df["done_reason"] == "collision"]),
        "timeouts": len(df[df["done_reason"] == "timeout"])
    }
    print("Evaluation Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")

    return df

if __name__ == "__main__":
    try:
        df = run_eval(
            env=env,
            model=model,
            model_path=args["model"],
            record_video=args["no_record_video"],
            episodes=args["episodes"]
        )
    finally:
        try:
            env.close()
        except Exception as e:
            print(f"Environment close failed: {e}")