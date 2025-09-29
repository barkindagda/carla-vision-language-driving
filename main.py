import warnings
import os
from datetime import datetime

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
from Models.CLIP import config
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from Models.CLIP.vlm_rewarded_ppo import VLMRewardedPPO
from environment.carla_env import CarlaEnv
# --- CHANGE 1: Import the new callback ---
from Models.CLIP.utils import HParamCallback, TensorboardCallback, PostRolloutLogCallback, write_json

parser = argparse.ArgumentParser(description="Trains a CARLA agent with VLMRewardedPPO")
parser.add_argument("--host", default="localhost", type=str, help="IP of the host server (default: 127.0.0.1)")
parser.add_argument("--port", default=2000, type=int, help="TCP port to listen to (default: 2000)")
parser.add_argument("--total_timesteps", type=int, default=100_000, help="Total timesteps to train for")
parser.add_argument("--start_carla", action="store_true", help="If True, start a CARLA server")
parser.add_argument("--no_render", action="store_false", help="If True, render the environment")
parser.add_argument("--num_checkpoints", type=int, default=5, help="Checkpoint number")
parser.add_argument("--log_dir", type=str, default="tensorboard", help="Directory to save logs")
parser.add_argument("--device", type=str, default="cuda:0", help="cpu, cuda:0, cuda:1, cuda:2")
parser.add_argument("--config", type=str, default="carla_ppo", help="Config to use (default: carla_ppo)")

args = vars(parser.parse_args())
CONFIG = config.set_config(args["config"])
CONFIG.algorithm_params.device = args["device"]

# Ensure segment_length matches vlm_frames
CONFIG.vlm_params.segment_length = 3  # Match CarlaEnv vlm_frames

os.makedirs(args["log_dir"], exist_ok=True)

# Initialize environment
env = DummyVecEnv([lambda: CarlaEnv(
    render_mode=None if args["no_render"] else "human",
    vlm_frames=CONFIG.vlm_params.segment_length  # Align with segment_length
)])

# Initialize model
model = VLMRewardedPPO(
    env=env,
    config=CONFIG,
    inference_only=False
)

# Set up logging
model_suffix = "{}_id{}".format(datetime.now().strftime("%Y%m%d_%H%M%S"), args['config'])
model_name = f'{model.__class__.__name__}_{model_suffix}'
model_dir = os.path.join(args["log_dir"], model_name)
new_logger = configure(model_dir, ["stdout", "csv", "tensorboard"])
model.set_logger(new_logger)
write_json(CONFIG, os.path.join(model_dir, 'config.json'))

# Train model
model.learn(
    total_timesteps=args["total_timesteps"],
    callback=[
        HParamCallback(CONFIG),
        TensorboardCallback(),
        PostRolloutLogCallback(), # --- CHANGE 2: Add the PostRolloutLogCallback here ---
        CheckpointCallback(
            save_freq=max(1, args["total_timesteps"] // args["num_checkpoints"]),
            save_path=model_dir,
            name_prefix="model"
        )
    ],
    reset_num_timesteps=False
)