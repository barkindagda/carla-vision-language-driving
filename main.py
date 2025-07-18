import os
import sys
import time
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import wandb
from datetime import datetime
import random
import glob

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.utils import get_linear_fn
from PIL import Image
from Models.resnet18_attention import ResnetAttention

# Assuming carla_env_w_symbolic.py is the correct, intended environment file
from environment.carla_env_w_symbolic import CarlaEnv as CarlaEnvironment
from Models.vlm_controller_symbolic import VLMController

try:
    from Models.feature_extractors.clip_image_encoder import CLIPAndSegFeatureExtractor
    print("Successfully imported CLIPAndSegFeatureExtractor.")
except ImportError:
    print("Warning: Could not import CLIPAndSegFeatureExtractor.")
    CLIPAndSegFeatureExtractor = None

PROJECT_ROOT = os.path.expanduser("~/BARKIN/carla-vision-language-driving")
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
CUSTOM_EXTRACTORS_DIR = os.path.join(PROJECT_ROOT, "Models", "feature_extractors")
if CUSTOM_EXTRACTORS_DIR not in sys.path:
    sys.path.append(CUSTOM_EXTRACTORS_DIR)

# --- Configuration ---
EXPERIMENT_DATE_TIME = datetime.now().strftime('%Y%m%d_%H%M%S')
EXPERIMENT_NAME_BASE = "carla_vlm_ppo_s1"
TOTAL_TIMESTEPS = 550000
CHECKPOINT_INTERVAL = 100
SEED = 42
EFFICINCY_PRIORITY = False
EXTRACTOR_CHOICE = "PEDESTRIAN_ATTENTIVE"

# --- Feature Extractor Setup ---
if EXTRACTOR_CHOICE == "CLIP_SEG_SPATIAL":
    FEATURE_EXTRACTOR_SETUP = {
        "type": "CLIP_SEG_SPATIAL", "features_dim": 256,
        "clip_vision_model_name": "openai/clip-vit-base-patch32", "clipseg_model_name": "CIDAS/clipseg-rd64-refined",
        "pedestrian_prompt": "a pedestrian",
    }
    EXTRACTOR_HEATMAP_SAVE_SUBDIR = "clip_heatmaps"
elif EXTRACTOR_CHOICE == "PEDESTRIAN_ATTENTIVE":
    FEATURE_EXTRACTOR_SETUP = {
        "type": "PEDESTRIAN_ATTENTIVE", "features_dim": 64,
        "resnet18_unet_weights_path": os.path.join(CUSTOM_EXTRACTORS_DIR, "resne18unet_weights.pt")
    }
    EXTRACTOR_HEATMAP_SAVE_SUBDIR = None
else:
    FEATURE_EXTRACTOR_SETUP = {"type": "DEFAULT_CNN", "features_dim": 512}
    EXTRACTOR_HEATMAP_SAVE_SUBDIR = None

# --- PPO & Environment Hyperparameters ---
EXPERIMENT_NAME = f"{EXPERIMENT_NAME_BASE}_{FEATURE_EXTRACTOR_SETUP['type']}_{EXPERIMENT_DATE_TIME}"
INITIAL_LR = 0.0003
FINAL_LR = 5e-5
linear_schedule = get_linear_fn(INITIAL_LR, FINAL_LR, 1.0)
PPO_CONFIG = {
    "learning_rate": linear_schedule, "n_steps": 1024, "batch_size": 256, "n_epochs": 10, "gamma": 0.99,
    "gae_lambda": 0.95, "clip_range": 0.1, "ent_coef": 0.01, "vf_coef": 0.25, "max_grad_norm": 0.5, "verbose": 0,
    "net_arch_mlp_head": dict(pi=[256, 128], vf=[512, 256]),
}
ENV_RENDER_MODE = None
ENV_USE_VLM_WEIGHTS = True
ENV_NORMALIZE_REWARDS = True
ENV_USE_SYMBOLIC_RULES = True # Flag for symbolic rules

# --- VLM Configuration ---
VLM_VERBOSE = True
VLM_MODEL_NAME = 'DAMO-NLP-SG/VideoLLaMA3-2B-Image'
VLM_UPDATE_FREQUENCY = 5
VLM_FRAMES_NEEDED = 3
VLM_MAX_TOKENS = 120

# --- Directory & Path Configuration ---
OUTPUT_DIR_BASE = "./experiments"
OUTPUT_DIR = os.path.join(OUTPUT_DIR_BASE, EXPERIMENT_NAME)
VLM_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "vlm_outputs")
FRAMES_DIR = os.path.join(VLM_OUTPUT_DIR, "frames")
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "models")
LOG_DIR = os.path.join(OUTPUT_DIR, "sb3_logs")
CHECKPOINT_PATH = os.path.join(MODEL_SAVE_PATH, "latest_checkpoint.zip")

# --- WandB Configuration ---
WANDB_PROJECT = "VLM-attention-PPO"
WANDB_ENTITY = None

# --- Helper Functions ---

def read_vlm_weights_from_log(json_file_path: str):
    """Reads the latest VLM weight decision from the log file."""
    try:
        if os.path.exists(json_file_path):
            with open(json_file_path, 'r') as f:
                log_data = json.load(f)
            if log_data and "decisions" in log_data and isinstance(log_data["decisions"], list):
                for decision in reversed(log_data["decisions"]):
                    if isinstance(decision, dict) and decision.get("task_type") == "weights" and "weights" in decision:
                        return decision["weights"]
        return None
    except Exception:
        return None

def get_memory_stats():
    """Gets current CUDA memory stats."""
    mem_stats = {}
    if not torch.cuda.is_available():
        return mem_stats
    for i in range(torch.cuda.device_count()):
        mem_stats[f"cuda:{i}_allocated_MB"] = torch.cuda.memory_allocated(i) / 1024**2
        mem_stats[f"cuda:{i}_reserved_MB"] = torch.cuda.memory_reserved(i) / 1024**2
    return mem_stats

def save_env_state(env, checkpoint_path):
    """Saves critical state from the environment to a pickle file."""
    if hasattr(env, 'get_state'):
        state = env.get_state()
        with open(checkpoint_path + '.env.pkl', 'wb') as f:
            pickle.dump(state, f)

def load_env_state(env, checkpoint_path):
    """Loads state into the environment from a pickle file."""
    env_state_path = checkpoint_path + '.env.pkl'
    if os.path.exists(env_state_path) and hasattr(env, 'set_state'):
        try:
            with open(env_state_path, 'rb') as f:
                state = pickle.load(f)
            env.set_state(state)
            print("Successfully loaded environment state.")
        except Exception as e:
            print(f"Warning: Could not load environment state: {e}")

# --- Custom Callback ---

class VLMSimpleCallback(BaseCallback):
    """A custom callback for logging, and for saving and loading training state."""
    def __init__(self, vlm_decision_log_path_getter, verbose=0):
        super().__init__(verbose)
        self.vlm_decision_log_path_getter = vlm_decision_log_path_getter
        self.episode_rewards_list = []
        self.episode_lengths_list = []
        self.current_episode_reward_sum = 0
        self.current_episode_step_count = 0
        self.last_vlm_weights_logged_step = -1
        self.last_cache_clear_step = 0
        self.last_checkpoint_step = 0

    def _on_step(self) -> bool:
        # Log step reward and action
        reward = self.locals['rewards'][0]
        self.current_episode_reward_sum += reward
        self.current_episode_step_count += 1
        if wandb.run:
            wandb.log({"Step/reward": float(reward)}, step=self.num_timesteps)
            try:
                action_val = float(self.locals['clipped_actions'][0])
                wandb.log({"Step/action_value": action_val}, step=self.num_timesteps)
            except (TypeError, IndexError):
                pass
            
        # Log VLM weights periodically
        if wandb.run and ENV_USE_VLM_WEIGHTS and (self.num_timesteps % 10 == 0) and (self.num_timesteps > self.last_vlm_weights_logged_step):
            vlm_log_file = self.vlm_decision_log_path_getter()
            if vlm_log_file:
                vlm_weights_data = read_vlm_weights_from_log(vlm_log_file)
                if vlm_weights_data:
                    wandb.log({f"VLM_Weights/{k}": float(v) for k, v in vlm_weights_data.items()}, step=self.num_timesteps)
                    self.last_vlm_weights_logged_step = self.num_timesteps
        
        # Clear CUDA cache and log memory periodically to prevent fragmentation
        if torch.cuda.is_available() and (self.num_timesteps - self.last_cache_clear_step >= 100):
            torch.cuda.empty_cache()
            if wandb.run:
                mem_stats = get_memory_stats()
                wandb.log({f"Memory/{k}": v for k, v in mem_stats.items()}, step=self.num_timesteps)
            self.last_cache_clear_step = self.num_timesteps

        # Save checkpoint periodically
        if self.num_timesteps > 0 and self.num_timesteps - self.last_checkpoint_step >= CHECKPOINT_INTERVAL:
            print(f"\n💾 Checkpointing at step {self.num_timesteps}...")
            self.save_training_state(CHECKPOINT_PATH)
            self.last_checkpoint_step = self.num_timesteps

        # Handle episode end
        if self.locals['dones'][0]:
            ep_rew = self.current_episode_reward_sum
            ep_len = self.current_episode_step_count
            avg_rew_per_step = ep_rew / max(1, ep_len)
            self.episode_rewards_list.append(ep_rew)
            self.episode_lengths_list.append(ep_len)
            print(f"\n🏁 Ep {len(self.episode_rewards_list)} Fin: Rew={ep_rew:.2f}, Len={ep_len}, AvgRew/Step={avg_rew_per_step:.3f}")
            if wandb.run:
                wandb.log({
                    "Episode/reward": ep_rew,
                    "Episode/length": ep_len,
                    "Episode/reward_per_step": avg_rew_per_step,
                    "Episode/count": len(self.episode_rewards_list)
                }, step=self.num_timesteps)
            self.current_episode_reward_sum = 0
            self.current_episode_step_count = 0
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return True

    def save_training_state(self, checkpoint_path):
        self.model.save(checkpoint_path)
        state = {
            'episode_rewards_list': self.episode_rewards_list, 'episode_lengths_list': self.episode_lengths_list,
            'current_episode_reward_sum': self.current_episode_reward_sum, 'current_episode_step_count': self.current_episode_step_count,
            'last_vlm_weights_logged_step': self.last_vlm_weights_logged_step, 'last_cache_clear_step': self.last_cache_clear_step,
            'last_checkpoint_step': self.num_timesteps
        }
        with open(checkpoint_path + '.cb.pkl', 'wb') as f:
            pickle.dump(state, f)
        save_env_state(self.training_env.envs[0], checkpoint_path)
        print(f"✅ Checkpoint fully saved: {checkpoint_path}")

    def load_training_state(self, checkpoint_path):
        cb_path = checkpoint_path + '.cb.pkl'
        if not os.path.exists(cb_path):
            print("Warning: No callback state file found. Starting fresh.")
            return
        with open(cb_path, 'rb') as f:
            state = pickle.load(f)
        self.episode_rewards_list = state.get('episode_rewards_list', [])
        self.episode_lengths_list = state.get('episode_lengths_list', [])
        self.current_episode_reward_sum = state.get('current_episode_reward_sum', 0)
        self.current_episode_step_count = state.get('current_episode_step_count', 0)
        self.last_vlm_weights_logged_step = state.get('last_vlm_weights_logged_step', -1)
        self.last_cache_clear_step = state.get('last_cache_clear_step', 0)
        self.last_checkpoint_step = state.get('last_checkpoint_step', 0)
        print(f"✅ Callback state loaded. Resuming from step {self.last_checkpoint_step}.")

# --- Setup and Main Execution ---

def setup_directories():
    """Ensures all necessary directories for the experiment exist."""
    print("--- Setting up experiment directories ---")
    dirs_to_create = [OUTPUT_DIR, VLM_OUTPUT_DIR, FRAMES_DIR, MODEL_SAVE_PATH, LOG_DIR]
    if EXTRACTOR_HEATMAP_SAVE_SUBDIR:
        dirs_to_create.append(os.path.join(OUTPUT_DIR, EXTRACTOR_HEATMAP_SAVE_SUBDIR))
    for directory in dirs_to_create:
        os.makedirs(directory, exist_ok=True)
        print(f"  - Directory ensured: {directory}")
    print("--- Directories setup complete ---\n")

vlm_controller_instance = None
def get_vlm_log_file_path_for_callback():
    return vlm_controller_instance.log_file if vlm_controller_instance else None

def main():
    """Main function to set up and run the training process."""
    global vlm_controller_instance
    print(f"🚗 CARLA VLM-Enhanced PPO Training\n" + "="*70)
    setup_directories()

    wandb_config = {k: v for k, v in globals().items() if isinstance(v, (str, int, float, bool, dict)) and k.isupper()}
    wandb.init(project=WANDB_PROJECT, name=EXPERIMENT_NAME, entity=WANDB_ENTITY,
               config=wandb_config, sync_tensorboard=True, monitor_gym=True, save_code=True)

    print("🧠 Creating VLM controller...")
    vlm_controller_instance = VLMController(
        model_name=VLM_MODEL_NAME, update_frequency=VLM_UPDATE_FREQUENCY, frames_needed=VLM_FRAMES_NEEDED,
        output_dir=VLM_OUTPUT_DIR, max_new_tokens=VLM_MAX_TOKENS, verbose=VLM_VERBOSE,
        efficiency_priority=EFFICINCY_PRIORITY
    )
    print(f"✅ VLM controller created. Log file: {vlm_controller_instance.log_file}")
    
    print("🌍 Creating CARLA environment...")
    def make_env_fn():
        env = CarlaEnvironment(
            render_mode=ENV_RENDER_MODE,
            use_vlm_weights=ENV_USE_VLM_WEIGHTS,
            normalize_rewards=ENV_NORMALIZE_REWARDS,
            use_symbolic_rewards=ENV_USE_SYMBOLIC_RULES
        )
        env.seed(SEED + random.randint(0, 100))
        env.frame_save_dir = FRAMES_DIR
        env.vlm_controller = vlm_controller_instance
        return env
    vec_env = DummyVecEnv([make_env_fn])
    vec_env = VecMonitor(vec_env, filename=os.path.join(LOG_DIR, "monitor.csv"))
    print("✅ CARLA VecEnv created.")

    policy_kwargs = {}
    if FEATURE_EXTRACTOR_SETUP["type"] == "PEDESTRIAN_ATTENTIVE":
        print("🖼️ Using Custom ResnetAttention Extractor.")
        policy_kwargs = dict(
            features_extractor_class=ResnetAttention,
            features_extractor_kwargs=dict(
                features_dim=FEATURE_EXTRACTOR_SETUP["features_dim"],
                resnet18_unet_weights_path=FEATURE_EXTRACTOR_SETUP["resnet18_unet_weights_path"]
            ),
            net_arch=PPO_CONFIG.get("net_arch_mlp_head"), activation_fn=nn.ReLU
        )
    else:
        print("🖼️ Using SB3 Default CNN.")
        policy_kwargs = {"net_arch": PPO_CONFIG.get("net_arch_mlp_head")}

    model_hyperparams = {k: v for k, v in PPO_CONFIG.items() if k != "net_arch_mlp_head"}
    
    callback = VLMSimpleCallback(vlm_decision_log_path_getter=get_vlm_log_file_path_for_callback, verbose=1)
    
    if os.path.exists(CHECKPOINT_PATH):
        print(f"✅ Resuming training from checkpoint: {CHECKPOINT_PATH}")
        model = PPO.load(CHECKPOINT_PATH, env=vec_env, custom_objects={'learning_rate': linear_schedule})
        callback.load_training_state(CHECKPOINT_PATH)
        load_env_state(vec_env.envs[0], CHECKPOINT_PATH)
    else:
        print("🤖 Starting new training session.")
        model = PPO("CnnPolicy", vec_env, tensorboard_log=LOG_DIR, policy_kwargs=policy_kwargs,
                    seed=SEED, **model_hyperparams)

    callback.model = model

    print(f"\n🚀 Starting PPO training from step {model.num_timesteps} for {TOTAL_TIMESTEPS:,} total timesteps...")
    start_time = time.time()
    
    # --- Main Training Loop with OOM Recovery ---
    while model.num_timesteps < TOTAL_TIMESTEPS:
        try:
            model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback,
                        reset_num_timesteps=False, tb_log_name=EXPERIMENT_NAME_BASE)
            break
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print("\n" + "="*80)
                print("⚠️ CUDA OOM detected! Attempting to recover from last checkpoint...")
                print("="*80 + "\n")
                torch.cuda.empty_cache()
                
                if os.path.exists(CHECKPOINT_PATH):
                    model = PPO.load(CHECKPOINT_PATH, env=vec_env, custom_objects={'learning_rate': linear_schedule})
                    callback.load_training_state(CHECKPOINT_PATH)
                    load_env_state(vec_env.envs[0], CHECKPOINT_PATH)
                    callback.model = model
                    print(f"✅ Successfully recovered. Resuming training from step ~{model.num_timesteps}.")
                else:
                    print("❌ No checkpoint found to recover from. Training cannot continue.")
                    raise
            else:
                print(f"An unexpected RuntimeError occurred: {e}")
                raise
    
    # --- Finalization ---
    final_model_path = os.path.join(MODEL_SAVE_PATH, "final_model.zip")
    model.save(final_model_path)
    print(f"\n💾 Final model saved: {final_model_path}")
    if wandb.run and hasattr(wandb, 'Artifact'):
        artifact = wandb.Artifact(f'{EXPERIMENT_NAME}-model', type='model')
        artifact.add_file(final_model_path)
        wandb.log_artifact(artifact)
        
    total_training_time = time.time() - start_time
    print(f"\n🎉 Training Session Concluded. Time: {total_training_time/3600:.2f}h")
    if callback.episode_rewards_list:
        mean_rew = np.mean(callback.episode_rewards_list[-100:]) if callback.episode_rewards_list else 0
        print(f"  - Mean Reward (last 100 eps): {mean_rew:.2f}")
        print(f"  - Best Ep Reward: {max(callback.episode_rewards_list):.2f}")
        
    vec_env.close()
    print("🗑️ CARLA VecEnv closed.")
    if wandb.run:
        wandb.run.finish()
    print("👌 WandB run finished.")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    main()