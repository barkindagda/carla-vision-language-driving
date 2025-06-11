import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import wandb
from datetime import datetime
import random
import glob

# Stable Baselines3 imports
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

# PIL and torchvision for image manipulation
from PIL import Image
from torchvision import transforms as T_vis
from Models.resnet18_attention import ResnetAttention

# --- Project-specific imports ---
PROJECT_ROOT = os.path.expanduser("~/BARKIN/carla-vision-language-driving")
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

CUSTOM_EXTRACTORS_DIR = os.path.join(PROJECT_ROOT, "Models", "feature_extractors")
if CUSTOM_EXTRACTORS_DIR not in sys.path:
    sys.path.append(CUSTOM_EXTRACTORS_DIR)

from environment.carla_env_w_symbolic import CarlaEnv as CarlaEnvironment
from Models.vlm_controller_symbolic import VLMController

# --- IMPORT THE FEATURE EXTRACTORS ---
try:
    # Kept your original import path
    from Models.feature_extractors.clip_image_encoder import CLIPAndSegFeatureExtractor
    print("Successfully imported CLIPAndSegFeatureExtractor.")
except ImportError:
    print("Warning: Could not import CLIPAndSegFeatureExtractor.")
    CLIPAndSegFeatureExtractor = None

# ============================================================================
# CONFIGURATION
# ============================================================================
EXPERIMENT_DATE_TIME = datetime.now().strftime('%Y%m%d_%H%M%S')
EXPERIMENT_NAME_BASE = "carla_vlm_ppo_s1"
TOTAL_TIMESTEPS = 250000
CHECKPOINT_INTERVAL = 20000
SEED = 42

# --- CHOOSE YOUR FEATURE EXTRACTOR ---
# Options: "CLIP_SEG_SPATIAL", "PEDESTRIAN_ATTENTIVE", "DEFAULT_CNN"
EXTRACTOR_CHOICE = "PEDESTRIAN_ATTENTIVE"

# --- Define configurations for each extractor type ---
if EXTRACTOR_CHOICE == "CLIP_SEG_SPATIAL":
    FEATURE_EXTRACTOR_SETUP = {
        "type": "CLIP_SEG_SPATIAL",
        "features_dim": 256,
        "clip_vision_model_name": "openai/clip-vit-base-patch32",
        "clipseg_model_name": "CIDAS/clipseg-rd64-refined",
        "pedestrian_prompt": "a pedestrian",
    }
    EXTRACTOR_HEATMAP_SAVE_SUBDIR = "clip_heatmaps"

# <<< CORRECTION 1: Changed "RESNET18" to "PEDESTRIAN_ATTENTIVE" to match your choice above.
elif EXTRACTOR_CHOICE == "PEDESTRIAN_ATTENTIVE":
    FEATURE_EXTRACTOR_SETUP = {
        # <<< CORRECTION 2: Changed "type" to "PEDESTRIAN_ATTENTIVE" for consistency.
        "type": "PEDESTRIAN_ATTENTIVE",
        "features_dim": 256, # The desired final output dimension
        "resnet18_unet_weights_path": os.path.join(CUSTOM_EXTRACTORS_DIR, "resne18unet_weights.pt")
    }
    EXTRACTOR_HEATMAP_SAVE_SUBDIR = None # This extractor does not save heatmaps

else: # Default case
    FEATURE_EXTRACTOR_SETUP = {"type": "DEFAULT_CNN", "features_dim": 512}
    EXTRACTOR_HEATMAP_SAVE_SUBDIR = None

EXPERIMENT_NAME = f"{EXPERIMENT_NAME_BASE}_{FEATURE_EXTRACTOR_SETUP['type']}_{EXPERIMENT_DATE_TIME}"


PPO_CONFIG = {
    "learning_rate": 3e-4, "n_steps": 2048, "batch_size": 32, "n_epochs": 5,
    "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.01,
    "vf_coef": 0.5, "max_grad_norm": 0.5, "verbose": 0,
    "net_arch_mlp_head": [dict(pi=[128], vf=[128])]
}
ENV_RENDER_MODE = None
ENV_USE_VLM_WEIGHTS = True
ENV_USE_VLM_ACTIONS = False
ENV_NORMALIZE_REWARDS = True
VLM_VERBOSE = True

OUTPUT_DIR_BASE = "./experiments"
OUTPUT_DIR = os.path.join(OUTPUT_DIR_BASE, EXPERIMENT_NAME)
VLM_MODEL_NAME='DAMO-NLP-SG/VideoLLaMA3-2B-Image'
VLM_UPDATE_FREQUENCY=5
VLM_FRAMES_NEEDED=3
VLM_MAX_TOKENS=312
VLM_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "vlm_outputs")
FRAMES_DIR = os.path.join(VLM_OUTPUT_DIR, "frames")
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "models")
LOG_DIR = os.path.join(OUTPUT_DIR, "sb3_logs")

WANDB_PROJECT = "carla-vlm-ppo-heatmaps"
WANDB_ENTITY = None

# ============================================================================
# HELPER FUNCTION FOR VLM WEIGHTS (Unchanged)
# ============================================================================
def read_vlm_weights_from_log(json_file_path: str):
    try:
        if os.path.exists(json_file_path):
            with open(json_file_path, 'r') as f:
                log_data = json.load(f)
            if log_data and "decisions" in log_data and isinstance(log_data["decisions"], list):
                for decision in reversed(log_data["decisions"]):
                    if isinstance(decision, dict) and \
                       decision.get("task_type") == "weights" and \
                       "weights" in decision and isinstance(decision["weights"], dict):
                        return decision["weights"]
        return None
    except (json.JSONDecodeError, IOError): return None
    except Exception: return None

# ============================================================================
# CALLBACK (Simplified)
# ============================================================================
class VLMSimpleCallback(BaseCallback):
    def __init__(self, vlm_decision_log_path_getter, verbose=0):
        super().__init__(verbose)
        self.vlm_decision_log_path_getter = vlm_decision_log_path_getter
        self.episode_rewards_list = []
        self.episode_lengths_list = []
        self.current_episode_reward_sum = 0
        self.current_episode_step_count = 0
        self.last_vlm_weights_logged_step = -1

    def _on_step(self) -> bool:
        reward = self.locals['rewards'][0]
        self.current_episode_reward_sum += reward
        self.current_episode_step_count += 1
        if wandb.run:
            wandb.log({"Step/reward": float(reward)}, step=self.num_timesteps)
            try:
                action_val = float(self.locals['actions'][0])
                wandb.log({"Step/action_value": action_val}, step=self.num_timesteps)
            except (TypeError, IndexError): pass

        if ENV_USE_VLM_WEIGHTS and (self.num_timesteps % 10 == 0) and (self.num_timesteps > self.last_vlm_weights_logged_step):
            vlm_log_file = self.vlm_decision_log_path_getter()
            if vlm_log_file:
                vlm_weights_data = read_vlm_weights_from_log(vlm_log_file)
                if vlm_weights_data:
                    wandb.log({f"VLM_Weights/{k}": float(v) for k, v in vlm_weights_data.items()}, step=self.num_timesteps)
                    self.last_vlm_weights_logged_step = self.num_timesteps

        if self.locals['dones'][0]:
            ep_rew = self.current_episode_reward_sum
            ep_len = self.current_episode_step_count
            self.episode_rewards_list.append(ep_rew)
            self.episode_lengths_list.append(ep_len)
            avg_rew_per_step = ep_rew / max(1, ep_len)
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

        if self.num_timesteps > 0 and self.num_timesteps % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = os.path.join(MODEL_SAVE_PATH, f"model_step_{self.num_timesteps}.zip")
            self.model.save(checkpoint_path)
            print(f"\n💾 Checkpoint saved: {checkpoint_path}")
        return True


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def setup_directories():
    print("--- Setting up experiment directories ---")
    dirs_to_create = [OUTPUT_DIR, VLM_OUTPUT_DIR, FRAMES_DIR, MODEL_SAVE_PATH, LOG_DIR]
    
    # <<< CORRECTION 3: Added an 'if' check to prevent the script from crashing
    # when EXTRACTOR_HEATMAP_SAVE_SUBDIR is None.
    if EXTRACTOR_HEATMAP_SAVE_SUBDIR:
        fe_internal_heatmap_dir = os.path.join(OUTPUT_DIR, EXTRACTOR_HEATMAP_SAVE_SUBDIR)
        dirs_to_create.append(fe_internal_heatmap_dir)

    for directory in dirs_to_create:
        os.makedirs(directory, exist_ok=True)
        print(f"  - Directory ensured: {directory}")
    print("--- Directories setup complete ---\n")


global vlm_controller_instance
vlm_controller_instance = None
def get_vlm_log_file_path_for_callback():
    global vlm_controller_instance
    return vlm_controller_instance.log_file if vlm_controller_instance else None


# ============================================================================
# MAIN FUNCTION
# ============================================================================
def main():
    global vlm_controller_instance

    print(f"🚗 CARLA VLM-Enhanced PPO Training\n" + "="*70)
    print(f"📁 Experiment: {EXPERIMENT_NAME}")
    print(f"🔩 Feature Extractor: {FEATURE_EXTRACTOR_SETUP['type']}")
    if EXTRACTOR_HEATMAP_SAVE_SUBDIR:
        print(f"🔥 Heatmaps will be saved by the extractor to: {EXTRACTOR_HEATMAP_SAVE_SUBDIR}")
    print("="*70)

    setup_directories()

    wandb.init(project=WANDB_PROJECT, name=EXPERIMENT_NAME, entity=WANDB_ENTITY,
               config=vars(), sync_tensorboard=True, monitor_gym=True, save_code=True)

    print("🧠 Creating VLM controller...")
    vlm_controller_instance = VLMController(
        model_name=VLM_MODEL_NAME, update_frequency=VLM_UPDATE_FREQUENCY,
        frames_needed=VLM_FRAMES_NEEDED, output_dir=VLM_OUTPUT_DIR,
        max_new_tokens=VLM_MAX_TOKENS, verbose=VLM_VERBOSE
    )
    print(f"✅ VLM controller created. Log file: {vlm_controller_instance.log_file}")

    print("🌍 Creating CARLA environment...")
    def make_env_fn():
        env = CarlaEnvironment(
            render_mode=ENV_RENDER_MODE, use_vlm_weights=ENV_USE_VLM_WEIGHTS,
            use_vlm_actions=ENV_USE_VLM_ACTIONS, normalize_rewards=ENV_NORMALIZE_REWARDS
        )
        env.seed(SEED + random.randint(0, 100))
        env.frame_save_dir = FRAMES_DIR
        env.vlm_controller = vlm_controller_instance
        return env

    vec_env = DummyVecEnv([make_env_fn])
    monitor_log_path = os.path.join(LOG_DIR, "monitor.csv")
    vec_env = VecMonitor(vec_env, filename=monitor_log_path)
    print(f"✅ CARLA VecEnv created. Monitor logs: {monitor_log_path}")

    # --- PPO Policy and kwargs ---
    policy_kwargs = {}
    extractor_type = FEATURE_EXTRACTOR_SETUP["type"]
    shared_features_dim = FEATURE_EXTRACTOR_SETUP["features_dim"]

    if extractor_type == "DEFAULT_CNN":
        print("🖼️ Using SB3 Default CNN.")
        policy_kwargs = {"net_arch": PPO_CONFIG.get("net_arch_mlp_head")}

    elif extractor_type == "CLIP_SEG_SPATIAL":
        print("🖼️ Using Custom CLIPAndSegFeatureExtractor.")
        if CLIPAndSegFeatureExtractor is None:
            raise ImportError("CLIPAndSegFeatureExtractor not available.")
        policy_kwargs = dict(
            features_extractor_class=CLIPAndSegFeatureExtractor,
            features_extractor_kwargs=dict(
                features_dim=shared_features_dim,
                clip_vision_model_name=FEATURE_EXTRACTOR_SETUP["clip_vision_model_name"],
                clipseg_model_name=FEATURE_EXTRACTOR_SETUP["clipseg_model_name"],
                pedestrian_prompt=FEATURE_EXTRACTOR_SETUP["pedestrian_prompt"],
                extractor_heatmap_save_dir=os.path.join(OUTPUT_DIR, EXTRACTOR_HEATMAP_SAVE_SUBDIR),
            ),
            net_arch=PPO_CONFIG.get("net_arch_mlp_head"),
            activation_fn=nn.ReLU
        )

    # <<< CORRECTION 4: The primary logical fix. This now correctly matches the EXTRACTOR_CHOICE
    # and uses your specified class name 'ResnetAttention'.
    elif extractor_type == "PEDESTRIAN_ATTENTIVE":
        print("🖼️ Using Custom ResnetAttention Extractor.")
        if ResnetAttention is None:
            raise ImportError("ResnetAttention not available. Ensure Models/resnet18_attention.py exists and is imported correctly.")
        policy_kwargs = dict(
            features_extractor_class=ResnetAttention,
            features_extractor_kwargs=dict(
                features_dim=shared_features_dim,
                resnet18_unet_weights_path=FEATURE_EXTRACTOR_SETUP["resnet18_unet_weights_path"]
            ),
            net_arch=PPO_CONFIG.get("net_arch_mlp_head"),
            activation_fn=nn.ReLU
        )

    else:
        raise ValueError(f"Unknown FEATURE_EXTRACTOR_SETUP type: {extractor_type}")

    print("🤖 Creating PPO model...")
    model_hyperparams = {k: v for k, v in PPO_CONFIG.items() if k != "net_arch_mlp_head"}
    model = PPO("CnnPolicy", vec_env, tensorboard_log=LOG_DIR, policy_kwargs=policy_kwargs,
                seed=SEED, **model_hyperparams)
    print(f"✅ PPO model created.")

    callback = VLMSimpleCallback(vlm_decision_log_path_getter=get_vlm_log_file_path_for_callback, verbose=1)

    print(f"\n🚀 Starting PPO training for {TOTAL_TIMESTEPS:,} timesteps...")
    start_time = time.time()
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback,
                    reset_num_timesteps=False, tb_log_name=EXPERIMENT_NAME_BASE)
        final_model_path = os.path.join(MODEL_SAVE_PATH, "final_model.zip")
        model.save(final_model_path)
        print(f"\n💾 Final model saved: {final_model_path}")
        if wandb.run and hasattr(wandb, 'Artifact'):
            artifact = wandb.Artifact(f'{EXPERIMENT_NAME}-model', type='model')
            artifact.add_file(final_model_path)
            wandb.log_artifact(artifact)

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user.")
        interrupted_model_path = os.path.join(MODEL_SAVE_PATH, "interrupted_model.zip")
        model.save(interrupted_model_path)
        print(f"💾 Interrupted model saved: {interrupted_model_path}")

    except Exception as e:
        print(f"\n❌ An error occurred during training: {e}")
        import traceback
        traceback.print_exc()
        error_model_path = os.path.join(MODEL_SAVE_PATH, "error_model.zip")
        if 'model' in locals():
            model.save(error_model_path)
            print(f"💾 Model state on error saved: {error_model_path}")

    finally:
        total_training_time = time.time() - start_time
        print(f"\n🎉 Training Session Concluded. Time: {total_training_time/3600:.2f}h")
        if callback.episode_rewards_list:
            mean_rew = np.mean(callback.episode_rewards_list[-100:])
            print(f"  - Mean Reward (last 100 eps): {mean_rew:.2f}")
            print(f"  - Best Ep Reward: {max(callback.episode_rewards_list):.2f}")
        vec_env.close()
        print("🗑️ CARLA VecEnv closed.")
        if wandb.run:
            wandb.run.finish()
            print("👌 WandB run finished.")


if __name__ == "__main__":
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    main()