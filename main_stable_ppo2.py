#!/usr/bin/env python3
"""
CARLA VLM-Enhanced PPO Training Script
Integrates VLM for reward weighting with PPO for autonomous driving.
Relies on modifications in CarlaEnv and VLMController for episodic state reset.
"""

import os
import sys
import time
import pickle
import json
import numpy as np
import torch
import wandb
from datetime import datetime
import random

# Stable Baselines3 imports
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

# --- Project-specific imports ---
PROJECT_ROOT = "/home/server01/Vinal/CARLA_0.9.15/VLM_Barkin/CarlaEnv"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from environment.carla_env_w_symbolic import CarlaEnv as CarlaEnvironment
from Models.vlm_controller_symbolic import VLMController

# ============================================================================
# CONFIGURATION
# ============================================================================

# --- Experiment Configuration ---
EXPERIMENT_DATE_TIME = datetime.now().strftime('%Y%m%d_%H%M%S')
EXPERIMENT_NAME_BASE = "carla_vlm_stable_ppo" # MODIFIED
EXPERIMENT_NAME = f"{EXPERIMENT_NAME_BASE}_{EXPERIMENT_DATE_TIME}"
TOTAL_TIMESTEPS = 500_000
CHECKPOINT_INTERVAL = 20000
SEED = 42

# --- PPO Configuration ---
PPO_CONFIG = {
    "learning_rate": 3e-4, "n_steps": 2048, "batch_size": 64, "n_epochs": 10,
    "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.01,
    "vf_coef": 0.5, "max_grad_norm": 0.5, "verbose": 0
}

# --- Environment Configuration ---
ENV_RENDER_MODE = None
ENV_USE_VLM_WEIGHTS = True
ENV_USE_VLM_ACTIONS = False
ENV_NORMALIZE_REWARDS = True

# --- VLM Controller Configuration ---
VLM_MODEL_NAME = "DAMO-NLP-SG/VideoLLaMA3-2B-Image"
VLM_UPDATE_FREQUENCY = 5
VLM_FRAMES_NEEDED = 3
VLM_MAX_TOKENS = 128
VLM_VERBOSE = True

# --- Paths ---
OUTPUT_DIR_BASE = "/home/server01/Vinal/CARLA_0.9.15/VLM_Barkin/VLM_Action/experiments"
OUTPUT_DIR = f"{OUTPUT_DIR_BASE}/{EXPERIMENT_NAME}"
VLM_OUTPUT_DIR = f"{OUTPUT_DIR}/vlm_outputs"
FRAMES_DIR = f"{VLM_OUTPUT_DIR}/frames"
MODEL_SAVE_PATH = f"{OUTPUT_DIR}/models" # Checkpoints will go into subfolder named by run_id or experiment_name
LOG_DIR = f"{OUTPUT_DIR}/sb3_logs"

# --- WandB Configuration ---
WANDB_PROJECT = "carla-vlm-ppo-experiments" # MODIFIED project name slightly for clarity
WANDB_ENTITY = None # Your Wandb username or team name

# ============================================================================
# HELPER FUNCTION FOR VLM WEIGHTS (Refined)
# (read_vlm_weights_from_log - same as before)
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
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_file_path}")
        return None
    except Exception as e:
        print(f"Error reading VLM decisions from {json_file_path}: {e}")
        return None

# ============================================================================
# CALLBACK (Naming of logged metrics updated)
# ============================================================================
class VLMPPOCallback(BaseCallback):
    def __init__(self, vlm_decision_log_path_getter, verbose=0):
        super(VLMPPOCallback, self).__init__(verbose)
        self.vlm_decision_log_path_getter = vlm_decision_log_path_getter
        self.episode_rewards_list = []
        self.episode_lengths_list = []
        self.current_episode_reward_sum = 0
        self.current_episode_step_count = 0
        self.diagnostics_printed = False
        self.action_diagnostics_printed_count = 0
        self.last_vlm_weights_logged_step = -1

    def _on_step(self) -> bool:
        if not self.diagnostics_printed and self.num_timesteps > 0:
            self._print_ppo_diagnostics()
            self.diagnostics_printed = True
            
        current_step_reward = self.locals['rewards'][0]
        self.current_episode_reward_sum += current_step_reward
        self.current_episode_step_count += 1
        
        # 1) Log Step_reward (as requested)
        wandb.log({"Step_reward": float(current_step_reward)}, step=self.num_timesteps)
        
        # 3) Log action_value at each step (as requested)
        if hasattr(self.locals, 'actions') and self.locals['actions'] is not None:
            try:
                action_val = float(self.locals['actions'][0])
                wandb.log({"action_value": action_val}, step=self.num_timesteps)
            except (TypeError, IndexError) as e:
                if self.verbose > 1: print(f"Warning: Could not log action_value at step {self.num_timesteps}. Error: {e}")

        if self.action_diagnostics_printed_count < 3 and self.num_timesteps in [10, 500, 1000]:
            self._print_action_diagnostics(self.locals['actions'])
            self.action_diagnostics_printed_count += 1
            
        # 2) Log vlm_weights (as requested)
        if ENV_USE_VLM_WEIGHTS and (self.num_timesteps % 10 == 0) and (self.num_timesteps > self.last_vlm_weights_logged_step):
            vlm_log_file = self.vlm_decision_log_path_getter()
            if vlm_log_file:
                vlm_weights_data = read_vlm_weights_from_log(vlm_log_file)
                if vlm_weights_data and isinstance(vlm_weights_data, dict):
                    ws = vlm_weights_data.get("safety")
                    wc = vlm_weights_data.get("comfort")
                    we = vlm_weights_data.get("efficiency")
                    if ws is not None and wc is not None and we is not None:
                        wandb.log({
                            "vlm_weights/safety": float(ws),
                            "vlm_weights/comfort": float(wc),
                            "vlm_weights/efficiency": float(we),
                        }, step=self.num_timesteps)
                        self.last_vlm_weights_logged_step = self.num_timesteps
                        if self.verbose > 0 and self.num_timesteps % 200 == 0:
                             print(f"🧠 Logged VLM Weights at step {self.num_timesteps}: Safety={ws:.2f}, Comfort={wc:.2f}, Efficiency={we:.2f}")
            # Frame debug info logging was removed as per previous request.

        if self.locals['dones'][0]:
            self.episode_rewards_list.append(self.current_episode_reward_sum)
            self.episode_lengths_list.append(self.current_episode_step_count)
            avg_reward_per_step_in_episode = self.current_episode_reward_sum / max(1, self.current_episode_step_count)
            
            print(f"\n🏁 Episode {len(self.episode_rewards_list)} Finished:")
            print(f"   Reward Sum: {self.current_episode_reward_sum:.2f}")
            print(f"   Length: {self.current_episode_step_count} steps")
            print(f"   Avg Reward/Step in Episode: {avg_reward_per_step_in_episode:.3f}")

            # MODIFIED: Logging episode metrics with "episode/" prefix or direct names
            wandb.log({
                "episode/reward": self.current_episode_reward_sum,
                "episode/length": self.current_episode_step_count,
                "episode/reward_per_step": avg_reward_per_step_in_episode,
                "episode/count": len(self.episode_rewards_list) # More direct than "total_completed"
            }, step=self.num_timesteps)

            # SB3's VecMonitor logs "rollout/ep_rew_mean" and "rollout/ep_len_mean"
            # which are rolling averages and generally preferred for trend analysis.
            # The above "episode/" logs are for exact values of just-completed episodes.

            self.current_episode_reward_sum = 0
            self.current_episode_step_count = 0
            
        if self.num_timesteps > 0 and self.num_timesteps % CHECKPOINT_INTERVAL == 0:
            # MODIFIED: Checkpoint name uses EXPERIMENT_NAME_BASE
            checkpoint_path = f"{MODEL_SAVE_PATH}/{EXPERIMENT_NAME_BASE}_step_{self.num_timesteps}.zip"
            self.model.save(checkpoint_path)
            print(f"\n💾 Checkpoint saved: {checkpoint_path} at step {self.num_timesteps}")
        return True

    def _print_ppo_diagnostics(self):
        # (This method remains for console output, no changes needed for metric naming here)
        print("\n" + "="*70)
        print(" 🔍 PPO Environment & Model Diagnostics (Initial Setup)")
        print("="*70)
        try:
            env = self.training_env.envs[0]
            while hasattr(env, 'env'): 
                env = env.env
            print(f"  Observation Space:")
            print(f"    Type: {type(env.observation_space)}, Shape: {env.observation_space.shape}, Dtype: {env.observation_space.dtype}")
            print(f"  Action Space:")
            print(f"    Type: {type(env.action_space)}, Shape: {env.action_space.shape}, Dtype: {env.action_space.dtype}")
            print(f"    Low: {env.action_space.low}, High: {env.action_space.high}")
            print(f"  Frame Saving Directory (CarlaEnv): {getattr(env, 'frame_save_dir', 'Not Configured')}")
            print(f"  VLM Controller in CarlaEnv: {'Present' if hasattr(env, 'vlm_controller') and env.vlm_controller else 'Not Present'}")
            if hasattr(env, 'vlm_controller') and env.vlm_controller:
                 print(f"  VLM Controller Log File: {getattr(env.vlm_controller, 'log_file', 'N/A')}")
            print(f"  PPO Model Policy: {type(self.model.policy).__name__}")
            sample_obs = self.training_env.observation_space.sample() 
            if isinstance(sample_obs, dict): 
                for key, value in sample_obs.items():
                    print(f"  Sample Obs '{key}': Shape={value.shape}, Dtype={value.dtype}, Min={value.min():.2f}, Max={value.max():.2f}")
            else: 
                print(f"  Sample Obs: Shape={sample_obs.shape}, Dtype={sample_obs.dtype}, Min={sample_obs.min():.2f}, Max={sample_obs.max():.2f}")
        except Exception as e:
            print(f"  Error during PPO diagnostics: {e}")
        print("="*70 + "\n")


    def _print_action_diagnostics(self, actions):
        # (This method remains for console output, no changes needed for metric naming here)
        # Wandb logging for "action_value" is now done in _on_step every step.
        print(f"\n---🎬 Action Diagnostics at Timestep: {self.num_timesteps} ---")
        try:
            action_val = float(actions[0]) 
            interpretation = "Unknown"
            if action_val < -0.7: interpretation = "Very Strong Braking 🛑🛑"
            elif action_val < -0.3: interpretation = "Moderate Braking 🛑"
            elif action_val < -0.05: interpretation = "Light Braking/Decel 🔽"
            elif action_val <= 0.05: interpretation = "Coasting/Maintain ➡️"
            elif action_val <= 0.3: interpretation = "Light Acceleration 🔼"
            elif action_val <= 0.7: interpretation = "Moderate Acceleration ↗️"
            else: interpretation = "Strong Acceleration ⏫"
            print(f"  Raw Action Output (from PPO): {actions}") 
            print(f"  Interpreted Value: {action_val:.4f} ({interpretation})")
        except Exception as e:
            print(f"  Error during action diagnostics: {e}")
        print("--- End Action Diagnostics ---\n")
        

    def _on_rollout_end(self) -> None:
        # (This method's fix for logger.get_mean() is retained, primarily for console prints)
        if self.verbose > 0:
            print(f"\n🔄 Rollout End (after {self.model.n_steps} steps). Timestep: {self.num_timesteps}")
            ep_rew_mean_val = None
            ep_len_mean_val = None
            if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value'):
                current_logger_values = self.model.logger.name_to_value
                ep_rew_mean_val = current_logger_values.get("rollout/ep_rew_mean")
                ep_len_mean_val = current_logger_values.get("rollout/ep_len_mean")
            if ep_rew_mean_val is not None:
                print(f"  SB3 Mean Episode Reward (this rollout, via logger.name_to_value): {ep_rew_mean_val:.2f}")
            else:
                if self.episode_rewards_list:
                    print(f"  Custom Mean Ep Reward (all completed): {np.mean(self.episode_rewards_list):.2f} (if available)")
                else:
                    print(f"  SB3 Mean Ep Reward (this rollout): Not available in logger.name_to_value at _on_rollout_end.")
            if ep_len_mean_val is not None:
                print(f"  SB3 Mean Ep Length (this rollout, via logger.name_to_value): {ep_len_mean_val:.2f}")
            else:
                if self.episode_lengths_list:
                    print(f"  Custom Mean Ep Length (all completed): {np.mean(self.episode_lengths_list):.2f} (if available)")
                else:
                    print(f"  SB3 Mean Ep Length (this rollout): Not available in logger.name_to_value at _on_rollout_end.")

# ============================================================================
# HELPER FUNCTIONS
# (setup_directories, get_vlm_log_file_path_for_callback - same as before)
# ============================================================================
def setup_directories():
    print("--- Setting up experiment directories ---")
    for directory in [OUTPUT_DIR, VLM_OUTPUT_DIR, FRAMES_DIR, MODEL_SAVE_PATH, LOG_DIR]:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"  Ensured directory exists: {directory}")
        except Exception as e:
            print(f"  Error creating directory {directory}: {e}")
    print("--- Directories setup complete ---\n")

global vlm_controller_instance # Declare global to be set in main
vlm_controller_instance = None 

def get_vlm_log_file_path_for_callback():
    global vlm_controller_instance 
    if vlm_controller_instance and hasattr(vlm_controller_instance, 'log_file'):
        return vlm_controller_instance.log_file
    return None

# ============================================================================
# MAIN FUNCTION (Updated experiment naming)
# ============================================================================
def main():
    global vlm_controller_instance # Allow main to set the global instance

    print(f"🚗 CARLA VLM-Enhanced PPO Training ({EXPERIMENT_NAME_BASE})") # MODIFIED
    print(f"📁 Experiment Full Name: {EXPERIMENT_NAME}")
    print(f"🕰️ Current Time: {EXPERIMENT_DATE_TIME}")
    print("="*70)
    
    setup_directories()
    
    try:
        wandb.init(
            project=WANDB_PROJECT, name=EXPERIMENT_NAME, entity=WANDB_ENTITY,
            config={
                "EXPERIMENT_NAME": EXPERIMENT_NAME, "TOTAL_TIMESTEPS": TOTAL_TIMESTEPS,
                "CHECKPOINT_INTERVAL": CHECKPOINT_INTERVAL, "SEED": SEED,
                "PPO_CONFIG": PPO_CONFIG, "ENV_RENDER_MODE": ENV_RENDER_MODE,
                "ENV_USE_VLM_WEIGHTS": ENV_USE_VLM_WEIGHTS, "ENV_USE_VLM_ACTIONS": ENV_USE_VLM_ACTIONS,
                "ENV_NORMALIZE_REWARDS": ENV_NORMALIZE_REWARDS, "VLM_MODEL_NAME": VLM_MODEL_NAME,
                "VLM_UPDATE_FREQUENCY": VLM_UPDATE_FREQUENCY, "VLM_FRAMES_NEEDED": VLM_FRAMES_NEEDED,
                "VLM_MAX_TOKENS": VLM_MAX_TOKENS, "OUTPUT_DIR": OUTPUT_DIR,
                "VLM_OUTPUT_DIR": VLM_OUTPUT_DIR, "FRAMES_DIR": FRAMES_DIR,
                "MODEL_SAVE_PATH": MODEL_SAVE_PATH, "LOG_DIR": LOG_DIR
            },
            sync_tensorboard=True, monitor_gym=True, save_code=True
        )
        print("📊 WandB logging initialized successfully.")
    except Exception as e:
        print(f"⚠️ Error initializing WandB: {e}. Training will continue without WandB logging.")
        class DummyWandb: # Basic dummy for code to run
            def __init__(self): self.run = type('DummyRun', (), {'id': f'local_{EXPERIMENT_DATE_TIME}', 'dir': OUTPUT_DIR})()
            def log(self, *args, **kwargs): pass
            def finish(self, *args, **kwargs): pass
            def Artifact(self, *args, **kwargs): _dummy_artifact = type('DummyArtifact', (), {}); _dummy_artifact.add_file = lambda *a, **kw: None; return _dummy_artifact
            def log_artifact(self, *args, **kwargs): pass
        # This is a very simplified dummy. wandb.sdk might not exist or work this way in all contexts.
        # Fallback if wandb fails needs careful handling if features depend on wandb.run.id etc.
        # For now, we assume wandb.log will just pass.
        if not hasattr(wandb, 'log'): wandb.log = lambda *args, **kwargs: None 
        if not hasattr(wandb, 'finish'): wandb.finish = lambda *args, **kwargs: None
        if not hasattr(wandb, 'Artifact'): wandb.Artifact = lambda *a, **k: type('DummyArt', (), {'add_file': lambda *x, **y: None, 'name':a[0]})()
        if not hasattr(wandb, 'log_artifact'): wandb.log_artifact = lambda *args, **kwargs: None
        if not hasattr(wandb, 'run'): wandb.run = type('DummyRun', (), {'id': f'local_{EXPERIMENT_DATE_TIME}', 'dir': OUTPUT_DIR})()


    print("🧠 Creating VLM controller...")
    vlm_controller_config = {
        "model_name": VLM_MODEL_NAME, "update_frequency": VLM_UPDATE_FREQUENCY,
        "frames_needed": VLM_FRAMES_NEEDED, "output_dir": VLM_OUTPUT_DIR,
        "max_new_tokens": VLM_MAX_TOKENS, "verbose": VLM_VERBOSE
    }
    vlm_controller_instance = VLMController(**vlm_controller_config)
    print(f"✅ VLM controller created. Log file: {vlm_controller_instance.log_file}")
    
    print("🌍 Creating CARLA environment...")
    def make_env_fn():
        env_instance = CarlaEnvironment(
            render_mode=ENV_RENDER_MODE, use_vlm_weights=ENV_USE_VLM_WEIGHTS,
            use_vlm_actions=ENV_USE_VLM_ACTIONS, normalize_rewards=ENV_NORMALIZE_REWARDS
        )
        env_instance.seed(SEED + random.randint(0,100))
        env_instance.frame_save_dir = FRAMES_DIR
        env_instance.vlm_controller = vlm_controller_instance
        return env_instance
    
    vec_env = DummyVecEnv([make_env_fn])
    # Ensure LOG_DIR for monitor file is specific to prevent conflicts if running multiple scripts
    monitor_log_path = os.path.join(LOG_DIR, f"monitor_{EXPERIMENT_NAME_BASE}_{EXPERIMENT_DATE_TIME}.csv")
    vec_env = VecMonitor(vec_env, filename=monitor_log_path)
    print(f"✅ CARLA VecEnv created and wrapped with VecMonitor. Monitor logs at: {monitor_log_path}")

    obs_space = vec_env.observation_space
    if len(obs_space.shape) == 3:
        policy_type = "CnnPolicy"
        policy_kwargs = PPO_CONFIG.get("policy_kwargs", {
            "net_arch": {"pi": [256], "vf": [256]}, "activation_fn": torch.nn.ReLU,
            "normalize_images": True
        })
        print(f"🖼️ Using CnnPolicy for image observations (Shape: {obs_space.shape})")
    elif len(obs_space.shape) == 1:
        policy_type = "MlpPolicy"
        policy_kwargs = PPO_CONFIG.get("policy_kwargs", {
             "net_arch": {"pi": [256, 256], "vf": [256, 256]}, "activation_fn": torch.nn.ReLU
        })
        print(f"📊 Using MlpPolicy for vector observations (Shape: {obs_space.shape})")
    else:
        raise ValueError(f"Unsupported observation space shape: {obs_space.shape}")

    print("🤖 Creating PPO model...")
    model_params = {key: value for key, value in PPO_CONFIG.items() if key != "policy_kwargs"}
    model = PPO(
        policy_type, vec_env, tensorboard_log=LOG_DIR,
        policy_kwargs=policy_kwargs, seed=SEED, **model_params
    )
    print(f"✅ PPO model created with {policy_type}.")
    
    callback = VLMPPOCallback(vlm_decision_log_path_getter=get_vlm_log_file_path_for_callback, verbose=1)
    
    print(f"\n🚀 Starting PPO training for {TOTAL_TIMESTEPS:,} timesteps...")
    print(f"   Logging to WandB project: {WANDB_PROJECT}, run: {EXPERIMENT_NAME}")
    print(f"   SB3 TensorBoard logs will be in: {LOG_DIR}")
    
    start_time = time.time()
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS, callback=callback,
            reset_num_timesteps=False, tb_log_name=EXPERIMENT_NAME_BASE # tb_log_name for subfolder in LOG_DIR
        )
        final_model_path = f"{MODEL_SAVE_PATH}/{EXPERIMENT_NAME_BASE}_final_model.zip" # MODIFIED
        model.save(final_model_path)
        print(f"\n💾 Final model saved: {final_model_path}")
        if wandb.run and hasattr(wandb, 'Artifact'): # Check if wandb is active
            artifact = wandb.Artifact(f'{EXPERIMENT_NAME_BASE}-final-model', type='model') # MODIFIED
            artifact.add_file(final_model_path)
            wandb.log_artifact(artifact)
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user (KeyboardInterrupt).")
        interrupted_model_path = f"{MODEL_SAVE_PATH}/{EXPERIMENT_NAME_BASE}_interrupted_model.zip" # MODIFIED
        model.save(interrupted_model_path)
        print(f"💾 Interrupted model saved: {interrupted_model_path}")
    except Exception as e:
        print(f"\n❌ An error occurred during training: {e}")
        import traceback
        traceback.print_exc()
        error_model_path = f"{MODEL_SAVE_PATH}/{EXPERIMENT_NAME_BASE}_error_model.zip" # MODIFIED
        model.save(error_model_path)
        print(f"💾 Model state on error saved: {error_model_path}")
    finally:
        total_training_time = time.time() - start_time
        print(f"\n🎉 Training Session Concluded.")
        print(f"   Total Training Time: {total_training_time/3600:.2f} hours ({total_training_time/60:.2f} minutes)")
        if hasattr(callback, 'episode_rewards_list') and callback.episode_rewards_list:
            print(f"   Total Episodes Completed: {len(callback.episode_rewards_list)}")
            if len(callback.episode_rewards_list) >=100:
                 print(f"   Mean Reward (last 100 eps): {np.mean(callback.episode_rewards_list[-100:]):.2f}")
            elif callback.episode_rewards_list:
                 print(f"   Mean Reward (all eps): {np.mean(callback.episode_rewards_list):.2f}")
            print(f"   Best Episode Reward: {max(callback.episode_rewards_list):.2f}")
        else:
            print("   No full episodes completed or callback not retaining rewards.")
        print(f"   Output data saved in: {OUTPUT_DIR}")
        try:
            vec_env.close()
            print("🗑️ CARLA VecEnv closed.")
        except Exception as e:
            print(f"⚠️ Error closing CARLA VecEnv: {e}")
        if hasattr(wandb, 'finish') and callable(wandb.finish): # Check if wandb was dummied
            wandb.finish()
            print("👌 WandB run finished.")
        else:
            print("ℹ️ WandB logging was not active or failed to initialize.")

if __name__ == "__main__":
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    main()