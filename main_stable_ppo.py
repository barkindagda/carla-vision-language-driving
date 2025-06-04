#!/usr/bin/env python3
"""
CARLA VLM-Enhanced PPO Training Script with Diagnostics
Simplified integration of VLM with PPO for autonomous driving
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

# Stable Baselines3 imports
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize, VecMonitor

# Import your CARLA environment and VLM controller
sys.path.append('/home/server01/Vinal/CARLA_0.9.15/VLM_Barkin/CarlaEnv')
from environment.carla_env_w_symbolic import CarlaEnv as CarlaEnvironment
from Models.vlm_controller_symbolic import VLMController

# ============================================================================
# CONFIGURATION
# ============================================================================

# Experiment Configuration
EXPERIMENT_NAME = f"carla_vlm_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
TOTAL_TIMESTEPS = 500_000
CHECKPOINT_INTERVAL = 10000

# PPO Configuration
PPO_CONFIG = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "verbose": 1
}

# Environment Configuration
ENV_RENDER_MODE = None  # Match JAX version; adjust if needed for frame capture
ENV_USE_VLM_WEIGHTS = True
ENV_USE_VLM_ACTIONS = False

# VLM Configuration
VLM_MODEL_NAME = "DAMO-NLP-SG/VideoLLaMA3-2B-Image"
VLM_UPDATE_FREQUENCY = 5
VLM_FRAMES_NEEDED = 3
VLM_MAX_TOKENS = 512
VLM_VERBOSE = True

# Paths
OUTPUT_DIR = f"/home/server01/Vinal/CARLA_0.9.15/VLM_Barkin/VLM_Action/experiments/{EXPERIMENT_NAME}"
VLM_OUTPUT_DIR = f"{OUTPUT_DIR}/vlm_outputs"
FRAMES_DIR = f"{VLM_OUTPUT_DIR}/frames/{EXPERIMENT_NAME}"  # Added to match JAX version
MODEL_SAVE_PATH = f"{OUTPUT_DIR}/models"
LOG_DIR = f"{OUTPUT_DIR}/logs"

# WandB Configuration
WANDB_PROJECT = "carla-vlm-ppo"

# ============================================================================
# HELPER FUNCTIONS FOR VLM INTEGRATION
# ============================================================================

def read_vlm_decisions_from_file(json_file_path):
    """Read VLM decisions from JSON file (matching JAX implementation)"""
    try:
        if os.path.exists(json_file_path):
            with open(json_file_path, 'r') as f:
                decisions = json.load(f)
                if decisions:  # Return the most recent decision
                    return decisions[-1] if isinstance(decisions, list) else decisions
        return None
    except Exception as e:
        print(f"Error reading VLM decisions: {e}")
        return None

# ============================================================================
# SIMPLIFIED CALLBACK FOR METRICS
# ============================================================================

class SimplifiedPPOCallback(BaseCallback):
    """Simplified callback focusing on essential PPO metrics and diagnostics"""
    
    def __init__(self, vlm_controller=None, verbose=0):
        super(SimplifiedPPOCallback, self).__init__(verbose)
        self.vlm_controller = vlm_controller
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_steps = 0
        self.diagnostics_printed = False
        self.action_diagnostics_printed = False
        self.weight_history = []
        
    def _on_step(self) -> bool:
        # Print input/output diagnostics only once at the beginning
        if not self.diagnostics_printed and self.num_timesteps > 0:
            self._print_ppo_diagnostics()
            self.diagnostics_printed = True
            
        # Track step rewards
        if hasattr(self.locals, 'rewards') and self.locals['rewards'] is not None:
            step_reward = float(self.locals['rewards'][0])
            self.current_episode_reward += step_reward
            self.current_episode_steps += 1
            
            # Log step reward to WandB
            wandb.log({
                "step_reward": step_reward,
                "episode_progress": self.current_episode_reward
            }, step=self.num_timesteps)
            
        # Print action diagnostics a few times
        if not self.action_diagnostics_printed and self.num_timesteps in [10, 50, 100]:
            self._print_action_diagnostics()
            if self.num_timesteps == 100:
                self.action_diagnostics_printed = True
            
        # VLM Weight Tracking and Frame Debugging
        if ENV_USE_VLM_WEIGHTS and self.num_timesteps % 50 == 0:
            try:
                # Get environment reference
                if hasattr(self.training_env, 'envs') and len(self.training_env.envs) > 0:
                    env = self.training_env.envs[0]
                    while hasattr(env, 'env'):
                        env = env.env
                    
                    # Method 1: Try to get VLM decision from JSON file
                    weights = None
                    if hasattr(env, 'vlm_controller') and hasattr(env.vlm_controller, 'log_file'):
                        vlm_decision = read_vlm_decisions_from_file(env.vlm_controller.log_file)
                        if vlm_decision and 'weights' in vlm_decision:
                            weights = vlm_decision['weights']
                    
                    # Method 2: Try to get from environment's current weights
                    if weights is None and hasattr(env, 'get_current_weights'):
                        weights = env.get_current_weights()
                    
                    # Method 3: Check previous VLM weights
                    if weights is None and hasattr(env, 'previous_vlm_weights'):
                        weights = env.previous_vlm_weights
                    
                    if weights and isinstance(weights, dict):
                        w1 = weights.get("w1", weights.get("safety", 1.0))
                        w2 = weights.get("w2", weights.get("comfort", 1.0)) 
                        w3 = weights.get("w3", weights.get("efficiency", 1.0))
                        
                        if not (w1 == 1.0 and w2 == 1.0 and w3 == 1.0):
                            wandb.log({
                                "vlm/safety_weight": w1,
                                "vlm/comfort_weight": w2,
                                "vlm/efficiency_weight": w3,
                                "vlm/weights_active": 1.0
                            }, step=self.num_timesteps)
                            
                            self.weight_history.append({
                                "step": self.num_timesteps,
                                "weights": {"w1": w1, "w2": w2, "w3": w3},
                                "justification": weights.get("justification", ""),
                                "timestamp": time.time()
                            })
                            
                            if self.verbose > 0 and self.num_timesteps % 200 == 0:
                                print(f"✅ VLM Weights Active - Safety: {w1:.3f}, Comfort: {w2:.3f}, Efficiency: {w3:.3f}")
                        else:
                            wandb.log({"vlm/weights_active": 0.0}, step=self.num_timesteps)
                    
                    # Enhanced Frame Debugging
                    if hasattr(env, 'frame_buffer'):
                        frame_count = len(env.frame_buffer) if env.frame_buffer else 0
                        wandb.log({"debug/frame_buffer_size": frame_count}, step=self.num_timesteps)
                        
                        # Check if frames are being saved
                        saved_frames = []
                        if hasattr(env, 'frame_save_dir') and os.path.exists(env.frame_save_dir):
                            saved_frames = [f for f in os.listdir(env.frame_save_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
                        
                        wandb.log({"debug/saved_frame_count": len(saved_frames)}, step=self.num_timesteps)
                        
                        if frame_count == 0 or len(saved_frames) == 0:
                            if self.verbose > 0:
                                print(f"⚠️ Frame issue at step {self.num_timesteps}:")
                                print(f"   Frame buffer size: {frame_count}")
                                print(f"   Saved frames: {len(saved_frames)}")
                                print(f"   Frame save dir: {getattr(env, 'frame_save_dir', 'Not set')}")
                        elif self.verbose > 1:
                            print(f"📸 Frame status: Buffer size = {frame_count}, Saved frames = {len(saved_frames)}")
                            
            except Exception as e:
                if self.verbose > 0:
                    print(f"Error tracking VLM weights or frames: {e}")

        # Check for episode end
        if hasattr(self.locals, 'dones') and self.locals['dones'] is not None:
            if self.locals['dones'][0]:
                self.episode_rewards.append(self.current_episode_reward)
                self.episode_lengths.append(self.current_episode_steps)
                
                avg_reward_per_step = self.current_episode_reward / max(1, self.current_episode_steps)
                
                wandb.log({
                    "episode_reward": self.current_episode_reward,
                    "episode_length": self.current_episode_steps,
                    "reward_per_step": avg_reward_per_step,
                    "episodes_completed": len(self.episode_rewards)
                }, step=self.num_timesteps)
                
                if len(self.episode_rewards) >= 10:
                    recent_rewards = self.episode_rewards[-10:]
                    wandb.log({
                        "mean_reward_10": np.mean(recent_rewards),
                        "std_reward_10": np.std(recent_rewards)
                    }, step=self.num_timesteps)
                
                if len(self.episode_rewards) >= 100:
                    recent_rewards = self.episode_rewards[-100:]
                    wandb.log({
                        "mean_reward_100": np.mean(recent_rewards)
                    }, step=self.num_timesteps)
                
                print(f"🎯 Episode {len(self.episode_rewards)}: "
                      f"Reward = {self.current_episode_reward:.2f} | "
                      f"Steps = {self.current_episode_steps} | "
                      f"Avg = {avg_reward_per_step:.3f}")
                
                self.current_episode_reward = 0
                self.current_episode_steps = 0
        
        # Save checkpoint
        if self.num_timesteps % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = f"{MODEL_SAVE_PATH}/ppo_step_{self.num_timesteps}.zip"
            self.model.save(checkpoint_path)
            wandb.save(checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")
        
        return True
    
    def _print_ppo_diagnostics(self):
        """Print PPO input/output diagnostics once"""
        print("\n" + "="*60)
        print("🔍 PPO DIAGNOSTICS")
        print("="*60)
        
        try:
            if hasattr(self.training_env, 'envs') and len(self.training_env.envs) > 0:
                env = self.training_env.envs[0]
                while hasattr(env, 'env'):
                    env = env.env
                
                print(f"📊 OBSERVATION SPACE:")
                print(f"   Type: {type(env.observation_space)}")
                print(f"   Shape: {env.observation_space.shape}")
                print(f"   Dtype: {env.observation_space.dtype}")
                print(f"   Low: {env.observation_space.low}")
                print(f"   High: {env.observation_space.high}")
                
                print(f"\n🎮 ACTION SPACE:")
                print(f"   Type: {type(env.action_space)}")
                print(f"   Shape: {env.action_space.shape}")
                print(f"   Dtype: {env.action_space.dtype}")
                print(f"   Low: {env.action_space.low} (Emergency Brake)")
                print(f"   High: {env.action_space.high} (Full Throttle)")
                
                print(f"\n📸 FRAME SAVING CONFIG:")
                print(f"   Frame save dir: {getattr(env, 'frame_save_dir', 'Not set')}")
                print(f"   VLM controller: {'Present' if hasattr(env, 'vlm_controller') else 'Not set'}")
                
                if hasattr(self.locals, 'obs_tensor') and self.locals['obs_tensor'] is not None:
                    obs = self.locals['obs_tensor']
                    print(f"\n📈 CURRENT OBSERVATION:")
                    print(f"   Tensor shape: {obs.shape}")
                    print(f"   Tensor dtype: {obs.dtype}")
                    print(f"   Value range: [{obs.min():.3f}, {obs.max():.3f}]")
                    flat_obs = obs.flatten()
                    if len(flat_obs) > 10:
                        print(f"   Sample values: {flat_obs[:10].tolist()}")
                    else:
                        print(f"   All values: {flat_obs.tolist()}")
                
        except Exception as e:
            print(f"Error in diagnostics: {e}")
        
        print("="*60 + "\n")
    
    def _print_action_diagnostics(self):
        """Print action diagnostics periodically"""
        try:
            if hasattr(self.locals, 'actions') and self.locals['actions'] is not None:
                actions = self.locals['actions']
                print(f"🎬 Step {self.num_timesteps} - PPO Action Output:")
                print(f"   Action: {actions}")
                print(f"   Action type: {type(actions)}")
                print(f"   Action shape: {actions.shape if hasattr(actions, 'shape') else 'No shape'}")
                if hasattr(actions, 'shape') and len(actions) > 0:
                    action_val = float(actions[0]) if hasattr(actions[0], 'item') else float(actions[0])
                    print(f"   Action value: {action_val:.4f}")
                    
                    if action_val < -0.5:
                        interpretation = "Strong Braking 🛑"
                    elif action_val < -0.1:
                        interpretation = "Light Braking 🔽"
                    elif action_val < 0.1:
                        interpretation = "Coasting ➡️"
                    elif action_val < 0.5:
                        interpretation = "Light Acceleration 🔼"
                    else:
                        interpretation = "Strong Acceleration ⬆️"
                    
                    print(f"   Interpretation: {interpretation}")
                    
                    wandb.log({
                        "action_value": action_val,
                        "action_magnitude": abs(action_val)
                    }, step=self.num_timesteps)
        except Exception as e:
            print(f"Error in action diagnostics: {e}")
    
    def _on_rollout_end(self) -> None:
        """Log rollout summary"""
        try:
            if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value'):
                ppo_metrics = self.model.logger.name_to_value
                metrics_to_log = {}
                for key, value in ppo_metrics.items():
                    if any(metric in key.lower() for metric in 
                          ['policy_loss', 'value_loss', 'entropy', 'explained_variance', 'learning_rate']):
                        metrics_to_log[f"ppo/{key}"] = value
                if metrics_to_log:
                    wandb.log(metrics_to_log, step=self.num_timesteps)
            
            if self.episode_rewards:
                wandb.log({
                    "rollout/episodes_this_rollout": len(self.episode_rewards),
                    "rollout/mean_episode_reward": np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else np.mean(self.episode_rewards),
                    "rollout/total_episodes": len(self.episode_rewards)
                }, step=self.num_timesteps)
                
                print(f"\n📊 Rollout Complete - Episodes: {len(self.episode_rewards)} | "
                      f"Mean Reward: {np.mean(self.episode_rewards[-10:]):.2f}")
                      
        except Exception as e:
            print(f"Error in rollout logging: {e}")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def setup_directories():
    """Create necessary directories"""
    for directory in [OUTPUT_DIR, VLM_OUTPUT_DIR, FRAMES_DIR, MODEL_SAVE_PATH, LOG_DIR]:
        os.makedirs(directory, exist_ok=True)
    print(f"✅ Experiment directory: {OUTPUT_DIR}")
    print(f"✅ Frames directory: {FRAMES_DIR}")

def add_vlm_method_to_env():
    """Add VLM integration methods to environment (matching JAX implementation)"""
    
    def get_current_weights(self):
        """Helper method to get current weights being used - tries multiple sources"""
        sources_to_try = [
            ('previous_vlm_weights', 'Previous VLM weights'),
            ('vlm_weights', 'Current VLM weights'),
            ('current_weights', 'Current weights'),
            ('last_vlm_weights', 'Last VLM weights')
        ]
        
        for attr_name, description in sources_to_try:
            if hasattr(self, attr_name):
                weights = getattr(self, attr_name)
                if weights is not None and isinstance(weights, dict):
                    if ENV_VLM_VERBOSE:
                        print(f"Found weights from {description}: {weights}")
                    return weights
        
        default_weights = getattr(self, 'default_weights', {"w1": 1.0, "w2": 1.0, "w3": 1.0})
        if ENV_VLM_VERBOSE:
            print(f"Using default weights: {default_weights}")
        return default_weights

    def get_vlm_decision_file_path(self):
        """Get the path to VLM decision JSON file"""
        if hasattr(self, 'vlm_controller') and hasattr(self.vlm_controller, 'log_file'):
            return self.vlm_controller.log_file
        return os.path.join(VLM_OUTPUT_DIR, f"vlm_decisions_{EXPERIMENT_NAME}.json")

    if not hasattr(CarlaEnvironment, 'get_current_weights'):
        setattr(CarlaEnvironment, 'get_current_weights', get_current_weights)
    if not hasattr(CarlaEnvironment, 'get_vlm_decision_file_path'):
        setattr(CarlaEnvironment, 'get_vlm_decision_file_path', get_vlm_decision_file_path)

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main training function with simplified diagnostics"""
    print("🚗 CARLA VLM-Enhanced PPO Training")
    print(f"📁 Experiment: {EXPERIMENT_NAME}")
    print("="*60)
    
    # Setup
    setup_directories()
    add_vlm_method_to_env()
    
    # Initialize WandB
    wandb.init(
        project=WANDB_PROJECT,
        name=EXPERIMENT_NAME,
        config={
            **PPO_CONFIG,
            "total_timesteps": TOTAL_TIMESTEPS,
            "env_use_vlm_weights": ENV_USE_VLM_WEIGHTS,
            "env_use_vlm_actions": ENV_USE_VLM_ACTIONS,
            "vlm_update_frequency": VLM_UPDATE_FREQUENCY,
            "vlm_model": VLM_MODEL_NAME,
            "experiment_name": EXPERIMENT_NAME,
            "frames_dir": FRAMES_DIR
        },
        tags=["PPO", "CARLA", "VLM", "autonomous-driving"]
    )
    print("📊 WandB logging initialized")
    
    # Create VLM controller
    vlm_config = {
        "model_name": VLM_MODEL_NAME,
        "update_frequency": VLM_UPDATE_FREQUENCY,
        "frames_needed": VLM_FRAMES_NEEDED,
        "output_dir": VLM_OUTPUT_DIR,
        "max_new_tokens": VLM_MAX_TOKENS,
        "verbose": VLM_VERBOSE
    }
    
    print("🧠 Creating VLM controller...")
    vlm_controller = VLMController(**vlm_config)
    print(f"✅ VLM controller created")
    
    # Create environment
    print("🌍 Creating CARLA environment...")
    
    def make_env():
        env = CarlaEnvironment(
            render_mode=ENV_RENDER_MODE,
            use_vlm_weights=ENV_USE_VLM_WEIGHTS,
            use_vlm_actions=ENV_USE_VLM_ACTIONS
        )
        env.frame_save_dir = FRAMES_DIR  # Set frame save directory
        env.vlm_controller = vlm_controller  # Set VLM controller
        env.episode_counter = 0  # Initialize episode counter
        return env
    
    vec_env = make_vec_env(make_env, n_envs=1, seed=42)
    vec_env = VecMonitor(vec_env, LOG_DIR)
    
    # Verify VLM and frame save dir on unwrapped env
    env = vec_env.envs[0]
    while hasattr(env, 'env'):
        env = env.env
    print(f"✅ Frame save dir set: {getattr(env, 'frame_save_dir', 'Not set')}")
    print(f"✅ VLM controller: {'Present' if hasattr(env, 'vlm_controller') else 'Not set'}")
    
    # Determine policy type based on observation space
    obs_space = vec_env.observation_space
    if len(obs_space.shape) == 3:
        policy_type = "CnnPolicy"
        policy_kwargs = {
            "net_arch": {"pi": [256, 256], "vf": [256, 256]},
            "activation_fn": torch.nn.ReLU,
            "normalize_images": True
        }
        print(f"🖼️ Using CnnPolicy for image observations: {obs_space.shape}")
    else:
        policy_type = "MlpPolicy"
        policy_kwargs = {
            "net_arch": {"pi": [256, 256], "vf": [256, 256]},
            "activation_fn": torch.nn.ReLU
        }
        print(f"📊 Using MlpPolicy for vector observations: {obs_space.shape}")
    
    # Create PPO model
    print("🤖 Creating PPO model...")
    model = PPO(
        policy_type,
        vec_env,
        tensorboard_log=LOG_DIR,
        policy_kwargs=policy_kwargs,
        **PPO_CONFIG
    )
    
    print(f"✅ PPO model created with {policy_type}")
    print(f"   Observation Space: {vec_env.observation_space}")
    print(f"   Action Space: {vec_env.action_space}")
    
    # Create callback
    callback = SimplifiedPPOCallback(verbose=1)
    
    # Start training
    print(f"\n🚀 Starting training for {TOTAL_TIMESTEPS:,} timesteps...")
    
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callback,
            reset_num_timesteps=False,
            tb_log_name="ppo_carla_vlm"
        )
        
        # Save final model
        final_model_path = f"{MODEL_SAVE_PATH}/ppo_final.zip"
        model.save(final_model_path)
        print(f"💾 Final model saved: {final_model_path}")
        
        # Save weight history
        if callback.weight_history:
            weight_history_path = f"{MODEL_SAVE_PATH}/weight_history_final.pkl"
            with open(weight_history_path, "wb") as f:
                pickle.dump(callback.weight_history, f)
            print(f"💾 Weight history saved: {weight_history_path}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted")
        model.save(f"{MODEL_SAVE_PATH}/ppo_interrupted.zip")
    
    # Training summary
    total_time = time.time() - start_time
    print(f"\n🎉 Training Complete!")
    print(f"   Time: {total_time/3600:.2f} hours")
    print(f"   Episodes: {len(callback.episode_rewards)}")
    if callback.episode_rewards:
        print(f"   Final Episode Reward: {callback.episode_rewards[-1]:.2f}")
        print(f"   Best Episode Reward: {max(callback.episode_rewards):.2f}")
    
    # Cleanup
    vec_env.close()
    wandb.finish()

if __name__ == "__main__":
    main()