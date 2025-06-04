#!/usr/bin/env python3
"""
CARLA VLM-Enhanced PPO Training Script with Diagnostics
Optimized integration of VLM with PPO for autonomous driving
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
from tqdm import tqdm

# Stable Baselines3 imports
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecMonitor

# Import your CARLA environment and VLM controller
sys.path.append('/home/server01/Vinal/CARLA_0.9.15/VLM_Barkin/CarlaEnv')
from environment.carla_env_w_symbolic import CarlaEnvironment
from Models.vlm_controller_symbolic import VLMController

# ============================================================================
# CONFIGURATION
# ============================================================================

# Experiment Configuration
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
EXPERIMENT_NAME = f"carla_vlm_ppo_{TIMESTAMP}"
TOTAL_TIMESTEPS = 100000
CHECKPOINT_INTERVAL = 10000
EARLY_CHECKPOINT_INTERVAL = 1000
EARLY_CHECKPOINT_THRESHOLD = 10000

# PPO Configuration
PPO_CONFIG = {
    "learning_rate": 1e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.3,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "verbose": 0,
    "policy": "MlpPolicy",
    "policy_kwargs": {
        "net_arch": {"pi": [256, 256], "vf": [256, 256]},
        "activation_fn": torch.nn.ReLU
    }
}

# Environment Configuration
ENV_RENDER_MODE = None
ENV_USE_VLM_WEIGHTS = True
ENV_USE_VLM_ACTIONS = False
ENV_OBS_SHAPE = (512,)
NORMALIZE_REWARD = True

# VLM Configuration
VLM_MODEL_NAME = "DAMO-NLP-SG/VideoLLaMA3-2B-Image"
VLM_UPDATE_FREQUENCY = 5
VLM_FRAMES_NEEDED = 3
VLM_MAX_TOKENS = 512
VLM_VERBOSE = False

# Paths
PROJECT_BASE_DIR = "/home/server01/Vinal/CARLA_0.9.15/VLM_Barkin/VLM_Action"
VLM_OUTPUT_DIR = os.path.join(PROJECT_BASE_DIR, "vlm_outputs")
FRAMES_DIR = os.path.join(VLM_OUTPUT_DIR, "frames", f"ppo_{TIMESTAMP}")
PPO_TRAINING_DIR = os.path.join(VLM_OUTPUT_DIR, "PPO_training", f"run_{TIMESTAMP}")
WEIGHTS_DIR = os.path.join(PPO_TRAINING_DIR, "weights")
CHECKPOINTS_DIR = os.path.join(PPO_TRAINING_DIR, "checkpoints")
LOGS_DIR = os.path.join(PPO_TRAINING_DIR, "logs")
VLM_LOG_FILE = os.path.join(VLM_OUTPUT_DIR, f"vlm_decisions_{TIMESTAMP}.json")

# WandB Configuration
WANDB_PROJECT = "carla-vlm-ppo"

# ============================================================================
# HELPER CLASSES AND FUNCTIONS
# ============================================================================

class RunningStat:
    """Tracks running mean and variance for reward normalization"""
    def __init__(self, epsilon=1e-4):
        self.mean = 0.0
        self.std = 1.0
        self.count = 0
        self.epsilon = epsilon
    
    def update(self, x):
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.std = np.sqrt(self.std**2 + (delta * delta2 - self.std**2) / self.count)
        self.std = max(self.std, self.epsilon)

def read_vlm_decisions_from_file(json_file_path):
    """Read VLM decisions from JSON file"""
    try:
        if os.path.exists(json_file_path):
            with open(json_file_path, 'r') as f:
                decisions = json.load(f)
                if decisions:
                    return decisions[-1] if isinstance(decisions, list) else decisions
        return None
    except Exception as e:
        wandb.log({"error/vlm_decision_read": str(e)})
        return None

# ============================================================================
# CUSTOM CALLBACK FOR METRICS AND PROGRESS
# ============================================================================

class ProgressPPOCallback(BaseCallback):
    """Callback for metrics, progress bar updates, and minimal console output"""
    
    def __init__(self, total_timesteps, progress_bar, vlm_controller=None):
        super(ProgressPPOCallback, self).__init__(verbose=0)
        self.total_timesteps = total_timesteps
        self.progress_bar = progress_bar
        self.vlm_controller = vlm_controller
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_steps = 0
        self.first_vlm_update_checked = False
        self.weight_history = []
        self.vlm_updates_per_episode = 0
        self.current_episode = 0
        self.reward_stat = RunningStat() if NORMALIZE_REWARD else None
    
    def _on_step(self) -> bool:
        self.progress_bar.update(1)
        
        if hasattr(self.locals, 'rewards') and self.locals['rewards'] is not None:
            raw_reward = float(self.locals['rewards'][0])
            
            # Reward processing
            clipped_reward = np.clip(raw_reward, -10.0, 10.0)
            if NORMALIZE_REWARD:
                self.reward_stat.update(clipped_reward)
                normalized_reward = np.clip(
                    (clipped_reward - self.reward_stat.mean) / (self.reward_stat.std + 1e-8),
                    -5.0, 5.0
                )
                step_reward = normalized_reward
            else:
                step_reward = clipped_reward
            
            self.current_episode_reward += step_reward
            self.current_episode_steps += 1
            
            wandb.log({
                "reward/raw": raw_reward,
                "reward/clipped": clipped_reward,
                "reward/normalized": step_reward if NORMALIZE_REWARD else clipped_reward,
                "episode_progress": self.current_episode_reward
            }, step=self.num_timesteps)
            
        if (not self.first_vlm_update_checked and 
            ENV_USE_VLM_WEIGHTS and 
            self.num_timesteps >= VLM_UPDATE_FREQUENCY and 
            self.num_timesteps % VLM_UPDATE_FREQUENCY == 0):
            try:
                if hasattr(self.training_env, 'envs') and len(self.training_env.envs) > 0:
                    env = self.training_env.envs[0]
                    while hasattr(env, 'env'):
                        env = env.env
                    
                    frame_count = len(env.frame_buffer) if hasattr(env, 'frame_buffer') and env.frame_buffer else 0
                    saved_frames = []
                    frame_paths = []
                    if hasattr(env, 'frame_save_dir') and os.path.exists(env.frame_save_dir):
                        saved_frames = [f for f in os.listdir(env.frame_save_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
                        frame_paths = [os.path.join(env.frame_save_dir, f) for f in saved_frames[-VLM_FRAMES_NEEDED:]]
                    
                    vlm_active = False
                    weights = None
                    if hasattr(env, 'vlm_controller') and hasattr(env.vlm_controller, 'log_file'):
                        vlm_decision = read_vlm_decisions_from_file(env.vlm_controller.log_file)
                        if vlm_decision and 'weights' in vlm_decision:
                            weights = vlm_decision['weights']
                            vlm_active = True
                            self.vlm_updates_per_episode += 1
                    
                    print(f"✅ First VLM Update (Step {self.num_timesteps}):")
                    print(f"   Frame Buffer: {frame_count} frames")
                    print(f"   Saved Frames: {len(saved_frames)} in {getattr(env, 'frame_save_dir', 'Not set')}")
                    if frame_paths:
                        print(f"   Processed Frames: {' → '.join([os.path.basename(p) for p in frame_paths])}")
                    print(f"   VLM Active: {'Yes' if vlm_active else 'No'}")
                    if weights:
                        print(f"   VLM Weights: Safety={weights.get('w1', weights.get('safety', 1.0)):.3f}, "
                              f"Comfort={weights.get('w2', weights.get('comfort', 1.0)):.3f}, "
                              f"Efficiency={weights.get('w3', weights.get('efficiency', 1.0)):.3f}")
                    
                    wandb.log({
                        "debug/frame_buffer_size": frame_count,
                        "debug/saved_frame_count": len(saved_frames),
                        "debug/vlm_active": 1.0 if vlm_active else 0.0
                    }, step=self.num_timesteps)
                    
                    if hasattr(env, 'vlm_controller'):
                        env.vlm_controller.first_update_verbose = False
                    
                    self.first_vlm_update_checked = True
                    
            except Exception as e:
                wandb.log({"error/first_vlm_check": str(e)})
        
        if ENV_USE_VLM_WEIGHTS and self.num_timesteps % 50 == 0:
            try:
                if hasattr(self.training_env, 'envs') and len(self.training_env.envs) > 0:
                    env = self.training_env.envs[0]
                    while hasattr(env, 'env'):
                        env = env.env
                    
                    weights = None
                    if hasattr(env, 'vlm_controller') and hasattr(env.vlm_controller, 'log_file'):
                        vlm_decision = read_vlm_decisions_from_file(env.vlm_controller.log_file)
                        if vlm_decision and 'weights' in vlm_decision:
                            weights = vlm_decision['weights']
                            if weights != self.weight_history[-1]['weights'] if self.weight_history else True:
                                self.vlm_updates_per_episode += 1
                    
                    if weights is None and hasattr(env, 'get_current_weights'):
                        weights = env.get_current_weights()
                    
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
                                "vlm/weights_active": 1.0,
                                "vlm/updates_per_episode": self.vlm_updates_per_episode
                            }, step=self.num_timesteps)
                            
                            self.weight_history.append({
                                "step": self.num_timesteps,
                                "weights": {"w1": w1, "w2": w2, "w3": w3},
                                "justification": weights.get("justification", ""),
                                "timestamp": time.time()
                            })
                        else:
                            wandb.log({"vlm/weights_active": 0.0}, step=self.num_timesteps)
                    
                    if hasattr(env, 'frame_buffer'):
                        frame_count = len(env.frame_buffer) if env.frame_buffer else 0
                        saved_frames = []
                        if hasattr(env, 'frame_save_dir') and os.path.exists(env.frame_save_dir):
                            saved_frames = [f for f in os.listdir(env.frame_save_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
                        
                        wandb.log({
                            "debug/frame_buffer_size": frame_count,
                            "debug/saved_frame_count": len(saved_frames)
                        }, step=self.num_timesteps)
                            
            except Exception as e:
                wandb.log({"error/vlm_tracking": str(e)})

        if hasattr(self.locals, 'dones') and self.locals['dones'] is not None:
            if self.locals['dones'][0]:
                # Reset VLM controller state for new episode
                try:
                    if hasattr(self.training_env, 'envs') and len(self.training_env.envs) > 0:
                        env = self.training_env.envs[0]
                        while hasattr(env, 'env'):
                            env = env.env
                        if hasattr(env, 'vlm_controller') and hasattr(env.vlm_controller, 'last_update_step'):
                            env.vlm_controller.last_update_step = 0
                except Exception as e:
                    wandb.log({"error/vlm_reset": str(e)})
                
                self.episode_rewards.append(self.current_episode_reward)
                self.episode_lengths.append(self.current_episode_steps)
                
                avg_reward_per_step = self.current_episode_reward / max(1, self.current_episode_steps)
                
                wandb.log({
                    "episode_reward": self.current_episode_reward,
                    "episode_length": self.current_episode_steps,
                    "reward_per_step": avg_reward_per_step,
                    "episodes": len(self.episode_rewards),
                    "vlm/updates_per_episode": self.vlm_updates_per_episode
                }, step=self.num_timesteps)
                
                if len(self.episode_rewards) >= 10:
                    wandb.log({
                        "mean_reward_10": np.mean(self.episode_rewards[-10:]),
                        "std_reward_10": np.std(self.episode_rewards[-10:])
                    }, step=self.num_timesteps)
                
                if len(self.episode_rewards) >= 100:
                    wandb.log({
                        "mean_reward_100": np.mean(self.episode_rewards[-100:])
                    }, step=self.num_timesteps)
                
                print(f"Episode {len(self.episode_rewards)}: Reward={self.current_episode_reward:.2f}, "
                      f"Steps={self.current_episode_steps}, Avg={avg_reward_per_step:.3f}, "
                      f"VLM Updates={self.vlm_updates_per_episode}")
                
                self.current_episode_reward = 0
                self.current_episode_steps = 0
                self.vlm_updates_per_episode = 0
                self.current_episode += 1
        
        save_checkpoint = (
            self.num_timesteps % CHECKPOINT_INTERVAL == 0 or
            self.num_timesteps == self.total_timesteps - 1 or
            (self.num_timesteps < EARLY_CHECKPOINT_THRESHOLD and
             self.num_timesteps % EARLY_CHECKPOINT_INTERVAL == 0)
        )
        
        if save_checkpoint:
            checkpoint_path = os.path.join(CHECKPOINTS_DIR, f"checkpoint_${self.num_timesteps}.zip")
            self.model.save(checkpoint_path)
            wandb.save(checkpoint_path)
            
            if self.weight_history:
                weight_history_path = os.path.join(WEIGHTS_DIR, f"weight_history_${self.num_timesteps}.pkl")
                with open(weight_history_path, "wb") as f:
                    pickle.dump(self.weight_history, f)
                wandb.save(weight_history_path)
        
        return True
    
    def _on_rollout_end(self):
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
        except Exception as e:
            wandb.log({"error/rollout_logging": str(e)})

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def setup_directories():
    """Create necessary directories"""
    for directory in [VLM_OUTPUT_DIR, FRAMES_DIR, PPO_TRAINING_DIR, WEIGHTS_DIR, CHECKPOINTS_DIR, LOGS_DIR]:
        os.makedirs(directory, exist_ok=True)

def add_vlm_method_to_env():
    """Add VLM integration methods to environment"""
    
    def get_current_weights(self):
        sources_to_try = [
            ('previous_vlm_weights', 'Previous VLM weights'),
            ('vlm_weights', 'Current VLM weights'),
            ('current_weights', 'Current weights'),
            ('last_vlm_weights', 'Last VLM weights')
        ]
        
        for attr_name, _ in sources_to_try:
            if hasattr(self, attr_name):
                weights = getattr(self, attr_name)
                if weights is not None and isinstance(weights, dict):
                    return weights
        
        return getattr(self, 'default_weights', {"w1": 1.0, "w2": 1.0, "w3": 1.0})

    def get_vlm_decision_file_path(self):
        if hasattr(self, 'vlm_controller') and hasattr(self.vlm_controller, 'log_file'):
            return self.vlm_controller.log_file
        return VLM_LOG_FILE

    if not hasattr(CarlaEnvironment, 'get_current_weights'):
        setattr(CarlaEnvironment, 'get_current_weights', get_current_weights)
    if not hasattr(CarlaEnvironment, 'get_vlm_decision_file_path'):
        setattr(CarlaEnvironment, 'get_vlm_decision_file_path', get_vlm_decision_file_path)

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main training function with optimized output"""
    print(f"🚖 CARLA VLM-Enhanced PPO Training: {EXPERIMENT_NAME}")
    
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
            "normalize_reward": NORMALIZE_REWARD,
            "experiment_name": EXPERIMENT_NAME,
            "frames_dir": FRAMES_DIR,
            "ppo_training_dir": PPO_TRAINING_DIR,
            "weights_dir": WEIGHTS_DIR,
            "checkpoints_dir": CHECKPOINTS_DIR,
            "logs_dir": LOGS_DIR
        },
        tags=["PPO", "CARLA", "VLM", "autonomous-driving"],
        dir=LOGS_DIR
    )
    
    # Create VLM controller
    vlm_config = {
        "model_name": VLM_MODEL_NAME,
        "update_frequency": VLM_UPDATE_FREQUENCY,
        "frames_needed": VLM_FRAMES_NEEDED,
        "output_dir": VLM_OUTPUT_DIR,
        "max_new_tokens": VLM_MAX_TOKENS,
        "verbose": VLM_VERBOSE,
        "log_file": VLM_LOG_FILE,
        "first_update_verbose": True
    }
    
    vlm_controller = VLMController(**vlm_config)
    
    # Create environment
    def make_env():
        env = CarlaEnvironment(
            render_mode=ENV_RENDER_MODE,
            use_vlm_weights=ENV_USE_VLM_WEIGHTS,
            use_vlm_actions=ENV_USE_VLM_ACTIONS
        )
        env.frame_save_dir = os.path.join(FRAMES_DIR)
        env.vlm_controller = vlm_controller
        env.episode_counter = 0
        env.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=ENV_OBS_SHAPE, dtype=np.float32
        )
        return env
    
    vec_env = make_vec_env(make_env, n_envs=1, seed=42)
    vec_env = VecMonitor(vec_env, LOGS_DIR)
    
    # Print summary
    print("\n🔍 PPO Model Summary:")
    print(f"   Observation Space: {vec_env.observation_space}")
    print(f"   Action Space: {vec_env.action_space}")
    
    # Create PPO model
    model = PPO(
        PPO_CONFIG["policy"],
        vec_env,
        tensorboard_log=LOGS_DIR,
        policy_kwargs=PPO_CONFIG["policy_kwargs"],
        **{k: v for k, v in PPO_CONFIG.items() if k not in ["policy", "policy_kwargs"]}
    )
    
    # Initialize progress bar
    progress_bar = tqdm(total=TOTAL_TIMESTEPS, desc="Training Progress", unit="step")
    
    # Create callback
    callback = ProgressPPOCallback(total_timesteps=TOTAL_TIMESTEPS, progress_bar=progress_bar, vlm_controller=vlm_controller)
    
    # Start training
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callback,
            reset_num_timesteps=False,
            tb_log_name="ppo_carla_vlm"
        )
        
        # Save final model
        final_model_path = os.path.join(CHECKPOINTS_DIR, "ppo_final_model.zip")
        model.save(final_model_path)
        wandb.save(final_model_path)
        
        # Save final weight history
        if callback.weight_history:
            weight_history_path = os.path.join(WEIGHTS_DIR, "weight_history_final.pkl")
            with open(weight_history_path, "wb") as f:
                pickle.dump(callback.weight_history, f)
            wandb.save(weight_history_path)
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        interrupted_model_path = os.path.join(CHECKPOINTS_DIR, "ppo_interrupted_model.zip")
        model.save(interrupted_model_path)
        wandb.save(interrupted_model_path)
        
        if callback.weight_history:
            weight_history_path = os.path.join(WEIGHTS_DIR, "weight_history_interrupted.pkl")
            with open(weight_history_path, "wb") as f:
                pickle.dump(callback.weight_history, f)
            wandb.save(weight_history_path)
    
    # Close progress bar
    progress_bar.close()
    
    # Training summary
    total_time = time.time() - start_time
    print(f"\n🎉 Training Completed!")
    print(f"   Duration: {total_time/3600:.2f} hours")
    print(f"   Episodes: {len(callback.episode_rewards)}")
    if callback.episode_rewards:
        print(f"   Final Episode Reward: {callback.episode_rewards[-1]:.2f}")
        print(f"   Best Episode Reward: {max(callback.episode_rewards):.2f}")
    
    # Cleanup
    vec_env.close()
    wandb.finish()

if __name__ == "__main__":
    main()