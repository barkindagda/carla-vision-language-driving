import os
import time
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
import flax.linen as nn
from flax.training.train_state import TrainState
import gymnasium as gym
import optax
import distrax
import wandb
from pathlib import Path
from functools import partial
import pickle
import datetime

# Import components
from environment.carla_env_w_symbolic import CarlaEnv, SymbolicRules
from Models.vlm_controller_symbolic import VLMController
from Models.PPO_agent import PPO, outer_loop, ActorNet, ValueNet

#############################################
# CONFIGURATION FLAGS - Easily adjustable
#############################################

# Experiment identification
EXPERIMENT_NAME = "carla_vlm_ppo"
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Environment settings
ENV_RENDER_MODE = None
ENV_VLM_FRAMES = 3
ENV_USE_SYMBOLIC_REWARDS = False    # Use symbolic rules for reward components
ENV_USE_VLM_WEIGHTS = True        # Use VLM to determine reward weights
ENV_USE_VLM_ACTIONS = False        # Use PPO for actions (not VLM)
ENV_NORMALIZE_REWARDS = True       # Apply reward normalization

# VLM controller settings
VLM_MODEL_NAME = "DAMO-NLP-SG/VideoLLaMA3-2B-Image"
VLM_UPDATE_FREQUENCY = 10          # Update weights every 10 timesteps
VLM_FRAMES_NEEDED = 3
VLM_MAX_TOKENS = 512
VLM_VERBOSE = False

# PPO training settings
PPO_TRAINING_STEPS = 200_000
PPO_NUM_ENVS = 1
PPO_ROLLOUT_STEPS = 1024  # Reduced for memory constraints
PPO_BATCH_SIZE = 32       # Reduced for memory constraints
PPO_EPOCHS = 10           # Reduced for memory constraints
PPO_CLIP_RANGE = 0.2
PPO_MAX_GRAD_NORM = 0.5
PPO_GAMMA = 0.99
PPO_LAMBDA = 0.95
PPO_LEARNING_RATE = 3e-4
PPO_VALUE_COEF = 0.5
PPO_ENTROPY_COEF = 0.01
PPO_LOG_VIDEO = False
PPO_ENABLE_LOGGING = True

# Base directory structure (matching existing project)
PROJECT_BASE_DIR = "/home/server01/Vinal/CARLA_0.9.15/VLM_Barkin/VLM_Action"
VLM_OUTPUT_DIR = os.path.join(PROJECT_BASE_DIR, "vlm_outputs")

# Run-specific directories with timestamps 
FRAMES_DIR = os.path.join(VLM_OUTPUT_DIR, "frames", f"ppo_{TIMESTAMP}")
PPO_TRAINING_DIR = os.path.join(VLM_OUTPUT_DIR, "PPO_training", f"run_{TIMESTAMP}")
WEIGHTS_DIR = os.path.join(PPO_TRAINING_DIR, "weights")
CHECKPOINTS_DIR = os.path.join(PPO_TRAINING_DIR, "checkpoints")
LOGS_DIR = os.path.join(PPO_TRAINING_DIR, "logs")
VLM_LOG_FILE = os.path.join(VLM_OUTPUT_DIR, f"vlm_decisions_{TIMESTAMP}.json")

# Checkpoint frequency (steps)
CHECKPOINT_INTERVAL = 5000
EARLY_CHECKPOINT_INTERVAL = 1000
EARLY_CHECKPOINT_THRESHOLD = 10000  # More frequent saves until this step

# Add the custom wrapper for reward conversion
class ScalarRewardWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Ensure reward is a scalar
        if isinstance(reward, (np.ndarray, list)):
            reward = float(reward[0] if isinstance(reward, list) else reward.item())
        return obs, reward, terminated, truncated, info

# Add a wrapper to resize observations to reduce memory
class ResizeObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, size=(192, 192)):
        super().__init__(env)
        self.size = size
        
        # Update observation space
        shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, 
            shape=(shape[0], size[0], size[1]), 
            dtype=np.uint8
        )
    def observation(self, observation):
        # Resize the observation using NumPy (more compatible than PIL)
        # Move channels to last dimension for resize
        obs_hwc = np.transpose(observation, (1, 2, 0))
        # Resize using OpenCV
        import cv2
        resized = cv2.resize(obs_hwc, self.size, interpolation=cv2.INTER_AREA)
        # Move channels back to first dimension
        return np.transpose(resized, (2, 0, 1))

def main():
    """
    Main training script that integrates CARLA environment, VLM controller, 
    and PPO agent for autonomous driving with symbolic rewards.
    """
    print("Starting CARLA VLM-guided PPO training...")
    
    # Create all output directories
    for directory in [VLM_OUTPUT_DIR, FRAMES_DIR, PPO_TRAINING_DIR, WEIGHTS_DIR, CHECKPOINTS_DIR, LOGS_DIR]:
        os.makedirs(directory, exist_ok=True)
    
    print(f"VLM output directory: {VLM_OUTPUT_DIR}")
    print(f"Frames will be saved to: {FRAMES_DIR}")
    print(f"PPO training directory: {PPO_TRAINING_DIR}")
    print(f"Weights will be saved to: {WEIGHTS_DIR}")
    print(f"Checkpoints will be saved to: {CHECKPOINTS_DIR}")
    
    #######################
    # Environment Setup #
    #######################
    
    # Environment configuration from flags
    env_config = {
        "render_mode": ENV_RENDER_MODE,
        "vlm_frames": ENV_VLM_FRAMES,
        "use_symbolic_rewards": ENV_USE_SYMBOLIC_REWARDS,
        "use_vlm_weights": ENV_USE_VLM_WEIGHTS,
        "use_vlm_actions": ENV_USE_VLM_ACTIONS,
        "normalize_rewards": ENV_NORMALIZE_REWARDS
    }
    
    # VLM controller configuration from flags
    vlm_config = {
        "model_name": VLM_MODEL_NAME,
        "update_frequency": VLM_UPDATE_FREQUENCY,
        "frames_needed": VLM_FRAMES_NEEDED,
        "output_dir": VLM_OUTPUT_DIR,
        "max_new_tokens": VLM_MAX_TOKENS,
        "verbose": VLM_VERBOSE
    }
    
    # PPO agent configuration from flags
    ppo_config = {
        "env_id": "CarlaEnv-v0",
        "training_steps": PPO_TRAINING_STEPS,
        "n_envs": PPO_NUM_ENVS,
        "rollout_steps": PPO_ROLLOUT_STEPS,
        "batch_size": PPO_BATCH_SIZE,
        "clip_range": PPO_CLIP_RANGE,
        "epochs": PPO_EPOCHS,
        "max_grad_norm": PPO_MAX_GRAD_NORM,
        "gamma": PPO_GAMMA,
        "vf_clip_range": np.inf,
        "ent_coef": PPO_ENTROPY_COEF,
        "gae_lambda": PPO_LAMBDA,
        "learning_rate": PPO_LEARNING_RATE,
        "vf_coef": PPO_VALUE_COEF,
        "log_video": PPO_LOG_VIDEO,
        "log": PPO_ENABLE_LOGGING
    }
    
    # Add method to CarlaEnv to get current weights
    def get_current_weights(self):
        """Helper method to get current weights being used"""
        if hasattr(self, 'previous_vlm_weights'):
            return self.previous_vlm_weights
        return self.default_weights
    
    # Add the method to CarlaEnv class
    setattr(CarlaEnv, 'get_current_weights', get_current_weights)
    
    # Register the CARLA environment with Gymnasium
    gym.register(
        id='CarlaEnv-v0',
        entry_point='environment.carla_env_w_symbolic:CarlaEnv',
        max_episode_steps=1000,
    )
    
    # Create VLM controller first, so we can pass it to environments
    vlm_controller = VLMController(**vlm_config)
    
    def make_wrapped_env():
        # Create the environment with all configuration parameters
        env = gym.make('CarlaEnv-v0', 
                    render_mode=ENV_RENDER_MODE, 
                    vlm_frames=ENV_VLM_FRAMES,
                    use_symbolic_rewards=ENV_USE_SYMBOLIC_REWARDS,
                    use_vlm_weights=ENV_USE_VLM_WEIGHTS,
                    use_vlm_actions=ENV_USE_VLM_ACTIONS,
                    normalize_rewards=ENV_NORMALIZE_REWARDS)
        
        # Get unwrapped env for direct access
        carla_env = env.unwrapped
        
        # Set frames directory for saving images
        carla_env.frame_save_dir = FRAMES_DIR
        
        # Configure the VLM controller
        carla_env.vlm_controller = vlm_controller
        
        # Initialize episode counter if not present
        if not hasattr(carla_env, 'episode_counter'):
            carla_env.episode_counter = 0
            
        # Add observation resize wrapper to reduce memory usage
        env = ResizeObservationWrapper(env, size=(192, 192))
        
        # Add scalar reward wrapper to ensure PPO compatibility
        env = ScalarRewardWrapper(env)
        
        # Add episode statistics tracking
        env = gym.wrappers.RecordEpisodeStatistics(env)
        
        return env
    
    # Create vectorized environment with wrapped environments
    print("Creating vectorized environment...")
    vectorized_env = gym.vector.SyncVectorEnv([make_wrapped_env for _ in range(PPO_NUM_ENVS)])
    
    # Store a reference to the unwrapped environment for direct access
    unwrapped_env = vectorized_env.envs[0].unwrapped
    
    # Initialize wandb for logging with all configuration details
    if PPO_ENABLE_LOGGING:
        # Combine all configs for comprehensive logging
        combined_config = {
            **ppo_config,
            **vlm_config,
            **env_config,
            "output_dir": PROJECT_BASE_DIR,
            "frames_dir": FRAMES_DIR,
            "weights_dir": WEIGHTS_DIR,
            "checkpoints_dir": CHECKPOINTS_DIR,
            "logs_dir": LOGS_DIR
        }
        
        print("Initializing wandb logging...")
        wandb.init(
            project=EXPERIMENT_NAME,
            name=f"vlm-ppo-{TIMESTAMP}",
            config=combined_config,
            tags=["PPO", "CARLA", "VLM", "SymbolicRules"],
            save_code=True,
            dir=LOGS_DIR
        )
    
    #######################
    # Initialize Networks #
    #######################
    
    print("Initializing PPO neural networks...")
    # Get initial observation for network initialization
    print("- Getting initial observation...")
    obs, _ = vectorized_env.reset(seed=42)  # Set seed for deterministic initialization
    key = random.PRNGKey(42)
    
    # Split random keys for actor and critic networks
    actor_key, value_key, key = random.split(key, num=3)
    
    # Create neural networks with CNN feature extraction
    print("- Creating actor and critic networks...")
    actor_net = ActorNet(vectorized_env.single_action_space.shape[0])
    value_net = ValueNet()
    
    # Verify that action space is correct for our needs (-1 to 1)
    action_space = vectorized_env.single_action_space
    if isinstance(action_space, gym.spaces.Box):
        if not (action_space.low[0] == -1.0 and action_space.high[0] == 1.0):
            print(f"WARNING: Action space range is {action_space.low[0]} to {action_space.high[0]}, expected -1.0 to 1.0")
    else:
        print(f"WARNING: Unexpected action space type: {type(action_space)}")
    
    # Create optimizer with learning rate schedule
    print("- Creating optimizers...")
    opt = optax.chain(
        optax.clip_by_global_norm(PPO_MAX_GRAD_NORM),
        optax.inject_hyperparams(optax.adamw)(
            learning_rate=optax.linear_schedule(
                init_value=PPO_LEARNING_RATE,
                end_value=PPO_LEARNING_RATE / 10,
                transition_steps=PPO_TRAINING_STEPS,
            ),
        ),
    )
  
    # Create train states for actor and critic
    print("- Initializing network parameters...")
    actor_ts = TrainState.create(
        apply_fn=actor_net.apply,
        params=actor_net.init(actor_key, obs),
        tx=opt)

    value_ts = TrainState.create(
        apply_fn=value_net.apply,
        params=value_net.init(value_key, obs),
        tx=opt)
    
    #############################
    # Initialize Training Loop #
    #############################
    
    # Create PPO agent
    print("Creating PPO agent...")
    ppo_agent = PPO(buffer_size=PPO_NUM_ENVS * PPO_ROLLOUT_STEPS, **ppo_config)
    
    # Initialize environment
    print("Resetting environment for training...")
    last_obs, _ = vectorized_env.reset()
    last_episode_starts = np.ones((PPO_NUM_ENVS,), dtype=bool)
    current_global_step = 0
    
    # Training metrics
    weight_history = []
    episode_counter = 0
    
    # Main training loop
    print("\n" + "="*50)
    print("STARTING TRAINING")
    print("="*50 + "\n")
    
    try:
        while current_global_step < ppo_agent.training_steps:
            print(f"\nStep: {current_global_step}/{ppo_agent.training_steps} (Episode: {episode_counter})")
            
            try:
                ###############################
                # PHASE 1: COLLECT EXPERIENCE
                ###############################
                print("Collecting rollout data...")
                rollout, rollout_info = ppo_agent.get_rollout(
                    actor_ts,
                    value_ts,
                    vectorized_env,
                    last_obs,
                    last_episode_starts,
                    key,
                )
                
                # Update step counter
                current_global_step += ppo_agent.rollout_steps * ppo_agent.n_envs
                
                ################################
                # PHASE 2: POLICY OPTIMIZATION
                ################################
                print("Updating policy with PPO...")
                actor_ts, value_ts, key, training_info = outer_loop(
                    key, actor_ts, value_ts, rollout, ppo_agent
                )
                
                # Combine logs
                full_logs = {**training_info, **rollout_info}
                
                ###############################
                # PHASE 3: VLM WEIGHT UPDATES
                ###############################
                # Only log VLM weights if they're being used
                if ENV_USE_VLM_WEIGHTS:
                    try:
                        # Get current weights - these are updated based on VLM_UPDATE_FREQUENCY
                        weights = unwrapped_env.get_current_weights()
                        
                        # Only log the actual weights, not fallbacks
                        if weights is not None:
                            full_logs.update({
                                "safety_weight": weights["w1"],
                                "comfort_weight": weights["w2"],
                                "efficiency_weight": weights["w3"],
                                "weight_justification": weights.get("justification", "")[:100]
                            })
                            
                            # Track weight history
                            weight_history.append({
                                "step": current_global_step,
                                "episode": episode_counter,
                                "weights": weights,
                                "timestamp": time.time()
                            })
                            
                            # Print actual VLM-determined weights
                            print(f"Current VLM Weights - Safety: {weights['w1']:.2f}, "
                                f"Comfort: {weights['w2']:.2f}, "
                                f"Efficiency: {weights['w3']:.2f}")
                    except Exception as e:
                        print(f"Error accessing VLM weights: {e}")
                
                # Add VLM update frequency to logs
                full_logs["vlm_update_frequency"] = VLM_UPDATE_FREQUENCY
                
                # Include information about reward normalization if enabled
                if ENV_NORMALIZE_REWARDS and hasattr(unwrapped_env, 'reward_running_mean'):
                    full_logs.update({
                        "reward_mean": float(unwrapped_env.reward_running_mean),
                        "reward_std": float(getattr(unwrapped_env, 'reward_running_std', 1.0))
                    })
                
                # Print training information
                print(f"Training metrics:")
                print(f"- Rollout mean reward: {full_logs.get('mean rollout reward', 0):.4f}")
                print(f"- Actor loss: {full_logs.get('actor_loss_total', 0):.4f}")
                print(f"- Value loss: {full_logs.get('value_loss_total', 0):.4f}")
                
                # Increment episode counter if we've had any terminated episodes
                if 'episodes' in rollout_info and rollout_info['episodes'] > 0:
                    new_episodes = rollout_info['episodes']
                    episode_counter += new_episodes
                    print(f"Completed {new_episodes} new episodes (total: {episode_counter})")
                    
                    # Update episode counter in unwrapped env for frame naming
                    if hasattr(unwrapped_env, 'episode_counter'):
                        unwrapped_env.episode_counter = episode_counter
                
                ################################
                # PHASE 4: LOGGING & CHECKPOINTS
                ################################
                if PPO_ENABLE_LOGGING:
                    # Log to wandb
                    wandb.log(full_logs, step=current_global_step)
                    
                    # Determine if we should save a checkpoint
                    save_checkpoint = (
                        current_global_step % CHECKPOINT_INTERVAL == 0 or 
                        current_global_step == ppo_agent.training_steps - 1 or
                        (current_global_step < EARLY_CHECKPOINT_THRESHOLD and 
                         current_global_step % EARLY_CHECKPOINT_INTERVAL == 0)
                    )
                    
                    if save_checkpoint:
                        print(f"Saving checkpoint at step {current_global_step}...")
                        checkpoint = {
                            "actor_params": actor_ts.params,
                            "value_params": value_ts.params,
                            "step": current_global_step,
                            "episode": episode_counter
                        }
                        checkpoint_path = os.path.join(CHECKPOINTS_DIR, f"checkpoint_{current_global_step}.pkl")
                        with open(checkpoint_path, "wb") as f:
                            pickle.dump(checkpoint, f)
                        
                        # Log checkpoint to wandb
                        wandb.save(checkpoint_path)
                        
                        # Save weight history
                        weight_history_path = os.path.join(WEIGHTS_DIR, f"weight_history_{current_global_step}.pkl")
                        with open(weight_history_path, "wb") as f:
                            pickle.dump(weight_history, f)
                            
                        print(f"Checkpoint saved to {checkpoint_path}")
                    
            except Exception as e:
                print(f"Error during rollout or training step: {e}")
                import traceback
                traceback.print_exc()
                
                # Reset the environment on error with a unique seed
                reset_seed = int(time.time()) % 10000
                print(f"Resetting environment with seed {reset_seed}")
                last_obs, _ = vectorized_env.reset(seed=reset_seed)
                last_episode_starts = np.ones((PPO_NUM_ENVS,), dtype=bool)
                time.sleep(2)  # Brief pause to allow system to stabilize
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving final checkpoint...")
        # Save final checkpoint
        checkpoint = {
            "actor_params": actor_ts.params,
            "value_params": value_ts.params,
            "step": current_global_step,
            "episode": episode_counter
        }
        with open(os.path.join(CHECKPOINTS_DIR, "checkpoint_interrupted.pkl"), "wb") as f:
            pickle.dump(checkpoint, f)
    
    finally:
        # Cleanup
        if PPO_ENABLE_LOGGING:
            wandb.finish()
        
        # Ensure environment is closed properly
        vectorized_env.close()
        
        print("Training complete!")
        
        # Save final weight history
        if weight_history:
            weight_history_path = os.path.join(CHECKPOINTS_DIR, "weight_history_final.pkl")
            with open(weight_history_path, "wb") as f:
                pickle.dump(weight_history, f)

if __name__ == "__main__":
    main()