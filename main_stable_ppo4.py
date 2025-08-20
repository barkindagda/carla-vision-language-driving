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
from torchvision import transforms as T_vis
from Models.resnet18_attention import ResnetAttention

PROJECT_ROOT = os.path.expanduser("~/BARKIN/carla-vision-language-driving")
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
CUSTOM_EXTRACTORS_DIR = os.path.join(PROJECT_ROOT, "Models", "feature_extractors")
if CUSTOM_EXTRACTORS_DIR not in sys.path:
    sys.path.append(CUSTOM_EXTRACTORS_DIR)
from environment.carla_env2 import CarlaEnv as CarlaEnvironment
from Models.vlm_controller_symbolic import VLMController

try:
    from Models.feature_extractors.clip_image_encoder import CLIPAndSegFeatureExtractor
    print("Successfully imported CLIPAndSegFeatureExtractor.")
except ImportError:
    print("Warning: Could not import CLIPAndSegFeatureExtractor.")
    CLIPAndSegFeatureExtractor = None

EXPERIMENT_DATE_TIME = datetime.now().strftime('%Y%m%d_%H%M%S')
EXPERIMENT_NAME_BASE = "carla_vlm_ppo_s1"

# ==== Core training budget ====
TOTAL_TIMESTEPS = 250000

CHECKPOINT_INTERVAL = 100
SEED = 42
EFFICIENCY_PRIORITY = False  # fixed spelling (was EFFICINCY_PRIORITY)
EXTRACTOR_CHOICE = "PEDESTRIAN_ATTENTIVE"  # Options: "DEFAULT_CNN", "CLIP_SEG_SPATIAL", "PEDESTRIAN_ATTENTIVE"

if EXTRACTOR_CHOICE == "CLIP_SEG_SPATIAL":
    FEATURE_EXTRACTOR_SETUP = {
        "type": "CLIP_SEG_SPATIAL",
        "features_dim": 256,
        "clip_vision_model_name": "openai/clip-vit-base-patch32",
        "clipseg_model_name": "CIDAS/clipseg-rd64-refined",
        "pedestrian_prompt": "a pedestrian",
    }
    EXTRACTOR_HEATMAP_SAVE_SUBDIR = "clip_heatmaps"
elif EXTRACTOR_CHOICE == "PEDESTRIAN_ATTENTIVE":
    FEATURE_EXTRACTOR_SETUP = {
        "type": "PEDESTRIAN_ATTENTIVE",
        "features_dim": 256,
        "resnet18_unet_weights_path": os.path.join(CUSTOM_EXTRACTORS_DIR, "resne18unet_weights.pt")
    }
    EXTRACTOR_HEATMAP_SAVE_SUBDIR = None
else:
    FEATURE_EXTRACTOR_SETUP = {"type": "DEFAULT_CNN", "features_dim": 512}
    EXTRACTOR_HEATMAP_SAVE_SUBDIR = None

EXPERIMENT_NAME = f"{EXPERIMENT_NAME_BASE}_{FEATURE_EXTRACTOR_SETUP['type']}_{EXPERIMENT_DATE_TIME}"

# ------------------------------------------------------------------
# Global LR endpoints (total‑budget scale; we’ll re‑derive on resume)
# ------------------------------------------------------------------
INITIAL_LR = 3e-4
FINAL_LR = 1e-4  # just formatting consistency

# PPO config skeleton (learning_rate will be injected dynamically)
PPO_CONFIG = {
    # "learning_rate": <patched at runtime>,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.25,
    "max_grad_norm": 0.5,
    "verbose": 0,
    "net_arch_mlp_head": dict(pi=[512,256,128], vf=[512,256,128])} # dict(pi=[256, 128], vf=[512, 256]),} ##

ENV_RENDER_MODE = None
ENV_USE_VLM_WEIGHTS = True
ENV_USE_VLM_ACTIONS = False
ENV_NORMALIZE_REWARDS = True
VLM_VERBOSE = True

OUTPUT_DIR_BASE = "./experiments"
OUTPUT_DIR = os.path.join(OUTPUT_DIR_BASE, EXPERIMENT_NAME)
VLM_MODEL_NAME = 'DAMO-NLP-SG/VideoLLaMA3-2B-Image'
VLM_UPDATE_FREQUENCY = 5
VLM_FRAMES_NEEDED = 3
VLM_MAX_TOKENS = 120
VLM_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "vlm_outputs")
FRAMES_DIR = os.path.join(VLM_OUTPUT_DIR, "frames")
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "models")
LOG_DIR = os.path.join(OUTPUT_DIR, "sb3_logs")
CHECKPOINT_PATH = os.path.join(MODEL_SAVE_PATH, "latest_checkpoint.zip")
WANDB_PROJECT = "VLM-attention-PPO"
WANDB_ENTITY = None

# ==================================================================
# Helpers
# ==================================================================

def read_vlm_weights_from_log(json_file_path: str):
    """Load most recent weight dict from VLM decision log."""
    try:
        if os.path.exists(json_file_path):
            with open(json_file_path, 'r') as f:
                log_data = json.load(f)
            if log_data and "decisions" in log_data and isinstance(log_data["decisions"], list):
                for decision in reversed(log_data["decisions"]):
                    if (
                        isinstance(decision, dict)
                        and decision.get("task_type") == "weights"
                        and isinstance(decision.get("weights"), dict)
                    ):
                        return decision["weights"]
        return None
    except (json.JSONDecodeError, IOError):
        return None
    except Exception:
        return None


def get_memory_stats():
    mem_stats = {}
    for i in range(torch.cuda.device_count()):
        mem_stats[f"cuda:{i}_allocated_MB"] = torch.cuda.memory_allocated(i) / 1024 / 1024
        mem_stats[f"cuda:{i}_reserved_MB"] = torch.cuda.memory_reserved(i) / 1024 / 1024
    return mem_stats


# ================================================================
#  LR CONTINUATION UTILITIES
# ================================================================
def interp_linear(a: float, b: float, t: float) -> float:
    """Linear interpolate between a and b at fraction t in [0,1]."""
    return a + (b - a) * np.clip(t, 0.0, 1.0)


def make_continuation_lr_schedule(global_start_step: int, global_total_steps: int,
                                  initial_lr: float, final_lr: float):
    """
    Build an SB3-compatible schedule that *continues* a global linear decay.

    SB3 calls schedule(progress_remaining) where:
        progress_remaining = 1.0 at beginning of *this* learn() call,
        → 0.0 at the end of the call.

    If we’ve already trained `global_start_step`, we compute the LR that the
    global linear decay would have at that point, then decay from there to
    final_lr across the remaining training.

    Returns:
        lr_schedule_fn  (progress_remaining -> lr)
        current_lr      (float) lr at start of resumed training
    """
    # Global fraction already consumed
    frac_done = global_start_step / float(global_total_steps)
    # LR at resume start on *global* schedule
    current_lr = interp_linear(initial_lr, final_lr, frac_done)

    # Now produce a schedule from current_lr -> final_lr across remaining chunk.
    # Equivalent to SB3 get_linear_fn(current_lr, final_lr, 1.0).
    lr_schedule_fn = get_linear_fn(current_lr, final_lr, 1.0)
    return lr_schedule_fn, current_lr


# ==================================================================
# Callback with state saving/loading
# ==================================================================
class VLMSimpleCallback(BaseCallback):
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
        self.extra_state = {}

    def _on_step(self) -> bool:
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

        # Clear CUDA cache and log memory every 100 timesteps
        if torch.cuda.is_available() and (self.num_timesteps - self.last_cache_clear_step >= 100):
            torch.cuda.empty_cache()
            mem_stats = get_memory_stats()
            print(f"[Step {self.num_timesteps}] Cleared CUDA cache. Memory usage: {mem_stats}")
            if wandb.run:
                wandb.log({f"Memory/{k}": v for k, v in mem_stats.items()}, step=self.num_timesteps)
            self.last_cache_clear_step = self.num_timesteps

        # Save checkpoint every CHECKPOINT_INTERVAL timesteps
        if self.num_timesteps - self.last_checkpoint_step >= CHECKPOINT_INTERVAL:
            self.save_training_state(CHECKPOINT_PATH)
            print(f"[Step {self.num_timesteps}] Checkpoint saved: {CHECKPOINT_PATH}")
            self.last_checkpoint_step = self.num_timesteps

        # Log VLM weights occasionally
        if ENV_USE_VLM_WEIGHTS and (self.num_timesteps % 10 == 0) and (self.num_timesteps > self.last_vlm_weights_logged_step):
            vlm_log_file = self.vlm_decision_log_path_getter()
            if vlm_log_file:
                vlm_weights_data = read_vlm_weights_from_log(vlm_log_file)
                if vlm_weights_data and wandb.run:
                    wandb.log({f"VLM_Weights/{k}": float(v) for k, v in vlm_weights_data.items()},
                              step=self.num_timesteps)
                    self.last_vlm_weights_logged_step = self.num_timesteps

        # Episode done?
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
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return True

    def save_training_state(self, checkpoint_path):
        # Save PPO model
        self.model.save(checkpoint_path)
        # Save callback state
        state = {
            'episode_rewards_list': self.episode_rewards_list,
            'episode_lengths_list': self.episode_lengths_list,
            'current_episode_reward_sum': self.current_episode_reward_sum,
            'current_episode_step_count': self.current_episode_step_count,
            'last_vlm_weights_logged_step': self.last_vlm_weights_logged_step,
            'last_cache_clear_step': self.last_cache_clear_step,
            'last_checkpoint_step': self.last_checkpoint_step,
            'extra_state': self.extra_state
        }
        with open(checkpoint_path + '.cb.pkl', 'wb') as f:
            pickle.dump(state, f)

    def load_training_state(self, checkpoint_path):
        with open(checkpoint_path + '.cb.pkl', 'rb') as f:
            state = pickle.load(f)
        self.episode_rewards_list = state['episode_rewards_list']
        self.episode_lengths_list = state['episode_lengths_list']
        self.current_episode_reward_sum = state['current_episode_reward_sum']
        self.current_episode_step_count = state['current_episode_step_count']
        self.last_vlm_weights_logged_step = state['last_vlm_weights_logged_step']
        self.last_cache_clear_step = state['last_cache_clear_step']
        self.last_checkpoint_step = state['last_checkpoint_step']
        self.extra_state = state['extra_state']


# ==================================================================
# Directory management
# ==================================================================
def setup_directories():
    print("--- Setting up experiment directories ---")
    dirs_to_create = [OUTPUT_DIR, VLM_OUTPUT_DIR, FRAMES_DIR, MODEL_SAVE_PATH, LOG_DIR]
    if EXTRACTOR_HEATMAP_SAVE_SUBDIR:
        fe_internal_heatmap_dir = os.path.join(OUTPUT_DIR, EXTRACTOR_HEATMAP_SAVE_SUBDIR)
        dirs_to_create.append(fe_internal_heatmap_dir)
    for directory in dirs_to_create:
        os.makedirs(directory, exist_ok=True)
        print(f"  - Directory ensured: {directory}")
    print("--- Directories setup complete ---\n")


# ==================================================================
# Global handle to VLM controller (single‑process use)
# ==================================================================
global vlm_controller_instance
vlm_controller_instance = None

def get_vlm_log_file_path_for_callback():
    global vlm_controller_instance
    return vlm_controller_instance.log_file if vlm_controller_instance else None


# ==================================================================
# Environment state persistence (optional but useful)
# ==================================================================
def save_env_state(env, checkpoint_path):
    # Save frame buffer and any other relevant state
    state = {}
    if hasattr(env, 'frame_buffer'):
        state['frame_buffer'] = env.frame_buffer
    if hasattr(env, 'vlm_controller') and hasattr(env.vlm_controller, 'get_state'):
        state['vlm_controller_state'] = env.vlm_controller.get_state()
    with open(checkpoint_path + '.env.pkl', 'wb') as f:
        pickle.dump(state, f)

def load_env_state(env, checkpoint_path):
    try:
        with open(checkpoint_path + '.env.pkl', 'rb') as f:
            state = pickle.load(f)
        if 'frame_buffer' in state and hasattr(env, 'frame_buffer'):
            env.frame_buffer = state['frame_buffer']
        if 'vlm_controller_state' in state and hasattr(env, 'vlm_controller') and hasattr(env.vlm_controller, 'set_state'):
            env.vlm_controller.set_state(state['vlm_controller_state'])
    except Exception as e:
        print(f"Warning: Could not load environment state: {e}")


# ==================================================================
# Main
# ==================================================================
def main():
    global vlm_controller_instance

    print(f"🚗 CARLA VLM-Enhanced PPO Training\n" + "="*70)
    print(f"📁 Experiment: {EXPERIMENT_NAME}")
    print(f"🔩 Feature Extractor: {FEATURE_EXTRACTOR_SETUP['type']}")
    if EXTRACTOR_HEATMAP_SAVE_SUBDIR:
        print(f"🔥 Heatmaps will be saved by the extractor to: {EXTRACTOR_HEATMAP_SAVE_SUBDIR}")
    print("="*70)

    setup_directories()

    # Minimal, clean W&B config (avoid dumping all globals)
    wandb_config = {
        "total_timesteps": TOTAL_TIMESTEPS,
        "seed": SEED,
        "feature_extractor": FEATURE_EXTRACTOR_SETUP,
        "ppo": {k: v for k, v in PPO_CONFIG.items() if k != "net_arch_mlp_head"},
        "net_arch_mlp_head": PPO_CONFIG["net_arch_mlp_head"],
        "env_flags": {
            "use_vlm_weights": ENV_USE_VLM_WEIGHTS,
            "use_vlm_actions": ENV_USE_VLM_ACTIONS,
            "normalize_rewards": ENV_NORMALIZE_REWARDS,
        },
    }
    wandb.init(
        project=WANDB_PROJECT,
        name=EXPERIMENT_NAME,
        entity=WANDB_ENTITY,
        config=wandb_config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )

    print("🧠 Creating VLM controller...")
    vlm_controller_instance = VLMController(
        model_name=VLM_MODEL_NAME,
        update_frequency=VLM_UPDATE_FREQUENCY,
        frames_needed=VLM_FRAMES_NEEDED,
        output_dir=VLM_OUTPUT_DIR,
        max_new_tokens=VLM_MAX_TOKENS,
        verbose=VLM_VERBOSE,
        efficiency_priority=EFFICIENCY_PRIORITY,
    )
    print(f"✅ VLM controller created. Log file: {vlm_controller_instance.log_file}")

    print("🌍 Creating CARLA environment...")
    def make_env_fn():
        env = CarlaEnvironment(
            render_mode=ENV_RENDER_MODE,
            use_vlm_weights=ENV_USE_VLM_WEIGHTS,
            use_vlm_actions=ENV_USE_VLM_ACTIONS,
            normalize_rewards=ENV_NORMALIZE_REWARDS,
        )
        env.seed(SEED + random.randint(0, 100))  # legacy Gym seeding; adjust if Gymnasium
        env.frame_save_dir = FRAMES_DIR
        env.vlm_controller = vlm_controller_instance
        return env

    vec_env = DummyVecEnv([make_env_fn])
    monitor_log_path = os.path.join(LOG_DIR, "monitor.csv")
    vec_env = VecMonitor(vec_env, filename=monitor_log_path)
    print(f"✅ CARLA VecEnv created. Monitor logs: {monitor_log_path}")

    # Policy kwargs (feature extractor selection)
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
            activation_fn=nn.ReLU,
        )

    elif extractor_type == "PEDESTRIAN_ATTENTIVE":
        print("🖼️ Using Custom ResnetAttention Extractor.")
        if ResnetAttention is None:
            raise ImportError("ResnetAttention not available. Ensure Models/resnet18_attention.py exists and is imported correctly.")
        policy_kwargs = dict(
            features_extractor_class=ResnetAttention,
            features_extractor_kwargs=dict(
                features_dim=shared_features_dim,
                resnet18_unet_weights_path=FEATURE_EXTRACTOR_SETUP["resnet18_unet_weights_path"],
            ),
            net_arch=PPO_CONFIG.get("net_arch_mlp_head"),
            activation_fn=nn.ReLU,
        )

    else:
        raise ValueError(f"Unknown FEATURE_EXTRACTOR_SETUP type: {extractor_type}")

    # --------------------------------------------------------------
    # Determine resume step BEFORE creating model so LR schedule is right
    # --------------------------------------------------------------
    start_timestep = 0
    has_checkpoint = os.path.exists(CHECKPOINT_PATH)

    if has_checkpoint:
        # load callback state first to learn last step
        try:
            with open(CHECKPOINT_PATH + '.cb.pkl', 'rb') as f:
                _tmp_state = pickle.load(f)
            start_timestep = int(_tmp_state.get('last_checkpoint_step', 0))
            print(f"Detected prior training step from callback state: {start_timestep}")
        except Exception as e:
            print(f"Could not read callback state pre‑model‑load ({e}); assuming fresh start.")

    # Build continuation learning‑rate schedule (global → remaining)
    lr_schedule_fn, current_lr = make_continuation_lr_schedule(
        global_start_step=start_timestep,
        global_total_steps=TOTAL_TIMESTEPS,
        initial_lr=INITIAL_LR,
        final_lr=FINAL_LR,
    )

    # Assemble PPO hyperparams, inserting schedule
    model_hyperparams = {k: v for k, v in PPO_CONFIG.items() if k not in ("net_arch_mlp_head", "learning_rate")}
    model_hyperparams["learning_rate"] = lr_schedule_fn

    print(f"🤖 Creating PPO model... (resume step={start_timestep}, start_lr={current_lr:.6g})")
    model = PPO(
        "CnnPolicy",
        vec_env,
        tensorboard_log=LOG_DIR,
        policy_kwargs=policy_kwargs,
        seed=SEED,
        **model_hyperparams,
    )
    print("✅ PPO model created.")

    # If checkpoint exists, load weights *into* the freshly constructed model
    if has_checkpoint:
        print(f"Loading model weights from checkpoint: {CHECKPOINT_PATH}")
        model = PPO.load(
            CHECKPOINT_PATH,
            env=vec_env,
            tensorboard_log=LOG_DIR,
            policy_kwargs=policy_kwargs,
            seed=SEED,
            **model_hyperparams,
        )
        # SB3 load will carry its saved optimizer LR; overwrite to continuation LR
        model.lr_schedule = lr_schedule_fn
        model.learning_rate = current_lr

    # Callback
    callback = VLMSimpleCallback(vlm_decision_log_path_getter=get_vlm_log_file_path_for_callback, verbose=1)
    callback.model = model  # set immediately

    # Load callback / env state if resuming
    if has_checkpoint:
        print("Loading callback state...")
        try:
            callback.load_training_state(CHECKPOINT_PATH)
        except Exception as e:
            print(f"Warning: failed to load callback state ({e}). Starting fresh stats.")
        print("Loading environment state...")
        try:
            load_env_state(vec_env.envs[0], CHECKPOINT_PATH)
        except Exception as e:
            print(f"Warning: failed to load env state ({e}).")

        # Ensure callback internal step matches resume
        callback.last_checkpoint_step = start_timestep

    # --------------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------------
    remaining = max(0, TOTAL_TIMESTEPS - start_timestep)
    print(f"\n🚀 Starting PPO training: target total={TOTAL_TIMESTEPS:,} | already={start_timestep:,} | remaining={remaining:,}")

    start_time = time.time()

    while remaining > 0:
        try:
            model.learn(
                total_timesteps=remaining,
                callback=callback,
                reset_num_timesteps=False,  # keep global counter continuity
                tb_log_name=EXPERIMENT_NAME_BASE,
            )
            break  # Training finished
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print("\n⚠️ CUDA OOM detected. Attempting to recover from checkpoint...")
                torch.cuda.empty_cache()
                if os.path.exists(CHECKPOINT_PATH):
                    # reload model + callback state
                    model = PPO.load(
                        CHECKPOINT_PATH,
                        env=vec_env,
                        tensorboard_log=LOG_DIR,
                        policy_kwargs=policy_kwargs,
                        seed=SEED,
                        **model_hyperparams,
                    )
                    # patch LR schedule again (in case of partial progress)
                    callback.load_training_state(CHECKPOINT_PATH)
                    start_timestep = callback.last_checkpoint_step
                    remaining = max(0, TOTAL_TIMESTEPS - start_timestep)
                    lr_schedule_fn, current_lr = make_continuation_lr_schedule(
                        global_start_step=start_timestep,
                        global_total_steps=TOTAL_TIMESTEPS,
                        initial_lr=INITIAL_LR,
                        final_lr=FINAL_LR,
                    )
                    model.lr_schedule = lr_schedule_fn
                    model.learning_rate = current_lr
                    callback.model = model
                    load_env_state(vec_env.envs[0], CHECKPOINT_PATH)
                    print(f"Resumed from checkpoint at step {start_timestep} (remaining={remaining}).")
                else:
                    print("No checkpoint found. Exiting.")
                    raise
            else:
                raise

        # If we caught an OOM and resumed, loop again with adjusted 'remaining'

        # Compute whether finished (model.learn may have advanced)
        # SB3 increments model.num_timesteps internally; we mirror that
        remaining = max(0, TOTAL_TIMESTEPS - int(model.num_timesteps))

    # --------------------------------------------------------------
    # SAVE FINAL MODEL
    # --------------------------------------------------------------
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
        mean_rew = np.mean(callback.episode_rewards_list[-100:])
        print(f"  - Mean Reward (last 100 eps): {mean_rew:.2f}")
        print(f"  - Best Ep Reward: {max(callback.episode_rewards_list):.2f}")

    vec_env.close()
    print("🗑️ CARLA VecEnv closed.")

    if wandb.run:
        wandb.run.finish()
        print("👌 WandB run finished.")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ==================================================================
# Entrypoint
# ==================================================================
if __name__ == "__main__":
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    main()