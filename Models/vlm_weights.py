import torch
import numpy as np
from typing import List, Dict
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
import os
import time
import json
import re

class VLMScorer:
    """
    Adjusts reward weights using VideoLLaMA based on driving context.
    This version processes a long rollout by breaking it into short, overlapping clips
    to provide dynamic, per-timestep reward weights that are normalized to enforce trade-offs.
    """
    def __init__(
        self,
        model_name: str = "DAMO-NLP-SG/VideoLLaMA3-2B-Image",
        device: str = "cuda",
        batch_size: int = 8,
        max_new_tokens: int = 32,
        output_dir: str = "./vlm_outputs",
        verbose: bool = True
    ):
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.output_dir = output_dir
        self.verbose = verbose
        os.makedirs(output_dir, exist_ok=True)
        self.log_file = os.path.join(output_dir, f"vlm_weights_{int(time.time())}.jsonl")
        self._load_model()

    def _load_model(self):
        """Load the VideoLLaMA model and processor."""
        if self.verbose:
            print(f"Loading VideoLLaMA model: {self.model_name}")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            if self.verbose:
                print(f"Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise RuntimeError(f"Failed to initialize VLM model: {e}")

    def _get_dynamic_instruction(self, vehicle_state: Dict) -> str:
        """Generate a prompt that asks the VLM to adjust reward weights."""
        speed_kmh = vehicle_state.get("speed_kmh", 0)
        context_info = f"""The self-driving vehicle's state at the end of this 3-frame clip is:
- Speed: {speed_kmh:.1f} km/h
- Acceleration: {vehicle_state.get('acceleration', 0):.2f} m/s²
"""
        instruction = f"""{context_info}
Your task is to act as a co-pilot. Based on the 3-frame video clip, suggest adjustments to the agent's reward priorities. The default weights are all 1.0.

Provide ONLY percentage adjustments in the format:
**SAFETY: [+/-]X%, COMFORT: [+/-]X%, EFFICIENCY: [+/-]X%**

- **Increase a weight** if that factor is critical right now.
- **Decrease a weight** if that factor is less important.
- Use **+0%** for no change.

**Examples:**
- A pedestrian is very close: Increase safety focus.
  `SAFETY: +50%, COMFORT: +0%, EFFICIENCY: -30%`
- The road ahead is completely empty and straight: Focus on making progress.
  `SAFETY: -20%, COMFORT: +10%, EFFICIENCY: +20%`
- Driving smoothly in moderate traffic: Maintain a balanced approach.
  `SAFETY: +0%, COMFORT: +0%, EFFICIENCY: +0%`

Provide your adjustments for the given clip.
"""
        return instruction

    def _parse_adjustments_from_text(self, text: str) -> Dict[str, float]:
        """
        Parse percentage adjustments, convert them to weights, and then normalize them
        so that their product is 1.
        """
        weights = {"safety": 1.0, "comfort": 1.0, "efficiency": 1.0}
        try:
            # Step 1: Parse VLM text to get initial weight adjustments
            pattern = r"(SAFETY|COMFORT|EFFICIENCY)\s*:\s*([+-])\s*(\d+)\s*%"
            matches = re.findall(pattern, text, re.IGNORECASE)
            
            for category, sign, value in matches:
                key = category.lower()
                adjustment = float(value) / 100.0
                if sign == '+':
                    weights[key] = 1.0 + adjustment
                elif sign == '-':
                    weights[key] = 1.0 - adjustment
            
            # Step 2: Clamp weights to a reasonable positive range
            for key in weights:
                weights[key] = np.clip(weights[key], 0.1, 2.0)

            # Step 3: Normalize the weights so their product is 1 (geometric mean normalization)
            w_s, w_c, w_e = weights["safety"], weights["comfort"], weights["efficiency"]
            product = w_s * w_c * w_e
            geo_mean = product ** (1/3)
            
            if geo_mean > 1e-6: # Avoid division by zero
                weights["safety"] = w_s / geo_mean
                weights["comfort"] = w_c / geo_mean
                weights["efficiency"] = w_e / geo_mean

        except Exception as e:
            if self.verbose:
                print(f"Error parsing or normalizing weights: {e}, text: '{text}'")
                return {"safety": 1.0, "comfort": 1.0, "efficiency": 1.0} # Return default on error
        
        return weights

    def _log_weights(self, weights: Dict[str, float], vehicle_state: Dict, sequence_id: str):
        """Log final weights to a JSONL file."""
        try:
            log_entry = {
                "timestamp": time.time(), "sequence_id": sequence_id,
                "vehicle_state": {k: float(v) if isinstance(v, (int, float, np.number)) else str(v) for k, v in vehicle_state.items()},
                "safety_weight": float(weights["safety"]),
                "comfort_weight": float(weights["comfort"]),
                "efficiency_weight": float(weights["efficiency"])
            }
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            if self.verbose: print(f"Error logging weights: {e}")
                
    def get_per_timestep_weights(self, frames_rollout: np.ndarray, speeds_rollout: np.ndarray, infos_rollout: List[List[Dict]], clip_size: int = 3, step_size: int = 1) -> np.ndarray:
        """
        Primary method called by the PPO algorithm.
        Returns: np.ndarray of shape (num_timesteps, 3) with weights [w1, w2, w3] for each timestep.
        """
        num_timesteps = len(frames_rollout)
        # Initialize with default weights of 1.0. Any timestep not in a clip will keep this default.
        timestep_weights = np.ones((num_timesteps, 3), dtype=np.float32)

        # Prepare clips from the full rollout
        clip_frames_batch, clip_speeds_batch, clip_infos_batch = [], [], []
        for i in range(0, num_timesteps - clip_size + 1, step_size):
            end_idx = i + clip_size
            clip_frames_batch.append(frames_rollout[i:end_idx])
            clip_speeds_batch.append(speeds_rollout[i:end_idx])
            clip_infos_batch.append(infos_rollout[i:end_idx])
            
        if not clip_frames_batch:
            if self.verbose: print("Rollout too short to create any clips.")
            return timestep_weights

        if self.verbose: print(f"Processing rollout of {num_timesteps} steps into {len(clip_frames_batch)} clips.")

        # Process all clips in batches to get their corresponding weights
        all_clip_weights = []
        for i in range(0, len(clip_frames_batch), self.batch_size):
            batch_frames = np.array(clip_frames_batch[i:i+self.batch_size])
            batch_speeds = np.array(clip_speeds_batch[i:i+self.batch_size])
            batch_infos = clip_infos_batch[i:i+self.batch_size]
            weights = self._get_clip_weights_batch(batch_frames, batch_speeds, batch_infos)
            all_clip_weights.extend(weights)

        # Correctly assign the weight for each clip only to the timesteps within that clip
        clip_idx = 0
        for i in range(0, num_timesteps - clip_size + 1, step_size):
            if clip_idx < len(all_clip_weights):
                weights_for_clip = all_clip_weights[clip_idx]
                # Apply this clip's weights to all frames it contains.
                # If clips overlap, the later clip's weights will overwrite the earlier ones.
                for j in range(clip_size):
                    timestep_weights[i+j] = weights_for_clip
                clip_idx += 1
        
        return timestep_weights

    def _get_clip_weights_batch(self, frames_batch: np.ndarray, speeds_batch: np.ndarray, infos_batch: List[List[Dict]]) -> np.ndarray:
        """Gets weights for a batch of short video clips."""
        batch_size = frames_batch.shape[0]
        weights_array = np.ones((batch_size, 3), dtype=np.float32)
        sequence_ids = [f"seq_{int(time.time())}_{i}" for i in range(batch_size)]

        conversations = []
        for i in range(batch_size):
            segment_frames = frames_batch[i]
            content = []
            
            indices = np.linspace(0, len(segment_frames) - 1, 3, dtype=int)
            for frame_idx in indices:
                frame = segment_frames[frame_idx]
                try:
                    if frame.shape[0] == 3: frame = frame.transpose(1, 2, 0)
                    content.append({"type": "image", "image": Image.fromarray(frame.astype(np.uint8))})
                except Exception as e:
                    if self.verbose: print(f"Error processing frame in batch {i}: {e}")
            
            last_info_in_segment = infos_batch[i][-1][0]
            vehicle_state = {
                "speed_kmh": float(speeds_batch[i][-1] * 3.6),
                "acceleration": last_info_in_segment.get("acceleration", 0.0),
                "distance_to_goal": last_info_in_segment.get("distance_to_goal", 0.0)
            }
            instruction = self._get_dynamic_instruction(vehicle_state)
            content.append({"type": "text", "text": instruction})
            conversations.append({"role": "user", "content": content})

        try:
            inputs = self.processor(conversation=conversations, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)

            generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
            output_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

            for i, text in enumerate(output_texts):
                parsed_weights = self._parse_adjustments_from_text(text.split("assistant")[-1])
                weights_array[i, 0] = parsed_weights["safety"]
                weights_array[i, 1] = parsed_weights["comfort"]
                weights_array[i, 2] = parsed_weights["efficiency"]
                # Log using the state from the last frame in the clip
                last_info_in_segment = infos_batch[i][-1][0]
                vehicle_state = { "speed_kmh": float(speeds_batch[i][-1] * 3.6), "acceleration": last_info_in_segment.get("acceleration", 0.0), "distance_to_goal": last_info_in_segment.get("distance_to_goal", 0.0) }
                self._log_weights(parsed_weights, vehicle_state, sequence_ids[i])

        except Exception as e:
            if self.verbose: print(f"Error processing VLM batch: {e}")
            # On error, weights_array remains at the default of 1.0
        return weights_array