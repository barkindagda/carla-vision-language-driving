import torch
import numpy as np
from typing import List, Dict
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
import os
import time
import json
import re
from tqdm import tqdm # Import tqdm

class VLMScorer:
    """
    Adjusts reward weights using VideoLLaMA. This version processes one clip at a time
    to ensure a low memory footprint and avoid OOM errors.
    """
    def __init__(
        self,
        model_name: str = "DAMO-NLP-SG/VideoLLaMA3-2B-Image",
        device: str = "cuda",
        batch_size: int = 1,
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
        """Load the VideoLLaMA model using 8-bit quantization and accelerate."""
        if self.verbose:
            print(f"Loading Quantized (8-bit) VideoLLaMA model: {self.model_name}")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                load_in_8bit=True,
                device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            if self.verbose:
                print(f"Quantized model loaded successfully across devices.")
        except Exception as e:
            print(f"Error loading quantized model: {e}")
            raise RuntimeError(f"Failed to initialize VLM model: {e}")

    def _get_dynamic_instruction(self, vehicle_state: Dict) -> str:
        speed_kmh = vehicle_state.get("speed_kmh", 0)
        context_info = f"""The self-driving vehicle's state at the end of this 3-frame clip is:
- Speed: {speed_kmh:.1f} km/h
- Acceleration: {vehicle_state.get('acceleration', 0):.2f} m/s²
"""
        instruction = f"""{context_info}
Your task is to act as a co-pilot. Based on the 3-frame video clip, suggest adjustments to the agent's reward priorities. The default weights are all 1.0.

Provide ONLY percentage adjustments in the format:
**SAFETY: [+/-]X%, COMFORT: [+/-]X%, EFFICIENCY: [+/-]X%**

**Examples:**
- A pedestrian is very close: Increase safety focus.
  `SAFETY: +50%, COMFORT: +0%, EFFICIENCY: -30%`
- The road ahead is completely empty and straight: Focus on making progress.
  `SAFETY: -20%, COMFORT: +10%, EFFICIENCY: +20%`

Provide your adjustments for the given clip.
"""
        return instruction

    def _parse_adjustments_from_text(self, text: str) -> Dict[str, float]:
        weights = {"safety": 1.0, "comfort": 1.0, "efficiency": 1.0}
        try:
            pattern = r"(SAFETY|COMFORT|EFFICIENCY)\s*:\s*([+-])\s*(\d+)\s*%"
            matches = re.findall(pattern, text, re.IGNORECASE)
            for category, sign, value in matches:
                key = category.lower()
                adjustment = float(value) / 100.0
                if sign == '+':
                    weights[key] = 1.0 + adjustment
                elif sign == '-':
                    weights[key] = 1.0 - adjustment
            for key in weights:
                weights[key] = np.clip(weights[key], 0.1, 2.0)
            w_s, w_c, w_e = weights["safety"], weights["comfort"], weights["efficiency"]
            product = w_s * w_c * w_e
            geo_mean = product ** (1/3)
            if geo_mean > 1e-6:
                weights["safety"] = w_s / geo_mean
                weights["comfort"] = w_c / geo_mean
                weights["efficiency"] = w_e / geo_mean
        except Exception as e:
            if self.verbose:
                print(f"Error parsing or normalizing weights: {e}, text: '{text}'")
                return {"safety": 1.0, "comfort": 1.0, "efficiency": 1.0}
        return weights

    def _log_weights(self, weights: Dict[str, float], vehicle_state: Dict, sequence_id: str):
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
        Processes the rollout one clip at a time to minimize memory usage.
        """
        num_timesteps = len(frames_rollout)
        timestep_weights = np.ones((num_timesteps, 3), dtype=np.float32)

        if num_timesteps < clip_size:
            if self.verbose: print("Rollout too short to create any clips.")
            return timestep_weights

        num_clips = len(range(0, num_timesteps - clip_size + 1, step_size))
        if self.verbose: 
            print(f"Processing rollout of {num_timesteps} steps into {num_clips} clips...")

        # Wrap the loop with tqdm for a progress bar
        loop_range = range(0, num_timesteps - clip_size + 1, step_size)
        for i in tqdm(loop_range, desc="Scoring VLM Clips", disable=not self.verbose):
            end_idx = i + clip_size
            
            # Create a batch of size 1 for the current clip
            clip_frames = frames_rollout[i:end_idx][np.newaxis, ...]
            clip_speeds = speeds_rollout[i:end_idx][np.newaxis, ...]
            clip_infos = [infos_rollout[i:end_idx]]
            
            # Score this single clip
            weights_for_clip = self._get_clip_weights_batch(clip_frames, clip_speeds, clip_infos)
            
            if weights_for_clip.shape[0] > 0:
                # Assign the resulting weights to all timesteps in the current clip window
                for j in range(clip_size):
                    timestep_weights[i+j] = weights_for_clip[0]

        return timestep_weights

    def _get_clip_weights_batch(self, frames_batch: np.ndarray, speeds_batch: np.ndarray, infos_batch: List[List[Dict]]) -> np.ndarray:
        """Gets weights for a batch of one or more short video clips."""
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
            with torch.no_grad():
                inputs = self.processor(conversation=conversations, return_tensors="pt", padding=True)
                
                inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)

                generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
                output_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

                for i, text in enumerate(output_texts):
                    parsed_weights = self._parse_adjustments_from_text(text.split("assistant")[-1])
                    weights_array[i, 0] = parsed_weights["safety"]
                    weights_array[i, 1] = parsed_weights["comfort"]
                    weights_array[i, 2] = parsed_weights["efficiency"]
                    
                    last_info_in_segment = infos_batch[i][-1][0]
                    vehicle_state = { "speed_kmh": float(speeds_batch[i][-1] * 3.6), "acceleration": last_info_in_segment.get("acceleration", 0.0), "distance_to_goal": last_info_in_segment.get("distance_to_goal", 0.0) }
                    self._log_weights(parsed_weights, vehicle_state, sequence_ids[i])
        except Exception as e:
            if self.verbose: print(f"Error processing VLM batch: {e}")
        return weights_array