import torch
import numpy as np
from typing import List, Dict
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
import os
import time
import json
import re
from tqdm import tqdm

class VLMScorer:
    """
    Assigns reward weights using VideoLLaMA for Language-Guided Potential Shaping.
    Outputs normalized weights based on numerical values in [0, 10] for safety, comfort, and efficiency,
    with geometric mean normalization to ensure their product is approximately 1.
    """
    def __init__(
        self,
        model_name: str = "DAMO-NLP-SG/VideoLLaMA3-2B-Image",
        device: str = "cuda",
        batch_size: int = 1,
        max_new_tokens: int = 64,  # Increased for detailed numerical output
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
                print(f"Model loaded on {self.model.device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise RuntimeError(f"Failed to initialize VLM model: {e}")

    def _get_dynamic_instruction(self, vehicle_state: Dict) -> str:
        """Generate a prompt for assigning numerical weights in [0, 10]."""
        speed_kmh = vehicle_state.get("speed_kmh", 0)
        acceleration = vehicle_state.get("acceleration", 0)
        distance_to_goal = vehicle_state.get("distance_to_goal", 0)

        context_info = f"""The self-driving vehicle's current state is:
- Speed: {speed_kmh:.1f} km/h
- Acceleration: {acceleration:.2f} m/s²
- Distance to goal: {distance_to_goal:.2f} meters
"""
        instruction = f"""{context_info}
You are a co-pilot for a self-driving vehicle in an emergency braking scenario with occluded pedestrians (e.g., pedestrians hidden behind obstacles like parked cars). The vehicle must detect risks in the 3-frame video clip (analyze for occluded areas, pedestrian movement, road conditions) and prioritize safety to avoid collisions, maintain comfort during braking, and ensure efficiency by resuming progress after braking. As the vehicle state changes (e.g., speed increases, acceleration varies, distance to goal decreases), the weights must adapt accordingly. Based on the clip and state, assign numerical weights in [0, 10] to safety, comfort, and efficiency. Default weights are 5.0. Weights will be normalized so their product is approximately 1, but assign raw values to reflect priorities: high safety (7.0-10.0) when collision risk is high (e.g., high speed close to goal or sudden deceleration suggesting risk), low safety (0.0-3.0) and high comfort/efficiency (7.0-10.0) when no risk is detected (e.g., low speed, smooth acceleration, far from goal).

Provide ONLY numerical weights in the format:
SAFETY: X.XX, COMFORT: X.XX, EFFICIENCY: X.XX

Objective Definitions:
- Safety: 7.0-10.0 if high collision risk with a pedestrian (e.g., high speed > 10 km/h when distance to goal < 5m, or negative acceleration indicating emergency brake); 0.0-3.0 if no risk (low speed < 5 km/h, positive or zero acceleration, distance to goal > 20m).
- Comfort: 0.0-3.0 if abrupt braking or jerky motion (|acceleration| > 1.5 m/s²); 7.0-10.0 if smooth driving (|acceleration| < 0.5 m/s²).
- Efficiency: 0.0-3.0 if stalled or slow (speed < 1 km/h); 7.0-10.0 if progressing well (speed > 5 km/h, positive acceleration).

The weights must change as the state changes—do not use the same weights repeatedly. Provide your weights for the given clip and state in the specified format. Do not include explanations or extra words.
"""
        return instruction

    def _parse_weights_from_text(self, text: str) -> Dict[str, float]:
        """Parse numerical weights from VLM output and normalize them."""
        weights = {"safety": 5.0, "comfort": 5.0, "efficiency": 5.0}
        try:
            pattern = r"(SAFETY|COMFORT|EFFICIENCY)\s*:\s*(\d+\.\d+)"
            matches = re.findall(pattern, text, re.IGNORECASE)
            if not matches:
                return weights
            for category, value in matches:
                key = category.lower()
                weight = float(value)
                weights[key] = max(min(weight, 10.0), 0.0)  # Clip to [0, 10]
            # Normalize weights using geometric mean to ensure product ~ 1
            w_s, w_c, w_e = weights["safety"], weights["comfort"], weights["efficiency"]
            product = w_s * w_c * w_e
            geo_mean = product ** (1/3) if product > 1e-6 else 1.0
            weights["safety"] = w_s / geo_mean
            weights["comfort"] = w_c / geo_mean
            weights["efficiency"] = w_e / geo_mean
        except Exception as e:
            return weights
        return weights

    def _log_weights(self, weights: Dict[str, float], vehicle_state: Dict, sequence_id: str, raw_output: str = ""):
        """Log weights to a JSONL file, including raw model output and raw weights."""
        try:
            log_entry = {
                "timestamp": time.time(),
                "sequence_id": sequence_id,
                "vehicle_state": {k: float(v) if isinstance(v, (int, float, np.number)) else str(v) for k, v in vehicle_state.items()},
                "safety_weight": float(weights["safety"]),
                "comfort_weight": float(weights["comfort"]),
                "efficiency_weight": float(weights["efficiency"]),
                "raw_model_output": raw_output,
                "raw_weights": {k: float(v) for k, v in weights.items()}
            }
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            if self.verbose:
                print(f"Error logging weights: {e}")

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
            # if self.verbose:
            #     print(f"Clip {i} frame shapes: {[frame.shape for frame in segment_frames[indices]]}")
            #     print(f"Clip {i} pixel mean: {[frame.mean() for frame in segment_frames[indices]]}")
            for frame_idx in indices:
                frame = segment_frames[frame_idx]
                try:
                    if frame.shape[0] == 3: frame = frame.transpose(1, 2, 0)
                    content.append({"type": "image", "image": Image.fromarray(frame.astype(np.uint8))})
                except Exception as e:
                    if self.verbose: print(f"Error processing frame in batch {i}: {e}")
                    blank_frame = np.zeros((384, 384, 3), dtype=np.uint8)
                    content.append({"type": "image", "image": Image.fromarray(blank_frame)})
            
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
                    parsed_weights = self._parse_weights_from_text(text.split("assistant")[-1])
                    weights_array[i, 0] = parsed_weights["safety"]
                    weights_array[i, 1] = parsed_weights["comfort"]
                    weights_array[i, 2] = parsed_weights["efficiency"]
                    
                    last_info_in_segment = infos_batch[i][-1][0]
                    vehicle_state = {
                        "speed_kmh": float(speeds_batch[i][-1] * 3.6),
                        "acceleration": last_info_in_segment.get("acceleration", 0.0),
                        "distance_to_goal": last_info_in_segment.get("distance_to_goal", 0.0)
                    }
                    self._log_weights(parsed_weights, vehicle_state, sequence_ids[i], text)
        except Exception as e:
            if self.verbose: print(f"Error processing VLM batch: {e}")
        return weights_array