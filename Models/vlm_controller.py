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
    Scores video segments using VideoLLaMA for Language-Guided Potential Shaping.
    Outputs only numerical scores in [0, 1] for safety, comfort, and efficiency.
    """
    def __init__(
        self,
        model_name: str = "DAMO-NLP-SG/VideoLLaMA3-2B-Image",
        device: str = "cuda",
        batch_size: int = 1,
        max_new_tokens: int = 32,  # Reduced due to numerical-only output
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
        self.log_file = os.path.join(output_dir, f"vlm_scores_{int(time.time())}.jsonl")
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
        """
        Generate a prompt for scoring a driving scene with example scores.
        """
        speed_kmh = vehicle_state.get("speed_kmh", 0)
        context_info = f"""The self-driving vehicle's current state is:
- Speed: {speed_kmh:.1f} km/h
- Acceleration: {vehicle_state.get('acceleration', 0):.2f} m/s²
- Distance to goal: {vehicle_state.get('distance_to_goal', 0):.2f} meters
"""

        instruction = f"""{context_info}Evaluate the 3-frame video clip for safety, comfort, and efficiency. Provide ONLY numerical scores in [0, 1] in the format:
SAFETY: X.XX, COMFORT: X.XX, EFFICIENCY: X.XX

Objective Definitions and Examples:
- Safety: 0.0 = collision or very close to obstacles; 0.9 = clear road, no obstacles within 10 meters.
- Comfort: 0.0 = abrupt acceleration/braking (|accel| > 2 m/s²); 0.9 = smooth driving (|accel| < 0.5 m/s²).
- Efficiency: 0.0 = stalled (speed < 1 km/h); 0.9 = optimal speed (~20 km/h) toward goal.

Example Responses:
- Clear road, steady 20 km/h, smooth: SAFETY: 0.90, COMFORT: 0.90, EFFICIENCY: 0.90
- Pedestrian 1m away, sudden brake: SAFETY: 0.10, COMFORT: 0.20, EFFICIENCY: 0.30
- Stalled, jerky motion: SAFETY: 0.50, COMFORT: 0.10, EFFICIENCY: 0.10

Provide your scores for the given clip and state in the specified format.
"""
        return instruction

 # In vlm_controller.py

    def _parse_scores_from_text(self, text: str) -> Dict[str, float]:
        """
        Parse numerical scores from the VLM output using regular expressions
        to handle conversational or multi-line text.
        """
        # Default scores to return if parsing fails for any category.
        scores = {"safety": 0.5, "comfort": 0.5, "efficiency": 0.5}
        
        try:
            # Define regex patterns to find "CATEGORY: X.XX"
            # The pattern looks for the category name, a colon, optional whitespace,
            # and captures the floating-point number.
            safety_pattern = r"SAFETY:\s*(\d+\.\d+)"
            comfort_pattern = r"COMFORT:\s*(\d+\.\d+)"
            efficiency_pattern = r"EFFICIENCY:\s*(\d+\.\d+)"

            # Search for each pattern in the full text
            safety_match = re.search(safety_pattern, text, re.IGNORECASE)
            comfort_match = re.search(comfort_pattern, text, re.IGNORECASE)
            efficiency_match = re.search(efficiency_pattern, text, re.IGNORECASE)

            # If a match is found, extract the captured number (group 1)
            if safety_match:
                scores["safety"] = max(min(float(safety_match.group(1)), 1.0), 0.0)
            if comfort_match:
                scores["comfort"] = max(min(float(comfort_match.group(1)), 1.0), 0.0)
            if efficiency_match:
                scores["efficiency"] = max(min(float(efficiency_match.group(1)), 1.0), 0.0)

        except Exception as e:
            if self.verbose:
                # This will now only print if the regex itself fails, which is rare.
                print(f"Error parsing scores with regex: {e}, text: {text}")
        
        return scores

    def _log_score(self, scores: Dict[str, float], vehicle_state: Dict, sequence_id: str):
        """Log scores to a JSONL file."""
        try:
            log_entry = {
                "timestamp": time.time(),
                "sequence_id": sequence_id,
                "vehicle_state": {k: float(v) if isinstance(v, (int, float)) else str(v) for k, v in vehicle_state.items()},
                "safety_score": float(scores["safety"]),
                "comfort_score": float(scores["comfort"]),
                "efficiency_score": float(scores["efficiency"])
            }
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            if self.verbose:
                print(f"Error logging score: {e}")

    def score_segment_batch(self, frames_batch: np.ndarray, speeds_batch: np.ndarray, infos_batch: List[List[Dict]]) -> np.ndarray:
        """
        Score a batch of segments.
        Args:
            frames_batch: Shape (batch_size, segment_length, 3, 384, 384)
            speeds_batch: Shape (batch_size, segment_length)
            infos_batch: List of lists of vehicle state dictionaries
        Returns:
            np.ndarray: Shape (batch_size, 3) with [safety, comfort, efficiency] scores
        """
        batch_size = frames_batch.shape[0]
        scores = np.zeros((batch_size, 3), dtype=np.float32)
        sequence_ids = [f"seq_{int(time.time())}_{i}" for i in range(batch_size)]

        conversations = []
        for i in range(batch_size):
            segment_frames = frames_batch[i]
            content = []

            num_input_frames = segment_frames.shape[0]
            if num_input_frames > 0:
                indices = np.linspace(0, num_input_frames - 1, 3, dtype=int)
                frames_to_process = segment_frames[indices]
            else:
                frames_to_process = []

            for frame in frames_to_process:
                try:
                    if frame.shape != (3, 384, 384):
                        if self.verbose:
                            print(f"Unexpected frame shape: {frame.shape}, attempting to reshape")
                        if frame.shape == (384, 384, 3):
                            frame = frame.transpose(2, 0, 1)
                        frame = np.clip(frame * 255 if frame.max() <= 1.0 else frame, 0, 255).astype(np.uint8)
                    frame = frame.transpose(1, 2, 0)
                    image = Image.fromarray(frame)
                    content.append({"type": "image", "image": image})
                except Exception as e:
                    if self.verbose:
                        print(f"Error processing a frame in batch {i}: {e}")
                    blank_frame = np.zeros((384, 384, 3), dtype=np.uint8)
                    content.append({"type": "image", "image": Image.fromarray(blank_frame)})

            try:
                segment_infos = infos_batch[i]
                info_dict = segment_infos[0] if isinstance(segment_infos, list) and segment_infos and isinstance(segment_infos[0], dict) else {}
                vehicle_state = {
                    "speed_kmh": float(speeds_batch[i].mean()),
                    "acceleration": info_dict.get("acceleration", 0.0),
                    "distance_to_goal": info_dict.get("distance_to_goal", 0.0)
                }
                instruction = self._get_dynamic_instruction(vehicle_state)
                content.append({"type": "text", "text": instruction})
                conversations.append({"role": "user", "content": content})
            except Exception as e:
                if self.verbose:
                    print(f"Error processing vehicle state for batch {i}: {e}")
                vehicle_state = {"speed_kmh": float(speeds_batch[i].mean()), "acceleration": 0.0, "distance_to_goal": 0.0}
                instruction = self._get_dynamic_instruction(vehicle_state)
                content.append({"type": "text", "text": instruction})
                conversations.append({"role": "user", "content": content})

        try:
            inputs = self.processor(conversation=conversations, return_tensors="pt")
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)

            generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)  # Deterministic output
            output_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

            for i, text in enumerate(output_texts):
                text = text.strip()
                parsed_scores = self._parse_scores_from_text(text)
                scores[i, 0] = parsed_scores["safety"]
                scores[i, 1] = parsed_scores["comfort"]
                scores[i, 2] = parsed_scores["efficiency"]
                vehicle_state = {
                    "speed_kmh": float(speeds_batch[i].mean()),
                    "acceleration": infos_batch[i][0].get("acceleration", 0.0),
                    "distance_to_goal": infos_batch[i][0].get("distance_to_goal", 0.0)
                }
                self._log_score(parsed_scores, vehicle_state, sequence_ids[i])

                # # Apply penalties (same as original)
                # if np.any(speeds_batch[i] > 10.0):
                #     scores[i, 1] *= 0.8  # Comfort penalty
                #     scores[i, 2] *= 0.8  # Efficiency penalty
                # for info in infos_batch[i]:
                #     if info.get("pedestrian_distance", float("inf")) < 2.0:
                #         scores[i, 0] *= 0.7  # Safety penalty
                #         scores[i, 1] *= 0.7  # Comfort penalty
        except Exception as e:
            if self.verbose:
                print(f"Error processing batch: {e}")
            scores = np.ones((batch_size, 3)) * 0.5
        return scores

    def score_segment(self, frames: np.ndarray, speeds: np.ndarray, infos: List[Dict]) -> np.ndarray:
        """Score a single segment."""
        return self.score_segment_batch(frames[np.newaxis, ...], speeds[np.newaxis, ...], [infos])[0]