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
    Generates safety, comfort, and efficiency scores in [0, 1] for each segment.
    """

    def __init__(
        self,
        model_name: str = "DAMO-NLP-SG/VideoLLaMA3-2B-Image",
        device: str = "cuda",
        batch_size: int = 1,
        max_new_tokens: int = 312,
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
        Generate a prompt for scoring a driving scene.
        """
        speed_kmh = vehicle_state.get("speed_kmh", 0)
        context_info = f"""The self-driving vehicle's current state is:
- Speed: {speed_kmh:.1f} km/h
- Acceleration: {vehicle_state.get('acceleration', 0):.2f} m/s²
- Distance to goal: {vehicle_state.get('distance_to_goal', 0):.2f} meters
"""

        # MODIFICATION: A much stricter prompt with a one-shot example to force consistency.
        # We are also asking for a [0.0, 1.0] score directly.
        instruction = """You are a precise driving analyst. Evaluate the 3-frame video clip based on the definitions below.

Objective Definitions:
- Safety: How safe the maneuver is. 0.0 is a collision; 1.0 is perfectly safe.
- Comfort: Ride smoothness. 0.0 is extremely jerky; 1.0 is perfectly smooth.
- Efficiency: Progress towards the goal. 0.0 is stalled or stuck; 1.0 is optimal progress.

You MUST provide your response ONLY in the following format, with no extra text or explanations before or after.

EXAMPLE RESPONSE:
SAFETY_VALUE: 0.9
SAFETY_JUSTIFICATION: The vehicle maintains a good distance from the car ahead.
COMFORT_VALUE: 0.8
COMFORT_JUSTIFICATION: The acceleration is smooth and controlled.
EFFICIENCY_VALUE: 0.6
EFFICIENCY_JUSTIFICATION: The vehicle is moving slower than the speed limit.

Now, provide your analysis for the given video clip and vehicle state.

"""
        return f"{context_info}{instruction}"

    def _parse_scores_from_text(self, text: str) -> Dict[str, float]:
        """
        MODIFICATION: A much more robust parser.
        It finds the objective keyword (case-insensitive) and then searches for the
        first number that appears after it.
        """
        scores = {"safety": 0.5, "comfort": 0.5, "efficiency": 0.5}
        
        # Make text lowercase for case-insensitive searching of keywords
        lower_text = text.lower()

        for objective in ["safety", "comfort", "efficiency"]:
            try:
                # Find the starting position of the keyword
                start_index = lower_text.find(objective)
                if start_index == -1:
                    if self.verbose:
                        print(f"Missing '{objective}' keyword in text.")
                    continue

                # Search for the first number pattern in the substring that follows the keyword
                search_area = text[start_index:]
                match = re.search(r'([-+]?\d*\.?\d+)', search_area)
                
                if match:
                    value = float(match.group(1))
                    # Clamp the value between 0 and 1, as requested in the new prompt
                    value = max(min(value, 1.0), 0.0)
                    scores[objective] = value
                else:
                    if self.verbose:
                        print(f"Found '{objective}' keyword but no number followed in text.")

            except Exception as e:
                if self.verbose:
                    print(f"Error parsing score for '{objective}': {e}")
                continue # Keep default score if an error occurs

        return scores


    def _log_score(self, scores: Dict[str, float], vehicle_state: Dict, sequence_id: str, text: str):
        """
        Logs score data to a .jsonl file in an efficient, append-only manner.
        """
        try:
            log_entry = {
                "timestamp": time.time(),
                "sequence_id": sequence_id,
                "vehicle_state": {k: v if isinstance(v, (int, float, str, bool, type(None))) else str(v) 
                                  for k, v in vehicle_state.items()},
                "safety_score": scores["safety"],
                "comfort_score": scores["comfort"],
                "efficiency_score": scores["efficiency"],
                "raw_output": text
            }
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            if self.verbose:
                print(f"Error logging score: {e}")

    # ... The score_segment_batch and score_segment methods remain unchanged ...
    def score_segment_batch(self, frames_batch: np.ndarray, speeds_batch: np.ndarray, infos_batch: List[List[Dict]]) -> np.ndarray:
        """
        Score a batch of segments.
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
                        if len(frame.shape) == 2:
                            frame = np.stack([frame] * 3)
                        elif frame.shape == (384, 384, 3):
                            frame = frame.transpose(2, 0, 1)
                    
                    frame = frame.transpose(1, 2, 0)
                    frame = np.clip(frame * 255 if frame.max() <= 1.0 else frame, 0, 255).astype(np.uint8)
                    image = Image.fromarray(frame)
                    content.append({"type": "image", "image": image})
                except Exception as e:
                    if self.verbose:
                        print(f"Error processing a frame in batch {i}: {e}")
                    blank_frame = np.zeros((384, 384, 3), dtype=np.uint8)
                    content.append({"type": "image", "image": Image.fromarray(blank_frame)})

            try:
                segment_infos = infos_batch[i]
                if isinstance(segment_infos, list) and segment_infos:
                    if isinstance(segment_infos[0], list):
                        info_dict = segment_infos[0][0] if segment_infos[0] else {}
                    else:
                        info_dict = segment_infos[0] if isinstance(segment_infos[0], dict) else {}
                else:
                    info_dict = {}

                vehicle_state = {
                    "speed_kmh": float(speeds_batch[i].mean()),
                    "acceleration": info_dict.get("acceleration", 0.0),
                    "distance_to_goal": info_dict.get("distance_to_goal", 0.0)
                }
                
                if self.verbose and not isinstance(info_dict, dict):
                    print(f"Warning: Invalid info format for batch {i}: {type(info_dict)}")
                    print(f"Info structure: {segment_infos[:2]}")

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
            inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)

            generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=True, temperature=0.7, top_p=0.9)
            output_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

            for i, text in enumerate(output_texts):
                text = text.strip()
                parsed_scores = self._parse_scores_from_text(text)
                scores[i, 0] = parsed_scores["safety"]
                scores[i, 1] = parsed_scores["comfort"]
                scores[i, 2] = parsed_scores["efficiency"]
                vehicle_state = {"speed_kmh": float(speeds_batch[i].mean()), "acceleration": infos_batch[i][0].get("acceleration", 0.0), "distance_to_goal": infos_batch[i][0].get("distance_to_goal", 0.0)}
                self._log_score(parsed_scores, vehicle_state, sequence_ids[i], text)

                if np.any(speeds_batch[i] > 10.0):
                    scores[i, 1] *= 0.8
                    scores[i, 2] *= 0.8
                for info in infos_batch[i]:
                    if info.get("pedestrian_distance", float("inf")) < 2.0:
                        scores[i, 0] *= 0.7
                        scores[i, 1] *= 0.7
        except Exception as e:
            if self.verbose:
                print(f"Error processing batch: {e}")
            scores = np.ones((batch_size, 3)) * 0.5
        return scores

    def score_segment(self, frames: np.ndarray, speeds: np.ndarray, infos: List[Dict]) -> np.ndarray:
        """
        Score a single segment.
        """
        frames_batch = frames[np.newaxis, ...]
        speeds_batch = speeds[np.newaxis, ...]
        infos_batch = [infos]
        return self.score_segment_batch(frames_batch, speeds_batch, infos_batch)[0]