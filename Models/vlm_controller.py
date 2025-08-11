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
        """
        Initialize the VLM scorer.
        Args:
            model_name: HuggingFace model name for VideoLLaMA.
            device: Device to run the model on (e.g., 'cuda' or 'cpu').
            batch_size: Number of segments to process in a batch.
            max_new_tokens: Maximum tokens for VLM generation.
            output_dir: Directory to save logs.
            verbose: Whether to print detailed logs.
        """
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.output_dir = output_dir
        self.verbose = verbose

        os.makedirs(output_dir, exist_ok=True)
        self.log_file = os.path.join(output_dir, f"vlm_scores_{int(time.time())}.json")
        with open(self.log_file, 'w') as f:
            json.dump({"initialization": time.time(), "scores": []}, f)

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
        Args:
            vehicle_state: Dictionary with state info (e.g., speed_kmh, acceleration).
        Returns:
            str: Prompt for the VLM.
        """
        speed_kmh = vehicle_state.get("speed_kmh", 0)
        context_info = f"""Current vehicle state:
 - Speed: {speed_kmh:.1f} km/h
 - Acceleration: {vehicle_state.get('acceleration', 0):.2f} m/s²
 - Distance to goal: {vehicle_state.get('distance_to_goal', 0):.2f} meters
 """

        # MODIFICATION: Changed the prompt to specify 3 frames.
        instruction = """Analyze the provided video clip (3 frames) of a self-driving car at a crosswalk. Evaluate the driving behavior for safety, comfort, and efficiency. For each objective, provide:

 - VALUE: A score between -1.0 (very poor) and +1.0 (excellent)
 - JUSTIFICATION: A brief explanation based on the frames and vehicle state

Make decisions based solely on the frames and vehicle state. Provide your response in this format:

SAFETY_VALUE: [number between -1.0 and +1.0]
SAFETY_JUSTIFICATION: [explanation]
COMFORT_VALUE: [number between -1.0 and +1.0]
COMFORT_JUSTIFICATION: [explanation]
EFFICIENCY_VALUE: [number between -1.0 and +1.0]
EFFICIENCY_JUSTIFICATION: [explanation]
 """
        return f"{context_info}\n\n{instruction}"

    def _parse_scores_from_text(self, text: str) -> Dict[str, float]:
        """
        Parse safety, comfort, and efficiency scores from VLM output.
        Args:
            text: Raw VLM output.
        Returns:
            Dict with 'safety', 'comfort', 'efficiency' scores in [0, 1].
        """
        scores = {"safety": 0.5, "comfort": 0.5, "efficiency": 0.5}
        try:
            for objective in ["SAFETY", "COMFORT", "EFFICIENCY"]:
                # Match numbers like -1.0, 0.5, +1.0, etc.
                value_match = re.search(rf'{objective}_VALUE:\s*([-+]?\d*\.?\d+)', text)
                if value_match:
                    try:
                        value = float(value_match.group(1))
                        # Ensure value is in [-1, 1]
                        value = max(min(value, 1.0), -1.0)
                        # Map [-1, 1] to [0, 1]
                        scores[objective.lower()] = (value + 1) / 2
                    except ValueError:
                        if self.verbose:
                            print(f"Invalid {objective}_VALUE in text: {text}")
                else:
                    if self.verbose:
                        print(f"Missing {objective}_VALUE in text: {text}")
        except Exception as e:
            if self.verbose:
                print(f"Error parsing scores: {e}, returning default 0.5")
        return scores

    def _log_score(self, scores: Dict[str, float], vehicle_state: Dict, sequence_id: str, text: str):
        """
        Log score data to file.
        """
        try:
            sanitized_state = {k: v if isinstance(v, (int, float, str, bool)) or v is None else str(v)
                               for k, v in vehicle_state.items()}
            log_entry = {
                "timestamp": time.time(),
                "sequence_id": sequence_id,
                "vehicle_state": sanitized_state,
                "safety_score": scores["safety"],
                "comfort_score": scores["comfort"],
                "efficiency_score": scores["efficiency"],
                "raw_output": text
            }
            try:
                with open(self.log_file, 'r') as f:
                    log_data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                log_data = {"initialization": time.time(), "scores": []}
            log_data["scores"].append(log_entry)
            temp_file = f"{self.log_file}.temp"
            with open(temp_file, 'w') as f:
                json.dump(log_data, f, indent=2)
            os.replace(temp_file, self.log_file)
        except Exception as e:
            if self.verbose:
                print(f"Error logging score: {e}")

    def score_segment_batch(self, frames_batch: np.ndarray, speeds_batch: np.ndarray, infos_batch: List[List[Dict]]) -> np.ndarray:
        """
        Score a batch of segments.
        Args:
            frames_batch: np.ndarray of shape [batch_size, segment_length, 3, 384, 384]
            speeds_batch: np.ndarray of shape [batch_size, segment_length]
            infos_batch: List of [batch_size, segment_length] info dicts
        Returns:
            np.ndarray: Scores for safety, comfort, efficiency of shape [batch_size, 3]
        """
        batch_size = frames_batch.shape[0]
        scores = np.zeros((batch_size, 3), dtype=np.float32)  # [safety, comfort, efficiency]
        sequence_ids = [f"seq_{int(time.time())}_{i}" for i in range(batch_size)]

        # Prepare batch inputs
        conversations = []
        for i in range(batch_size):
            segment_frames = frames_batch[i]  # [segment_length, 3, 384, 384]
            content = []
            
            # MODIFICATION: Select exactly 3 frames (first, middle, last) for processing.
            # This ensures the VLM always sees 3 frames as requested in the prompt.
            num_input_frames = segment_frames.shape[0]
            if num_input_frames > 0:
                indices = np.linspace(0, num_input_frames - 1, 3, dtype=int)
                frames_to_process = segment_frames[indices]
            else:
                frames_to_process = [] # Handle empty case

            # Process each of the 3 selected frames in the segment
            for frame in frames_to_process:
                try:
                    if frame.shape != (3, 384, 384):
                        if self.verbose:
                            print(f"Unexpected frame shape: {frame.shape}, attempting to reshape")
                        if len(frame.shape) == 2:  # [384, 384]
                            frame = np.stack([frame] * 3)  # Convert to RGB
                        elif frame.shape == (384, 384, 3):
                            frame = frame.transpose(2, 0, 1)  # HWC to CHW
                    
                    # Convert from CHW to HWC for PIL
                    frame = frame.transpose(1, 2, 0)  # [384, 384, 3]
                    
                    # Ensure uint8 range
                    frame = np.clip(frame * 255 if frame.max() <= 1.0 else frame, 0, 255).astype(np.uint8)
                    
                    # Convert to PIL Image
                    image = Image.fromarray(frame)
                    content.append({"type": "image", "image": image})
                    
                except Exception as e:
                    if self.verbose:
                        print(f"Error processing a frame in batch {i}: {e}")
                    # Use blank frame as fallback
                    blank_frame = np.zeros((384, 384, 3), dtype=np.uint8)
                    content.append({"type": "image", "image": Image.fromarray(blank_frame)})

            try:
                # Safely extract info from infos_batch
                segment_infos = infos_batch[i]
                if isinstance(segment_infos, list) and segment_infos:
                    if isinstance(segment_infos[0], list):
                        # Handle nested list case
                        info_dict = segment_infos[0][0] if segment_infos[0] else {}
                    else:
                        # Handle flat list case
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
                    print(f"Info structure: {segment_infos[:2]}")  # Print first two elements for debugging

                instruction = self._get_dynamic_instruction(vehicle_state)
                content.append({"type": "text", "text": instruction})
                conversations.append({"role": "user", "content": content})

            except Exception as e:
                if self.verbose:
                    print(f"Error processing vehicle state for batch {i}: {e}")
                vehicle_state = {
                    "speed_kmh": float(speeds_batch[i].mean()),
                    "acceleration": 0.0,
                    "distance_to_goal": 0.0
                }
                instruction = self._get_dynamic_instruction(vehicle_state)
                content.append({"type": "text", "text": instruction})
                conversations.append({"role": "user", "content": content})

        # Process batch
        try:
            inputs = self.processor(conversation=conversations, return_tensors="pt")
            inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
                      for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)

            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
            output_texts = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )

            # Parse and log scores
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
                self._log_score(parsed_scores, vehicle_state, sequence_ids[i], text)

                # Adjust scores based on speeds and infos
                if np.any(speeds_batch[i] > 10.0):
                    scores[i, 1] *= 0.8  # Penalize comfort for high speed
                    scores[i, 2] *= 0.8  # Penalize efficiency
                for info in infos_batch[i]:
                    if info.get("pedestrian_distance", float("inf")) < 2.0:
                        scores[i, 0] *= 0.7  # Penalize safety for close pedestrians
                        scores[i, 1] *= 0.7  # Penalize comfort

        except Exception as e:
            if self.verbose:
                print(f"Error processing batch: {e}")
            scores = np.ones((batch_size, 3)) * 0.5

        return scores

    def score_segment(self, frames: np.ndarray, speeds: np.ndarray, infos: List[Dict]) -> np.ndarray:
        """
        Score a single segment.
        Args:
            frames: np.ndarray of shape [segment_length, 3, 384, 384]
            speeds: np.ndarray of shape [segment_length]
            infos: List of info dicts
        Returns:
            np.ndarray: Scores for [safety, comfort, efficiency] in [0, 1]
        """
        frames_batch = frames[np.newaxis, ...]
        speeds_batch = speeds[np.newaxis, ...]
        infos_batch = [infos]
        return self.score_segment_batch(frames_batch, speeds_batch, infos_batch)[0]