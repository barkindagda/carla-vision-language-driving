import os
import time
import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

class VLMController:
    """
    Vision Language Model controller for determining reward weights for an
    autonomous vehicle agent using VideoLLaMA.
    """
    def __init__(
            self,
            model_name="DAMO-NLP-SG/VideoLLaMA3-2B-Image",
            update_frequency=5,
            frames_needed=3,
            output_dir="/home/cavlab/CARLA_0.9.15/VLM_Barkin/CarlaEnv/vlm_outputs",
            max_new_tokens=512,
            verbose=True,
            efficiency_priority=False
    ):
        """
        Initialize the VLM Controller.
        Args:
            model_name: HuggingFace model name for VideoLLaMA.
            update_frequency: How often to update weights (in timesteps).
            frames_needed: Number of frames to use for each decision.
            output_dir: Directory to save outputs and logs.
            max_new_tokens: Maximum tokens to generate in model responses.
            verbose: Whether to print detailed logs.
            efficiency_priority: Flag to use efficiency-focused instructions.
        """
        self.model_name = model_name
        self.update_frequency = update_frequency
        self.frames_needed = frames_needed
        self.output_dir = output_dir
        self.max_new_tokens = max_new_tokens
        self.verbose = verbose
        self.efficiency_priority = efficiency_priority
        self.last_update_timestep = 0

        # Create timestamped output directory
        timestamp = int(time.time())
        self.session_dir = os.path.join(output_dir, f"session_{timestamp}")
        os.makedirs(self.session_dir, exist_ok=True)
        
        self.model = None
        self.processor = None

        self.log_file = os.path.join(self.session_dir, "vlm_decisions.json")
        with open(self.log_file, 'w') as f:
            json.dump({"initialization": time.time(), "decisions": []}, f)

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
                print(f"Model loaded successfully on {self.model.device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise RuntimeError(f"Failed to initialize VLM model: {e}")

    def _process_frames_for_weights(self, frame_paths, vehicle_state):
        """
        Process a sequence of frames through VideoLLaMA to get reward weights.
        Args:
            frame_paths: List of paths to frame images.
            vehicle_state: Dictionary with current vehicle state.
        Returns:
            dict: Processing results for weights.
        """
        try:
            sequence_id = f"weights_{int(time.time())}"
            if self.verbose:
                print(f"Processing sequence {sequence_id}: {frame_paths[0]} → {frame_paths[-1]}")

            instruction = self._get_weights_instruction(vehicle_state)

            content = []
            for i, frame_path in enumerate(frame_paths):
                content.append({"type": "text", "text": f"Frame{i + 1}: "})
                content.append({"type": "image", "image": {"image_path": frame_path}})
            content.append({"type": "text", "text": instruction})

            conversation = [{"role": "user", "content": content}]
            inputs = self.processor(conversation=conversation, return_tensors="pt")
            inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)

            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )

            output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            parsed_result = self._parse_weights_from_text(output_text)

            return {
                "success": True,
                "raw_text": output_text,
                "task_type": "weights",
                **parsed_result,
                "sequence_id": sequence_id
            }
        except Exception as e:
            print(f"Error processing frames for weights: {e}")
            fallback = {
                "safety_weight": 0.7,
                "comfort_weight": 0.2,
                "efficiency_weight": 0.1,
                "justification": f"Default weights due to error: {str(e)}"
            }
            return {
                "success": False,
                "raw_text": f"Error: {str(e)}",
                "task_type": "weights",
                **fallback,
                "sequence_id": f"error_{int(time.time())}"
            }

    def reset_episode_state(self):
        """Reset state at the beginning of a new episode."""
        self.last_update_timestep = 0 - self.update_frequency
        if hasattr(self, 'previous_weights'):
            del self.previous_weights

    def get_reward_weights(self, vehicle_state, frame_buffer, carla_env=None):
        """
        Determine appropriate weights for reward components based on the current scene.
        Respects update_frequency to avoid excessive VLM calls.
        
        Args:
            vehicle_state: Current vehicle state information.
            frame_buffer: List of frame paths.
            carla_env: Optional reference to the CarlaEnv for logging control.
        
        Returns:
            dict: Weights for different reward components (w1, w2, w3).
        """
        default_weights_output = {
            "w1": 0.7, "w2": 0.2, "w3": 0.1, "justification": "Default weights"
        }
        
        current_timestep = vehicle_state.get("timestep", 0) if vehicle_state else carla_env.timestep
        
        time_since_last_update = current_timestep - self.last_update_timestep
        should_update = time_since_last_update >= self.update_frequency
        
        if not should_update:
            if self.verbose:
                print(f"[VLM] Skipping weight update at timestep {current_timestep} (next at {self.last_update_timestep + self.update_frequency})")
            return getattr(self, 'previous_weights', default_weights_output)
        
        if not all([vehicle_state, frame_buffer, len(frame_buffer) >= self.frames_needed]):
            if self.verbose:
                print(f"[VLM] Not enough data for weight update at timestep {current_timestep}.")
            return default_weights_output
        
        if self.verbose:
            print(f"[VLM] Performing weight update at timestep {current_timestep}.")
        
        try:
            result = self._process_frames_for_weights(frame_buffer[-self.frames_needed:], vehicle_state)
            
            extracted_weights = {
                "safety_weight": result.get("safety_weight", 0.7),
                "comfort_weight": result.get("comfort_weight", 0.2),
                "efficiency_weight": result.get("efficiency_weight", 0.1),
                "justification": result.get("justification", "Default justification")
            }
            
            weights_output = {
                "w1": extracted_weights["safety_weight"],
                "w2": extracted_weights["comfort_weight"],
                "w3": extracted_weights["efficiency_weight"],
                "justification": extracted_weights["justification"]
            }
            
            self.last_update_timestep = current_timestep
            self.previous_weights = weights_output.copy()
            
            self._log_decision(result, vehicle_state, carla_env)
            
            if self.verbose:
                print(f"[VLM] New weights: Safety={weights_output['w1']:.2f}, Comfort={weights_output['w2']:.2f}, Efficiency={weights_output['w3']:.2f}")
            
            return weights_output
            
        except Exception as e:
            print(f"[VLM] Error in get_reward_weights: {e}")
            return default_weights_output

    def _get_weights_instruction(self, vehicle_state):
        """Generate an instruction for determining reward weights."""
        context_info = f"""Current vehicle state:
- Speed: {vehicle_state["speed_kmh"]:.1f} km/h
- Acceleration: {vehicle_state.get("acceleration", 0):.2f} m/s²
- Distance to goal: {vehicle_state.get("distance_to_goal", 0):.2f} meters
- Timestep: {vehicle_state.get('timestep', 0)}
"""
        
        weights_instruction_safety = """You are an AI assistant helping to train an autonomous driving agent.

**YOUR ROLE**: Analyze the driving situation in the video frames and determine the optimal reward weights for a PPO reinforcement learning agent.

**TASK**: Assign weights to three components:
1.  **Safety weight (w1)**: Prioritize avoiding pedestrians and hazards.
2.  **Comfort weight (w2)**: Prioritize smooth, comfortable driving for passengers.
3.  **Efficiency weight (w3)**: Prioritize making efficient progress toward the destination.

**GUIDELINES**:
- The weights guide the PPO agent's learning. They will be normalized, so they don't need to sum to 1.0.
- **Increase safety weight** significantly when pedestrians or other hazards are present or likely.
- **Increase comfort weight** during normal, safe driving conditions.
- **Increase efficiency weight** when the path ahead is clear and safe.

Provide your response strictly in this format:
SAFETY_WEIGHT: [0.0-1.0]
COMFORT_WEIGHT: [0.0-1.0]
EFFICIENCY_WEIGHT: [0.0-1.0]
JUSTIFICATION: [Explain your reasoning based on the visual information and vehicle context.]
"""
        
        weights_instruction_efficiency = """You are an AI assistant helping to train a performance-oriented autonomous driving agent.

**YOUR ROLE**: Analyze the driving situation and set reward weights to encourage a balance between speed and safety.

**TASK**: Assign weights to three components:
1.  **Safety weight (w1)**: Ensure collision avoidance, but do not be overly cautious.
2.  **Comfort weight (w2)**: Promote a smooth ride.
3.  **Efficiency weight (w3)**: Strongly prioritize timely progress toward the goal.

**GUIDELINES**:
- The weights will be normalized.
- Prioritize **comfort and efficiency** in normal, safe situations.
- Only increase the **safety weight** if there is a clear and immediate risk of collision. Avoid high safety weights for distant or non-threatening hazards.

Provide your response strictly in this format:
SAFETY_WEIGHT: [0.0-1.0]
COMFORT_WEIGHT: [0.0-1.0]
EFFICIENCY_WEIGHT: [0.0-1.0]
JUSTIFICATION: [Explain why you chose this balance of weights based on the scene.]
"""
        
        instruction = weights_instruction_efficiency if self.efficiency_priority else weights_instruction_safety
        return f"{context_info}\n\n{instruction}"

    def _parse_weights_from_text(self, text):
        """Parse weight information from VLM output text."""
        safety_weight = 1.0
        comfort_weight = 0.5
        efficiency_weight = 0.5
        justification = "Default weight justification"

        try:
            patterns = {
                'safety': [r'SAFETY_WEIGHT\s*[:=]\s*(\d+\.?\d*)'],
                'comfort': [r'COMFORT_WEIGHT\s*[:=]\s*(\d+\.?\d*)'],
                'efficiency': [r'EFFICIENCY_WEIGHT\s*[:=]\s*(\d+\.?\d*)']
            }

            def find_value(key, txt):
                for p in patterns[key]:
                    match = re.search(p, txt, re.IGNORECASE)
                    if match:
                        return float(match.group(1))
                return None

            s_w = find_value('safety', text)
            c_w = find_value('comfort', text)
            e_w = find_value('efficiency', text)

            if s_w is not None: safety_weight = s_w
            if c_w is not None: comfort_weight = c_w
            if e_w is not None: efficiency_weight = e_w

            # Normalize weights before returning
            total = safety_weight + comfort_weight + efficiency_weight
            if total > 0:
                safety_weight /= total
                comfort_weight /= total
                efficiency_weight /= total

            # Get justification
            justification_match = re.search(r'JUSTIFICATION\s*:\s*(.*)', text, re.DOTALL | re.IGNORECASE)
            if justification_match:
                justification = justification_match.group(1).strip()
            
            justification = justification[:500]  # Limit justification length
            
            # Round values for clean logging
            safety_weight = round(safety_weight, 4)
            comfort_weight = round(comfort_weight, 4)
            efficiency_weight = round(efficiency_weight, 4)
            
            if self.verbose:
                print(f"Parsed normalized weights: safety={safety_weight}, comfort={comfort_weight}, efficiency={efficiency_weight}")
                
        except Exception as e:
            print(f"Error parsing weights from text: {e}. Using default weights.")

        return {
            "safety_weight": safety_weight,
            "comfort_weight": comfort_weight,
            "efficiency_weight": efficiency_weight,
            "justification": justification
        }

    def _log_decision(self, vlm_result, vehicle_state, carla_env=None):
        """Log essential VLM weight decision data."""
        should_log = (carla_env and carla_env.use_vlm_weights) or (not carla_env)
        if not should_log:
            return
        
        try:
            log_entry = {
                "episode": vehicle_state.get("episode", 0),
                "timestep": vehicle_state.get("timestep", 0),
                "task_type": "weights",
                "vehicle_info": {
                    "speed_kmh": round(vehicle_state.get("speed_kmh", 0), 2),
                    "pedestrian_distance": round(vehicle_state.get("pedestrian_distance", float('inf')), 2),
                    "distance_to_goal": round(vehicle_state.get("distance_to_goal", 0), 2),
                },
                "weights": {
                    "safety": round(vlm_result.get("safety_weight", 0.7), 4),
                    "comfort": round(vlm_result.get("comfort_weight", 0.2), 4),
                    "efficiency": round(vlm_result.get("efficiency_weight", 0.1), 4),
                },
                "justification": vlm_result.get("justification", "")
            }
            
            if self.verbose and "raw_text" in vlm_result:
                log_entry["raw_response"] = vlm_result["raw_text"]
            
            try:
                with open(self.log_file, 'r') as f:
                    log_data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                log_data = {"initialization": time.time(), "decisions": []}
                print("Created new log file.")

            log_data["decisions"].append(log_entry)
            
            temp_file = f"{self.log_file}.temp"
            with open(temp_file, 'w') as f:
                json.dump(log_data, f, indent=2)
            os.replace(temp_file, self.log_file)
            
            if self.verbose:
                print(f"Logged weight decision to {self.log_file}")
                
        except Exception as e:
            print(f"Error logging decision: {e}")