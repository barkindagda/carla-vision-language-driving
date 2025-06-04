import os
import time
import json
import re
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

class VLMController:
    """
    Vision Language Model controller for autonomous vehicle decision-making
    using VideoLLaMA to process sequences of frames from CARLA.
    """
    def __init__(
            self,
            model_name="DAMO-NLP-SG/VideoLLaMA3-2B-Image",
            update_frequency=5,
            frames_needed=3,
            output_dir="/home/cavlab/CARLA_0.9.15/VLM_Barkin/CarlaEnv/vlm_outputs",
            max_new_tokens=512,
            verbose=True
    ):
        """
        Initialize the VLM Controller.
        Args:
            model_name: HuggingFace model name for VideoLLaMA
            update_frequency: How often to update decisions (in timesteps)
            frames_needed: Number of frames to use for each decision
            output_dir: Directory to save outputs and logs
            max_new_tokens: Maximum tokens to generate in model responses
            verbose: Whether to print detailed logs
        """
        self.model_name = model_name
        self.update_frequency = update_frequency
        self.frames_needed = frames_needed
        self.output_dir = output_dir
        self.max_new_tokens = max_new_tokens
        self.verbose = verbose
        self.last_update_timestep = 0 # Initialize here

        # Create timestamped output directory
        timestamp = int(time.time())
        self.session_dir = os.path.join(output_dir, f"session_{timestamp}")
        os.makedirs(self.session_dir, exist_ok=True)
        
        self.model = None
        self.processor = None
        self.last_update_timestep = 0
        self.current_action_text = "MAINTAIN"
        self.current_action_value = 0.3
        self.current_justification = "Starting the journey safely."

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

    def process_if_needed(self, carla_env):
        """
        Process frames through VLM if it's time for an update.
        Args:
            carla_env: CARLA environment instance
        Returns:
            bool: True if VLM processing was performed
        """
        if (len(carla_env.frame_buffer) >= self.frames_needed and
                carla_env.timestep - self.last_update_timestep >= self.update_frequency):
            vehicle_state = carla_env.get_current_vehicle_state()
            
            # Only process actions if VLM actions are enabled
            if carla_env.use_vlm_actions:
                vlm_result = self.process_frames(carla_env.frame_buffer[-self.frames_needed:], vehicle_state, task="action")

                if vlm_result["success"]:
                    self.current_action_text = vlm_result["action_text"]
                    self.current_action_value = vlm_result["action_value"]
                    self.current_justification = vlm_result["justification"]

                    carla_env.current_vlm_action = self.current_action_text
                    carla_env.current_vlm_justification = self.current_justification
                    carla_env.current_vlm_action_value = self.current_action_value

                    # Log the action decision
                    self._log_decision(vlm_result, vehicle_state, carla_env)

            # Only process weights if VLM weights are enabled
            if carla_env.use_vlm_weights and carla_env.timestep % self.update_frequency == 0:
                weights = self.get_reward_weights(vehicle_state, carla_env.frame_buffer)
                # The get_reward_weights method already calls _log_decision
                
            self.last_update_timestep = carla_env.timestep
            return True
        return False

    def process_frames(self, frame_paths, vehicle_state, task="action"):
        """
        Process a sequence of frames through VideoLLaMA for either action or weights.
        Args:
            frame_paths: List of paths to frame images
            vehicle_state: Dictionary with current vehicle state
            task: "action" for driving action, "weights" for reward weights
        Returns:
            dict: Processing results
        """
        try:
            sequence_id = f"{task}_{int(time.time())}"
            if self.verbose:
                print(f"Processing sequence {sequence_id}: {frame_paths[0]} → {frame_paths[-1]}")

            instruction = (self._get_dynamic_instruction(vehicle_state) if task == "action"
                        else self._get_weights_instruction(vehicle_state))

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
            parsed_result = (self._parse_action_from_text(output_text) if task == "action"
                        else self._parse_weights_from_text(output_text))

            return {
                "success": True,
                "raw_text": output_text,
                "task_type": task,  # Add task type to result
                **parsed_result,
                "sequence_id": sequence_id
            }
        except Exception as e:
            print(f"Error processing frames: {e}")
            fallback = {
                "action_text": self.current_action_text,
                "action_value": self.current_action_value,
                "justification": f"Fallback due to error: {str(e)}"
            } if task == "action" else {
                "safety_weight": 0.7,
                "comfort_weight": 0.2,
                "efficiency_weight": 0.1,
                "justification": f"Default weights due to error: {str(e)}"
            }
            return {
                "success": False,
                "raw_text": f"Error: {str(e)}",
                "task_type": task,  # Add task type to result
                **fallback,
                "sequence_id": f"error_{int(time.time())}"
            }
    def reset_episode_state(self):
        self.last_update_timestep=0-self.update_frequency
            
    def get_reward_weights(self, vehicle_state, frame_buffer=None, carla_env=None):
        """
        Determine appropriate weights for reward components based on current scene.
        Respects update_frequency to avoid excessive VLM calls.
        
        Args:
            vehicle_state: Current vehicle state information
            frame_buffer: List of frame paths (if None, use default weights)
            carla_env: Optional reference to the CarlaEnv for logging control
        
        Returns:
            dict: Weights for different reward components using keys w1, w2, w3
        """
        # Default weights in the expected output format
        default_weights_output = {
            "w1": 0.7,  # Safety weight
            "w2": 0.2,  # Comfort weight
            "w3": 0.1,  # Efficiency weight
            "justification": "Default weights"
        }
        
        # Get current timestep from vehicle_state or carla_env
        current_timestep = vehicle_state.get("timestep", 0) if vehicle_state else 0
        if carla_env:
            current_timestep = carla_env.timestep
        
        # Check if we should update based on frequency
        time_since_last_update = current_timestep - self.last_update_timestep
        should_update = time_since_last_update >= self.update_frequency
        
        # If we shouldn't update yet, return the previous weights
        if not should_update:
            if self.verbose:
                print(f"[VLM] Skipping weight update at timestep {current_timestep} " +
                    f"(last update: {self.last_update_timestep}, " +
                    f"next update at: {self.last_update_timestep + self.update_frequency})")
            
            # Return previous weights if available, otherwise defaults
            if hasattr(self, 'previous_weights') and self.previous_weights:
                return self.previous_weights
            else:
                return default_weights_output
        
        # If we should update but don't have enough data, return defaults
        if vehicle_state is None or frame_buffer is None or len(frame_buffer) < self.frames_needed:
            if self.verbose:
                print(f"[VLM] Not enough frames for weight update at timestep {current_timestep} " +
                    f"({len(frame_buffer) if frame_buffer else 0}/{self.frames_needed})")
            
            return default_weights_output
        
        # If we reach here, we should perform an update
        if self.verbose:
            print(f"[VLM] Performing weight update at timestep {current_timestep} " +
                f"(waited {time_since_last_update} steps)")
        
        try:
            # Process frames to get weights
            result = self.process_frames(frame_buffer[-self.frames_needed:], vehicle_state, task="weights")
            
            # Ensure all required keys are present, use defaults for any missing keys
            extracted_weights = {
                "safety_weight": result.get("safety_weight", 0.7),
                "comfort_weight": result.get("comfort_weight", 0.2),
                "efficiency_weight": result.get("efficiency_weight", 0.1),
                "justification": result.get("justification", "Default justification")
            }
            
            # Convert to w1, w2, w3 format
            weights_output = {
                "w1": extracted_weights["safety_weight"],
                "w2": extracted_weights["comfort_weight"],
                "w3": extracted_weights["efficiency_weight"],
                "justification": extracted_weights["justification"]
            }
            
            # Update timestamp and store weights for future reference
            self.last_update_timestep = current_timestep
            self.previous_weights = weights_output.copy()
            
            # Log the weight decision
            self._log_decision(result, vehicle_state, carla_env)
            
            if self.verbose:
                print(f"[VLM] New weights: Safety={weights_output['w1']:.2f}, " +
                    f"Comfort={weights_output['w2']:.2f}, Efficiency={weights_output['w3']:.2f}")
            
            return weights_output
            
        except Exception as e:
            print(f"[VLM] Error in get_reward_weights: {e}")
            # Return default weights in the expected format
            return default_weights_output
    def _get_weights_instruction(self, vehicle_state):
        """
        Generate an instruction for determining reward weights.
        Args:
            vehicle_state: Current state information from the environment
        Returns:
            str: Instruction for the VLM
        """
        speed_kmh = vehicle_state["speed_kmh"]
        acceleration = vehicle_state.get("acceleration", 0)
        distance_to_goal = vehicle_state.get("distance_to_goal", 0)
        timestep = vehicle_state.get('timestep', 0)

        # Remove pedestrian information as requested
        context_info = f"""Current vehicle state:
- Speed: {speed_kmh:.1f} km/h
- Acceleration: {acceleration:.2f} m/s²
- Distance to goal: {distance_to_goal:.2f} meters
- Timestep: {timestep}
"""

        weights_instruction = """You are assisting in training a Proximal Policy Optimization (PPO) reinforcement learning agent for autonomous driving by determining appropriate reward weights.

**YOUR ROLE**: You analyze the current driving situation and determine the optimal reward weights that the PPO agent should use to learn good driving behavior.

**TASK**: Based on the current driving situation shown in the frames, assign weights to three reward components:
1. Safety weight (w1): Prioritize pedestrian safety
2. Comfort weight (w2): Prioritize passenger comfort (smooth driving)
3. Efficiency weight (w3): Prioritize efficient progress

**GUIDELINES**:
- The weights will be used to calculate a composite reward signal that trains the PPO agent
- Weights don't need to sum to 1.0 (they will be normalized)
- Higher safety weights when hazards are visible or in potentially dangerous situations
- Higher comfort weights during normal driving with passengers
- Higher efficiency weights when path is clear
- Analyze frames carefully

Provide response in format:
SAFETY_WEIGHT: [0.0-1]
COMFORT_WEIGHT: [0.0-1]
EFFICIENCY_WEIGHT: [0.0-1]
JUSTIFICATION: [Explanation based on frames and context]
"""
        return f"{context_info}\n\n{weights_instruction}"

    def _parse_weights_from_text(self, text):
        """
        Parse weight information from VLM output text.
        Args:
            text: Raw model output text
        Returns:
            dict: Parsed weight information
        """
        safety_weight = 0.7
        comfort_weight = 0.2
        efficiency_weight = 0.1
        justification = "Default weight justification"

        try:
            # Try to find weights using different patterns (handles both formats)
            safety_patterns = [r'SAFETY_WEIGHT\s*:\s*(\d+\.\d+)', r'SAFETY_WEIGHT\s*=\s*(\d+\.\d+)', r'safety weight\s*:\s*(\d+\.\d+)']
            comfort_patterns = [r'COMFORT_WEIGHT\s*:\s*(\d+\.\d+)', r'COMFORT_WEIGHT\s*=\s*(\d+\.\d+)', r'comfort weight\s*:\s*(\d+\.\d+)']
            efficiency_patterns = [r'EFFICIENCY_WEIGHT\s*:\s*(\d+\.\d+)', r'EFFICIENCY_WEIGHT\s*=\s*(\d+\.\d+)', r'efficiency weight\s*:\s*(\d+\.\d+)']

            # Try each pattern
            for pattern in safety_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    safety_weight = float(match.group(1))
                    break
                    
            for pattern in comfort_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    comfort_weight = float(match.group(1))
                    break
                    
            for pattern in efficiency_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    efficiency_weight = float(match.group(1))
                    break

            # Normalize weights before returning
            total = safety_weight + comfort_weight + efficiency_weight
            if total > 0:
                safety_weight = safety_weight / total
                comfort_weight = comfort_weight / total
                efficiency_weight = efficiency_weight / total

            # Get justification - try various patterns
            justification_match = re.search(r'JUSTIFICATION\s*:\s*(.*?)(?:\n\n|$)', text, re.DOTALL | re.IGNORECASE)
            if justification_match:
                justification = justification_match.group(1).strip()
            else:
                # If no match with the specific pattern, try grabbing text after "JUSTIFICATION:"
                parts = re.split(r'JUSTIFICATION\s*:', text, flags=re.IGNORECASE)
                if len(parts) > 1:
                    justification = parts[1].strip()
                    
            # Limit justification length
            justification = justification[:500]  # Limit to 500 chars to avoid excessive logging
                    
            # Round values to avoid floating point precision issues
            safety_weight = round(safety_weight, 4)
            comfort_weight = round(comfort_weight, 4)
            efficiency_weight = round(efficiency_weight, 4)
            
            if self.verbose:
                print(f"Parsed weights: safety={safety_weight}, comfort={comfort_weight}, efficiency={efficiency_weight}")
                
        except Exception as e:
            print(f"Error parsing weights from text: {e}")
            print(f"Using default weights instead.")

        return {
            "safety_weight": safety_weight,
            "comfort_weight": comfort_weight,
            "efficiency_weight": efficiency_weight,
            "justification": justification
        }

    def _get_dynamic_instruction(self, vehicle_state):
        """
        Generate a dynamic instruction for action based on vehicle state.
        Args:
            vehicle_state: Current state information from the environment
        Returns:
            str: Instruction for the VLM
        """
        speed_kmh = vehicle_state["speed_kmh"]
        reward_info = ""
        if "current_rewards" in vehicle_state and vehicle_state["current_rewards"]:
            rewards = vehicle_state["current_rewards"]
            reward_info = f"""
    Recent Reward Breakdown:
    - Safety: {rewards.get('safety_reward', 0):.2f}
    - Progress: {rewards.get('progress_reward', 0):.2f}
    - Smoothness: {rewards.get('smoothness_reward', 0):.2f}
    - Collision: {rewards.get('collision_penalty', 0):.2f}
    - Total: {rewards.get('total_reward', 0):.2f}
    """
        trend_info = ""
        if all(key in vehicle_state for key in ["safety_trend", "progress_trend", "smoothness_trend"]):
            trend_info = f"""
    Performance Trends:
    - Safety trend: {vehicle_state["safety_trend"]}
    - Progress trend: {vehicle_state["progress_trend"]}
    - Smoothness trend: {vehicle_state["smoothness_trend"]}
    """
        context_info = f"""Previous action: {self.current_action_text} (value: {self.current_action_value:.2f})
    Justification: {self.current_justification}

    Current vehicle state:
    - Speed: {speed_kmh:.1f} km/h
    - Acceleration: {vehicle_state.get('acceleration', 0):.2f} m/s²
    - Distance to goal: {vehicle_state.get('distance_to_goal', 0):.2f} meters
    - Timestep: {vehicle_state.get('timestep', 0)}
    {reward_info}
    {trend_info}
    """
        base_instruction = """You are assisting an autonomous vehicle. Examine the frames and determine the best driving action.

    **GOAL**: Optimize driving behavior to maximize total reward by balancing safety, efficiency, and comfort.

    **INSTRUCTIONS**:
    1. **Reward Optimization**:
    - Maximize total reward
    - Safety reward: Keep close to zero (negative near pedestrians)
    - Progress reward: Maximize when safe
    - Smoothness reward: Avoid sudden changes
    - Collision penalty: Avoid at all costs (-200)

    2. **Balanced Priorities**:
    - Safety: Essential but balanced with progress
    - Efficiency: Complete journey quickly when safe
    - Comfort: Smooth acceleration/deceleration

    3. **Action Values** (-1.0 to +1.0):
    - Negative: Brake (-1.0 hard, 0 light)
    - Zero: Maintain speed
    - Positive: Accelerate (+1.0 strong)

    4. **Pedestrian Response**:
    - Close pedestrian: Hard braking (-0.7 to -0.9)
    - Medium distance: Medium braking (-0.4 to -0.6)
    - Far away: Gentle braking (-0.1 to -0.3)
    - No pedestrian: Accelerate (+0.5 to +0.8)
    - Be cautious around occlusions

    5. **Vehicle Control**:
    - -1.0 to -0.8: BRAKE_HARD
    - -0.7 to -0.3: BRAKE_GENTLY
    - -0.2 to -0.1: DECELERATE
    - 0.0 to 0.4: MAINTAIN
    - 0.5 to 1.0: ACCELERATE

    Provide response in format:
    ACTION: [BRAKE_HARD/BRAKE_GENTLY/DECELERATE/MAINTAIN/ACCELERATE]
    VALUE: [-1.0 to +1.0]
    JUSTIFICATION: [Explanation based on frames and reward]
    """
        return f"{context_info}\n\n{base_instruction}"

    def _parse_action_from_text(self, text):
        """
        Parse action information from VLM output text.
        Args:
            text: Raw model output text
        Returns:
            dict: Parsed action information
        """
        action_text = "MAINTAIN"
        action_value = 0.0
        justification = "Default justification"

        try:
            action_match = re.search(r'ACTION:\s*(BRAKE_HARD|BRAKE_GENTLY|DECELERATE|MAINTAIN|ACCELERATE)', text)
            if action_match:
                action_text = action_match.group(1)

            value_match = re.search(r'VALUE:\s*(-?\d+\.?\d*)', text)
            if value_match:
                action_value = float(value_match.group(1))
                action_value = max(min(action_value, 1.0), -1.0)
            else:
                action_value_map = {
                    "BRAKE_HARD": -0.9,
                    "BRAKE_GENTLY": -0.5,
                    "DECELERATE": -0.2,
                    "MAINTAIN": 0.3,
                    "ACCELERATE": 0.7
                }
                action_value = action_value_map.get(action_text, 0.0)

            justification_match = re.search(r'JUSTIFICATION:\s*(.*?)(?:\n|$)', text, re.DOTALL)
            if justification_match:
                justification = justification_match.group(1).strip()
                if len(justification) < 20 and len(text.split("JUSTIFICATION:", 1)) > 1:
                    justification = text.split("JUSTIFICATION:", 1)[1].strip()
        except Exception as e:
            print(f"Error parsing action: {e}")
            if "brake hard" in text.lower() or "emergency" in text.lower():
                action_text, action_value = "BRAKE_HARD", -0.9
            elif "brake" in text.lower() or "slow down" in text.lower():
                action_text, action_value = "BRAKE_GENTLY", -0.5
            elif "deceler" in text.lower():
                action_text, action_value = "DECELERATE", -0.2
            elif "maintain" in text.lower() or "current speed" in text.lower():
                action_text, action_value = "MAINTAIN", 0.3
            elif "acceler" in text.lower() or "speed up" in text.lower():
                action_text, action_value = "ACCELERATE", 0.7
            justification = "Parsed from context due to format error"

        return {
            "action_text": action_text,
            "action_value": action_value,
            "justification": justification
        }
    def _log_decision(self, vlm_result, vehicle_state, carla_env=None):
        """
        Log essential VLM decision data in a clean, focused format.
        
        Args:
            vlm_result: Results from VLM processing
            vehicle_state: Current vehicle state
            carla_env: Optional reference to the CarlaEnv to check settings
        """
        # Check if logging is appropriate
        should_log = False
        if carla_env:
            if vlm_result.get("task_type") == "weights" and carla_env.use_vlm_weights:
                should_log = True
            elif vlm_result.get("task_type") == "action" and carla_env.use_vlm_actions:
                should_log = True
        else:
            should_log = True
        
        if not should_log:
            return
        
        try:
            # Create a simplified log entry with only essential information
            log_entry = {
                "episode": vehicle_state.get("episode", 0) if vehicle_state else 0,
                "timestep": vehicle_state.get("timestep", 0) if vehicle_state else 0,
                "task_type": vlm_result.get("task_type", "unknown"),
            }
            
            # Add minimal vehicle state information
            log_entry["vehicle_info"] = {
                "speed_kmh": round(vehicle_state.get("speed_kmh", 0), 3),
                "pedestrian_distance": round(vehicle_state.get("pedestrian_distance", float('inf')), 2),
                "distance_to_goal": round(vehicle_state.get("distance_to_goal", 0), 2),
                "collision_detected": vehicle_state.get("collision_detected", False)
            }
            
            # Add only the current reward (not history)
            if "current_rewards" in vehicle_state:
                current_rewards = vehicle_state["current_rewards"]
                log_entry["reward"] = {
                    "safety": round(current_rewards.get("safety_reward", 0), 4),
                    "comfort": round(current_rewards.get("comfort_reward", 0), 4),
                    "efficiency": round(current_rewards.get("efficiency_reward", 0), 4),
                    "total": round(current_rewards.get("total_reward", 0), 4)
                }
            
            # Add weight or action information based on task type
            if vlm_result.get("task_type") == "weights":
                # Just log the normalized weights
                log_entry["weights"] = {
                    "safety": round(vlm_result.get("safety_weight", 0.7), 3),
                    "comfort": round(vlm_result.get("comfort_weight", 0.2), 3),
                    "efficiency": round(vlm_result.get("efficiency_weight", 0.1), 3),
                }
            elif vlm_result.get("task_type") == "action":
                log_entry["action"] = {
                    "text": vlm_result.get("action_text", "UNKNOWN"),
                    "value": round(vlm_result.get("action_value", 0.0), 3)
                }
            
            # Add complete justification without truncation
            log_entry["justification"] = vlm_result.get("justification", "")
            
            # Add raw text if in verbose mode and it contains additional information
            if self.verbose:
                raw_text = vlm_result.get("raw_text", "")
                if raw_text:
                    log_entry["raw_response"] = raw_text
            
            # Load existing log data
            try:
                with open(self.log_file, 'r') as f:
                    log_data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                log_data = {"initialization": time.time(), "decisions": []}
                print(f"Created new log file due to corruption or missing file")

            # Add new entry and save
            log_data["decisions"].append(log_entry)
            
            # Use atomic write to prevent corruption
            temp_file = f"{self.log_file}.temp"
            with open(temp_file, 'w') as f:
                json.dump(log_data, f, indent=2)
            os.replace(temp_file, self.log_file)
            
            if self.verbose:
                print(f"Logged {vlm_result.get('task_type', 'decision')} to {self.log_file}")
                    
        except Exception as e:
            print(f"Error logging decision: {e}")