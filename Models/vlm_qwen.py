import os
import time
import json
import re
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


class VLMController:
    """
    Vision Language Model controller for autonomous vehicle decision-making
    using Qwen2.5-VL to process sequences of frames from CARLA.
    """

    def __init__(
            self,
            model_name="Qwen/Qwen2.5-VL-72B-Instruct",
            update_frequency=5,
            frames_needed=3,
            output_dir="/home/cavlab/CARLA_0.9.15/VLM_Barkin/CarlaEnv/vlm_outputs",
            max_new_tokens=512,
            verbose=True
    ):
        """
        Initialize the VLM Controller.

        Args:
            model_name: HuggingFace model name for Qwen2.5-VL
            update_frequency: How often to update decisions (in timesteps)
            frames_needed: Number of frames to use for each decision
            output_dir: Directory to save outputs and logs
            max_new_tokens: Maximum tokens to generate in model responses
            verbose: Whether to print detailed logs
        """
        # Basic configuration
        self.model_name = model_name
        self.update_frequency = update_frequency
        self.frames_needed = frames_needed
        self.output_dir = output_dir
        self.max_new_tokens = max_new_tokens
        self.verbose = verbose

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Controller state
        self.model = None
        self.processor = None
        self.last_update_timestep = 0
        self.current_action_text = "MAINTAIN"
        self.current_action_value = 0.0  # Default value for MAINTAIN
        self.current_justification = "Starting the journey safely."

        # Initialize logging
        self.log_file = os.path.join(output_dir, f"vlm_decisions_{int(time.time())}.json")
        with open(self.log_file, 'w') as f:
            json.dump({"initialization": time.time(), "decisions": []}, f)

        # Load the model
        self._load_model()

    def _load_model(self):
        """Load the Qwen2.5-VL model and processor."""
        if self.verbose:
            print(f"Loading Qwen2.5-VL model: {self.model_name}")

        try:
            # Load model
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
            )

            # Load processor with default pixel range for performance
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
        # Only process if we have enough frames AND it's time for an update
        if (len(carla_env.frame_buffer) >= self.frames_needed and
                (carla_env.timestep - self.last_update_timestep >= self.update_frequency)):

            # Get current vehicle state
            vehicle_state = carla_env.get_current_vehicle_state()

            # Process with VLM
            vlm_result = self.process_frames(carla_env.frame_buffer[-self.frames_needed:], vehicle_state)

            if vlm_result["success"]:
                # Extract the action info
                self.current_action_text = vlm_result["action_text"]
                self.current_action_value = vlm_result["action_value"]
                self.current_justification = vlm_result["justification"]

                # Update the environment with new values
                carla_env.current_vlm_action = self.current_action_text
                carla_env.current_vlm_justification = self.current_justification
                carla_env.current_action_value = self.current_action_value

                # Record update time
                self.last_update_timestep = carla_env.timestep

                # Log for debugging
                if self.verbose:
                    print(
                        f"[VLM] New action: {self.current_action_text} ({self.current_action_value:.2f}) - {self.current_justification}")

                # Log to file
                self._log_decision(vlm_result, vehicle_state)

                return True

        return False

    def process_frames(self, frame_paths, vehicle_state):
        """
        Process a sequence of frames through Qwen2.5-VL.

        Args:
            frame_paths: List of paths to frame images
            vehicle_state: Dictionary with current vehicle state

        Returns:
            dict: Processing results including recommended action
        """
        try:
            # Create unique sequence ID
            sequence_id = f"seq_{int(time.time())}"

            if self.verbose:
                print(f"Processing sequence {sequence_id}: {frame_paths[0]} → {frame_paths[-1]}")

            # Get instruction with vehicle state incorporated
            instruction = self._get_dynamic_instruction(vehicle_state)

            # Create content array with frame paths
            content = []
            for i, frame_path in enumerate(frame_paths):
                content.append({"type": "image", "image": f"file://{frame_path}"})
                content.append({"type": "text", "text": f"Frame{i + 1}: "})
            content.append({"type": "text", "text": instruction})

            # Create conversation format
            messages = [{"role": "user", "content": content}]

            # Prepare inputs for inference
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)

            # Generate output
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )

            # Trim generated IDs to get only the new tokens
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            # Decode output
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0].strip()

            # Parse the output to extract action information
            parsed_result = self._parse_action_from_text(output_text)

            return {
                "success": True,
                "raw_text": output_text,
                "action_text": parsed_result["action_text"],
                "action_value": parsed_result["action_value"],
                "justification": parsed_result["justification"],
                "sequence_id": sequence_id
            }

        except Exception as e:
            print(f"Error processing frames: {e}")

            # Create fallback response
            return {
                "success": False,
                "raw_text": f"Error: {str(e)}",
                "action_text": self.current_action_text,
                "action_value": self.current_action_value,
                "justification": f"Fallback due to error: {str(e)}",
                "sequence_id": f"error_{int(time.time())}"
            }

    def _get_dynamic_instruction(self, vehicle_state):
        """
        Generate a dynamic instruction based on vehicle state,
        without direct access to pedestrian distance information.
        
        Args:
            vehicle_state: Current state information from the environment
            
        Returns:
            str: Instruction for the VLM
        """
        # Extract key state information (excluding pedestrian distance)
        speed_kmh = vehicle_state["speed_kmh"]
        
        # Extract reward information if available
        reward_info = ""
        if "current_rewards" in vehicle_state and vehicle_state["current_rewards"]:
            rewards = vehicle_state["current_rewards"]
            reward_info = f"""
    Recent Reward Breakdown:
    - Safety: {rewards.get('safety_reward', 0):.2f} (negative values indicate unsafe proximity to pedestrians)
    - Progress: {rewards.get('progress_reward', 0):.2f} (positive values indicate good speed when safe)
    - Smoothness: {rewards.get('smoothness_reward', 0):.2f} (negative values indicate jerky driving)
    - Collision: {rewards.get('collision_penalty', 0):.2f} (large negative value if collision occurred)
    - Total: {rewards.get('total_reward', 0):.2f}
    """
        
        # Add trend information if available
        trend_info = ""
        if all(key in vehicle_state for key in ["safety_trend", "progress_trend", "smoothness_trend"]):
            trend_info = f"""
    Performance Trends:
    - Safety trend: {vehicle_state["safety_trend"]}
    - Progress trend: {vehicle_state["progress_trend"]}
    - Smoothness trend: {vehicle_state["smoothness_trend"]}
    """
        
        # Create context-aware instruction WITHOUT pedestrian distance information
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
        
        # Base instruction (same as original)
        base_instruction = """You are assisting an autonomous vehicle. Examine the frames and determine the best driving action.

    **GOAL**: Optimize driving behavior to maximize total reward by balancing safety, efficiency, and comfort.

    **IMPORTANT INSTRUCTIONS**:

    1. **Reward Optimization (HIGHEST PRIORITY)**:
    - Your primary goal is to maximize the total reward
    - Safety reward: Should be as close to zero as possible (becomes negative when near pedestrians)
    - Progress reward: Maximize this when it's safe (higher speed when no pedestrians)
    - Smoothness reward: Avoid sudden acceleration/braking changes to keep this close to zero
    - Collision penalty: Must be avoided at all costs (-200 penalty)

    2. **Balanced Priorities**:
    - Safety: Pedestrian safety is essential but should be balanced with efficient progress
    - Efficiency: Complete the journey quickly when safe to do so
    - Comfort: Provide smooth acceleration/deceleration for passenger comfort
    - Use the reward signals to find the optimal balance between these priorities

    3. **Action Values**:
    - Action values range from -1.0 to +1.0:
        - Negative values (-1.0 to 0): Apply brakes (closer to -1.0 means harder braking)
        - Zero (0): Maintain current speed with no acceleration or braking
        - Positive values (0 to +1.0): Apply throttle (closer to +1.0 means stronger acceleration)

    4. **Pedestrian Response Guidelines**:
    - You must rely on what you can see in the frames to detect pedestrians
    - If pedestrian is clearly visible and close: Apply harder braking (-0.7 to -0.9)
    - If pedestrian is partially visible or at medium distance: Apply medium braking (-0.4 to -0.6)
    - If pedestrian is barely visible or far away: Apply gentle braking (-0.1 to -0.3)
    - If no pedestrian is visible: Accelerate to make efficient progress (+0.5 to +0.8)
    - Be cautious around occlusions (vehicles, barriers) that might hide pedestrians

    5. **Vehicle Control Interpretation**:
    Action value → Vehicle behavior:
    - -1.0 to -0.8: BRAKE_HARD (emergency stop)
    - -0.7 to -0.3: BRAKE_GENTLY (moderate braking)
    - -0.2 to -0.1: DECELERATE (slight braking)
    - 0.0: MAINTAIN (maintain current speed)
    - 0.1 to 0.5: ACCELERATE (increase speed gently)
    - 0.5 to 1.0: ACCELERATE_HARD (increase speed more aggressively)

    6. **Reward-Based Decision Making**:
    - Watch for reward trends to adjust your strategy
    - If safety reward is declining: Increase caution around multiple pedestrians
    - If progress reward is declining: Increase speed when no pedestrians are visible
    - If smoothness reward is declining: Make more gradual acceleration/deceleration changes
    - Remember that total reward is the ultimate measure of success

    Provide your response in the following format:

    ACTION: [DECELERATE/DO_NOTHING/ACCELERATE/]
    VALUE: [number between -1.0 and +1.0]
    JUSTIFICATION: [Brief explanation of your decision based on what you see in the frames and reward optimization]

    Make your decisions based solely on what you can see in the frames and the reward feedback.
    """
        # Combine context and base instruction
        return f"{context_info}\n\n{base_instruction}"

    def _parse_action_from_text(self, text):
        """
        Parse action information from the VLM output text.

        Args:
            text: Raw model output text

        Returns:
            dict: Parsed action information
        """
        # Default values
        action_text = "MAINTAIN"
        action_value = 0.0
        justification = "Default justification"

        try:
            # Extract action text
            # Corrected line
            action_match = re.search(r'ACTION:\s*(BRAKE_HARD|BRAKE_GENTLY|DECELERATE|MAINTAIN|ACCELERATE|ACCELERATE_HARD)', text, re.IGNORECASE)
            if action_match:
                action_text = action_match.group(1).upper()

            # Extract action value
            value_match = re.search(r'VALUE:\s*(-?\d+\.?\d*)', text)
            if value_match:
                action_value = float(value_match.group(1))
                # Ensure value is in valid range
                action_value = max(min(action_value, 1.0), -1.0)
            else:
                # If no value found, assign based on action text
                action_value_map = {
                    "BRAKE_HARD": -0.9,
                    "BRAKE_GENTLY": -0.5,
                    "DECELERATE": -0.2,
                    "MAINTAIN": 0.0,
                    "ACCELERATE": 0.5,
                    "ACCELERATE_HARD": 0.8
                }
                action_value = action_value_map.get(action_text, 0.0)

            # Extract justification
            justification_match = re.search(r'JUSTIFICATION:\s*(.*?)(?:\n|$)', text, re.DOTALL)
            if justification_match:
                justification = justification_match.group(1).strip()

            # If justification is too short, try to extract more context
            if len(justification) < 20:
                full_text_after_justification = text.split("JUSTIFICATION:", 1)
                if len(full_text_after_justification) > 1:
                    justification = full_text_after_justification[1].strip()

        except Exception as e:
            print(f"Error parsing action from text: {e}")
            # If parsing fails, infer from text matching
            text_lower = text.lower()
            if "brake hard" in text_lower or "emergency" in text_lower:
                action_text = "BRAKE_HARD"
                action_value = -0.9
            elif "brake gently" in text_lower or "slow down" in text_lower:
                action_text = "BRAKE_GENTLY"
                action_value = -0.5
            elif "decelerate" in text_lower:
                action_text = "DECELERATE"
                action_value = -0.2
            elif "maintain" in text_lower or "current speed" in text_lower:
                action_text = "MAINTAIN"
                action_value = 0.0
            elif "accelerate hard" in text_lower:
                action_text = "ACCELERATE_HARD"
                action_value = 0.8
            elif "accelerate" in text_lower or "speed up" in text_lower:
                action_text = "ACCELERATE"
                action_value = 0.5

            justification = "Parsed from context due to format error"

        return {
            "action_text": action_text,
            "action_value": action_value,
            "justification": justification
        }

    def _log_decision(self, vlm_result, vehicle_state):
        """
        Log decision data to file.

        Args:
            vlm_result: Results from VLM processing
            vehicle_state: Current vehicle state
        """
        try:
            # Create a sanitized version of vehicle_state
            sanitized_state = {}
            for key, value in vehicle_state.items():
                if isinstance(value, (int, float, str, bool)) or value is None:
                    sanitized_state[key] = value
                else:
                    sanitized_state[key] = str(value)
            
            # Create the log entry with sanitized data
            log_entry = {
                "timestamp": time.time(),
                "sequence_id": vlm_result.get("sequence_id", "unknown"),
                "vehicle_state": sanitized_state,
                "action_text": vlm_result.get("action_text", "UNKNOWN"),
                "action_value": float(vlm_result.get("action_value", 0.0)),
                "justification": str(vlm_result.get("justification", "")),
                "raw_response": str(vlm_result.get("raw_text", ""))
            }
            
            # Load existing log
            try:
                with open(self.log_file, 'r') as f:
                    log_data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                log_data = {"initialization": time.time(), "decisions": []}
                print(f"Created new log file due to corruption or missing file")
            
            # Add new decision
            log_data["decisions"].append(log_entry)
            
            # Save updated log
            temp_file = f"{self.log_file}.temp"
            with open(temp_file, 'w') as f:
                json.dump(log_data, f, indent=2)
            
            os.replace(temp_file, self.log_file)
            
        except Exception as e:
            print(f"Error logging decision: {e}")