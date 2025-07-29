import os
import time
import json
import re
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

class VLMController:
    """
    Vision Language Model controller for autonomous vehicle decision-making
    using LLaVA-OneVision to process sequences of frames from CARLA.
    """

    def __init__(
            self,
            model_name="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
            update_frequency=5,
            frames_needed=3,
            output_dir="./vlm_outputs",
            max_new_tokens=512,
            verbose=True
    ):
        """
        Initialize the VLM Controller.

        Args:
            model_name: HuggingFace model name for LLaVA-OneVision.
            update_frequency: How often to update decisions (in timesteps).
            frames_needed: Number of frames to use for each decision.
            output_dir: Directory to save outputs and logs.
            max_new_tokens: Maximum tokens to generate in model responses.
            verbose: Whether to print detailed logs.
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
        self.log_file = os.path.join(output_dir, f"vlm_decisions_llava_{int(time.time())}.json")
        with open(self.log_file, 'w') as f:
            json.dump({"initialization": time.time(), "decisions": []}, f)

        # Load the model
        self._load_model()

    def _load_model(self):
        """Load the LLaVA-OneVision model and processor."""
        if self.verbose:
            print(f"[INFO] Loading LLaVA-OneVision model: {self.model_name}")

        try:
            # Load model
            self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16, # LLaVA works well with float16
                device_map="auto",
                trust_remote_code=True,
            )

            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            if self.verbose:
                print(f"[INFO] Model loaded successfully on {self.model.device}")

        except Exception as e:
            print(f"[ERROR] Error loading model: {e}")
            raise RuntimeError(f"Failed to initialize VLM model: {e}")

    def process_if_needed(self, carla_env):
        """
        Process frames through VLM if it's time for an update.
        """
        # Only process if we have enough frames AND it's time for an update
        if (len(carla_env.frame_buffer) >= self.frames_needed and
                (carla_env.timestep - self.last_update_timestep >= self.update_frequency)):

            vehicle_state = carla_env.get_current_vehicle_state()
            vlm_result = self.process_frames(carla_env.frame_buffer[-self.frames_needed:], vehicle_state)

            if vlm_result["success"]:
                self.current_action_text = vlm_result["action_text"]
                self.current_action_value = vlm_result["action_value"]
                self.current_justification = vlm_result["justification"]

                # Update the environment
                carla_env.current_vlm_action = self.current_action_text
                carla_env.current_vlm_justification = self.current_justification
                carla_env.current_action_value = self.current_action_value
                self.last_update_timestep = carla_env.timestep

                if self.verbose:
                    print(
                        f"[VLM] New action: {self.current_action_text} ({self.current_action_value:.2f}) - {self.current_justification}")

                self._log_decision(vlm_result, vehicle_state)
                return True
        return False

    def process_frames(self, frame_paths, vehicle_state):
        """
        Process a sequence of frames through LLaVA-OneVision.
        """
        try:
            sequence_id = f"seq_{int(time.time())}"
            if self.verbose:
                print(f"[INFO] Processing sequence {sequence_id}: {frame_paths[0]} → {frame_paths[-1]}")

            instruction = self._get_dynamic_instruction(vehicle_state)

            # Build the content list for the user's turn
            content_list = []
            for i, frame_path in enumerate(frame_paths):
                content_list.append({"type": "text", "text": f"Frame {i + 1}:"})
                # The LLaVA processor can often handle local file paths directly
                content_list.append({"type": "image", "url": frame_path})
            
            content_list.append({"type": "text", "text": instruction})

            # Create the final conversation structure
            conversation = [{"role": "user", "content": content_list}]

            # Prepare inputs using the chat template
            inputs = self.processor.apply_chat_template(
                [conversation],  # Must be a list of conversations
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                padding=True,
                return_tensors="pt"
            ).to(self.model.device, torch.float16)

            # Generate output
            generate_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            
            # The generated IDs include the prompt, so we must slice them
            input_token_len = inputs.input_ids.shape[1]
            output_ids = generate_ids[:, input_token_len:]

            # Decode the generated tokens only
            output_text = self.processor.batch_decode(
                output_ids, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0].strip()
            
            if self.verbose:
                print(f"--- VLM RAW OUTPUT ---\n{output_text}\n----------------------")

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
            print(f"[ERROR] Error processing frames: {e}")
            return {
                "success": False, "raw_text": f"Error: {e}",
                "action_text": self.current_action_text, "action_value": self.current_action_value,
                "justification": f"Fallback due to error: {e}", "sequence_id": f"error_{int(time.time())}"
            }

    def _get_dynamic_instruction(self, vehicle_state):
        """
        Generates the dynamic instruction prompt for the VLM.
        This function is model-agnostic and can be reused.
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
    - Total: {rewards.get('total_reward', 0):.2f}
    """
        
        context_info = f"""Previous action: {self.current_action_text} (value: {self.current_action_value:.2f})
    Justification: {self.current_justification}

    Current vehicle state:
    - Speed: {speed_kmh:.1f} km/h
    {reward_info}
    """

        base_instruction = """You are an autonomous vehicle assistant. Analyze the sequence of frames to understand motion and determine the best driving action to maximize total reward by balancing safety and efficiency.

    **Instructions**:
    1.  **Safety First**: Avoid collisions at all costs. If you see a pedestrian, brake. The closer the pedestrian, the harder you brake.
    2.  **Efficiency**: When the path is clear, accelerate moderately to make progress.
    3.  **Action Values**: Your action value must be between -1.0 (max brake) and +1.0 (max throttle). 0.0 means maintain speed.
    4.  **Reward Optimization**: Use the reward signals to guide your decisions. A negative safety reward means you are too close to a pedestrian.

    Provide your response strictly in the following format:

    ACTION: [BRAKE_HARD/BRAKE_GENTLY/DECELERATE/MAINTAIN/ACCELERATE/ACCELERATE_HARD]
    VALUE: [number between -1.0 and 1.0]
    JUSTIFICATION: [Brief explanation of your decision based on the frames and rewards.]
    """
        return f"{context_info}\n\n{base_instruction}"

    def _parse_action_from_text(self, text):
        """
        Parses the structured action and justification from the model's text output.
        This function is model-agnostic and can be reused.
        """
        try:
            action_match = re.search(r'ACTION:\s*(\S+)', text, re.IGNORECASE)
            value_match = re.search(r'VALUE:\s*(-?\d+\.?\d*)', text, re.IGNORECASE)
            justification_match = re.search(r'JUSTIFICATION:\s*(.*)', text, re.IGNORECASE | re.DOTALL)

            action_text = action_match.group(1).upper() if action_match else "MAINTAIN"
            action_value = float(value_match.group(1)) if value_match else 0.0
            justification = justification_match.group(1).strip() if justification_match else "No justification provided."

            # Clamp the action value to the valid range [-1.0, 1.0]
            action_value = max(-1.0, min(1.0, action_value))

            return {"action_text": action_text, "action_value": action_value, "justification": justification}
        except Exception as e:
            print(f"[ERROR] Failed to parse VLM output: '{text}'. Error: {e}")
            return {"action_text": "MAINTAIN", "action_value": 0.0, "justification": "Error during parsing."}

    def _log_decision(self, vlm_result, vehicle_state):
        """Logs the VLM decision to a JSON file."""
        try:
            with open(self.log_file, 'r+') as f:
                log_data = json.load(f)
                # Sanitize state for JSON serialization
                sanitized_state = {k: str(v) for k, v in vehicle_state.items()}
                log_data["decisions"].append({
                    "timestamp": time.time(), "sequence_id": vlm_result.get("sequence_id"),
                    "vehicle_state": sanitized_state, "action_text": vlm_result.get("action_text"),
                    "action_value": vlm_result.get("action_value"), "justification": vlm_result.get("justification"),
                    "raw_response": vlm_result.get("raw_text")
                })
                f.seek(0)
                json.dump(log_data, f, indent=4)
        except Exception as e:
            print(f"[ERROR] Failed to log decision: {e}")