import os
import time
import json
import re
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


class VLMDescriptor:
    """
    Simplified Vision Language Model descriptor for generating contrasting language goals
    (positive and negative) based on driving scenes from CARLA.
    """

    def __init__(
            self,
            model_name="DAMO-NLP-SG/VideoLLaMA3-2B-Image",
            output_dir="/home/cavlab/CARLA_0.9.15/VLM_Barkin/CarlaEnv/vlm_outputs",
            max_new_tokens=512,
            verbose=True
    ):
        """
        Initialize the VLM Descriptor.

        Args:
            model_name: HuggingFace model name for VideoLLaMA
            output_dir: Directory to save outputs and logs
            max_new_tokens: Maximum tokens to generate in model responses
            verbose: Whether to print detailed logs
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.max_new_tokens = max_new_tokens
        self.verbose = verbose

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize logging
        self.log_file = os.path.join(output_dir, f"vlm_descriptions_{int(time.time())}.json")
        with open(self.log_file, 'w') as f:
            json.dump({"initialization": time.time(), "descriptions": []}, f)

        # Load the model
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

    def describe_scene(self, frame_paths, vehicle_state):
        """
        Process driving scene frames and generate contrasting language goals.

        Args:
            frame_paths: List of paths to frame images
            vehicle_state: Dictionary with current vehicle state

        Returns:
            dict: Positive and negative language goals
        """
        try:
            sequence_id = f"seq_{int(time.time())}"
            if self.verbose:
                print(f"Processing sequence {sequence_id}: {frame_paths[0]} → {frame_paths[-1]}")

            # Get instruction with vehicle state
            instruction = self._get_dynamic_instruction(vehicle_state)

            # Create content array with frames
            content = []
            for i, frame_path in enumerate(frame_paths):
                content.append({"type": "text", "text": f"Frame{i + 1}: "})
                content.append({"type": "image", "image": {"image_path": frame_path}})
            content.append({"type": "text", "text": instruction})

            # Create conversation format
            conversation = [{"role": "user", "content": content}]

            # Process inputs
            inputs = self.processor(conversation=conversation, return_tensors="pt")
            inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
                      for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)

            # Generate output
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
            output_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0].strip()

            # Parse contrasting goals
            parsed_result = self._parse_goals_from_text(output_text)

            # Log results
            self._log_description(parsed_result, vehicle_state, sequence_id)

            return {
                "success": True,
                "positive_goal": parsed_result["positive_goal"],
                "negative_goal": parsed_result["negative_goal"],
                "sequence_id": sequence_id
            }

        except Exception as e:
            print(f"Error processing frames: {e}")
            return {
                "success": False,
                "positive_goal": "Unable to generate positive goal due to processing error.",
                "negative_goal": "Unable to generate negative goal due to processing error.",
                "sequence_id": f"error_{int(time.time())}"
            }

    def _get_dynamic_instruction(self, vehicle_state):
        """
        Generate a dynamic instruction for contrasting language goals.

        Args:
            vehicle_state: Current state information from the environment

        Returns:
            str: Instruction for the VLM
        """
        speed_kmh = vehicle_state.get("speed_kmh", 0)
        context_info = f"""Current vehicle state:
    - Speed: {speed_kmh:.1f} km/h
    - Acceleration: {vehicle_state.get('acceleration', 0):.2f} m/s²
    - Distance to goal: {vehicle_state.get('distance_to_goal', 0):.2f} meters
    """

        instruction = """You are analyzing driving scenes for an autonomous vehicle. Based on the provided frames and vehicle state, generate contrasting language goals:

    1. **Positive Goal**: Describe the desired driving behavior that optimizes safety, efficiency, and comfort.
    2. **Negative Goal**: Describe an undesired driving behavior that would compromise safety, efficiency, or comfort.

    **Guidelines**:
    - Use the frames to identify key elements (e.g., pedestrians, vehicles, road conditions).
    - Positive goal should promote safe, smooth, and efficient driving.
    - Negative goal should highlight unsafe, inefficient, or uncomfortable actions.
    - Keep descriptions concise and specific to the scene.

    Provide your response in the following format:

    POSITIVE_GOAL: [Describe desired behavior]
    NEGATIVE_GOAL: [Describe undesired behavior]
    """
        return f"{context_info}\n\n{instruction}"

    def _parse_goals_from_text(self, text):
        """
        Parse positive and negative goals from VLM output.

        Args:
            text: Raw model output text

        Returns:
            dict: Parsed positive and negative goals
        """
        positive_goal = "Maintain current speed and direction for safe and efficient progress."
        negative_goal = "Sudden braking or swerving, risking safety and comfort."

        try:
            positive_match = re.search(r'POSITIVE_GOAL:\s*(.*?)(?:\n|$)', text, re.DOTALL)
            if positive_match:
                positive_goal = positive_match.group(1).strip()

            negative_match = re.search(r'NEGATIVE_GOAL:\s*(.*?)(?:\n|$)', text, re.DOTALL)
            if negative_match:
                negative_goal = negative_match.group(1).strip()

        except Exception as e:
            print(f"Error parsing goals: {e}")

        return {
            "positive_goal": positive_goal,
            "negative_goal": negative_goal
        }

    def _log_description(self, result, vehicle_state, sequence_id):
        """
        Log description data to file.

        Args:
            result: Results from VLM processing
            vehicle_state: Current vehicle state
            sequence_id: Unique identifier for the sequence
        """
        try:
            sanitized_state = {k: v if isinstance(v, (int, float, str, bool)) or v is None else str(v)
                              for k, v in vehicle_state.items()}
            log_entry = {
                "timestamp": time.time(),
                "sequence_id": sequence_id,
                "vehicle_state": sanitized_state,
                "positive_goal": str(result.get("positive_goal", "")),
                "negative_goal": str(result.get("negative_goal", ""))
            }

            try:
                with open(self.log_file, 'r') as f:
                    log_data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                log_data = {"initialization": time.time(), "descriptions": []}

            log_data["descriptions"].append(log_entry)
            temp_file = f"{self.log_file}.temp"
            with open(temp_file, 'w') as f:
                json.dump(log_data, f, indent=2)
            os.replace(temp_file, self.log_file)

        except Exception as e:
            print(f"Error logging description: {e}")