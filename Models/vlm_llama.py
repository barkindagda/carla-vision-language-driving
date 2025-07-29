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
    Supports two modes: lane_change (full control) and longitudinal (acceleration only).
    """

    def __init__(
            self,
            model_name="DAMO-NLP-SG/VideoLLaMA3-2B-Image",
            update_frequency=3,
            frames_needed=3,
            output_dir="/home/server01/BARKIN/carla-vision-language-driving/vlm_outputs",
            max_new_tokens=512,
            verbose=True,
            mode="lane_change"  # Default mode: lane_change or longitudinal
    ):
        self.model_name = model_name
        self.update_frequency = update_frequency
        self.frames_needed = frames_needed
        self.output_dir = output_dir
        self.max_new_tokens = max_new_tokens
        self.verbose = verbose
        self.mode = mode.lower()
        self.current_action_text = "ACCELERATE | MAINTAIN_LANE" if mode == "lane_change" else "ACCELERATE"
        self.current_action_value = [0.5, 0.0] if mode == "lane_change" else 0.5
        self.current_justification = "Initializing with safe forward movement."
        self.last_update_timestep = 0
        self.log_file = os.path.join(output_dir, f"vlm_decisions_{int(time.time())}.json")
        
        os.makedirs(output_dir, exist_ok=True)
        with open(self.log_file, 'w') as f:
            json.dump({"initialization": time.time(), "decisions": []}, f)
        
        self._load_model()

    def _load_model(self):
        if self.verbose:
            print(f"Loading VideoLLaMA model: {self.model_name}")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
            )
            self.processor = AutoProcessor.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            print(f"Model loaded successfully on {self.model.device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise RuntimeError(f"Failed to initialize VLM model: {e}")

    def process_if_needed(self, carla_env):
        if (len(carla_env.frame_buffer) >= self.frames_needed and
                (carla_env.timestep - self.last_update_timestep >= self.update_frequency)):
            vehicle_state = carla_env.get_current_vehicle_state()
            vlm_result = self.process_frames(carla_env.frame_buffer[-self.frames_needed:], vehicle_state)
            if vlm_result["success"]:
                self.current_action_text = vlm_result["action_text"]
                self.current_action_value = vlm_result["action_value"]
                self.current_justification = vlm_result["justification"]
                carla_env.current_vlm_action = self.current_action_text
                carla_env.current_vlm_justification = self.current_justification
                carla_env.current_action_value = self.current_action_value
                self.last_update_timestep = carla_env.timestep
                if self.verbose:
                    yaw = vehicle_state.get("yaw", 90.0)
                    if self.mode == "lane_change":
                        print(
                            f"[VLM] New action: {self.current_action_text} "
                            f"(throttle/brake: {self.current_action_value[0]:.2f}, "
                            f"steer: {self.current_action_value[1]:.2f}, "
                            f"yaw: {yaw:.1f}°) - {self.current_justification}"
                        )
                    else:
                        print(
                            f"[VLM] New action: {self.current_action_text} "
                            f"(throttle/brake: {self.current_action_value:.2f}, "
                            f"yaw: {yaw:.1f}°) - {self.current_justification}"
                        )
                self._log_decision(vlm_result, vehicle_state)
                return True
        return False

    def process_frames(self, frame_paths, vehicle_state):
        try:
            sequence_id = f"seq_{int(time.time())}"
            if self.verbose:
                print(f"Processing sequence {sequence_id}: {frame_paths[0]} → {frame_paths[-1]}")
            instruction = self._get_dynamic_instruction(vehicle_state)
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
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=True, temperature=0.7, top_p=0.9
            )
            output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            print(f"[DEBUG] Raw VLM output: {output_text}")
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
            default_action = "ACCELERATE | MAINTAIN_LANE" if self.mode == "lane_change" else "ACCELERATE"
            default_value = [0.5, 0.0] if self.mode == "lane_change" else 0.5
            return {
                "success": False,
                "raw_text": f"Error: {str(e)}",
                "action_text": default_action,
                "action_value": default_value,
                "justification": f"Fallback due to error: {str(e)}",
                "sequence_id": f"error_{int(time.time())}"
            }

    def _get_dynamic_instruction(self, vehicle_state):
        speed_kmh = vehicle_state.get("speed_kmh", 0.0)
        reward_info = ""
        if "current_rewards" in vehicle_state and vehicle_state["current_rewards"]:
            rewards = vehicle_state["current_rewards"]
            reward_info = (
                f"Recent Reward Breakdown:\n"
                f"- Safety: {rewards.get('safety_reward', 0.0):.2f} (negative near obstacles/pedestrians)\n"
                f"- Progress: {rewards.get('progress_reward', 0.0):.2f} (positive for good speed when safe)\n"
                f"- Smoothness: {rewards.get('smoothness_reward', 0.0):.2f} (negative for jerky driving)\n"
                f"- {'Lane Change: ' + str(rewards.get('lane_change_reward', 0.0)) + ' (positive for necessary, negative for unsafe)' if self.mode == 'lane_change' else ''}\n"
                f"- Collision: {rewards.get('collision_penalty', 0.0):.2f} (large negative for collisions)\n"
                f"- Total: {rewards.get('total_reward', 0.0):.2f}\n"
            )

        if self.mode == "lane_change":
            dist_to_obstacle = vehicle_state.get("distance_to_obstacle", float('inf'))
            right_lane_x = -52.073021
            left_lane_x = -48.64543151855469
            current_x = vehicle_state.get("vehicle_location_x", 0.0)
            yaw = vehicle_state.get("yaw", 90.0)
            yaw_deviation = yaw - 90.0
            is_in_right_lane = abs(current_x - right_lane_x) < 1.0
            is_in_left_lane = abs(current_x - left_lane_x) < 1.0
            nearby_vehicles_count = vehicle_state.get("nearby_vehicles_count", 0)
            pedestrian_detected = vehicle_state.get("pedestrian_detected", False)
            pedestrian_distance = vehicle_state.get("pedestrian_distance", float('inf'))
            pedestrian_distance_category = vehicle_state.get("pedestrian_distance_category", "NONE")
            right_boundary = -53.5
            left_boundary = -47.0
            is_near_right_boundary = current_x < right_boundary
            is_near_left_boundary = current_x > left_boundary
            nearest_lane_x = right_lane_x if is_in_right_lane else left_lane_x
            x_deviation = current_x - nearest_lane_x
            yaw_steer_correction = max(min(-yaw_deviation * 0.05, 0.3), -0.3)
            x_steer_correction = max(min(-x_deviation * 0.1, 0.3), -0.3)
            steer_correction = yaw_steer_correction if abs(yaw_deviation) > 2.0 else x_steer_correction

            instruction = (
                "You are controlling an autonomous vehicle in a CARLA simulation. "
                "Based on the provided images and vehicle state, decide the best action to optimize safety, efficiency, orientation alignment, and stay within road boundaries. "
                f"Current state:\n"
                f"- Speed: {speed_kmh:.1f} km/h\n"
                f"- Distance to obstacle (ambulance): {dist_to_obstacle:.1f} meters\n"
                f"- Pedestrian detected: {pedestrian_detected} (distance: {pedestrian_distance:.1f} m, category: {pedestrian_distance_category})\n"
                f"- Position X: {current_x:.2f} (right lane: -52.07, left lane: -48.65, boundaries: -53.5 to -47.0)\n"
                f"- In right lane: {is_in_right_lane}\n"
                f"- In left lane: {is_in_left_lane}\n"
                f"- Near right boundary (x < -53.5): {is_near_right_boundary}\n"
                f"- Near left boundary (x > -47.0): {is_near_left_boundary}\n"
                f"- Yaw: {yaw:.1f}° (target: 90° for straight road, deviation: {yaw_deviation:.1f}°)\n"
                f"- Nearby vehicles: {nearby_vehicles_count}\n"
                f"- Previous action: {self.current_action_text} (throttle/brake: {self.current_action_value[0]:.2f}, steer: {self.current_action_value[1]:.2f})\n"
                f"{reward_info}\n"
                "Guidelines:\n"
                "1. **Reward Optimization**:\n"
                "   - Maximize total_reward by prioritizing safety (keep safety_reward near 0), then progress (increase when safe), and smoothness (minimize jerky driving).\n"
                "   - Perform lane changes only when necessary (positive lane_change_reward) and safe (no nearby vehicles, no pedestrians near).\n"
                "   - Avoid collision_penalty (-200) at all costs.\n"
                "2. **Road Boundary Enforcement**:\n"
                f"   - Keep vehicle within x = -53.5 to -47.0. Current x_deviation from lane center: {x_deviation:.2f} meters.\n"
                f"   - If x < -53.5, use CHANGE_LEFT ({x_steer_correction:.2f} suggested) to return to lane.\n"
                f"   - If x > -47.0 and nearby_vehicles_count > 0 or oncoming vehicles are visible in frames, use CHANGE_RIGHT (0.5 suggested) to correct orientation and avoid crossing into oncoming traffic.\n"
                f"   - If x > -47.0 and no oncoming vehicles, use CHANGE_RIGHT ({x_steer_correction:.2f} suggested) to return to lane.\n"
                "3. **Longitudinal Control**:\n"
                "   - If obstacle < 20 meters or pedestrian_distance < 7.5 meters and in right lane, use BRAKE_GENTLY (-0.5).\n"
                "   - If speed < 20 km/h and no obstacle within 20 meters and no pedestrian within 7.5 meters, use ACCELERATE (0.5).\n"
                "   - If speed > 30 km/h and obstacle < 30 meters or pedestrian_distance < 10 meters, use DECELERATE (-0.2).\n"
                "4. **Orientation and Lateral Control**:\n"
                f"   - Target yaw is 90° (straight along road). Current deviation: {yaw_deviation:.1f}°.\n"
                "   - If in right lane and obstacle < 20 meters, use CHANGE_LEFT (-0.5) to initiate lane change, provided nearby_vehicles_count = 0 and pedestrian_distance > 10 meters.\n"
                "   - After lane change (x ≈ -48.65) or if not near boundaries, correct orientation:\n"
                f"     - If yaw > 92° (deviation > 2°), use CHANGE_RIGHT ({yaw_steer_correction:.2f} suggested) to reduce yaw.\n"
                f"     - If yaw < 88° (deviation < -2°), use CHANGE_LEFT ({yaw_steer_correction:.2f} suggested) to increase yaw.\n"
                f"     - If yaw ≈ 90° (|deviation| ≤ 2°) and x ≈ -52.07 or -48.65, use MAINTAIN_LANE (0.0).\n"
                "   - Avoid lane changes if nearby_vehicles_count > 0 or pedestrian_distance < 10 meters.\n"
                "5. **Action Values**:\n"
                "   - Throttle/Brake: [-1.0, 1.0], negative for braking, positive for acceleration.\n"
                "   - Steer: [-1.0, 1.0], negative for left, positive for right, use proportional steering for boundary or yaw correction.\n"
                "Output in the following format:\n"
                "ACTION: [BRAKE_HARD/BRAKE_GENTLY/DECELERATE/MAINTAIN/ACCELERATE/ACCELERATE_HARD] | [CHANGE_LEFT/CHANGE_RIGHT/MAINTAIN_LANE]\n"
                f"VALUE: [throttle_brake, {steer_correction:.2f}] (use suggested steer for boundary or yaw correction, or 0.5 for oncoming traffic correction)\n"
                "JUSTIFICATION: [Explanation based on frames, state, yaw, boundaries, pedestrians, oncoming vehicles, and rewards]\n"
            )
        else:  # longitudinal mode
            trend_info = ""
            if all(key in vehicle_state for key in ["safety_trend", "progress_trend", "smoothness_trend"]):
                trend_info = (
                    f"Performance Trends:\n"
                    f"- Safety trend: {vehicle_state['safety_trend']}\n"
                    f"- Progress trend: {vehicle_state['progress_trend']}\n"
                    f"- Smoothness trend: {vehicle_state['smoothness_trend']}\n"
                )
            instruction = (
                f"Previous action: {self.current_action_text} (value: {self.current_action_value:.2f})\n"
                f"Justification: {self.current_justification}\n"
                f"Current vehicle state:\n"
                f"- Speed: {speed_kmh:.1f} km/h\n"
                f"- Acceleration: {vehicle_state.get('acceleration', 0):.2f} m/s²\n"
                f"- Distance to goal: {vehicle_state.get('distance_to_goal', 0):.2f} meters\n"
                f"- Timestep: {vehicle_state.get('timestep', 0)}\n"
                f"{reward_info}\n"
                f"{trend_info}\n"
                "You are assisting an autonomous vehicle in longitudinal control mode (no steering). "
                "Examine the frames and determine the best driving action to optimize safety, efficiency, and comfort.\n"
                "**GOAL**: Maximize total reward by balancing safety, efficiency, and comfort.\n"
                "**IMPORTANT INSTRUCTIONS**:\n"
                "1. **Reward Optimization (HIGHEST PRIORITY)**:\n"
                "   - Maximize total reward.\n"
                "   - Safety reward: Keep close to zero (negative when near pedestrians).\n"
                "   - Progress reward: Maximize when safe (higher speed when no pedestrians).\n"
                "   - Smoothness reward: Avoid sudden acceleration/braking changes.\n"
                "   - Collision penalty: Avoid at all costs (-200 penalty).\n"
                "2. **Balanced Priorities**:\n"
                "   - Safety: Pedestrian safety is essential but balance with efficient progress.\n"
                "   - Efficiency: Complete the journey quickly when safe.\n"
                "   - Comfort: Provide smooth acceleration/deceleration.\n"
                "3. **Action Values**:\n"
                "   - Action values range from -1.0 to +1.0:\n"
                "     - Negative (-1.0 to 0): Braking (closer to -1.0 means harder braking).\n"
                "     - Zero (0): Maintain current speed.\n"
                "     - Positive (0 to +1.0): Acceleration (closer to +1.0 means stronger acceleration).\n"
                "4. **Pedestrian Response Guidelines**:\n"
                "   - Rely on frames to detect pedestrians.\n"
                "   - If pedestrian is close: BRAKE_HARD (-0.7 to -0.9).\n"
                "   - If pedestrian is at medium distance: BRAKE_GENTLY (-0.4 to -0.6).\n"
                "   - If pedestrian is far: DECELERATE (-0.1 to -0.3).\n"
                "   - If no pedestrian: ACCELERATE (+0.5 to +0.8).\n"
                "   - Be cautious around occlusions.\n"
                "5. **Vehicle Control Interpretation**:\n"
                "   - -1.0 to -0.8: BRAKE_HARD (emergency stop)\n"
                "   - -0.7 to -0.3: BRAKE_GENTLY (moderate braking)\n"
                "   - -0.2 to -0.1: DECELERATE (slight braking)\n"
                "   - 0.0: MAINTAIN (maintain current speed)\n"
                "   - 0.1 to 0.5: ACCELERATE (increase speed gently)\n"
                "   - 0.5 to 1.0: ACCELERATE_HARD (increase speed aggressively)\n"
                "6. **Reward-Based Decision Making**:\n"
                "   - Adjust strategy based on reward trends.\n"
                "   - Declining safety reward: Increase caution.\n"
                "   - Declining progress reward: Increase speed when safe.\n"
                "   - Declining smoothness reward: Make gradual changes.\n"
                "Output in the following format:\n"
                "ACTION: [BRAKE_HARD/BRAKE_GENTLY/DECELERATE/MAINTAIN/ACCELERATE/ACCELERATE_HARD]\n"
                "VALUE: [number between -1.0 and +1.0]\n"
                "JUSTIFICATION: [Explanation based on frames, state, and reward optimization]\n"
            )
        return instruction

    def _parse_action_from_text(self, text):
        if self.mode == "lane_change":
            action_text = "ACCELERATE | MAINTAIN_LANE"
            action_value = [0.5, 0.0]
            justification = "Default: Safe forward movement."
            try:
                action_match = re.search(r'ACTION:\s*(BRAKE_HARD|BRAKE_GENTLY|DECELERATE|MAINTAIN|ACCELERATE|ACCELERATE_HARD)\s*\|\s*(CHANGE_LEFT|CHANGE_RIGHT|MAINTAIN_LANE)', text, re.IGNORECASE)
                if action_match:
                    action_text = f"{action_match.group(1)} | {action_match.group(2)}"
                value_match = re.search(r'VALUE:\s*\[\s*(-?\d*\.?\d*)\s*,\s*(-?\d*\.?\d*)\s*\]', text)
                if value_match:
                    throttle_value = float(value_match.group(1))
                    steer_value = float(value_match.group(2))
                    action_value = [
                        max(min(throttle_value, 1.0), -1.0),
                        max(min(steer_value, 1.0), -1.0)
                    ]
                else:
                    action_value_map = {
                        "BRAKE_HARD": [-0.9, 0.0],
                        "BRAKE_GENTLY": [-0.5, 0.0],
                        "DECELERATE": [-0.2, 0.0],
                        "MAINTAIN": [0.0, 0.0],
                        "ACCELERATE": [0.5, 0.0],
                        "ACCELERATE_HARD": [0.9, 0.0],
                        "CHANGE_LEFT": [0.0, -0.5],
                        "CHANGE_RIGHT": [0.0, 0.5],
                        "MAINTAIN_LANE": [0.0, 0.0]
                    }
                    longitudinal_action = action_text.split(" | ")[0]
                    lateral_action = action_text.split(" | ")[1]
                    action_value = [
                        action_value_map.get(longitudinal_action, [0.5, 0.0])[0],
                        action_value_map.get(lateral_action, [0.0, 0.0])[1]
                    ]
                justification_match = re.search(r'JUSTIFICATION:\s*(.*?)(?:\n|$)', text, re.DOTALL)
                justification = justification_match.group(1).strip() if justification_match else "No justification provided."
            except Exception as e:
                print(f"[ERROR] Parsing failed: {e}, Raw text: {text}")
                justification = f"Parsing error: {str(e)}, defaulting to safe forward movement."
        else:  # longitudinal mode
            action_text = "ACCELERATE"
            action_value = 0.5
            justification = "Default: Safe forward movement."
            try:
                action_match = re.search(r'ACTION:\s*(BRAKE_HARD|BRAKE_GENTLY|DECELERATE|MAINTAIN|ACCELERATE|ACCELERATE_HARD)', text, re.IGNORECASE)
                if action_match:
                    action_text = action_match.group(1)
                value_match = re.search(r'VALUE:\s*(-?\d+\.?\d*)', text)
                if value_match:
                    action_value = max(min(float(value_match.group(1)), 1.0), -1.0)
                else:
                    action_value_map = {
                        "BRAKE_HARD": -0.9,
                        "BRAKE_GENTLY": -0.5,
                        "DECELERATE": -0.2,
                        "MAINTAIN": 0.0,
                        "ACCELERATE": 0.5,
                        "ACCELERATE_HARD": 0.9
                    }
                    action_value = action_value_map.get(action_text, 0.5)
                justification_match = re.search(r'JUSTIFICATION:\s*(.*?)(?:\n|$)', text, re.DOTALL)
                if justification_match:
                    justification = justification_match.group(1).strip()
                    if len(justification) < 20:
                        full_text_after_justification = text.split("JUSTIFICATION:", 1)
                        if len(full_text_after_justification) > 1:
                            justification = full_text_after_justification[1].strip()
            except Exception as e:
                print(f"Error parsing action from text: {e}")
                if "brake hard" in text.lower() or "emergency" in text.lower():
                    action_text = "BRAKE_HARD"
                    action_value = -0.9
                elif "brake" in text.lower() or "slow down" in text.lower():
                    action_text = "BRAKE_GENTLY"
                    action_value = -0.5
                elif "deceler" in text.lower():
                    action_text = "DECELERATE"
                    action_value = -0.2
                elif "maintain" in text.lower() or "current speed" in text.lower():
                    action_text = "MAINTAIN"
                    action_value = 0.0
                elif "acceler" in text.lower() or "speed up" in text.lower():
                    action_text = "ACCELERATE"
                    action_value = 0.5
                justification = "Parsed from context due to format error"
        return {
            "action_text": action_text,
            "action_value": action_value,
            "justification": justification
        }

    def _log_decision(self, vlm_result, vehicle_state):
        try:
            sanitized_state = {k: v if isinstance(v, (int, float, str, bool)) or v is None else str(v) for k, v in vehicle_state.items()}
            log_entry = {
                "timestamp": time.time(),
                "sequence_id": vlm_result.get("sequence_id", "unknown"),
                "vehicle_state": sanitized_state,
                "action_text": vlm_result.get("action_text", "UNKNOWN"),
                "action_value": vlm_result.get("action_value", [0.5, 0.0] if self.mode == "lane_change" else 0.5),
                "justification": str(vlm_result.get("justification", "")),
                "raw_response": str(vlm_result.get("raw_text", "")),
                "mode": self.mode
            }
            with open(self.log_file, 'r') as f:
                log_data = json.load(f)
            log_data["decisions"].append(log_entry)
            temp_file = f"{self.log_file}.temp"
            with open(temp_file, 'w') as f:
                json.dump(log_data, f, indent=2)
            os.replace(temp_file, self.log_file)
        except Exception as e:
            print(f"Error logging decision: {e}")