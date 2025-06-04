import os
import glob
import json
import re
from PIL import Image, ImageDraw, ImageFont

# Configuration
FRAMES_DIR = '/home/server01/Vinal/CARLA_0.9.15/VLM_Barkin/VLM_Action/vlm_outputs/frames/example3'  # Directory containing frame images
JSON_FILE = '/home/server01/Vinal/CARLA_0.9.15/VLM_Barkin/VLM_Action/vlm_outputs/vlm_decisions_example3.json'  # JSON file with decisions
OUTPUT_GIF = '/home/server01/Vinal/CARLA_0.9.15/VLM_Barkin/VLM_Action/vlm_outputs/example3.gif'  # Output GIF filename
DURATION = 200  # ms
FONT_SIZE = 22
TEXT_COLOR = (255, 255, 255)
BG_COLOR = (0, 0, 0, 180)
TARGET_SIZE = (740, 740)

def load_decisions(json_file):
    """Loads decisions from JSON file and sorts them chronologically."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract and sort decisions chronologically
    decisions_list = []
    for decision in data["decisions"]:
        # Extract better justification if default is used
        justification = decision["justification"]
        raw_response = decision.get("raw_response", "")
        
        # If justification is default, try to extract better one from raw_response
        if justification == "Default justification" and raw_response:
            # Try to find justification in raw_response
            if "JUSTIFICATION:" in raw_response:
                try:
                    justification = raw_response.split("JUSTIFICATION:")[1].strip()
                except:
                    pass
        
        # Extract total reward from current_rewards string if available
        total_reward = None
        current_rewards_str = decision["vehicle_state"].get("current_rewards", "")
        if current_rewards_str:
            # Try to extract total_reward using regex
            match = re.search(r"'total_reward': np\.float64\(([-+]?[0-9]*\.?[0-9]+)\)", current_rewards_str)
            if match:
                total_reward = float(match.group(1))
        
        decisions_list.append({
            "action_text": decision["action_text"],
            "action_value": decision["action_value"],
            "justification": justification,
            "vehicle_state": decision["vehicle_state"],
            "total_reward": total_reward
        })
    
    # Sort decisions by timestamp
    decisions_list.sort(key=lambda x: x["vehicle_state"].get("timestep", 0))
    
    print(f"Loaded {len(decisions_list)} decisions from JSON.")
    return decisions_list

def extract_frame_index(filename):
    """Extracts frame index from filename."""
    match = re.search(r'_(\d+)_opencv', filename)
    if match:
        return int(match.group(1))
    return None

def get_decision_for_frame(frame_index, decisions_list, frames_per_decision=3):
    """Maps a frame to its corresponding decision based on order/index."""
    # Calculate which decision this frame corresponds to
    decision_index = frame_index // frames_per_decision
    
    # Check if this decision index exists
    if 0 <= decision_index < len(decisions_list):
        return decisions_list[decision_index]
    
    # If we're past the last decision, return the last one
    if frame_index >= len(decisions_list) * frames_per_decision and decisions_list:
        return decisions_list[-1]
    
    return None

def add_text_overlay(image, text_lines):
    """Adds a text overlay on an image."""
    draw = ImageDraw.Draw(image, 'RGBA')
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", FONT_SIZE)
    except:
        font = ImageFont.load_default()

    max_width = 0
    total_height = 0
    line_heights = []

    for line in text_lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        max_width = max(max_width, width)
        line_heights.append(height)
        total_height += height + 8

    padding = 16
    draw.rectangle(
        [(0, 0), (max_width + 2 * padding, total_height + 2 * padding)],
        fill=BG_COLOR
    )

    y_position = padding
    for i, line in enumerate(text_lines):
        draw.text((padding, y_position), line, fill=TEXT_COLOR, font=font)
        y_position += line_heights[i] + 8

    return image

def format_justification(justification, max_chars=80):
    """Format justification text with proper wrapping."""
    if not justification or justification == "Default justification":
        return ["No detailed justification available"]
    
    # Split text into lines that fit within max_chars
    words = justification.split()
    lines = []
    current_line = ""
    
    for word in words:
        if len(current_line) + len(word) + 1 <= max_chars:
            if current_line:
                current_line += " " + word
            else:
                current_line = word
        else:
            lines.append(current_line)
            current_line = word
    
    if current_line:
        lines.append(current_line)
    
    return lines

def create_gif():
    """Creates the GIF from frames with text overlays."""
    frames = []
    decisions_list = load_decisions(JSON_FILE)

    # Get the sorted list of frame files
    frame_files = sorted(glob.glob(os.path.join(FRAMES_DIR, '*.png')), 
                          key=lambda x: extract_frame_index(os.path.basename(x)) or -1)
    print(f"Found {len(frame_files)} total frame images.")

    for frame_file in frame_files:
        base_name = os.path.basename(frame_file)

        try:
            # Extract frame index from filename
            frame_index = extract_frame_index(base_name)
            
            if frame_index is None:
                print(f"Could not extract frame index from {base_name}")
                continue
                
            # Open the image and resize it
            img = Image.open(frame_file).convert('RGBA')
            img = img.resize(TARGET_SIZE, Image.LANCZOS)

            # Get the corresponding decision based on frame index
            decision = get_decision_for_frame(frame_index, decisions_list, frames_per_decision=3)
            
            if decision:
                # Get vehicle state info for display
                vehicle_state = decision["vehicle_state"]
                speed_kmh = vehicle_state.get("speed_kmh", 0)
                
                # Format justification with proper line breaks
                justification_lines = format_justification(decision['justification'])
                
                # Create text overlay with simplified information
                text_lines = [
                    f"Action: {decision['action_text']} ({decision['action_value']:.2f})",
                    f"Speed: {speed_kmh:.2f} km/h"
                ]
                
                # Add total reward if available
                if decision["total_reward"] is not None:
                    text_lines.append(f"Total Reward: {decision['total_reward']:.4f}")
                
                # Add blank line before justification
                text_lines.append("")
                text_lines.append("Justification:")
                
                # Add formatted justification lines
                text_lines.extend(justification_lines)
                
                img = add_text_overlay(img, text_lines)
            
            else:
                # Default overlay if no matching decision found
                img = add_text_overlay(img, [
                    f"Frame: {frame_index}", 
                    "(No matching decision)"
                ])

            # Convert RGBA to RGB (remove alpha channel) for saving as GIF
            if img.mode == 'RGBA':
                background = Image.new('RGB', img.size, (0, 0, 0))
                background.paste(img, mask=img.split()[3])
                img = background

            frames.append(img)

            # Print progress every 50 frames
            if len(frames) % 50 == 0:
                print(f"Processed {len(frames)} frames...")

        except Exception as e:
            print(f"Error processing {base_name}: {e}")
            continue

    if frames:
        # Save the frames as a GIF
        frames[0].save(
            OUTPUT_GIF,
            save_all=True,
            append_images=frames[1:],
            optimize=False,
            duration=DURATION,
            loop=0
        )
        print(f"GIF saved as {OUTPUT_GIF}")
    else:
        print("No frames were processed successfully.")

if __name__ == "__main__":
    create_gif()