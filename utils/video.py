import cv2
import os
import re
from PIL import Image
import glob

def extract_timestamp(filename):
    # Extract timestamp from filename (e.g., frame_1753558213069686_136_opencv_detection.png)
    match = re.search(r'frame_(\d+)_(\d+)_opencv_detection\.png', filename)
    if match:
        return int(match.group(1))
    return 0

def create_video(input_folder, output_file, fps=30, output_format='mp4'):
    # Get list of image files
    images = glob.glob(os.path.join(input_folder, 'frame_*_opencv_detection.png'))
    # Sort images by timestamp
    images.sort(key=extract_timestamp)
    
    if not images:
        print("No images found in the specified folder.")
        return

    # Read the first image to get dimensions
    first_frame = cv2.imread(images[0])
    height, width, _ = first_frame.shape

    if output_format.lower() == 'mp4':
        # Initializeಸਤ

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

        # Write frames to video
        for image in images:
            frame = cv2.imread(image)
            video_writer.write(frame)

        video_writer.release()
        print(f"Video saved as {output_file}")

    elif output_format.lower() == 'gif':
        # Convert images to PIL format for GIF
        pil_images = [Image.fromarray(cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)) for image in images]
        
        # Save GIF
        pil_images[0].save(
            output_file,
            save_all=True,
            append_images=pil_images[1:],
            duration=int(1000/fps),
            loop=0
        )
        print(f"GIF saved as {output_file}")

def main():
    input_folder = "/home/server01/BARKIN/carla-vision-language-driving/vlm_outputs/frames/example_lateral_success"
    output_file = "./output_lateral.mp4"  # Default output file name
    fps = 30
    
    if output_file.endswith('.mp4'):
        create_video(input_folder, output_file, fps, 'mp4')
    elif output_file.endswith('.gif'):
        create_video(input_folder, output_file, fps, 'gif')
    else:
        print("Unsupported output format. Please use .mp4 or .gif")

if __name__ == "__main__":
    main()