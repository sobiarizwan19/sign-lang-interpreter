import cv2
import os
from datetime import datetime

def extract_frames(video_path, frame_gap=5, output_folder="extracted_frames"):
    """
    Extract frames from video at specified intervals
    
    Args:
        video_path: Path to input video
        frame_gap: Extract every Nth frame
        output_folder: Base folder for extracted frames
    
    Returns:
        tuple: (list of frame paths, output directory)
    """
    # Create output folder
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_folder, f"{video_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    frame_count = 0
    frame_paths = []
    
    print(f"Extracting frames from {video_path}...")
    print(f"Frame gap: {frame_gap}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_gap == 0:
            frame_path = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
        
        frame_count += 1
    
    cap.release()
    print(f"✅ Extracted {len(frame_paths)} frames to {output_dir}")
    
    return frame_paths, output_dir