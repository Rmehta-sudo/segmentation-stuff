#!/usr/bin/env python3
"""
Frame Extraction Script
Extracts frames from a video file and saves them to frames-{video_name}/ directory
"""

import cv2
import os
import sys
from pathlib import Path
import argparse

def extract_frames(video_path, output_dir=None, fps=None, start_frame=0, max_frames=None):
    """
    Extract frames from video file
    
    Args:
        video_path (str): Path to input video file
        output_dir (str): Custom output directory (optional)
        fps (float): Extract at specific fps (optional, extracts all frames if None)
        start_frame (int): Frame to start extraction from
        max_frames (int): Maximum number of frames to extract
    
    Returns:
        str: Path to output directory
    """
    
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Generate output directory name
    if output_dir is None:
        video_name = video_path.stem  # filename without extension
        output_dir = f"frames-{video_name}"
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video: {video_path}")
    print(f"Total frames: {total_frames}")
    print(f"Video FPS: {video_fps:.2f}")
    print(f"Output directory: {output_path}")
    
    # Calculate frame sampling
    if fps is not None and fps < video_fps:
        frame_step = int(video_fps / fps)
        print(f"Extracting every {frame_step} frames (target fps: {fps})")
    else:
        frame_step = 1
        print("Extracting all frames")
    
    # Set starting frame
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        print(f"Starting from frame {start_frame}")
    
    frame_count = 0
    saved_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            
            # Check if we should extract this frame
            if (current_frame_num - start_frame) % frame_step == 0:
                # Save frame
                frame_filename = f"{saved_count+1:05d}.jpg"
                frame_path = output_path / frame_filename
                
                success = cv2.imwrite(str(frame_path), frame)
                if success:
                    saved_count += 1
                    if saved_count % 50 == 0:  # Progress update every 50 frames
                        print(f"Extracted {saved_count} frames...")
                else:
                    print(f"Failed to save frame: {frame_path}")
            
            frame_count += 1
            
            # Check if we've reached max_frames limit
            if max_frames is not None and saved_count >= max_frames:
                print(f"Reached maximum frame limit: {max_frames}")
                break
    
    except KeyboardInterrupt:
        print("\nExtraction interrupted by user")
    
    finally:
        cap.release()
    
    print(f"Extraction complete!")
    print(f"Total frames processed: {frame_count}")
    print(f"Frames saved: {saved_count}")
    print(f"Saved to: {output_path.absolute()}")
    
    return str(output_path)

def main():
    parser = argparse.ArgumentParser(description='Extract frames from video files')
    parser.add_argument('video_path', help='Path to input video file')
    parser.add_argument('--output', '-o', help='Custom output directory name')
    parser.add_argument('--fps', type=float, help='Target FPS for extraction (extracts all frames if not specified)')
    parser.add_argument('--start', type=int, default=0, help='Starting frame number (default: 0)')
    parser.add_argument('--max-frames', type=int, help='Maximum number of frames to extract')
    parser.add_argument('--list-info', action='store_true', help='Just show video info without extracting')
    
    args = parser.parse_args()
    
    # Just show video info
    if args.list_info:
        cap = cv2.VideoCapture(args.video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {args.video_path}")
            return 1
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"Video Info:")
        print(f"  Path: {args.video_path}")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Total frames: {total_frames}")
        print(f"  Duration: {duration:.2f} seconds")
        
        cap.release()
        return 0
    
    try:
        extract_frames(
            video_path=args.video_path,
            output_dir=args.output,
            fps=args.fps,
            start_frame=args.start,
            max_frames=args.max_frames
        )
        return 0
    
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    # If run without arguments, show usage examples
    if len(sys.argv) == 1:
        print("Frame Extraction Script")
        print("======================")
        print()
        print("Usage examples:")
        print("  python extract_frames.py video.mp4                    # Extract all frames")
        print("  python extract_frames.py video.mp4 --fps 10           # Extract at 10 fps")
        print("  python extract_frames.py video.mp4 --max-frames 100   # Extract first 100 frames")
        print("  python extract_frames.py video.mp4 --start 50         # Start from frame 50")
        print("  python extract_frames.py video.mp4 --list-info        # Show video info only")
        print("  python extract_frames.py video.mp4 -o custom_folder   # Custom output folder")
        print()
        print("Output: Creates 'frames-{video_name}/' directory with numbered JPG files")
        sys.exit(0)
    
    sys.exit(main())