"""
Quick Start Script for Testing YOLOv8 Detection
This script helps you quickly test the setup without command-line arguments.
"""

from pathlib import Path
import sys

# Add parent directory to path to access vision module
sys.path.append(str(Path(__file__).parent.parent))

from vision.traffic_analyzer import process_video_with_tracking


def main():
    """Quick test with default paths."""
    
    # Get project root (two levels up from this file)
    project_root = Path(__file__).parent.parent.parent
    
    # Define paths relative to project root
    input_dir = project_root / "data" / "videos" / "raw"
    output_dir = project_root / "data" / "videos" / "processed"
    
    # Check if input directory has videos
    video_extensions = [".mp4", ".avi", ".mov", ".mkv"]
    video_files = []
    for ext in video_extensions:
        video_files.extend(list(input_dir.glob(f"*{ext}")))
    
    if not video_files:
        print("❌ No video files found in data/videos/raw/")
        print("\nPlease:")
        print("1. Download sample videos (see data/videos/README.md)")
        print("2. Place them in data/videos/raw/ directory")
        print("3. Run this script again")
        return
    
    print(f"✓ Found {len(video_files)} video(s) in data/videos/raw/")
    print("\nAvailable videos:")
    for i, video in enumerate(video_files, 1):
        print(f"  {i}. {video.name}")
    
    # Use first video
    input_video = video_files[0]
    output_video = output_dir / f"output_{input_video.stem}.mp4"
    
    print(f"\n🎬 Processing: {input_video.name}")
    print(f"📁 Output will be saved to: {output_video}")
    print("\nStarting detection... (this may take a few minutes)")
    print("-" * 60)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process video
    try:
        process_video_with_tracking(
            str(input_video),
            str(output_video),
            model_name="yolov8n.pt"
        )
        print("-" * 60)
        print("\n✅ Success! Check the output video:")
        print(f"   {output_video}")
        print("\nTo process more videos, use:")
        print(f"   python src/vision/yolo_basic.py -i <input> -o <output>")
        
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure all dependencies are installed: pip install -r requirements.txt")
        print("2. Check if video file is valid (try playing it)")
        print("3. See SETUP.md for detailed troubleshooting")


if __name__ == "__main__":
    print("=" * 60)
    print("YOLOv8 Vehicle Detection - Quick Start")
    print("=" * 60)
    main()
