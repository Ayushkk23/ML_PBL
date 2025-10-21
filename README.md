# ML_PBL
Traffic Analysis PoC using YOLOv8, Object Tracking, and Anomaly Detection

## Project Overview
This project implements an intelligent traffic monitoring system for PCMC/Pune using:
- **YOLOv8**: State-of-the-art object detection for vehicle detection
- **ByteTrack**: Advanced multi-object tracking with unique IDs
- **Anomaly Detection**: Stalled vehicle detection and vehicle counting
- **Supervision**: Computer vision utilities for annotation

## Features
- ✅ Real-time vehicle detection (cars, motorcycles, buses, trucks)
- ✅ Vehicle tracking with unique IDs using ByteTrack
- ✅ Stalled vehicle detection (vehicles stationary for 2.5+ seconds)
- ✅ Vehicle counting across virtual lines
- ✅ Visual annotations with bounding boxes and labels
- ⏳ Interactive web dashboard (Streamlit)
- ⏳ Alert system for traffic anomalies

## Quick Start

### 1. Install Dependencies

```bash
# Python 3.10+ required (Python 3.13 supported)
pip install -r requirements.txt
```

### 2. Get Traffic Videos

```bash
# Install video downloader
pip install yt-dlp

# Download a traffic video from YouTube
yt-dlp -f "best[height<=720]" -o "data/videos/raw/traffic_test.mp4" <YOUTUBE_URL>
```

See [data/videos/README.md](data/videos/README.md) for more details.

### 3. Run Detection & Tracking

```bash
# Automatic - processes all videos in data/videos/raw/
python src/logic/quick_start.py

# Manual - specify input/output
python src/vision/traffic_analyzer.py -i data/videos/raw/traffic.mp4 -o data/videos/processed/output.mp4
```

## What You Get

The output video includes:
- 🎯 **Vehicle Tracking**: Each vehicle has a unique ID
- 🚨 **Stalled Detection**: Red boxes for vehicles stopped 2.5+ seconds
- 📊 **Vehicle Counting**: Yellow line counts vehicles crossing it
- 📈 **Live Statistics**: Frame count, vehicles counted, stalled count

## Project Structure
```
ML_PBL/
├── data/videos/
│   ├── raw/                     # Place input videos here
│   ├── processed/               # Output videos saved here
│   └── README.md                # Video download guide
├── src/
│   ├── vision/
│   │   └── traffic_analyzer.py # ✅ Main detection & tracking script
│   └── logic/
│       └── quick_start.py      # ✅ Easy testing script
├── requirements.txt             # All dependencies
├── SETUP.md                     # Detailed setup guide
└── README.md                    # This file
```

## Implementation Status

### ✅ Completed Features
- **Vehicle Detection**: YOLOv8 nano model for cars, motorcycles, buses, trucks
- **Multi-Object Tracking**: ByteTrack for consistent vehicle IDs
- **Stalled Vehicle Detection**: 25px radius, 75 frames (~2.5 seconds)
- **Vehicle Counting**: Horizontal line crossing detection
- **Visual Output**: Annotated videos with bounding boxes and labels

### ⏳ Future Enhancements
- Streamlit web dashboard
- Real-time webcam support
- Alert system (email/SMS)
- Database logging
- Multi-camera support

## Technology Stack
- **Python 3.10+**
- **PyTorch**: Deep learning framework
- **YOLOv8 (Ultralytics)**: Object detection model
- **Supervision**: Tracking and annotation
- **OpenCV**: Video processing
- **Streamlit**: Dashboard framework

## Configuration

### Adjust Detection Parameters

Edit `src/vision/traffic_analyzer.py`:

```python
# Stalled vehicle detection
STALLED_RADIUS_PIXELS = 25      # Movement radius threshold
STALLED_FRAME_THRESHOLD = 75    # Frames to confirm stalled (~2.5s at 30fps)

# Use different YOLOv8 models
# yolov8n.pt - Fastest (default)
# yolov8s.pt - Small
# yolov8m.pt - Medium (more accurate)
# yolov8l.pt - Large
# yolov8x.pt - Extra large (most accurate, slowest)
```

### Example Commands

```bash
# Process with larger model (better accuracy)
python src/vision/traffic_analyzer.py -i input.mp4 -o output.mp4 --model yolov8m.pt

# Quick test with default settings
python src/logic/quick_start.py
```

## How It Works

### 1. Detection
YOLOv8 detects vehicles (car, motorcycle, bus, truck) in each frame

### 2. Tracking
ByteTrack assigns and maintains unique IDs across frames

### 3. Stalled Detection
- Tracks vehicle positions over time
- Calculates movement within recent frames
- Flags vehicles stationary for 2.5+ seconds

### 4. Counting
- Defines horizontal counting line at 60% video height
- Detects when vehicle center crosses the line
- Prevents double-counting using track IDs

## Output Visualization

- **Green boxes**: Normal moving vehicles
- **Red boxes**: Stalled vehicles (with `[STALLED]` label)
- **Yellow line**: Counting line
- **Labels**: Track ID, vehicle type, confidence score
- **Stats overlay**: Frame count, vehicles counted, stalled count

## Contributing
1. Create a feature branch
2. Make changes
3. Test thoroughly
4. Submit pull request

## Resources
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Supervision Library](https://supervision.roboflow.com/)
- [Setup Guide](SETUP.md)
- [Video Collection Guide](data/videos/README.md)

## License
Educational/Research Project

## Contact
Repository: [Ayushkk23/ML_PBL](https://github.com/Ayushkk23/ML_PBL)
