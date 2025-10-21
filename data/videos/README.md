# Sample Traffic Videos - Setup Instructions

## Directory Structure
```
data/videos/
├── raw/          # Place downloaded videos here
└── processed/    # Output videos will be saved here
```

## Person C - Video Collection Task

### Requirements
- **Quantity**: 3-5 high-quality videos
- **Quality**: 720p (1280x720) or 1080p (1920x1080)
- **Location**: PCMC/Pune traffic areas
- **Camera Angle**: Stationary camera (no panning/moving)
- **Duration**: 1-3 minutes per video (optimal for testing)

### Recommended Search Terms for YouTube
- "Pune traffic live"
- "PCMC traffic camera"
- "Pune road traffic stationary"
- "Pimpri Chinchwad traffic"
- "Pune highway traffic"
- "Pune traffic junction"

### Good Video Characteristics
✅ **Look for:**
- Fixed camera position (no movement)
- Clear view of vehicles
- Good lighting (daytime preferred)
- Multiple vehicle types (cars, bikes, buses, trucks)
- Busy traffic scenes with vehicle movement
- Clear road markings/lanes

❌ **Avoid:**
- Moving/handheld camera footage
- Poor lighting (night videos may have low detection accuracy)
- Heavily obstructed views
- Dashcam footage (constantly moving perspective)
- Too much camera shake

### Download Instructions

#### Option 1: Using yt-dlp (Recommended)
```bash
# Install yt-dlp
pip install yt-dlp

# Download a video (720p)
yt-dlp -f "best[height<=720]" -o "data/videos/raw/pune_traffic_01.mp4" <YOUTUBE_URL>

# Download a video (1080p)
yt-dlp -f "best[height<=1080]" -o "data/videos/raw/pune_traffic_02.mp4" <YOUTUBE_URL>
```

#### Option 2: Online Downloaders
- Use online tools like: 
  - y2mate.com
  - savefrom.net
  - clipconverter.cc
- Select MP4 format and 720p/1080p quality
- Save to `data/videos/raw/` directory

### Naming Convention
Use descriptive names:
- `pune_traffic_01_junction.mp4`
- `pune_traffic_02_highway.mp4`
- `pcmc_traffic_03_roundabout.mp4`

### Example Video Sources (Curated List)
Search YouTube for channels that post:
- Pune traffic updates
- Indian road traffic compilations
- Traffic monitoring channels
- City traffic live streams

### File Size Guidelines
- 720p (1-3 min): ~50-150 MB
- 1080p (1-3 min): ~100-300 MB

### Copyright & Attribution
- Ensure videos are for educational/research purposes
- Keep a text file (`video_sources.txt`) with:
  - Video filename
  - YouTube URL
  - Channel name
  - Date downloaded

### After Downloading
1. Place all videos in `data/videos/raw/`
2. Create a `video_sources.txt` file with attribution
3. Test one video with the detection script:
   ```bash
   python src/vision/yolo_basic.py --input data/videos/raw/pune_traffic_01.mp4 --output data/videos/processed/output_01.mp4
   ```

### Storage & Sharing
- **Local Storage**: Keep raw videos in this folder
- **Git**: Do NOT commit video files to git (already in .gitignore)
- **Shared Drive**: Upload to Google Drive/OneDrive and share link with team
  - Create shared folder: "ML_PBL_Traffic_Videos"
  - Share read access with team members

### Quality Check
After downloading, verify:
- [ ] Video plays smoothly
- [ ] Resolution is 720p or 1080p (check properties)
- [ ] Video is stationary camera view
- [ ] Multiple vehicles visible
- [ ] Good lighting/visibility
- [ ] Duration is 1-3 minutes

## Test Video Processing
Once you have videos, test the detection script:

```bash
# Activate your environment first
# conda activate ml_pbl  (or)
# .\venv\Scripts\Activate.ps1

# Run detection
python src/vision/yolo_basic.py --input data/videos/raw/your_video.mp4 --output data/videos/processed/output.mp4

# View the output video
# Open data/videos/processed/output.mp4 in any video player
```

## Expected Results
- Output video will have bounding boxes around vehicles
- Frame counter and vehicle count displayed
- Processing time: ~1-2 minutes for a 30-second video (depends on hardware)

---

**Note**: First run will download the YOLOv8 model (~6MB for yolov8n.pt) automatically.
