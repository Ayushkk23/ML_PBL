# Development Environment Setup Guide

## Prerequisites
- Python 3.8 or higher
- pip or conda
- Git

## Setup Options

### Option 1: Using Conda (Recommended)

```bash
# Create conda environment
conda create -n ml_pbl python=3.10 -y

# Activate environment
conda activate ml_pbl

# Install PyTorch (with CUDA if you have a compatible GPU)
# For CUDA 11.8:
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# For CPU only:
conda install pytorch torchvision cpuonly -c pytorch

# Install other requirements
pip install -r requirements.txt
```

### Option 2: Using venv (Python Virtual Environment)

```bash
# Create virtual environment
python -m venv venv

# Activate environment (Windows)
.\venv\Scripts\Activate.ps1

# Activate environment (Linux/Mac)
source venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

## Verify Installation

```bash
# Check Python version
python --version

# Check if packages are installed
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import supervision as sv; print(f'Supervision: {sv.__version__}')"
python -c "from ultralytics import YOLO; print('Ultralytics: OK')"
```

## GPU Support (Optional but Recommended)

### Check if GPU is available:
```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

### If you have NVIDIA GPU:
1. Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
2. Install cuDNN: https://developer.nvidia.com/cudnn
3. Reinstall PyTorch with CUDA support (see conda command above)

## Project Structure

```
ML_PBL/
├── data/
│   └── videos/
│       ├── raw/           # Place input videos here
│       └── processed/     # Output videos saved here
├── src/
│   ├── vision/
│   │   └── yolo_basic.py  # YOLOv8 detection script (Person A)
│   └── logic/
│       └── anomaly_detection.py  # Anomaly logic (Person B)
├── requirements.txt
├── README.md
└── SETUP.md (this file)
```

## Running the Scripts

### Person A - Vision Script (YOLOv8 Detection)

```bash
# Basic usage
python src/vision/yolo_basic.py --input data/videos/raw/traffic.mp4 --output data/videos/processed/output.mp4

# With different model (larger = more accurate but slower)
python src/vision/yolo_basic.py -i input.mp4 -o output.mp4 --model yolov8s.pt

# Model options (in order of speed/accuracy trade-off):
# yolov8n.pt (nano - fastest, least accurate)
# yolov8s.pt (small)
# yolov8m.pt (medium)
# yolov8l.pt (large)
# yolov8x.pt (extra large - slowest, most accurate)
```

### Person B - Logic (Anomaly Detection)

The `anomaly_detection.py` file contains pseudocode and design for:
- Stalled vehicle detection
- Vehicle counting across a line
- Integration guidelines

This will be implemented in the next phase.

## Troubleshooting

### Import Errors
```bash
# If you get "module not found" errors:
pip install --upgrade -r requirements.txt
```

### CUDA/GPU Issues
```bash
# Reinstall PyTorch (CPU version)
pip uninstall torch torchvision
pip install torch torchvision
```

### Video Codec Issues
```bash
# Install additional codecs
pip install opencv-python-headless
# or
conda install -c conda-forge opencv
```

### Slow Processing
- Use smaller model: `yolov8n.pt` instead of larger models
- Process every Nth frame (modify script)
- Use GPU acceleration (see GPU Support section)

## Dependencies List

### Essential Packages:
- **torch** (2.5.1): PyTorch deep learning framework
- **torchvision** (0.20.1): Computer vision models and utilities
- **ultralytics** (8.3.63): YOLOv8 implementation
- **opencv-python** (4.12.0.88): Computer vision library
- **supervision** (0.26.1): Object detection utilities (tracking, annotation)
- **streamlit** (1.50.0): Web dashboard framework (for future UI)

### Supporting Packages:
- numpy: Numerical computations
- pandas: Data manipulation
- matplotlib: Visualization
- pillow: Image processing

## Next Steps

1. **Environment Setup** (Person C):
   - ✅ Create requirements.txt
   - ✅ Document setup process
   - ⏳ Download sample videos

2. **Vision Module** (Person A):
   - ✅ YOLOv8 detection script
   - ⏳ Test with sample videos
   - ⏳ Add tracking functionality

3. **Logic Module** (Person B):
   - ✅ Pseudocode for anomaly detection
   - ⏳ Implement stalled vehicle detection
   - ⏳ Implement counting logic

4. **Integration**:
   - ⏳ Combine vision + logic
   - ⏳ Create Streamlit dashboard
   - ⏳ Deploy PoC

## Resources

- **YOLOv8 Documentation**: https://docs.ultralytics.com/
- **Supervision Library**: https://supervision.roboflow.com/
- **OpenCV Tutorials**: https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html
- **Streamlit Docs**: https://docs.streamlit.io/

## Contact

For issues or questions:
- Create an issue in the repository
- Contact team members via project communication channel
