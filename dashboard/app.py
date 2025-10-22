import streamlit as st
import cv2
import tempfile
import time
from pathlib import Path
import sys, os

# Add backend folder to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import streaming version
from src.vision.traffic_analyzer import process_video_stream

# --------------------------
# Streamlit App
# --------------------------

st.title("PCMC Traffic Monitoring Dashboard")
st.write("Upload a traffic video to see vehicle detection, counts, and stalled vehicle alerts.")

uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "avi"])

if uploaded_file:
    # Save uploaded video temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    input_path = tfile.name  # ✅ define input_path here

    st.info("Processing video with YOLO tracker... this may take a while, please wait.")

    # Placeholders for dashboard updates
    frame_placeholder = st.empty()
    car_count_placeholder = st.empty()
    stalled_placeholder = st.empty()

    # Live video processing
    for frame, stats in process_video_stream(input_path):
        # Display each frame
        frame_placeholder.image(frame, channels="BGR")

        # Update live stats
        car_count_placeholder.metric("Vehicles Counted", stats["total_vehicles_counted"])
        stalled_placeholder.metric("Stalled Vehicles", stats["currently_stalled"])

        # Slight delay for UI smoothness
        time.sleep(0.03)

    st.success("✅ Video processing completed!")
