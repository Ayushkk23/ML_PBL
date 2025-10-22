import streamlit as st
import cv2
import tempfile
import os
from pathlib import Path
import time
import sys
import os

# Add the parent directory of dashboard to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.vision.traffic_analyzer import process_video_with_tracking


# --------------------------
# Import the backend TrafficAnalyzer
# --------------------------
from src.vision.traffic_analyzer import process_video_with_tracking

# --------------------------
# Initialize detector
# --------------------------

st.title("PCMC Traffic Monitoring Dashboard")
st.write("Upload a traffic video to see vehicle detection, counts, and stalled vehicle alerts.")

uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "avi"])

if uploaded_file:
    # Save uploaded video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    input_path = tfile.name

    # Prepare output video path
    output_path = str(Path(input_path).with_name("processed_output.mp4"))

    st.info("Processing video with backend YOLO tracker... this may take a while.")

    # Run detection using your teammates' backend
    # If your backend only processes full video, we can call it here
    process_video_with_tracking(input_path, output_path)
  # Replace with actual method name

    st.success("Processing complete!")

    # Read processed video and display frame-by-frame
    cap = cv2.VideoCapture(output_path)

    # Placeholders
    car_count_placeholder = st.empty()
    truck_count_placeholder = st.empty()
    stalled_placeholder = st.empty()
    frame_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # If your backend outputs metadata per frame, you can read it here
        # For now, we assume processed video already has bounding boxes drawn

        frame_placeholder.image(frame, channels="BGR")

        # Optional: If your backend outputs counts/stalled info as JSON or CSV,
        # you can read and display them like this:
        # car_count_placeholder.metric("Total Cars", counts.get("car", 0))
        # truck_count_placeholder.metric("Total Trucks", counts.get("truck", 0))
        # stalled_placeholder.text(stalled_text)

        time.sleep(0.03)

    cap.release()
    st.info(f"Processed video saved at: {output_path}")
