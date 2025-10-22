import streamlit as st
import cv2
# import backend functions
from backend_module import detect_and_track  

st.title("PCMC Traffic Dashboard")

uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi"])

if uploaded_file:
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())

    cap = cv2.VideoCapture("temp_video.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results, counts, stalled = detect_and_track(frame)
        
        # Display frame
        st.image(frame, channels="BGR")
        st.metric("Total Cars", counts.get("car", 0))
        st.write("Stalled Vehicles:")
        for v in stalled:
            st.write(f"ID {v['track_id']} at {v['bbox']}")
    
    cap.release()
