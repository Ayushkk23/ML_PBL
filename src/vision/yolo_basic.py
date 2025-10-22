from ultralytics import YOLO
import cv2
import argparse
import os

print("🔍 Debug Info:")
print("Current working directory:", os.getcwd())

def run_yolo(video_path, output_path="output.mp4"):
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(video_path)  # use the passed video_path
    print("Trying to open video:", video_path)
    if not cap.isOpened():
        print("⚠️ Failed to open video. File exists?", os.path.exists(video_path))
        return
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)
        cv2.imshow("YOLOv8 Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Done. Output saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, default="data/videos/processed/output.mp4", help="Path to save output video")
    args = parser.parse_args()
    run_yolo(args.input, args.output)
