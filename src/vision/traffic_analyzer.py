"""
YOLOv8 with Tracking and Anomaly Detection
Complete implementation with vehicle tracking, stalled detection, and counting.
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from typing import Dict, List, Set, Tuple


class AnomalyDetectionConfig:
    """Configuration parameters for anomaly detection."""
    
    # Stalled vehicle detection parameters
    STALLED_RADIUS_PIXELS = 25  # Maximum movement radius to be considered stalled
    STALLED_FRAME_THRESHOLD = 75  # Number of consecutive frames to confirm stalled (~2.5 seconds at 30fps)
    
    # Counting line parameters (will be set based on video dimensions)
    COUNTING_LINE_START = None  # (x1, y1)
    COUNTING_LINE_END = None    # (x2, y2)
    
    # Tracking parameters
    MAX_HISTORY_LENGTH = 100  # Keep last N positions for each vehicle


class VehicleAnomalyDetector:
    """Detects stalled vehicles and counts vehicles crossing a line."""
    
    def __init__(self, config: AnomalyDetectionConfig):
        self.config = config
        
        # Track history: {track_id: [(x, y, frame_num), ...]}
        self.track_history: Dict[int, List[Tuple[float, float, int]]] = {}
        
        # Vehicles that have been counted
        self.counted_ids: Set[int] = set()
        
        # Stalled vehicle candidates: {track_id: first_stalled_frame}
        self.stalled_candidates: Dict[int, int] = {}
        
        # Confirmed stalled vehicles
        self.stalled_vehicles: Set[int] = set()
        
        # Frame counter
        self.frame_counter: int = 0
        
        # Vehicle count
        self.vehicle_count: int = 0
    
    def update(self, detections: sv.Detections, frame_num: int):
        """
        Update tracker with new detections.
        
        Args:
            detections: Detections object with tracker_id
            frame_num: Current frame number
        """
        self.frame_counter = frame_num
        
        if detections.tracker_id is None:
            return
        
        # Process each detection
        for i in range(len(detections)):
            track_id = int(detections.tracker_id[i])
            bbox = detections.xyxy[i]
            
            # Calculate center point
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            
            # Initialize track history if new
            if track_id not in self.track_history:
                self.track_history[track_id] = []
            
            # Get previous position for line crossing check
            prev_pos = None
            if len(self.track_history[track_id]) > 0:
                prev_pos = self.track_history[track_id][-1][:2]
            
            # Add current position
            self.track_history[track_id].append((cx, cy, frame_num))
            
            # Keep only recent history
            if len(self.track_history[track_id]) > self.config.MAX_HISTORY_LENGTH:
                self.track_history[track_id].pop(0)
            
            # Check for line crossing
            if prev_pos is not None and self.config.COUNTING_LINE_START is not None:
                self.check_line_crossing(track_id, prev_pos, (cx, cy))
            
            # Check if stalled
            self.check_stalled(track_id)
    
    def check_line_crossing(self, track_id: int, prev_pos: Tuple[float, float], 
                           curr_pos: Tuple[float, float]) -> bool:
        """Check if vehicle crossed the counting line."""
        if track_id in self.counted_ids:
            return False
        
        if self.config.COUNTING_LINE_START is None:
            return False
        
        # Simplified line crossing check (horizontal line)
        x1, y1 = self.config.COUNTING_LINE_START
        x2, y2 = self.config.COUNTING_LINE_END
        
        prev_y = prev_pos[1]
        curr_y = curr_pos[1]
        
        # Check if crossed the horizontal line from either direction
        if (prev_y < y1 and curr_y >= y1) or (prev_y > y1 and curr_y <= y1):
            self.counted_ids.add(track_id)
            self.vehicle_count += 1
            return True
        
        return False
    
    def check_stalled(self, track_id: int) -> bool:
        """Check if vehicle is stalled."""
        if track_id not in self.track_history:
            return False
        
        positions = self.track_history[track_id]
        
        # Need at least STALLED_FRAME_THRESHOLD positions
        if len(positions) < self.config.STALLED_FRAME_THRESHOLD:
            return False
        
        # Get recent positions
        recent_positions = positions[-self.config.STALLED_FRAME_THRESHOLD:]
        
        # Calculate mean position
        mean_x = np.mean([pos[0] for pos in recent_positions])
        mean_y = np.mean([pos[1] for pos in recent_positions])
        
        # Calculate max distance from mean
        max_dist = max(
            np.sqrt((pos[0] - mean_x)**2 + (pos[1] - mean_y)**2)
            for pos in recent_positions
        )
        
        # Check if stalled
        if max_dist < self.config.STALLED_RADIUS_PIXELS:
            if track_id not in self.stalled_candidates:
                self.stalled_candidates[track_id] = self.frame_counter
            
            # Check if stalled long enough
            frames_stalled = self.frame_counter - self.stalled_candidates[track_id]
            if frames_stalled >= self.config.STALLED_FRAME_THRESHOLD:
                self.stalled_vehicles.add(track_id)
                return True
        else:
            # Vehicle is moving, remove from candidates
            self.stalled_candidates.pop(track_id, None)
            self.stalled_vehicles.discard(track_id)
        
        return track_id in self.stalled_vehicles
    
    def is_stalled(self, track_id: int) -> bool:
        """Check if a vehicle is currently stalled."""
        return track_id in self.stalled_vehicles
    
    def get_statistics(self) -> Dict:
        """Get current tracking statistics."""
        return {
            'total_vehicles_counted': self.vehicle_count,
            'currently_stalled': len(self.stalled_vehicles),
            'stalled_vehicle_ids': list(self.stalled_vehicles),
            'active_tracks': len(self.track_history),
            'frame_number': self.frame_counter
        }


def process_video_with_tracking(input_video_path: str, output_video_path: str, 
                                model_name: str = "yolov8n.pt"):
    """
    Process video with YOLOv8 detection, tracking, and anomaly detection.
    
    Args:
        input_video_path: Path to input video file
        output_video_path: Path to save output video
        model_name: YOLOv8 model name
    """
    
    # Load YOLOv8 model
    print(f"Loading YOLOv8 model: {model_name}")
    model = YOLO(model_name)
    
    # Open video
    video_info = sv.VideoInfo.from_video_path(input_video_path)
    print(f"Video Info: {video_info.width}x{video_info.height} @ {video_info.fps} FPS")
    print(f"Total frames: {video_info.total_frames}")
    
    # Initialize tracking with better parameters
    byte_tracker = sv.ByteTrack(
        track_activation_threshold=0.25,  # Lower threshold for better detection
        lost_track_buffer=30,              # Keep tracks longer before losing them
        minimum_matching_threshold=0.8,    # Higher matching for better consistency
        frame_rate=video_info.fps
    )
    
    # Initialize anomaly detection
    config = AnomalyDetectionConfig()
    
    # Set counting line (horizontal line at 60% height)
    line_y = int(video_info.height * 0.6)
    config.COUNTING_LINE_START = (0, line_y)
    config.COUNTING_LINE_END = (video_info.width, line_y)
    
    anomaly_detector = VehicleAnomalyDetector(config)
    
    # Initialize annotators
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)
    
    # Line annotator for counting line
    line_zone = sv.LineZone(
        start=sv.Point(config.COUNTING_LINE_START[0], config.COUNTING_LINE_START[1]),
        end=sv.Point(config.COUNTING_LINE_END[0], config.COUNTING_LINE_END[1])
    )
    
    # Define vehicle classes from COCO dataset
    vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
    
    # Process video
    frame_generator = sv.get_video_frames_generator(input_video_path)
    
    print("\nProcessing video with tracking and anomaly detection...")
    print(f"Counting line at y={line_y}")
    print(f"Stalled detection: {config.STALLED_RADIUS_PIXELS}px radius, {config.STALLED_FRAME_THRESHOLD} frames")
    print("-" * 60)
    
    with sv.VideoSink(output_video_path, video_info) as sink:
        for frame_idx, frame in enumerate(frame_generator):
            # Run YOLOv8 inference
            results = model(frame, verbose=False)[0]
            
            # Convert to supervision Detections
            detections = sv.Detections.from_ultralytics(results)
            
            # Filter only vehicle classes
            detections = detections[
                [class_id in vehicle_classes for class_id in detections.class_id]
            ]
            
            # Update tracking
            detections = byte_tracker.update_with_detections(detections)
            
            # Update anomaly detection
            anomaly_detector.update(detections, frame_idx)
            
            # Prepare visualization
            annotated_frame = frame.copy()
            
            # Draw counting line
            cv2.line(
                annotated_frame,
                config.COUNTING_LINE_START,
                config.COUNTING_LINE_END,
                (0, 255, 255),  # Yellow line
                2
            )
            
            # Draw bounding boxes
            if len(detections) > 0:
                # Create labels with track IDs
                labels = []
                colors = []
                
                for i in range(len(detections)):
                    track_id = int(detections.tracker_id[i])
                    class_id = detections.class_id[i]
                    confidence = detections.confidence[i]
                    
                    # Check if stalled
                    is_stalled = anomaly_detector.is_stalled(track_id)
                    
                    if is_stalled:
                        label = f"ID:{track_id} {model.names[class_id]} {confidence:.2f} [STALLED]"
                        colors.append((0, 0, 255))  # Red for stalled
                    else:
                        label = f"ID:{track_id} {model.names[class_id]} {confidence:.2f}"
                        colors.append((0, 255, 0))  # Green for normal
                    
                    labels.append(label)
                
                # Annotate with custom colors for stalled vehicles
                for i in range(len(detections)):
                    bbox = detections.xyxy[i]
                    cv2.rectangle(
                        annotated_frame,
                        (int(bbox[0]), int(bbox[1])),
                        (int(bbox[2]), int(bbox[3])),
                        colors[i],
                        2
                    )
                
                # Add labels
                annotated_frame = label_annotator.annotate(
                    scene=annotated_frame,
                    detections=detections,
                    labels=labels
                )
            
            # Get statistics
            stats = anomaly_detector.get_statistics()
            
            # Add statistics overlay
            y_offset = 30
            cv2.putText(
                annotated_frame,
                f"Frame: {frame_idx + 1}/{video_info.total_frames}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            
            y_offset += 30
            cv2.putText(
                annotated_frame,
                f"Vehicles Counted: {stats['total_vehicles_counted']}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )
            
            y_offset += 30
            cv2.putText(
                annotated_frame,
                f"Active Tracks: {stats['active_tracks']}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            
            y_offset += 30
            cv2.putText(
                annotated_frame,
                f"Stalled Vehicles: {stats['currently_stalled']}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255) if stats['currently_stalled'] > 0 else (255, 255, 255),
                2
            )
            
            # Write frame
            sink.write_frame(annotated_frame)
            
            # Progress update
            if (frame_idx + 1) % 30 == 0:
                print(f"Processed {frame_idx + 1}/{video_info.total_frames} frames... "
                      f"Counted: {stats['total_vehicles_counted']}, "
                      f"Stalled: {stats['currently_stalled']}")
    
    # Final statistics
    final_stats = anomaly_detector.get_statistics()
    print("-" * 60)
    print("\n✓ Processing complete!")
    print(f"\nFinal Statistics:")
    print(f"  Total vehicles counted: {final_stats['total_vehicles_counted']}")
    print(f"  Total tracks: {final_stats['active_tracks']}")
    print(f"  Currently stalled: {final_stats['currently_stalled']}")
    if final_stats['stalled_vehicle_ids']:
        print(f"  Stalled vehicle IDs: {final_stats['stalled_vehicle_ids']}")
    print(f"\nOutput saved to: {output_video_path}")


def main():
    """Main function to parse arguments and run video processing."""
    parser = argparse.ArgumentParser(
        description="YOLOv8 with Tracking and Anomaly Detection"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to input video file"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Path to output video file"
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="yolov8n.pt",
        help="YOLOv8 model name (default: yolov8n.pt)"
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not Path(args.input).exists():
        print(f"Error: Input video file not found: {args.input}")
        return
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Process video
    process_video_with_tracking(args.input, args.output, args.model)


if __name__ == "__main__":
    main()
