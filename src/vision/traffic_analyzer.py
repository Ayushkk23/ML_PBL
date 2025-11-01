"""
YOLOv8 with Tracking and Anomaly Detection
Complete implementation with vehicle tracking, stalled detection, and counting.
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO  # type: ignore # Pylance has incorrect type information for this import
from typing import Dict, List, Set, Tuple


class AnomalyDetectionConfig:
    """Configuration parameters for anomaly detection with adaptive settings."""
    
    def __init__(self, video_info: sv.VideoInfo):
        """Initialize configuration with video-specific parameters."""
        self.video_info = video_info
        
        # Adaptive stalled vehicle parameters based on frame rate
        fps = max(1, video_info.fps)  # Prevent division by zero
        self.STALLED_RADIUS_PIXELS = int(video_info.width * 0.03)  # 3% of frame width
        self.STALLED_FRAME_THRESHOLD = int(2.0 * fps)  # 2.0 seconds worth of frames
        
        # Counting line parameters (will be set dynamically)
        self.COUNTING_LINE_START: Tuple[int, int] | None = None
        self.COUNTING_LINE_END: Tuple[int, int] | None = None
        
        # Adaptive tracking parameters
        self.MAX_HISTORY_LENGTH = int(3 * fps)  # 3 seconds of history
        
        # Adaptive movement thresholds based on video dimensions
        min_dimension = min(video_info.width, video_info.height)
        self.MIN_MOVEMENT_THRESHOLD = max(2, int(min_dimension * 0.002))  # 0.2% of min dimension
        self.MAX_DISTANCE_FROM_LINE = int(video_info.height * 0.15)  # Reduced to 15% for more precise detection
        
        # Additional parameters for lower line position
        self.MIN_VEHICLE_SIZE = int(video_info.height * 0.05)  # Minimum size for valid vehicle detection
        
        # Adaptive detection thresholds based on resolution and frame rate
        base_confidence = 0.25  # Base confidence threshold
        
        # Adjust for resolution
        if video_info.width * video_info.height > 1920 * 1080:  # High resolution
            base_confidence *= 1.2  # Increase confidence for high-res
        else:
            base_confidence *= 0.8  # Decrease confidence for low-res
            
        # Adjust for frame rate
        if video_info.fps < 25:  # Low frame rate
            base_confidence *= 0.9  # More lenient for low FPS
            
        self.CONFIDENCE_THRESHOLD = base_confidence
        self.IOU_THRESHOLD = 0.3  # More lenient IOU threshold
        
        # Initialize counting line
        self._setup_counting_line()
    
    def _setup_counting_line(self):
        """Set up counting line based on video dimensions and standard vehicle height."""
        width, height = self.video_info.width, self.video_info.height
        
        # Place line at approximately 5ft height from bottom
        # Assuming standard vehicle height is about 5ft and appears in lower third of frame
        # We'll place the line at around 75-80% of frame height from top (or 20-25% from bottom)
        line_y = int(height * 0.8)  # This places line at lower part of frame
        
        # Calculate padding based on video width
        padding = int(width * 0.05)  # 5% padding on each side
        
        self.COUNTING_LINE_START = (padding, line_y)
        self.COUNTING_LINE_END = (width - padding, line_y)


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
        """Check if vehicle crossed the counting line with improved validation."""
        if track_id in self.counted_ids:
            return False
        
        if self.config.COUNTING_LINE_START is None or self.config.COUNTING_LINE_END is None:
            return False
        
        x1, y1 = self.config.COUNTING_LINE_START
        x2, y2 = self.config.COUNTING_LINE_END
        
        # Helper functions
        def distance(p1, p2):
            """Calculate Euclidean distance between two points."""
            return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        
        def point_to_line_distance(point, line_start, line_end):
            """Calculate the distance from a point to a line segment."""
            px, py = point
            x1, y1 = line_start
            x2, y2 = line_end
            
            # Calculate the squared length of the line segment
            line_length_sq = (x2 - x1)**2 + (y2 - y1)**2
            
            if line_length_sq == 0:
                return distance(point, line_start)
            
            # Calculate projection
            t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / line_length_sq))
            
            # Calculate closest point on line
            proj_x = x1 + t * (x2 - x1)
            proj_y = y1 + t * (y2 - y1)
            
            return distance((px, py), (proj_x, proj_y))
        
        def has_crossed_line(p1, p2, line_start, line_end):
            """Determine if a movement from p1 to p2 crosses the line."""
            def sign(p1, p2, p3):
                return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
            
            # Check if line segments intersect
            d1 = sign(p1, line_start, line_end)
            d2 = sign(p2, line_start, line_end)
            d3 = sign(p1, p2, line_start)
            d4 = sign(p1, p2, line_end)
            
            return ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
                   ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0))
        
        # Calculate current movement and distances
        movement = distance(prev_pos, curr_pos)
        dist_to_line = point_to_line_distance(curr_pos, self.config.COUNTING_LINE_START, self.config.COUNTING_LINE_END)
        
        # Check if movement is significant enough
        if movement < self.config.MIN_MOVEMENT_THRESHOLD:
            return False
            
        # Check if we're close enough to the line
        if dist_to_line > self.config.MAX_DISTANCE_FROM_LINE:
            return False
        
        # Check if the path crosses the line
        if has_crossed_line(prev_pos, curr_pos, self.config.COUNTING_LINE_START, self.config.COUNTING_LINE_END):
            # Check if the crossing is valid
            if has_crossed_line(prev_pos, curr_pos, self.config.COUNTING_LINE_START, self.config.COUNTING_LINE_END):
                # Calculate movement direction vector
                movement_vector = np.array([curr_pos[0] - prev_pos[0], curr_pos[1] - prev_pos[1]])
                movement_magnitude = np.linalg.norm(movement_vector)
                
                if movement_magnitude > 0:
                    # Normalize movement vector
                    movement_vector = movement_vector / movement_magnitude
                    
                    # Calculate line direction vector
                    line_vector = np.array([x2 - x1, y2 - y1])
                    line_magnitude = np.linalg.norm(line_vector)
                    
                    if line_magnitude > 0:
                        # Normalize line vector
                        line_vector = line_vector / line_magnitude
                        
                        # Calculate angle between vectors
                        dot_product = np.dot(movement_vector, line_vector)
                        angle = np.arccos(min(abs(dot_product), 1.0))
                        
                        # Check if angle is within acceptable range (30-150 degrees)
                        if 0.523 <= angle <= 2.618:  # pi/6 to 5pi/6
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
    
    # Initialize tracking with adaptive parameters based on video properties
    track_threshold = 0.3 if video_info.fps >= 25 else 0.25  # Lower threshold for low FPS
    lost_buffer = max(10, int(video_info.fps * 0.5))  # Adaptive buffer based on FPS
    matching_threshold = 0.25 if video_info.fps >= 25 else 0.2  # Adjust for low FPS
    
    byte_tracker = sv.ByteTrack(
        track_activation_threshold=track_threshold,
        lost_track_buffer=lost_buffer,
        minimum_matching_threshold=matching_threshold,
        frame_rate=video_info.fps
    )
    
    # Initialize anomaly detection with video-specific configuration
    config = AnomalyDetectionConfig(video_info)
    
    anomaly_detector = VehicleAnomalyDetector(config)
    
    # Initialize annotators
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)
    
    # Line annotator for counting line
    if config.COUNTING_LINE_START is not None and config.COUNTING_LINE_END is not None:
        line_zone = sv.LineZone(
            start=sv.Point(config.COUNTING_LINE_START[0], config.COUNTING_LINE_START[1]),
            end=sv.Point(config.COUNTING_LINE_END[0], config.COUNTING_LINE_END[1])
        )
    
    # Define vehicle classes from COCO dataset with adaptive confidence
    vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
    
    # Process video
    frame_generator = sv.get_video_frames_generator(input_video_path)
    
    print("\nProcessing video with tracking and anomaly detection...")
    print(f"Video properties: {video_info.width}x{video_info.height} @ {video_info.fps} FPS")
    print(f"Counting line at y={config.COUNTING_LINE_START[1] if config.COUNTING_LINE_START else 'Not set'}")
    print(f"Stalled detection: {config.STALLED_RADIUS_PIXELS}px radius, {config.STALLED_FRAME_THRESHOLD} frames")
    print("-" * 60)
    
    with sv.VideoSink(output_video_path, video_info) as sink:
        for frame_idx, frame in enumerate(frame_generator):
            # Run YOLOv8 inference with adaptive parameters
            results = model(
                frame,
                conf=config.CONFIDENCE_THRESHOLD,
                iou=config.IOU_THRESHOLD,
                verbose=False
            )[0]
            
            # Convert to supervision Detections
            detections = sv.Detections.from_ultralytics(results)
            
            # Adaptive filtering based on video properties
            if hasattr(detections, 'class_id') and detections.class_id is not None and detections.confidence is not None:
                # Create mask for both class and confidence
                class_mask = np.array([class_id in vehicle_classes for class_id in detections.class_id])
                
                # Enhanced detection filtering
                if detections.xyxy is not None:
                    # Calculate detection heights
                    detection_heights = detections.xyxy[:, 3] - detections.xyxy[:, 1]
                    
                    # Calculate relative sizes of detections
                    sizes = (detections.xyxy[:, 2] - detections.xyxy[:, 0]) * \
                           detection_heights / (video_info.width * video_info.height)
                    
                    # Create size mask for minimum vehicle size
                    size_mask = detection_heights >= config.MIN_VEHICLE_SIZE
                    
                    # Adjust confidence thresholds based on size and position
                    base_thresholds = np.where(
                        sizes > 0.01,  # Large objects (>1% of frame)
                        config.CONFIDENCE_THRESHOLD * 0.8,  # Lower threshold for large objects
                        config.CONFIDENCE_THRESHOLD  # Normal threshold
                    )
                    
                    # Additional position-based adjustment
                    y_centers = (detections.xyxy[:, 1] + detections.xyxy[:, 3]) / 2
                    near_line_mask = np.abs(y_centers - config.COUNTING_LINE_START[1]) < config.MAX_DISTANCE_FROM_LINE
                    
                    # Combine all masks
                    conf_mask = (detections.confidence >= base_thresholds) & size_mask & near_line_mask
                else:
                    conf_mask = detections.confidence >= config.CONFIDENCE_THRESHOLD
                
                mask = class_mask & conf_mask
                detections = detections[mask]
            
            # Update tracking
            if isinstance(detections, sv.Detections):
                detections = byte_tracker.update_with_detections(detections)
                
                # Update anomaly detection
                anomaly_detector.update(detections, frame_idx)
            
            # Prepare visualization
            annotated_frame = frame.copy()
            
            # Draw counting line
            if config.COUNTING_LINE_START is not None and config.COUNTING_LINE_END is not None:
                cv2.line(
                    annotated_frame,
                    config.COUNTING_LINE_START,
                    config.COUNTING_LINE_END,
                    (0, 255, 255),  # Yellow line
                    2
                )
            
            # Draw bounding boxes
            if isinstance(detections, sv.Detections) and len(detections) > 0:
                # Create labels with track IDs
                labels = []
                colors = []
                
                if (detections.tracker_id is not None and 
                    detections.class_id is not None and 
                    detections.confidence is not None):
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
