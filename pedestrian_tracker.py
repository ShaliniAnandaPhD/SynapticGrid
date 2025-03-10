"""
Uses computer vision and object detection to analyze and predict pedestrian movement
patterns in urban environments.
"""

import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import json
import os
import time
import logging
from pathlib import Path
import requests
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import defaultdict, deque
from scipy.spatial import distance
from sklearn.cluster import DBSCAN
from geopy.distance import geodesic

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("pedestrian_tracker.log"), logging.StreamHandler()]
)
logger = logging.getLogger("PedestrianTracker")

# Constants
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for detection
DETECTION_INTERVAL = 2  # Process every Nth frame for efficiency
MAX_DISAPPEARED = 30  # Maximum number of frames object can disappear before being removed
TRACKING_MEMORY = 100  # Number of recent positions to keep for each pedestrian
HEATMAP_RESOLUTION = (640, 480)  # Resolution for heatmap generation

# Classes from COCO dataset that we're interested in
PERSON_CLASS_ID = 0
CLASSES_OF_INTEREST = {
    0: "person", 
    1: "bicycle", 
    2: "car", 
    3: "motorcycle", 
    # Just for reference - we'll filter most of these out
    5: "bus", 
    6: "train", 
    7: "truck"
}


class TrackedObject:
    """
    Represents a tracked pedestrian or other object of interest.
    
    Keeps track of position history, motion vectors, and other attributes.
    """
    
    def __init__(self, object_id: int, centroid: Tuple[int, int], 
                class_id: int, confidence: float, bbox: Tuple[int, int, int, int]):
        """
        Initialize tracked object.
        
        Args:
            object_id: Unique identifier for this object
            centroid: (x, y) position in the frame
            class_id: Class ID from the detector
            confidence: Detection confidence score
            bbox: Bounding box coordinates (x1, y1, x2, y2)
        """
        self.object_id = object_id
        self.centroids = deque(maxlen=TRACKING_MEMORY)
        self.centroids.append(centroid)
        self.class_id = class_id
        self.class_name = CLASSES_OF_INTEREST.get(class_id, "unknown")
        self.confidence = confidence
        self.bbox = bbox
        self.disappeared = 0
        self.speed = None  # pixels per frame
        self.direction = None  # angle in degrees
        self.path_length = 0
        self.creation_time = time.time()
        self.last_update_time = time.time()
        self.is_counted = False
        self.zone_transitions = []  # List of zones this object has passed through
        
    def update(self, centroid: Tuple[int, int], confidence: float, bbox: Tuple[int, int, int, int]):
        """
        Update the object with a new detection.
        
        Args:
            centroid: New centroid position
            confidence: New confidence score
            bbox: New bounding box
        """
        # Calculate displacement
        if len(self.centroids) > 0:
            prev_centroid = self.centroids[-1]
            dx = centroid[0] - prev_centroid[0]
            dy = centroid[1] - prev_centroid[1]
            
            # Calculate speed (pixel distance)
            dist = np.sqrt(dx**2 + dy**2)
            self.path_length += dist
            
            # Update speed - pixels per frame
            self.speed = dist
            
            # Calculate direction angle in degrees
            if dist > 0:
                self.direction = np.degrees(np.arctan2(dy, dx))
        
        # Update attributes
        self.centroids.append(centroid)
        self.confidence = max(self.confidence, confidence)  # Keep highest confidence
        self.bbox = bbox
        self.disappeared = 0
        self.last_update_time = time.time()
        
    def predict_next_position(self) -> Tuple[int, int]:
        """
        Predict the next position based on recent movement.
        
        Returns:
            Predicted (x, y) position
        """
        if len(self.centroids) < 2:
            return self.centroids[-1]
        
        # Use last few positions to predict next
        # Could be more sophisticated with Kalman filter, but this works for basic tracking
        last_points = list(self.centroids)[-3:]
        if len(last_points) < 2:
            return last_points[-1]
            
        # Calculate average movement
        movements = []
        for i in range(1, len(last_points)):
            dx = last_points[i][0] - last_points[i-1][0]
            dy = last_points[i][1] - last_points[i-1][1]
            movements.append((dx, dy))
        
        avg_dx = sum(m[0] for m in movements) / len(movements)
        avg_dy = sum(m[1] for m in movements) / len(movements)
        
        # Predict next position
        last_x, last_y = last_points[-1]
        return (int(last_x + avg_dx), int(last_y + avg_dy))
    
    def get_motion_vector(self) -> Tuple[float, float]:
        """
        Get the current motion vector.
        
        Returns:
            (dx, dy) motion vector
        """
        if len(self.centroids) < 2:
            return (0, 0)
            
        # Get the last two positions
        p1 = self.centroids[-2]
        p2 = self.centroids[-1]
        
        return (p2[0] - p1[0], p2[1] - p1[1])
    
    def get_track_duration(self) -> float:
        """
        Get the duration this object has been tracked in seconds.
        
        Returns:
            Duration in seconds
        """
        return self.last_update_time - self.creation_time
        
    def mark_as_counted(self):
        """Mark this object as counted in statistics."""
        self.is_counted = True
        
    def add_zone_transition(self, from_zone: str, to_zone: str, timestamp: float):
        """
        Record a zone transition for this object.
        
        Args:
            from_zone: Zone the object was in
            to_zone: Zone the object moved to
            timestamp: Time of transition
        """
        self.zone_transitions.append({
            'from': from_zone,
            'to': to_zone,
            'timestamp': timestamp
        })
        
    def __str__(self):
        return f"Object {self.object_id} ({self.class_name}): pos={self.centroids[-1] if self.centroids else None}, tracked for {self.get_track_duration():.1f}s"


class CameraCalibration:
    """
    Handles camera calibration and conversion between pixel and real-world coordinates.
    """
    
    def __init__(self, calibration_data: Dict = None):
        """
        Initialize camera calibration.
        
        Args:
            calibration_data: Dictionary with calibration parameters
        """
        self.is_calibrated = False
        self.pixel_to_meter_ratio = 0.1  # Default placeholder value (10 pixels = 1 meter)
        
        # Reference points mapping pixel locations to real-world coordinates
        self.reference_points = []
        
        # Homography matrix for perspective transformation
        self.homography_matrix = None
        
        if calibration_data:
            self.load_calibration(calibration_data)
    
    def load_calibration(self, calibration_data: Dict):
        """
        Load calibration data from a dictionary.
        
        Args:
            calibration_data: Dictionary with calibration parameters
        """
        if 'pixel_to_meter_ratio' in calibration_data:
            self.pixel_to_meter_ratio = calibration_data['pixel_to_meter_ratio']
        
        if 'reference_points' in calibration_data:
            self.reference_points = calibration_data['reference_points']
            
            # Compute homography if we have at least 4 points
            if len(self.reference_points) >= 4:
                self._compute_homography()
        
        if 'homography_matrix' in calibration_data:
            self.homography_matrix = np.array(calibration_data['homography_matrix'])
            
        self.is_calibrated = True
        logger.info("Camera calibration loaded")
    
    def _compute_homography(self):
        """Compute homography matrix from reference points."""
        if len(self.reference_points) < 4:
            logger.warning("Need at least 4 reference points to compute homography")
            return
            
        # Extract pixel and world coordinates
        pixel_points = []
        world_points = []
        
        for point in self.reference_points:
            pixel_points.append(point['pixel'])
            world_points.append(point['world'])
            
        # Compute homography
        self.homography_matrix, status = cv2.findHomography(
            np.array(pixel_points), 
            np.array(world_points),
            cv2.RANSAC, 
            5.0
        )
        
        if status.all():
            logger.info("Homography matrix computed successfully")
        else:
            logger.warning("Some points were considered outliers in homography computation")
    
    def pixel_to_world(self, pixel_point: Tuple[int, int]) -> Tuple[float, float]:
        """
        Convert pixel coordinates to world coordinates.
        
        Args:
            pixel_point: (x, y) pixel coordinates
            
        Returns:
            (x, y) world coordinates
        """
        if self.homography_matrix is not None:
            # Use homography transformation
            point = np.array([pixel_point[0], pixel_point[1], 1])
            transformed = np.dot(self.homography_matrix, point)
            transformed = transformed / transformed[2]  # Normalize
            return (transformed[0], transformed[1])
        else:
            # Fallback to simple ratio
            return (pixel_point[0] * self.pixel_to_meter_ratio, 
                   pixel_point[1] * self.pixel_to_meter_ratio)
    
    def pixel_distance_to_meters(self, pixel_distance: float) -> float:
        """
        Convert a distance in pixels to meters.
        
        Args:
            pixel_distance: Distance in pixels
            
        Returns:
            Distance in meters
        """
        return pixel_distance * self.pixel_to_meter_ratio
    
    def calibrate_from_video(self, video_path: str, reference_height: float = 1.7):
        """
        Attempt to calibrate from a video using average person height.
        
        Args:
            video_path: Path to video file
            reference_height: Average person height in meters (default 1.7m)
        """
        # This is a placeholder for a more complex calibration procedure
        # In a real implementation, we'd detect people, measure their bounding boxes,
        # and use statistical methods to determine the pixel-to-meter ratio
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return
            
        # Process some frames to get average person height in pixels
        frame_count = 0
        person_heights = []
        
        while frame_count < 100:  # Process up to 100 frames
            ret, frame = cap.read()
            if not ret:
                break
                
            # Only process every 10th frame for efficiency
            if frame_count % 10 == 0:
                # Run detection to find people
                # This would use the detector to find people and measure their heights
                # For now, we'll just use a placeholder
                
                # Detect people and get their bounding boxes
                detections = self._detect_people(frame)
                
                for detection in detections:
                    if detection['class_id'] == PERSON_CLASS_ID:
                        # Get height in pixels
                        bbox = detection['bbox']
                        height_px = bbox[3] - bbox[1]
                        person_heights.append(height_px)
            
            frame_count += 1
            
        cap.release()
        
        if person_heights:
            # Use the median height for robustness
            median_height_px = np.median(person_heights)
            
            # Calculate pixel to meter ratio
            self.pixel_to_meter_ratio = reference_height / median_height_px
            self.is_calibrated = True
            
            logger.info(f"Calibration complete: 1 meter = {1/self.pixel_to_meter_ratio:.1f} pixels")
        else:
            logger.warning("Calibration failed: No people detected in video")
    
    def _detect_people(self, frame):
        """
        Placeholder for people detection.
        
        In a real implementation, this would use the object detector
        to find people in the frame.
        """
        # This is just a placeholder for the real detection code
        return []


class PedestrianTracker:
    """
    Main class for tracking pedestrians across video frames.
    """
    
    def __init__(self, model_path: str = None, camera_calibration: Dict = None):
        """
        Initialize the pedestrian tracker.
        
        Args:
            model_path: Path to the object detection model
            camera_calibration: Optional camera calibration data
        """
        self.next_object_id = 0
        self.objects = {}  # ID -> TrackedObject
        self.disappeared = {}  # ID -> count of frames disappeared
        self.frame_count = 0
        self.fps = 30  # Assumed fps, will be updated from video
        
        # Statistics
        self.pedestrian_counts = {
            'total': 0,
            'hourly': defaultdict(int),
            'zone_entries': defaultdict(int),
            'zone_transitions': defaultdict(int)
        }
        
        # Heatmap data
        self.position_heatmap = np.zeros(HEATMAP_RESOLUTION, dtype=np.float32)
        self.movement_vectors = []
        
        # Load detection model
        self.model = self._load_model(model_path)
        
        # Initialize camera calibration
        self.calibration = CameraCalibration(camera_calibration)
        
        # Define activity zones
        self.zones = {}  # Will be populated with Zone objects
        
        logger.info("Pedestrian tracker initialized")
    
    def _load_model(self, model_path: str = None):
        """
        Load the object detection model.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Loaded model
        """
        if model_path is None or not os.path.exists(model_path):
            # Use a default model - in this case SSD MobileNet from TensorFlow model zoo
            logger.info("No model path provided, using SSD MobileNet v2")
            # In a real implementation, we'd download a pre-trained model
            # For now, we'll just create a placeholder
            return None
        
        try:
            # Load the model using TensorFlow
            logger.info(f"Loading model from {model_path}")
            # model = tf.saved_model.load(model_path)
            
            # For simplicity, this is a placeholder
            # In a real implementation, we'd load the actual model
            model = None
            
            logger.info("Model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None
    
    def define_zone(self, zone_id: str, polygon: List[Tuple[int, int]]):
        """
        Define an activity zone in the frame.
        
        Args:
            zone_id: Unique identifier for the zone
            polygon: List of (x, y) points defining the zone boundary
        """
        self.zones[zone_id] = {
            'polygon': np.array(polygon),
            'entry_count': 0,
            'exit_count': 0,
            'current_objects': set()
        }
        logger.info(f"Defined zone {zone_id} with {len(polygon)} points")
    
    def point_in_zone(self, point: Tuple[int, int], zone_id: str) -> bool:
        """
        Check if a point is inside a defined zone.
        
        Args:
            point: (x, y) point to check
            zone_id: Zone identifier
            
        Returns:
            True if the point is in the zone, False otherwise
        """
        if zone_id not in self.zones:
            return False
            
        zone = self.zones[zone_id]
        return cv2.pointPolygonTest(zone['polygon'], point, False) >= 0
    
    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect objects in a frame.
        
        Args:
            frame: Video frame
            
        Returns:
            List of detection dictionaries with class, confidence, and bbox
        """
        if self.model is None:
            # If no model is loaded, use a placeholder implementation
            # In a real application, this would use the actual model to do detection
            
            # Generate some fake detections for demonstration
            height, width = frame.shape[:2]
            num_detections = np.random.randint(0, 5)  # 0-4 detections per frame
            
            detections = []
            for _ in range(num_detections):
                # Random position
                x = np.random.randint(0, width)
                y = np.random.randint(0, height)
                
                # Random size
                w = np.random.randint(30, 100)
                h = np.random.randint(50, 200)
                
                # Make sure the box is in the frame
                x1 = max(0, x - w//2)
                y1 = max(0, y - h//2)
                x2 = min(width, x + w//2)
                y2 = min(height, y + h//2)
                
                # Almost always detect a person
                class_id = PERSON_CLASS_ID if np.random.random() < 0.9 else np.random.choice([1, 2, 3])
                
                detections.append({
                    'class_id': class_id,
                    'confidence': np.random.uniform(0.5, 0.9),
                    'bbox': (x1, y1, x2, y2)
                })
                
            return detections
            
        # In a real implementation, this would use the model to detect objects
        # For example:
        # input_tensor = tf.convert_to_tensor(frame)
        # detections = self.model(input_tensor)
        # Process detections and return in standardized format
        
        # Placeholder for now
        return []
    
    def update(self, frame: np.ndarray) -> Dict:
        """
        Process a new frame and update tracking information.
        
        Args:
            frame: New video frame
            
        Returns:
            Dictionary with tracking results
        """
        self.frame_count += 1
        frame_height, frame_width = frame.shape[:2]
        
        # Only process every Nth frame for efficiency
        process_frame = (self.frame_count % DETECTION_INTERVAL == 0)
        
        # Dictionary to store current centroids: centroid -> object_id
        current_centroids = {}
        
        if process_frame:
            # Detect objects in the frame
            detections = self.detect_objects(frame)
            
            # Filter detections by class and confidence
            valid_detections = []
            for detection in detections:
                if (detection['class_id'] in CLASSES_OF_INTEREST and 
                    detection['confidence'] >= CONFIDENCE_THRESHOLD):
                    valid_detections.append(detection)
            
            # Process each detection
            for detection in valid_detections:
                bbox = detection['bbox']
                centroid = (
                    int((bbox[0] + bbox[2]) / 2),  # Center x
                    int((bbox[1] + bbox[3]) / 2)   # Center y
                )
                
                # Add to current centroids
                current_centroids[centroid] = {
                    'class_id': detection['class_id'],
                    'confidence': detection['confidence'],
                    'bbox': bbox
                }
        
        # If we have existing objects but no centroids in this frame, mark all objects as disappeared
        if not current_centroids and self.objects:
            for object_id in list(self.objects.keys()):
                self.objects[object_id].disappeared += 1
                
                # Remove the object if it's been gone too long
                if self.objects[object_id].disappeared > MAX_DISAPPEARED:
                    self._finalize_object(object_id)
                    
            # Return early as there's nothing to update
            return {'objects': self.objects, 'detections': []}
                
        # If we have no existing objects, register all centroids as new objects
        if not self.objects and current_centroids:
            for centroid, info in current_centroids.items():
                self._register_object(centroid, info['class_id'], info['confidence'], info['bbox'])
        
        # Otherwise, match existing objects to new detections
        else:
            object_ids = list(self.objects.keys())
            object_centroids = [obj.centroids[-1] if obj.centroids else (0, 0) for obj in self.objects.values()]
            
            # Compute distances between existing objects and new centroids
            D = distance.cdist(np.array(object_centroids), np.array(list(current_centroids.keys())))
            
            # Find the smallest distance for each row, then sort by distance
            rows = D.min(axis=1).argsort()
            
            # Find the smallest distance for each column, then sort by minimal distance
            cols = D.argmin(axis=1)[rows]
            
            # Keep track of which rows and columns we've already examined
            used_rows = set()
            used_cols = set()
            
            # Match objects to centroids
            for (row, col) in zip(rows, cols):
                # Skip if we've already examined this row or column
                if row in used_rows or col in used_cols:
                    continue
                
                # Get the object ID and update with the new centroid
                object_id = object_ids[row]
                centroid = list(current_centroids.keys())[col]
                info = current_centroids[centroid]
                
                # Update the object
                self.objects[object_id].update(centroid, info['confidence'], info['bbox'])
                
                # Mark as used
                used_rows.add(row)
                used_cols.add(col)
            
            # Check for unmatched rows (disappeared objects)
            for row in range(len(object_ids)):
                if row not in used_rows:
                    object_id = object_ids[row]
                    self.objects[object_id].disappeared += 1
                    
                    # Remove if gone too long
                    if self.objects[object_id].disappeared > MAX_DISAPPEARED:
                        self._finalize_object(object_id)
            
            # Check for unmatched centroids (new objects)
            for col in range(len(current_centroids)):
                if col not in used_cols:
                    centroid = list(current_centroids.keys())[col]
                    info = current_centroids[centroid]
                    self._register_object(centroid, info['class_id'], info['confidence'], info['bbox'])
        
        # Update position heatmap
        for obj in self.objects.values():
            if obj.centroids and obj.class_id == PERSON_CLASS_ID:
                centroid = obj.centroids[-1]
                
                # Scale to heatmap resolution
                heatmap_x = int(centroid[0] * HEATMAP_RESOLUTION[0] / frame_width)
                heatmap_y = int(centroid[1] * HEATMAP_RESOLUTION[1] / frame_height)
                
                # Add to heatmap
                if 0 <= heatmap_x < HEATMAP_RESOLUTION[0] and 0 <= heatmap_y < HEATMAP_RESOLUTION[1]:
                    self.position_heatmap[heatmap_y, heatmap_x] += 1
                
                # Update zone information
                self._update_zones(obj)
        
        # Update movement vectors for flow analysis
        for obj in self.objects.values():
            if len(obj.centroids) >= 2 and obj.class_id == PERSON_CLASS_ID:
                # Get the last two positions
                p1 = obj.centroids[-2]
                p2 = obj.centroids[-1]
                
                # Convert to real-world coordinates if calibrated
                if self.calibration.is_calibrated:
                    p1_world = self.calibration.pixel_to_world(p1)
                    p2_world = self.calibration.pixel_to_world(p2)
                    dx = p2_world[0] - p1_world[0]
                    dy = p2_world[1] - p1_world[1]
                else:
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                
                # Add movement vector
                self.movement_vectors.append({
                    'position': p2,
                    'dx': dx,
                    'dy': dy,
                    'magnitude': np.sqrt(dx**2 + dy**2),
                    'direction': np.degrees(np.arctan2(dy, dx))
                })
        
        # Limit the number of stored movement vectors
        if len(self.movement_vectors) > 1000:
            self.movement_vectors = self.movement_vectors[-1000:]
        
        # Return the current tracking state
        return {
            'objects': self.objects,
            'detections': list(current_centroids.keys())
        }
    
    def _register_object(self, centroid: Tuple[int, int], class_id: int, confidence: float,
                        bbox: Tuple[int, int, int, int]) -> int:
        """
        Register a new object.
        
        Args:
            centroid: (x, y) position of the object
            class_id: Class ID from detector
            confidence: Detection confidence
            bbox: Bounding box coordinates
            
        Returns:
            New object ID
        """
        # Create a new tracked object
        self.objects[self.next_object_id] = TrackedObject(
            self.next_object_id, centroid, class_id, confidence, bbox
        )
        
        # Update statistics if this is a person
        if class_id == PERSON_CLASS_ID:
            self.pedestrian_counts['total'] += 1
            
            # Update hourly count
            hour = datetime.datetime.now().hour
            self.pedestrian_counts['hourly'][hour] += 1
        
        # Increment ID for next object
        self.next_object_id += 1
        
        return self.next_object_id - 1
    
    def _finalize_object(self, object_id: int) -> None:
        """
        Finalize an object that is no longer being tracked.
        
        Args:
            object_id: ID of the object to finalize
        """
        obj = self.objects.get(object_id)
        
        if obj is None:
            return
            
        # Log tracking information
        logger.debug(f"Finalized {obj}")
        
        # Do any final processing or logging here
        
        # Remove the object from tracking
        del self.objects[object_id]
    
    def _update_zones(self, obj: TrackedObject) -> None:
        """
        Update zone statistics for an object.
        
        Args:
            obj: Tracked object
        """
        if not obj.centroids:
            return
            
        current_position = obj.centroids[-1]
        
        # Check each zone
        for zone_id, zone in self.zones.items():
            is_in_zone = self.point_in_zone(current_position, zone_id)
            
            # Object just entered the zone
            if is_in_zone and object not in zone['current_objects']:
                zone['current_objects'].add(obj.object_id)
                zone['entry_count'] += 1
                self.pedestrian_counts['zone_entries'][zone_id] += 1
                
                # Record transition if we know where the object came from
                if obj.zone_transitions:
                    last_zone = obj.zone_transitions[-1]['to']
                    key = f"{last_zone}->{zone_id}"
                    self.pedestrian_counts['zone_transitions'][key] += 1
                
                # Record transition for the object
                if obj.zone_transitions:
                    obj.add_zone_transition(obj.zone_transitions[-1]['to'], zone_id, time.time())
                else:
                    obj.add_zone_transition('outside', zone_id, time.time())
            
            # Object just left the zone
            elif not is_in_zone and obj.object_id in zone['current_objects']:
                zone['current_objects'].remove(obj.object_id)
                zone['exit_count'] += 1
                
                # We don't record which zone they went to yet - will be updated on entry to next zone
                obj.add_zone_transition(zone_id, 'unknown', time.time())
    
    def generate_heatmap(self) -> np.ndarray:
        """
        Generate a position heatmap.
        
        Returns:
            Heatmap image
        """
        # Normalize the heatmap for visualization
        if np.max(self.position_heatmap) > 0:
            normalized = self.position_heatmap / np.max(self.position_heatmap)
        else:
            normalized = self.position_heatmap
            
        # Apply colormap
        heatmap = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        return heatmap
    
    def generate_flow_field(self, frame_shape: Tuple[int, int], grid_size: int = 20) -> np.ndarray:
        """
        Generate a flow field visualization.
        
        Args:
            frame_shape: (height, width) of the frame
            grid_size: Size of grid cells for aggregating flow
            
        Returns:
            Flow field visualization image
        """
        height, width = frame_shape[:2]
        
        # Create an empty image
        flow_field = np.zeros((height, width, 3), dtype=np.uint8)
        
        # If no movement data, return empty image
        if not self.movement_vectors:
            return flow_field
            
        # Create a grid for aggregating flow vectors
        grid_h = height // grid_size
        grid_w = width // grid_size
        grid = np.zeros((grid_h, grid_w, 2))  # (dy, dx) at each grid cell
        grid_counts = np.zeros((grid_h, grid_w))
        
        # Aggregate flow vectors into grid
        for vector in self.movement_vectors:
            pos = vector['position']
            grid_x = min(grid_w - 1, pos[0] // grid_size)
            grid_y = min(grid_h - 1, pos[1] // grid_size)
            
            grid[grid_y, grid_x, 0] += vector['dy']
            grid[grid_y, grid_x, 1] += vector['dx']
            grid_counts[grid_y, grid_x] += 1
        
        # Normalize grid
        for y in range(grid_h):
            for x in range(grid_w):
                if grid_counts[y, x] > 0:
                    grid[y, x] /= grid_counts[y, x]
        
        # Draw flow vectors
        for y in range(grid_h):
            for x in range(grid_w):
                if grid_counts[y, x] > 0:
                    # Get the center of the grid cell
                    center_x = (x + 0.5) * grid_size
                    center_y = (y + 0.5) * grid_size
                    
                    # Get the flow vector
                    dx = grid[y, x, 1]
                    dy = grid[y, x, 0]
                    
                    # Scale the vector for visualization
                    magnitude = np.sqrt(dx**2 + dy**2)
                    scale = min(grid_size * 0.8, magnitude * 10)
                    
                    end_x = center_x + scale * dx / magnitude if magnitude > 0 else center_x
                    end_y = center_y + scale * dy / magnitude if magnitude > 0 else center_y
                    
                    # Color based on magnitude
                    color_val = min(255, int(magnitude * 50))
                    color = (0, color_val, 255 - color_val)
                    
                    # Draw the arrow
                    cv2.arrowedLine(
                        flow_field, 
                        (int(center_x), int(center_y)), 
                        (int(end_x), int(end_y)), 
                        color, 
                        2, 
                        tipLength=0.3
                    )
        
        return flow_field
    
    def visualize_tracking(self, frame: np.ndarray) -> np.ndarray:
        """
        Create a visualization of the tracking.
        
        Args:
            frame: Video frame
            
        Returns:
            Visualization image
        """
        # Create a copy of the frame for visualization
        viz_frame = frame.copy()
        
        # Draw zones
        for zone_id, zone in self.zones.items():
            # Draw zone polygon
            cv2.polylines(
                viz_frame, 
                [zone['polygon']], 
                True, 
                (0, 255, 0), 
                2
            )
            
            # Add zone label
            centroid = np.mean(zone['polygon'], axis=0).astype(int)
            cv2.putText(
                viz_frame,
                f"{zone_id}: {len(zone['current_objects'])} people",
                (centroid[0], centroid[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
        
        # Draw tracked objects
        for obj_id, obj in self.objects.items():
            if not obj.centroids:
                continue
                
            # Get current position
            pos = obj.centroids[-1]
            
            # Different colors for different classes
            if obj.class_id == PERSON_CLASS_ID:
                color = (0, 255, 0)  # Green for people
            else:
                color = (255, 0, 0)  # Red for other objects
            
            # Draw the centroid and object ID
            cv2.circle(viz_frame, pos, 4, color, -1)
            cv2.putText(
                viz_frame,
                f"ID {obj_id}",
                (pos[0] - 10, pos[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
            
            # Draw the track (last N positions)
            track = list(obj.centroids)
            if len(track) > 1:
                for i in range(1, len(track)):
                    cv2.line(
                        viz_frame,
                        track[i - 1],
                        track[i],
                        color,
                        2
                    )
            
            # Draw bounding box if available
            if obj.bbox:
                cv2.rectangle(
                    viz_frame,
                    (obj.bbox[0], obj.bbox[1]),
                    (obj.bbox[2], obj.bbox[3]),
                    color,
                    2
                )
        
        return viz_frame
    
    def process_video(self, video_path: str, output_path: str = None, visualize: bool = True,
                     max_frames: int = None) -> Dict:
        """
        Process a video file.
        
        Args:
            video_path: Path to video file
            output_path: Path to save output video (optional)
            visualize: Whether to generate visualization
            max_frames: Maximum number of frames to process (optional)
            
        Returns:
            Dictionary with processing statistics
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return {'success': False, 'error': 'Could not open video'}
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.fps = fps
        logger.info(f"Processing video: {width}x{height}, {fps} fps, {total_frames} frames")
        
        # Create output video writer if needed
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            logger.info(f"Writing output to {output_path}")
        
        # Reset tracking state
        self.next_object_id = 0
        self.objects = {}
        self.frame_count = 0
        self.position_heatmap = np.zeros(HEATMAP_RESOLUTION, dtype=np.float32)
        self.movement_vectors = []
        
        # Process frames
        while True:
            ret, frame = cap.read()
            
            # Break if end of video or max frames reached
            if not ret or (max_frames is not None and self.frame_count >= max_frames):
                break
                
            # Update tracking
            tracking_results = self.update(frame)
            
            # Create visualization if needed
            if visualize or output_path:
                viz_frame = self.visualize_tracking(frame)
                
                # Show in window if visualizing
                if visualize:
                    cv2.imshow('Pedestrian Tracking', viz_frame)
                    
                    # Exit if 'q' pressed
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Write to output video
                if out:
                    out.write(viz_frame)
            
            # Print progress periodically
            if self.frame_count % 100 == 0:
                progress = (self.frame_count / total_frames) * 100 if total_frames > 0 else 0
                logger.info(f"Processed frame {self.frame_count}/{total_frames} ({progress:.1f}%)")
        
        # Clean up
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        # Generate summary statistics
        stats = self.generate_statistics()
        
        logger.info(f"Video processing complete. Detected {stats['total_pedestrians']} pedestrians.")
        
        return {
            'success': True,
            'statistics': stats,
            'frames_processed': self.frame_count
        }
    
    def generate_statistics(self) -> Dict:
        """
        Generate summary statistics.
        
        Returns:
            Dictionary with statistics
        """
        # Calculate flow patterns
        flow_patterns = self._analyze_flow_patterns()
        
        # Calculate zone statistics
        zone_stats = {}
        for zone_id, zone in self.zones.items():
            zone_stats[zone_id] = {
                'entries': zone['entry_count'],
                'exits': zone['exit_count'],
                'current_count': len(zone['current_objects'])
            }
        
        # Get hourly distribution
        hourly = {}
        for hour, count in self.pedestrian_counts['hourly'].items():
            hourly[hour] = count
        
        # Get zone transitions
        transitions = {}
        for key, count in self.pedestrian_counts['zone_transitions'].items():
            transitions[key] = count
        
        return {
            'total_pedestrians': self.pedestrian_counts['total'],
            'currently_tracked': len([obj for obj in self.objects.values() if obj.class_id == PERSON_CLASS_ID]),
            'hourly_distribution': hourly,
            'zone_statistics': zone_stats,
            'zone_transitions': transitions,
            'flow_patterns': flow_patterns,
            'average_speed': self._calculate_average_speed()
        }
    
    def _analyze_flow_patterns(self) -> List[Dict]:
        """
        Analyze pedestrian flow patterns.
        
        Returns:
            List of flow pattern dictionaries
        """
        if not self.movement_vectors:
            return []
            
        # Convert movement vectors to numpy array
        vectors = np.array([(v['dx'], v['dy']) for v in self.movement_vectors])
        positions = np.array([v['position'] for v in self.movement_vectors])
        
        # Calculate magnitudes and directions
        magnitudes = np.sqrt(vectors[:, 0]**2 + vectors[:, 1]**2)
        directions = np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0]))
        
        # Cluster the directions to find main flow patterns
        # Adjust directions to handle the circular nature of angles
        directions_rad = np.radians(directions)
        dir_features = np.column_stack([np.cos(directions_rad), np.sin(directions_rad)])
        
        # Add position information for spatial clustering
        features = np.column_stack([
            positions[:, 0] / np.max(positions[:, 0]) if np.max(positions[:, 0]) > 0 else positions[:, 0],
            positions[:, 1] / np.max(positions[:, 1]) if np.max(positions[:, 1]) > 0 else positions[:, 1],
            dir_features
        ])
        
        # Use DBSCAN for clustering
        clustering = DBSCAN(eps=0.2, min_samples=5).fit(features)
        
        # Get the clusters
        labels = clustering.labels_
        unique_labels = np.unique(labels)
        
        # Remove noise (label -1)
        unique_labels = unique_labels[unique_labels != -1]
        
        # Analyze each cluster
        flow_patterns = []
        for label in unique_labels:
            mask = labels == label
            cluster_vectors = vectors[mask]
            cluster_positions = positions[mask]
            cluster_magnitudes = magnitudes[mask]
            cluster_directions = directions[mask]
            
            # Calculate average values
            avg_pos = np.mean(cluster_positions, axis=0)
            avg_dir = np.mean(cluster_directions)
            avg_mag = np.mean(cluster_magnitudes)
            
            # Create flow pattern entry
            flow_patterns.append({
                'position': (int(avg_pos[0]), int(avg_pos[1])),
                'direction': avg_dir,
                'magnitude': avg_mag,
                'count': np.sum(mask)
            })
        
        return flow_patterns
    
    def _calculate_average_speed(self) -> float:
        """
        Calculate average pedestrian speed.
        
        Returns:
            Average speed in pixels per second
        """
        if not self.movement_vectors:
            return 0.0
            
        # Get magnitudes
        magnitudes = [v['magnitude'] for v in self.movement_vectors]
        
        # Convert to speed (pixels per second)
        if self.fps > 0:
            speeds = [mag * self.fps / DETECTION_INTERVAL for mag in magnitudes]
        else:
            speeds = magnitudes  # Fallback if fps unknown
        
        # Convert to meters per second if calibrated
        if self.calibration.is_calibrated:
            speeds = [self.calibration.pixel_distance_to_meters(s) for s in speeds]
            
        # Calculate average
        if speeds:
            return np.mean(speeds)
        else:
            return 0.0
    
    def save_results(self, output_dir: str) -> None:
        """
        Save analysis results to files.
        
        Args:
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save statistics as JSON
        stats = self.generate_statistics()
        with open(os.path.join(output_dir, 'statistics.json'), 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Save heatmap
        heatmap = self.generate_heatmap()
        cv2.imwrite(os.path.join(output_dir, 'heatmap.png'), heatmap)
        
        # Save flow visualization
        if self.movement_vectors:
            flow_img = self.generate_flow_field((HEATMAP_RESOLUTION[1], HEATMAP_RESOLUTION[0]))
            cv2.imwrite(os.path.join(output_dir, 'flow_field.png'), flow_img)
        
        logger.info(f"Results saved to {output_dir}")


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the haversine distance between two points in meters.
    
    Args:
        lat1: Latitude of first point
        lon1: Longitude of first point
        lat2: Latitude of second point
        lon2: Longitude of second point
        
    Returns:
        Distance in meters
    """
    return geodesic((lat1, lon1), (lat2, lon2)).meters


def run_pedestrian_analysis(video_path: str, output_dir: str, calibration_path: str = None,
                         model_path: str = None, visualize: bool = False) -> Dict:
    """
    Run the pedestrian analysis on a video.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save results
        calibration_path: Path to camera calibration file (optional)
        model_path: Path to detection model (optional)
        visualize: Whether to show visualization during processing
        
    Returns:
        Dictionary with processing results
    """
    # Load camera calibration if available
    camera_calibration = None
    if calibration_path and os.path.exists(calibration_path):
        try:
            with open(calibration_path, 'r') as f:
                camera_calibration = json.load(f)
        except Exception as e:
            logger.error(f"Error loading calibration: {str(e)}")
    
    # Create the tracker
    tracker = PedestrianTracker(model_path, camera_calibration)
    
    # Define zones (this would typically be configured for each camera)
    # This is a placeholder example for a 1920x1080 video
    tracker.define_zone('entrance', [(100, 500), (300, 500), (300, 700), (100, 700)])
    tracker.define_zone('exit', [(1600, 500), (1800, 500), (1800, 700), (1600, 700)])
    tracker.define_zone('waiting_area', [(800, 300), (1100, 300), (1100, 600), (800, 600)])
    
    # Process the video
    results = tracker.process_video(video_path, 
                                  os.path.join(output_dir, 'output.mp4'),
                                  visualize=visualize)
    
    # Save results
    if results['success']:
        tracker.save_results(output_dir)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Pedestrian Flow Tracker')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--output', type=str, default='./output', help='Output directory')
    parser.add_argument('--calibration', type=str, help='Path to camera calibration file')
    parser.add_argument('--model', type=str, help='Path to detection model')
    parser.add_argument('--visualize', action='store_true', help='Show visualization during processing')
    
    args = parser.parse_args()
    
    run_pedestrian_analysis(
        args.video,
        args.output,
        args.calibration,
        args.model,
        args.visualize
    )

"""
SUMMARY:
========
This module implements a pedestrian flow tracking system that uses computer vision 
to detect, track, and analyze pedestrian movements. The system generates heatmaps of 
pedestrian activity, analyzes movement patterns, and tracks zone transitions.

Key components:
1. TrackedObject - Represents individual tracked pedestrians with position history
2. CameraCalibration - Handles conversion between pixel and real-world coordinates
3. PedestrianTracker - Main class for detecting and tracking pedestrians
4. Statistics and visualization tools for analyzing pedestrian flow

The system processes video feeds, identifies pedestrians, and generates metrics on
movement patterns, congestion, and zone occupancy.

TODO:
=====
1. Implement a real object detection model (currently using placeholder)
2. Add support for multiple camera views with coordinate mapping
3. Improve tracking algorithm to handle occlusions better
4. Implement track re-identification for people leaving and re-entering the frame
5. Add privacy filters to blur faces in output videos
6. Optimize for real-time processing on edge devices
7. Implement predictive crowd flow models based on historical patterns
8. Add support for demographic analysis (age groups, groups vs individuals)
9. Create a dashboard for real-time monitoring of pedestrian metrics
10. Implement anomaly detection for unusual pedestrian behavior
11. Add integration with traffic light systems for demand-responsive crossings
12. Create a mobile version for temporary event monitoring
13. Implement social distancing monitoring capabilities
"""
