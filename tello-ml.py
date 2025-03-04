"""
TelloML - Machine learning capabilities for Tello drone
Implements feature extraction, object detection, and autonomous flight features
"""

import numpy as np
import cv2
import os
import time
from datetime import datetime
import json

class TelloML:
    """
    Machine learning capabilities for Tello drone
    - Feature extraction from video feed
    - Object detection and tracking
    - Mapping and environment understanding
    - Autonomous decision making
    """
    
    def __init__(self):
        """Initialize ML components"""
        # Feature extraction
        self.feature_detector = cv2.SIFT_create()
        self.features = []
        
        # Object detection
        self.object_detector = self._create_object_detector()
        self.detected_objects = []
        
        # Mapping
        self.occupancy_grid = np.zeros((20, 20))  # 20x20 grid representing 10m x 10m area
        
        # Model directories
        self.models_dir = "tello_models"
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
    
    def _create_object_detector(self):
        """Create a simple object detector (mock implementation)"""
        # This would normally load a trained model like YOLO or SSD
        # But for this placeholder, we'll use a simple color-based detector
        
        class SimpleDetector:
            def detect(self, frame):
                """Detect objects based on color"""
                if frame is None:
                    return []
                
                # Convert to HSV
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                
                # Define color ranges for detection
                color_ranges = [
                    # Red objects
                    (np.array([0, 120, 70]), np.array([10, 255, 255]), "red"),
                    (np.array([170, 120, 70]), np.array([180, 255, 255]), "red"),
                    # Blue objects
                    (np.array([100, 150, 70]), np.array([130, 255, 255]), "blue"),
                    # Green objects
                    (np.array([40, 100, 70]), np.array([80, 255, 255]), "green")
                ]
                
                results = []
                
                for lower, upper, label in color_ranges:
                    # Create mask
                    mask = cv2.inRange(hsv, lower, upper)
                    
                    # Find contours
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Process larger contours
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if area > 500:  # Ignore small regions
                            x, y, w, h = cv2.boundingRect(contour)
                            confidence = min(1.0, area / 10000)  # Scale confidence by area
                            results.append({
                                "label": label,
                                "confidence": confidence,
                                "bbox": (x, y, w, h)
                            })
                
                return results
        
        return SimpleDetector()
    
    def extract_features(self, frame):
        """Extract features from video frame"""
        if frame is None:
            return []
            
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect keypoints
        keypoints, descriptors = self.feature_detector.detectAndCompute(gray, None)
        
        # Store features
        self.features = [(kp.pt, kp.size, kp.angle) for kp in keypoints]
        
        return self.features
    
    def detect_objects(self, frame):
        """Detect objects in the video frame"""
        if frame is None:
            return []
            
        # Run object detection
        self.detected_objects = self.object_detector.detect(frame)
        
        return self.detected_objects
    
    def update_occupancy_grid(self, frame, drone_position, drone_altitude, drone_yaw):
        """Update occupancy grid based on current observations"""
        if frame is None:
            return
            
        # Detect objects
        objects = self.detect_objects(frame)
        
        # Convert drone position to grid coordinates
        grid_center_x = self.occupancy_grid.shape[1] // 2
        grid_center_y = self.occupancy_grid.shape[0] // 2
        
        # For each detected object, update occupancy grid
        for obj in objects:
            # Get object position in frame
            x, y, w, h = obj["bbox"]
            
            # Calculate object position relative to drone
            # This is a simplification - would need camera calibration for accurate mapping
            frame_center_x = frame.shape[1] // 2
            frame_center_y = frame.shape[0] // 2
            
            # Calculate angle to object
            dx = (x + w/2) - frame_center_x
            dy = (y + h/2) - frame_center_y
            
            # Convert to angle
            angle_rad = np.arctan2(dx, frame_center_y - (y + h/2))
            object_angle = (drone_yaw + np.degrees(angle_rad)) % 360
            
            # Estimate distance (simplified)
            object_size = max(w, h)
            object_distance = 100 / (object_size + 1)  # Approximate inverse relationship
            
            # Convert to grid coordinates
            object_grid_x = int(grid_center_x + np.cos(np.radians(object_angle)) * object_distance / 100)
            object_grid_y = int(grid_center_y + np.sin(np.radians(object_angle)) * object_distance / 100)
            
            # Update occupancy grid if in bounds
            if (0 <= object_grid_x < self.occupancy_grid.shape[1] and 
                0 <= object_grid_y < self.occupancy_grid.shape[0]):
                
                # Increase occupancy (obstacle detected)
                self.occupancy_grid[object_grid_y, object_grid_x] = min(
                    1.0, 
                    self.occupancy_grid[object_grid_y, object_grid_x] + 0.2
                )
    
    def make_navigation_decision(self, drone_state, target_position):
        """Make autonomous navigation decision"""
        # Simple obstacle avoidance based on occupancy grid
        # Returns recommended movement vector (dx, dy, dz, dyaw)
        
        # Extract drone position
        x_pos = (drone_state["x"] - 500) / 100  # Convert to meters
        y_pos = (drone_state["y"] - 500) / 100
        z_pos = drone_state["z"] / 100
        yaw = drone_state["yaw"]
        
        # Target position
        target_x, target_y = target_position
        
        # Vector to target
        dx = target_x - x_pos
        dy = target_y - y_pos
        
        # Distance to target
        distance = np.sqrt(dx**2 + dy**2)
        
        # If we've reached the target
        if distance < 0.2:  # Within 20cm
            return (0, 0, 0, 0)
        
        # Calculate angle to target
        angle_to_target = np.degrees(np.arctan2(dy, dx))
        heading_error = (angle_to_target - yaw) % 360
        if heading_error > 180:
            heading_error -= 360
        
        # Prioritize rotation if we're not facing the right direction
        if abs(heading_error) > 15:
            yaw_speed = max(-30, min(30, heading_error / 3))
            return (0, 0, 0, yaw_speed)
        
        # Check for obstacles in the path
        grid_center_x = self.occupancy_grid.shape[1] // 2
        grid_center_y = self.occupancy_grid.shape[0] // 2
        
        # Sample points along the path
        obstacle_detected = False
        obstacle_direction = (0, 0)
        
        for step in range(1, 11):
            check_dist = step * 0.5  # Check up to 5m ahead in 0.5m increments
            if check_dist > distance:
                break
                
            # Position to check
            check_x = x_pos + (dx / distance) * check_dist
            check_y = y_pos + (dy / distance) * check_dist
            
            # Convert to grid
            grid_x = int(grid_center_x + check_x)
            grid_y = int(grid_center_y + check_y)
            
            # Check if in bounds
            if (0 <= grid_x < self.occupancy_grid.shape[1] and 
                0 <= grid_y < self.occupancy_grid.shape[0]):
                
                # Check occupancy
                if self.occupancy_grid[grid_y, grid_x] > 0.5:
                    obstacle_detected = True
                    # Direction to obstacle (for avoidance)
                    obstacle_direction = (grid_x - grid_center_x, grid_y - grid_center_y)
                    break
        
        # If obstacle detected, avoid it
        if obstacle_detected:
            # Calculate perpendicular direction (to go around obstacle)
            perp_x = -obstacle_direction[1]
            perp_y = obstacle_direction[0]
            
            # Normalize
            perp_len = np.sqrt(perp_x**2 + perp_y**2)
            if perp_len > 0:
                perp_x /= perp_len
                perp_y /= perp_len
            
            # Blend with original direction (70% avoid, 30% toward goal)
            blend_x = 0.7 * perp_x + 0.3 * (dx / distance)
            blend_y = 0.7 * perp_y + 0.3 * (dy / distance)
            
            # Calculate speeds (max 30 in any direction)
            speed_scale = min(30, distance * 100)
            lr_speed = int(blend_x * speed_scale)
            fb_speed = int(blend_y * speed_scale)
            
            return (lr_speed, fb_speed, 0, 0)
        
        # No obstacle, move directly toward target
        speed_scale = min(30, distance * 100)
        # Forward speed
        fb_speed = int(speed_scale)
        
        return (0, fb_speed, 0, 0)
    
    def save_models(self):
        """Save trained models and maps"""
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save occupancy grid
        grid_file = os.path.join(self.models_dir, f"occupancy_grid_{timestamp}.npy")
        np.save(grid_file, self.occupancy_grid)
        
        # Save feature descriptor database (mockup)
        feature_file = os.path.join(self.models_dir, f"features_{timestamp}.json")
        with open(feature_file, 'w') as f:
            # Save simplified representation
            json.dump({
                "timestamp": timestamp,
                "feature_count": len(self.features),
                "feature_summary": "SIFT features extracted from flight"
            }, f)
        
        print(f"Models saved to {self.models_dir}")
        return True
    
    def visualize_results(self, frame):
        """Create visualization with detected objects and features"""
        if frame is None:
            return frame
            
        # Create copy for visualization
        viz_frame = frame.copy()
        
        # Draw detected objects
        for obj in self.detected_objects:
            x, y, w, h = obj["bbox"]
            label = obj["label"]
            conf = obj["confidence"]
            
            # Color based on label
            color = (0, 0, 255)  # Default red
            if label == "blue":
                color = (255, 0, 0)
            elif label == "green":
                color = (0, 255, 0)
            
            # Draw bounding box
            cv2.rectangle(viz_frame, (x, y), (x+w, y+h), color, 2)
            
            # Draw label
            cv2.putText(viz_frame, f"{label} {conf:.2f}", (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw features
        for i, (pt, size, angle) in enumerate(self.features[:20]):  # Limit to first 20
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(viz_frame, (x, y), int(size), (0, 255, 255), 1)
            
            # Draw orientation line
            end_x = int(x + size * np.cos(np.radians(angle)))
            end_y = int(y + size * np.sin(np.radians(angle)))
            cv2.line(viz_frame, (x, y), (end_x, end_y), (0, 255, 255), 1)
        
        # Add text with counts
        cv2.putText(viz_frame, f"Objects: {len(self.detected_objects)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                   
        cv2.putText(viz_frame, f"Features: {len(self.features)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return viz_frame

class TelloDigitalTwinWithML(TelloDigitalTwin):
    """
    Extended Tello Digital Twin with ML capabilities
    """
    
    def __init__(self, use_mock=True):
        """Initialize with ML components"""
        super().__init__(use_mock)
        
        # Add ML component
        self.ml = TelloML()
        
        # ML feature flags
        self.feature_extraction_active = False
        self.object_detection_active = False
        self.mapping_active = False
        self.autonomous_mode = False
        
        # Target for autonomous navigation
        self.autonomous_target = None
    
    def update_ml(self):
        """Update ML components"""
        if self.frame is None:
            return
            
        # Extract features if enabled
        if self.feature_extraction_active:
            self.ml.extract_features(self.frame)
        
        # Detect objects if enabled
        if self.object_detection_active:
            self.ml.detect_objects(self.frame)
        
        # Update mapping if enabled
        if self.mapping_active:
            drone_state = {
                "x": self.x,
                "y": self.y,
                "z": self.z,
                "yaw": self.yaw
            }
            self.ml.update_occupancy_grid(self.frame, drone_state, self.z, self.yaw)
        
        # Autonomous navigation if enabled
        if self.autonomous_mode and self.autonomous_target is not None and self.is_flying:
            # Make navigation decision
            drone_state = {
                "x": self.x,
                "y": self.y,
                "z": self.z,
                "yaw": self.yaw
            }
            
            # Get control commands
            lr, fb, ud, yaw = self.ml.make_navigation_decision(drone_state, self.autonomous_target)
            
            # Send commands to drone
            if self.drone:
                self.drone.send_rc_control(lr, fb, ud, yaw)
    
    def update_video(self):
        """Override to add ML visualization"""
        super().update_video()
        
        # Add ML visualization if any ML feature is active
        if (self.frame is not None and 
            (self.feature_extraction_active or self.object_detection_active)):
            self.frame = self.ml.visualize_results(self.frame)
    
    def run(self):
        """Override run to add ML updates"""
        print("Starting Tello Digital Twin with ML")
        last_time = time.time()
        
        try:
            while True:
                # Calculate time difference
                current_time = time.time()
                dt = current_time - last_time
                last_time = current_time
                
                # Update state
                self.update_state(dt)
                
                # Update video
                self.update_video()
                
                # Update ML
                self.update_ml()
                
                # Sleep to reduce CPU usage
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("Digital Twin stopped by user")
        finally:
            # Save flight data
            self.save_flight_data()
            
            # Save ML models
            if hasattr(self, 'ml'):
                self.ml.save_models()
            
            # Land the drone if flying
            if self.is_flying and self.drone:
                print("Landing drone...")
                self.drone.land()
                self.is_flying = False


if __name__ == "__main__":
    # Test the ML-enabled digital twin
    twin = TelloDigitalTwinWithML(use_mock=True)
    twin.connect()
    
    # Start video
    twin.video_stream = True
    
    # Enable ML features
    twin.feature_extraction_active = True
    twin.object_detection_active = True
    twin.mapping_active = True
    
    # Simulate takeoff
    twin.is_flying = True
    twin.z = 100  # 1m height
    
    # Run for a while
    try:
        twin.run()
    except KeyboardInterrupt:
        print("Test stopped by user")
