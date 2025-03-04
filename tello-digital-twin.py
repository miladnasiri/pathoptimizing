"""
TelloDigitalTwin - A simplified digital twin for the Tello drone
Can be used for simulation or with a real drone
"""

import time
import math
import os
import cv2
import numpy as np
from datetime import datetime
import json
import threading

class TelloMockDrone:
    """Mock drone for simulation when no physical drone is available"""
    
    def __init__(self):
        self.connected = False
        self.battery = 100
        
    def connect(self):
        """Connect to the mock drone"""
        self.connected = True
        print("Connected to mock Tello drone")
        return True
        
    def get_battery(self):
        """Get battery percentage"""
        # Simulate battery drain
        if self.connected:
            self.battery = max(0, self.battery - 0.01)
        return int(self.battery)
        
    def takeoff(self):
        """Simulate takeoff"""
        print("Mock Tello takeoff")
        return True
        
    def land(self):
        """Simulate landing"""
        print("Mock Tello landing")
        return True
        
    def emergency(self):
        """Simulate emergency stop"""
        print("Mock Tello emergency stop")
        return True
        
    def send_rc_control(self, left_right, forward_backward, up_down, yaw):
        """Simulate RC control"""
        print(f"RC Control: LR={left_right}, FB={forward_backward}, UD={up_down}, Yaw={yaw}")
        return True
        
    def streamon(self):
        """Start video stream"""
        print("Mock video stream started")
        return True
        
    def streamoff(self):
        """Stop video stream"""
        print("Mock video stream stopped")
        return True
        
    def get_frame_read(self):
        """Get frame object"""
        return MockFrameRead()

class MockFrameRead:
    """Mock frame reader for simulation"""
    
    def __init__(self):
        self.frame = np.zeros((720, 960, 3), dtype=np.uint8)
        # Create a simple pattern for the frame
        cv2.putText(self.frame, "MOCK TELLO FEED", (300, 360), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
    def frame(self):
        """Get current frame"""
        return self.frame

class TelloDigitalTwin:
    """
    Digital twin for Tello drone - can be used with real drone or in simulation
    """
    
    def __init__(self, use_mock=True):
        """Initialize the digital twin"""
        self.use_mock = use_mock
        self.drone = None
        
        # Physical state
        self.x = 500  # cm - 500 is the center of the coordinate system
        self.y = 500  # cm
        self.z = 0    # cm
        self.yaw = 0  # degrees
        self.speed = 30  # cm/s
        self.battery = 100  # percentage
        
        # Status flags
        self.is_flying = False
        self.video_stream = False
        self.mode = "IDLE"
        
        # Frame data
        self.frame = None
        self.frame_read = None
        
        # Create necessary directories
        self.flight_dir = "tello_data"
        if not os.path.exists(self.flight_dir):
            os.makedirs(self.flight_dir)
            
        # Flight path data for logging
        self.path = []
        self.altitude = []
        self.timestamps = []
        self.battery_log = []
        self.start_time = None
            
        # Try to connect to the drone
        self._connect_drone()
        
    def _connect_drone(self):
        """Connect to the Tello drone or initialize mock"""
        if self.use_mock:
            self.drone = TelloMockDrone()
            print("Using mock Tello drone for simulation")
        else:
            try:
                from djitellopy import Tello
                self.drone = Tello()
                print("Tello SDK initialized")
            except ImportError:
                print("djitellopy not installed, defaulting to mock drone")
                self.drone = TelloMockDrone()
                self.use_mock = True
    
    def connect(self):
        """Connect to the drone"""
        if self.drone:
            self.drone.connect()
            self.battery = self.drone.get_battery()
            print(f"Connected to Tello drone. Battery: {self.battery}%")
            return True
        return False
    
    def update_state(self, dt):
        """Update the digital twin state"""
        # Record flight path
        if self.is_flying:
            self.path.append((self.x, self.y))
            self.altitude.append(self.z)
            
            if self.start_time is None:
                self.start_time = time.time()
                
            self.timestamps.append(time.time() - self.start_time)
            self.battery_log.append(self.battery)
            
            # Update battery level
            if self.drone:
                self.battery = self.drone.get_battery()
    
    def update_video(self):
        """Update video frame"""
        if self.video_stream and self.drone:
            if self.frame_read is None:
                self.frame_read = self.drone.get_frame_read()
                
            # Get video frame
            if isinstance(self.frame_read, MockFrameRead):
                self.frame = self.frame_read.frame
            else:
                self.frame = self.frame_read.frame
                
            # Add overlay with drone info
            if self.frame is not None:
                # Create copy to avoid modifying original
                frame_with_overlay = self.frame.copy()
                
                # Add text overlay with position and battery
                cv2.putText(frame_with_overlay, 
                           f"POS: ({(self.x-500)/100:.2f}m, {(self.y-500)/100:.2f}m, {self.z/100:.2f}m)", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                           
                cv2.putText(frame_with_overlay, 
                           f"YAW: {self.yaw}° BATT: {self.battery}%", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                self.frame = frame_with_overlay
    
    def save_flight_data(self):
        """Save flight data to file"""
        if len(self.path) == 0:
            return
            
        # Create timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"flight_{timestamp}.json"
        
        # Prepare data
        flight_data = {
            "timestamp": timestamp,
            "path": self.path,
            "altitude": self.altitude,
            "timestamps": self.timestamps,
            "battery": self.battery_log,
            "duration": self.timestamps[-1] if self.timestamps else 0
        }
        
        # Save to file
        filepath = os.path.join(self.flight_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(flight_data, f)
            
        print(f"Flight data saved to {filepath}")
        
        # Reset data
        self.path = []
        self.altitude = []
        self.timestamps = []
        self.battery_log = []
        self.start_time = None
    
    def run(self):
        """Run the digital twin main loop"""
        print("Starting Tello Digital Twin")
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
                
                # Simulated key handling would go here
                
                # Sleep to reduce CPU usage
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("Digital Twin stopped by user")
        finally:
            # Save flight data
            self.save_flight_data()
            
            # Land the drone if flying
            if self.is_flying and self.drone:
                print("Landing drone...")
                self.drone.land()
                self.is_flying = False


if __name__ == "__main__":
    # Test the digital twin
    twin = TelloDigitalTwin(use_mock=True)
    twin.connect()
    
    # Start video
    twin.video_stream = True
    
    # Simulate takeoff
    twin.is_flying = True
    twin.z = 100  # 1m height
    
    # Run for a while
    try:
        twin.run()
    except KeyboardInterrupt:
        print("Test stopped by user")
