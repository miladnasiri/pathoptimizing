"""
TelloMissions - Mission controller and predefined missions for Tello drones
"""

import time
import math
import os
import json
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from datetime import datetime
import cv2

class TelloMission:
    """Base class for Tello drone missions"""
    
    def __init__(self, controller, name="Unnamed Mission"):
        """
        Initialize a mission
        
        Args:
            controller: TelloDigitalTwin or compatible controller
            name: Name of the mission
        """
        self.controller = controller
        self.name = name
        self.start_time = None
        self.end_time = None
        self.thread = None
        self.abort_flag = False
        
        # Mission data for logging
        self.mission_data = {
            "name": name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mission_type": "base",
            "parameters": {},
            "path": [],
            "altitude": [],
            "timestamps": [],
            "battery": [],
            "success": False,
            "duration": 0,
            "error": None
        }
    
    def start(self):
        """Start the mission in a separate thread"""
        if self.thread is not None and self.thread.is_alive():
            print(f"Mission '{self.name}' is already running")
            return False
            
        self.thread = threading.Thread(target=self._run_mission)
        self.thread.daemon = True
        self.thread.start()
        return True
    
    def abort(self):
        """Abort the mission"""
        if self.thread is not None and self.thread.is_alive():
            self.abort_flag = True
            print(f"Aborting mission '{self.name}'")
            return True
        return False
    
    def _run_mission(self):
        """Run the mission and handle cleanup"""
        self.start_time = time.time()
        self.abort_flag = False
        
        try:
            # Execute mission-specific logic
            self._execute_mission()
        except Exception as e:
            print(f"Error in mission '{self.name}': {e}")
            self.mission_data["error"] = str(e)
        finally:
            # Record final state
            self.end_time = time.time()
            self.mission_data["duration"] = self.end_time - self.start_time
            
            # Save mission data
            self._save_mission_data()
    
    def _execute_mission(self):
        """To be implemented by subclasses"""
        # Default implementation - just hover for 5 seconds
        print(f"Executing mission '{self.name}'")
        
        # Record initial state
        self._record_data_point()
        
        # Hover for 5 seconds
        start_time = time.time()
        while time.time() - start_time < 5 and not self.abort_flag:
            # Record data every 0.5 seconds
            time.sleep(0.5)
            self._record_data_point()
        
        # Mark as successful if not aborted
        if not self.abort_flag:
            self.mission_data["success"] = True
            
        # Complete the mission
        self._complete_mission()
    
    def _record_data_point(self):
        """Record current drone state"""
        if not hasattr(self.controller, 'x'):
            return
            
        # Record position
        self.mission_data["path"].append((self.controller.x, self.controller.y))
        
        # Record altitude
        self.mission_data["altitude"].append(self.controller.z)
        
        # Record timestamp
        if self.start_time is not None:
            self.mission_data["timestamps"].append(time.time() - self.start_time)
        
        # Record battery
        self.mission_data["battery"].append(self.controller.battery)
    
    def _complete_mission(self):
        """Mark mission as complete and save data"""
        print(f"Mission '{self.name}' completed")
        
        # Check if the mission was successful
        if not self.abort_flag and self.mission_data.get("error") is None:
            self.mission_data["success"] = True
    
    def _save_mission_data(self):
        """Save mission data to file"""
        # Create missions directory if it doesn't exist
        missions_dir = os.path.join(self.controller.flight_dir, "missions")
        if not os.path.exists(missions_dir):
            os.makedirs(missions_dir)
            
        # Create filename with timestamp
        timestamp = self.mission_data["timestamp"].replace(" ", "_").replace(":", "-")
        filename = f"mission_{timestamp}.json"
        
        # Save to file
        filepath = os.path.join(missions_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.mission_data, f)
            
        print(f"Mission data saved to {filepath}")


class RectangleMission(TelloMission):
    """Fly in a rectangle pattern"""
    
    def __init__(self, controller, width=1.0, height=1.0, altitude=1.0):
        """
        Initialize rectangle mission
        
        Args:
            controller: TelloDigitalTwin controller
            width: Width of rectangle in meters
            height: Height of rectangle in meters
            altitude: Flight altitude in meters
        """
        super().__init__(controller, name="Rectangle Mission")
        self.width = width
        self.height = height
        self.altitude = altitude
        
        # Update mission data
        self.mission_data["mission_type"] = "rectangle"
        self.mission_data["parameters"] = {
            "width": width,
            "height": height,
            "altitude": altitude
        }
    
    def _execute_mission(self):
        """Execute rectangle mission"""
        if not self.controller.is_flying:
            print("Drone is not flying. Mission aborted.")
            return
            
        # Record initial state
        self._record_data_point()
        
        # Convert to cm for drone control
        width_cm = int(self.width * 100)
        height_cm = int(self.height * 100)
        altitude_cm = int(self.altitude * 100)
        
        try:
            # Adjust altitude if needed
            if abs(self.controller.z - altitude_cm) > 20:
                print(f"Adjusting altitude to {self.altitude}m")
                
                # Calculate direction and distance
                alt_diff = altitude_cm - self.controller.z
                direction = 1 if alt_diff > 0 else -1
                distance = abs(alt_diff)
                
                # Move in increments
                while distance > 0 and not self.abort_flag:
                    move_dist = min(50, distance)  # Max 50cm at a time
                    self.controller.drone.send_rc_control(0, 0, direction * 30, 0)
                    time.sleep(move_dist / 30)  # Approximation
                    self.controller.drone.send_rc_control(0, 0, 0, 0)
                    time.sleep(0.5)
                    
                    self._record_data_point()
                    distance -= move_dist
            
            # Fly rectangle
            if not self.abort_flag:
                print("Flying forward")
                self.controller.drone.send_rc_control(0, 30, 0, 0)
                time.sleep(width_cm / 30)  # Approximation
                self.controller.drone.send_rc_control(0, 0, 0, 0)
                time.sleep(0.5)
                self._record_data_point()
            
            if not self.abort_flag:
                print("Flying right")
                self.controller.drone.send_rc_control(30, 0, 0, 0)
                time.sleep(height_cm / 30)  # Approximation
                self.controller.drone.send_rc_control(0, 0, 0, 0)
                time.sleep(0.5)
                self._record_data_point()
            
            if not self.abort_flag:
                print("Flying backward")
                self.controller.drone.send_rc_control(0, -30, 0, 0)
                time.sleep(width_cm / 30)  # Approximation
                self.controller.drone.send_rc_control(0, 0, 0, 0)
                time.sleep(0.5)
                self._record_data_point()
            
            if not self.abort_flag:
                print("Flying left")
                self.controller.drone.send_rc_control(-30, 0, 0, 0)
                time.sleep(height_cm / 30)  # Approximation
                self.controller.drone.send_rc_control(0, 0, 0, 0)
                time.sleep(0.5)
                self._record_data_point()
            
            # Mark as successful if completed
            if not self.abort_flag:
                self.mission_data["success"] = True
                
        except Exception as e:
            print(f"Error executing rectangle mission: {e}")
            self.mission_data["error"] = str(e)
        finally:
            # Ensure drone stops
            try:
                self.controller.drone.send_rc_control(0, 0, 0, 0)
            except:
                pass
            
            # Complete the mission
            self._complete_mission()


class SpiralMission(TelloMission):
    """Fly in an ascending spiral pattern"""
    
    def __init__(self, controller, radius=1.0, height=1.0, rotations=2):
        """
        Initialize spiral mission
        
        Args:
            controller: TelloDigitalTwin controller
            radius: Radius of spiral in meters
            height: Total height gain in meters
            rotations: Number of rotations
        """
        super().__init__(controller, name="Spiral Mission")
        self.radius = radius
        self.height = height
        self.rotations = rotations
        
        # Update mission data
        self.mission_data["mission_type"] = "spiral"
        self.mission_data["parameters"] = {
            "radius": radius,
            "height": height,
            "rotations": rotations
        }
    
    def _execute_mission(self):
        """Execute spiral mission"""
        if not self.controller.is_flying:
            print("Drone is not flying. Mission aborted.")
            return
            
        # Record initial state
        self._record_data_point()
        
        try:
            # Fly spiral pattern
            start_height = self.controller.z
            target_height = start_height + (self.height * 100)  # Convert to cm
            
            # Calculate parameters
            steps = 20 * self.rotations  # 20 steps per rotation
            angle_step = 2 * math.pi / 20  # Angle change per step
            height_step = (self.height * 100) / steps  # Height change per step
            
            # Execute spiral
            current_angle = 0
            for i in range(steps):
                if self.abort_flag:
                    break
                    
                # Calculate position in spiral
                x_vel = int(30 * math.cos(current_angle))
                y_vel = int(30 * math.sin(current_angle))
                z_vel = int(height_step / 0.5)  # Adjust for time
                
                # Send command
                self.controller.drone.send_rc_control(x_vel, y_vel, z_vel, 0)
                time.sleep(0.5)  # Move for half a second
                self._record_data_point()
                
                # Update angle for next step
                current_angle += angle_step
            
            # Stop at the end
            self.controller.drone.send_rc_control(0, 0, 0, 0)
            time.sleep(0.5)
            self._record_data_point()
            
            # Mark as successful if completed
            if not self.abort_flag:
                self.mission_data["success"] = True
                
        except Exception as e:
            print(f"Error executing spiral mission: {e}")
            self.mission_data["error"] = str(e)
        finally:
            # Ensure drone stops
            try:
                self.controller.drone.send_rc_control(0, 0, 0, 0)
            except:
                pass
            
            # Complete the mission
            self._complete_mission()


class TelloMissionController:
    """
    Manages and executes missions for Tello drone
    """
    
    def __init__(self, controller):
        """
        Initialize mission controller
        
        Args:
            controller: TelloDigitalTwin or compatible controller
        """
        self.controller = controller
        self.current_mission = None
        self.mission_queue = []
        self.mission_history = []
        
    def add_mission(self, mission):
        """
        Add a mission to the queue
        
        Args:
            mission: TelloMission instance
        """
        self.mission_queue.append(mission)
        return len(self.mission_queue)
    
    def clear_queue(self):
        """Clear mission queue"""
        self.mission_queue = []
    
    def start_mission(self, mission=None):
        """
        Start a mission
        
        Args:
            mission: TelloMission to start, or None to start next in queue
        """
        # If a mission is already running, abort it
        if self.current_mission and getattr(self.current_mission, 'thread', None) and self.current_mission.thread.is_alive():
            print("A mission is already running. Aborting current mission.")
            self.current_mission.abort()
            time.sleep(1)  # Give some time for the mission to abort
        
        # Determine which mission to start
        if mission is not None:
            self.current_mission = mission
        elif self.mission_queue:
            self.current_mission = self.mission_queue.pop(0)
        else:
            print("No mission to start")
            return False
        
        # Start the mission
        success = self.current_mission.start()
        if success:
            print(f"Started mission: {self.current_mission.name}")
            self.mission_history.append(self.current_mission)
        return success
    
    def abort_current_mission(self):
        """Abort the current mission"""
        if self.current_mission:
            return self.current_mission.abort()
        return False
    
    def create_mission_ui(self, parent):
        """
        Create UI for mission control
        
        Args:
            parent: Parent tkinter frame
            
        Returns:
            Frame with mission controls
        """
        # Create main frame
        mission_frame = ttk.Frame(parent)
        
        # Create mission selection
        select_frame = ttk.LabelFrame(mission_frame, text="Select Mission")
        select_frame.pack(fill=tk.X, expand=False, padx=10, pady=10)
        
        # Mission type
        ttk.Label(select_frame, text="Mission Type:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.mission_var = tk.StringVar(value="rectangle")
        mission_dropdown = ttk.Combobox(select_frame, textvariable=self.mission_var, 
                                        values=["rectangle", "spiral", "custom"])
        mission_dropdown.grid(row=0, column=1, padx=5, pady=5)
        
        # Parameters frame - will change based on mission type
        self.param_frame = ttk.LabelFrame(mission_frame, text="Mission Parameters")
        self.param_frame.pack(fill=tk.X, expand=False, padx=10, pady=10)
        
        # Create parameter fields
        self._update_param_fields("rectangle")
        
        # Update parameters when mission type changes
        mission_dropdown.bind("<<ComboboxSelected>>", 
                             lambda e: self._update_param_fields(mission_dropdown.get()))
        
        # Control buttons
        control_frame = ttk.Frame(mission_frame)
        control_frame.pack(fill=tk.X, expand=False, padx=10, pady=10)
        
        ttk.Button(control_frame, text="Add to Queue", 
                  command=self._add_current_to_queue).pack(side=tk.LEFT, padx=5)
                  
        ttk.Button(control_frame, text="Start Mission", 
                  command=self._start_current_mission).pack(side=tk.LEFT, padx=5)
                  
        ttk.Button(control_frame, text="Abort Mission", 
                  command=self.abort_current_mission).pack(side=tk.LEFT, padx=5)
                  
        ttk.Button(control_frame, text="Clear Queue", 
                  command=self.clear_queue).pack(side=tk.LEFT, padx=5)
        
        # Queue display
        queue_frame = ttk.LabelFrame(mission_frame, text="Mission Queue")
        queue_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.queue_listbox = tk.Listbox(queue_frame)
        self.queue_listbox.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        queue_scroll = ttk.Scrollbar(queue_frame, orient=tk.VERTICAL, command=self.queue_listbox.yview)
        queue_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.queue_listbox.configure(yscrollcommand=queue_scroll.set)
        
        return mission_frame
    
    def _update_param_fields(self, mission_type):
        """Update parameter fields based on mission type"""
        # Clear current parameters
        for widget in self.param_frame.winfo_children():
            widget.destroy()
            
        # Create parameters based on mission type
        if mission_type == "rectangle":
            # Width
            ttk.Label(self.param_frame, text="Width (m):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
            self.width_var = tk.StringVar(value="1.0")
            ttk.Entry(self.param_frame, textvariable=self.width_var, width=10).grid(row=0, column=1, padx=5, pady=5)
            
            # Height
            ttk.Label(self.param_frame, text="Height (m):").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
            self.height_var = tk.StringVar(value="0.8")
            ttk.Entry(self.param_frame, textvariable=self.height_var, width=10).grid(row=0, column=3, padx=5, pady=5)
            
            # Altitude
            ttk.Label(self.param_frame, text="Altitude (m):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
            self.altitude_var = tk.StringVar(value="1.0")
            ttk.Entry(self.param_frame, textvariable=self.altitude_var, width=10).grid(row=1, column=1, padx=5, pady=5)
            
        elif mission_type == "spiral":
            # Radius
            ttk.Label(self.param_frame, text="Radius (m):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
            self.radius_var = tk.StringVar(value="1.0")
            ttk.Entry(self.param_frame, textvariable=self.radius_var, width=10).grid(row=0, column=1, padx=5, pady=5)
            
            # Height
            ttk.Label(self.param_frame, text="Height (m):").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
            self.height_var = tk.StringVar(value="1.0")
            ttk.Entry(self.param_frame, textvariable=self.height_var, width=10).grid(row=0, column=3, padx=5, pady=5)
            
            # Rotations
            ttk.Label(self.param_frame, text="Rotations:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
            self.rotations_var = tk.StringVar(value="2")
            ttk.Entry(self.param_frame, textvariable=self.rotations_var, width=10).grid(row=1, column=1, padx=5, pady=5)
            
        elif mission_type == "custom":
            # Custom mission parameters
            ttk.Label(self.param_frame, text="Custom missions can be programmed in code").grid(
                row=0, column=0, columnspan=4, sticky=tk.W, padx=5, pady=20)
    
    def _add_current_to_queue(self):
        """Add current mission settings to queue"""
        mission_type = self.mission_var.get()
        
        try:
            if mission_type == "rectangle":
                width = float(self.width_var.get())
                height = float(self.height_var.get())
                altitude = float(self.altitude_var.get())
                
                mission = RectangleMission(self.controller, width, height, altitude)
                self.add_mission(mission)
                self.queue_listbox.insert(tk.END, f"Rectangle: {width}x{height}m at {altitude}m")
                
            elif mission_type == "spiral":
                radius = float(self.radius_var.get())
                height = float(self.height_var.get())
                rotations = float(self.rotations_var.get())
                
                mission = SpiralMission(self.controller, radius, height, rotations)
                self.add_mission(mission)
                self.queue_listbox.insert(tk.END, f"Spiral: r={radius}m, h={height}m, {rotations} rotations")
                
            elif mission_type == "custom":
                messagebox.showinfo("Custom Mission", 
                                  "Custom missions must be programmed directly. See TelloMissions.py")
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numbers for all parameters")
    
    def _start_current_mission(self):
        """Start mission with current settings"""
        mission_type = self.mission_var.get()
        
        try:
            if mission_type == "rectangle":
                width = float(self.width_var.get())
                height = float(self.height_var.get())
                altitude = float(self.altitude_var.get())
                
                mission = RectangleMission(self.controller, width, height, altitude)
                self.start_mission(mission)
                
            elif mission_type == "spiral":
                radius = float(self.radius_var.get())
                height = float(self.height_var.get())
                rotations = float(self.rotations_var.get())
                
                mission = SpiralMission(self.controller, radius, height, rotations)
                self.start_mission(mission)
                
            elif mission_type == "custom":
                messagebox.showinfo("Custom Mission", 
                                  "Custom missions must be programmed directly. See TelloMissions.py")
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numbers for all parameters")


def add_mission_controller(controller):
    """
    Add mission controller to a drone controller
    
    Args:
        controller: TelloDigitalTwin or compatible controller
        
    Returns:
        Updated controller with mission_controller attribute
    """
    controller.mission_controller = TelloMissionController(controller)
    return controller


if __name__ == "__main__":
    # Test with a mock controller
    class MockController:
        def __init__(self):
            self.x = 500
            self.y = 500
            self.z = 100
            self.is_flying = True
            self.flight_dir = "tello_data"
            self.battery = 100
            
            class MockDrone:
                def send_rc_control(self, *args):
                    print(f"Drone RC control: {args}")
            
            self.drone = MockDrone()
            
            # Create data directory
            if not os.path.exists(self.flight_dir):
                os.makedirs(self.flight_dir)
    
    # Create controller and add mission controller
    controller = MockController()
    controller = add_mission_controller(controller)
    
    # Create and start a rectangle mission
    mission = RectangleMission(controller, width=1.0, height=0.8, altitude=1.0)
    controller.mission_controller.start_mission(mission)
    
    # Wait for mission to complete
    time.sleep(10)
