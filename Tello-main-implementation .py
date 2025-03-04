import cv2
import numpy as np
import time
import os
import sys
import argparse
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
import math

# Import our Tello modules
try:
    from TelloDigitalTwin import TelloDigitalTwin
    from TelloML import TelloML, TelloDigitalTwinWithML
    from TelloMissions import add_mission_controller, TelloMissionController
except ImportError:
    print("Tello modules not found. Make sure the files are in the same directory.")
    sys.exit(1)

# Import keyboard module
try:
    import KeyPressModule as kp
except ImportError:
    print("KeyPressModule not found. Make sure it's in the same directory.")
    sys.exit(1)

class OptimalPathFinder:
    """
    Class for finding optimal paths to targets using gradient descent principles
    - Implements adaptive learning rate
    - Path optimization strategies
    - Obstacle avoidance
    """
    
    def __init__(self, controller):
        self.controller = controller
        self.learning_rate = 0.1
        self.momentum = 0.8
        self.max_iterations = 100
        self.convergence_threshold = 0.01
        self.previous_gradient = (0, 0)
        self.path_history = []
        
    def adaptive_learning_rate(self, iteration, max_iterations, min_lr=0.01, max_lr=0.5):
        """
        Implements an adaptive learning rate that starts higher and decreases
        - Similar to what's shown in the gradient descent visualization
        """
        # Exponential decay
        decay = 5 * iteration / max_iterations
        lr = max_lr * math.exp(-decay)
        return max(min_lr, lr)
    
    def find_path_to_target(self, start_pos, target_pos, obstacles=None):
        """
        Find optimal path from start to target using gradient descent principles
        - Adjusts learning rate based on progress
        - Uses momentum to avoid local minima
        - Avoids obstacles if provided
        """
        current_pos = start_pos
        path = [current_pos]
        velocities = [(0, 0)]  # For momentum
        
        for i in range(self.max_iterations):
            # Calculate gradient (direction to target)
            dx = target_pos[0] - current_pos[0]
            dy = target_pos[1] - current_pos[1]
            
            # Normalize gradient
            distance = math.sqrt(dx**2 + dy**2)
            
            # Check if we've reached the target
            if distance < self.convergence_threshold:
                break
                
            # Normalize direction
            if distance > 0:
                dx /= distance
                dy /= distance
            
            # Apply obstacle avoidance if needed
            if obstacles:
                dx, dy = self._avoid_obstacles(current_pos, dx, dy, obstacles)
            
            # Get adaptive learning rate
            lr = self.adaptive_learning_rate(i, self.max_iterations)
            
            # Apply momentum
            vx = self.momentum * velocities[-1][0] + lr * dx
            vy = self.momentum * velocities[-1][1] + lr * dy
            
            # Update position
            new_x = current_pos[0] + vx
            new_y = current_pos[1] + vy
            
            # Save new position and velocity
            current_pos = (new_x, new_y)
            velocities.append((vx, vy))
            path.append(current_pos)
            
            # Save for analysis
            self.path_history.append({
                'iteration': i,
                'position': current_pos,
                'learning_rate': lr,
                'distance': distance,
                'gradient': (dx, dy),
                'velocity': (vx, vy)
            })
        
        return path
    
    def _avoid_obstacles(self, pos, dx, dy, obstacles, avoidance_radius=1.0):
        """Apply repulsive forces from obstacles"""
        total_fx, total_fy = 0, 0
        
        for obstacle in obstacles:
            # Calculate distance to obstacle
            obs_x, obs_y = obstacle[:2]
            obs_radius = obstacle[2] if len(obstacle) > 2 else 0.5
            
            # Vector from position to obstacle
            to_obs_x = obs_x - pos[0]
            to_obs_y = obs_y - pos[1]
            dist = math.sqrt(to_obs_x**2 + to_obs_y**2)
            
            # Only apply force if within avoidance radius
            if dist < avoidance_radius + obs_radius:
                # Normalize
                if dist > 0:
                    to_obs_x /= dist
                    to_obs_y /= dist
                
                # Repulsive force (inversely proportional to distance)
                force = max(0, avoidance_radius + obs_radius - dist) / (avoidance_radius + obs_radius)
                
                # Accumulate forces
                total_fx -= to_obs_x * force * 2.0  # Weight repulsion higher
                total_fy -= to_obs_y * force * 2.0
        
        # Combine with original direction (with weights)
        combined_x = dx + total_fx
        combined_y = dy + total_fy
        
        # Normalize the combined vector
        magnitude = math.sqrt(combined_x**2 + combined_y**2)
        if magnitude > 0:
            combined_x /= magnitude
            combined_y /= magnitude
        
        return combined_x, combined_y
    
    def visualize_path_finding(self, start_pos, target_pos, obstacles=None, path=None):
        """Create visualization of the path finding process"""
        if not path:
            path = self.find_path_to_target(start_pos, target_pos, obstacles)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot path on the left
        path_array = np.array(path)
        ax1.plot(path_array[:, 0], path_array[:, 1], 'b-', alpha=0.7)
        ax1.scatter(path_array[:, 0], path_array[:, 1], c=range(len(path)), cmap='viridis', s=30)
        ax1.scatter(start_pos[0], start_pos[1], color='green', s=100, marker='o', label='Start')
        ax1.scatter(target_pos[0], target_pos[1], color='red', s=100, marker='*', label='Target')
        
        # Plot obstacles if any
        if obstacles:
            for obs in obstacles:
                obs_x, obs_y = obs[:2]
                obs_radius = obs[2] if len(obs) > 2 else 0.5
                circle = plt.Circle((obs_x, obs_y), obs_radius, color='red', alpha=0.5)
                ax1.add_patch(circle)
        
        ax1.set_title('Path Finding')
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.legend()
        ax1.grid(True)
        
        # Plot learning metrics on the right
        if self.path_history:
            iterations = [entry['iteration'] for entry in self.path_history]
            distances = [entry['distance'] for entry in self.path_history]
            learning_rates = [entry['learning_rate'] for entry in self.path_history]
            
            ax2.plot(iterations, distances, 'b-', label='Distance to Target')
            ax2.set_ylabel('Distance', color='b')
            ax2.tick_params(axis='y', labelcolor='b')
            ax2.set_title('Convergence Metrics')
            ax2.set_xlabel('Iteration')
            ax2.grid(True)
            
            # Twin axis for learning rate
            ax2_twin = ax2.twinx()
            ax2_twin.plot(iterations, learning_rates, 'r-', label='Learning Rate')
            ax2_twin.set_ylabel('Learning Rate', color='r')
            ax2_twin.tick_params(axis='y', labelcolor='r')
            
            # Add combined legend
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2_twin.get_legend_handles_labels()
            ax2_twin.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        return fig

class TelloController:
    """Main controller for Tello drone with GUI"""
    
    def __init__(self):
        # Parse command line arguments
        self.args = self._parse_arguments()
        
        # Initialize drone controller
        if self.args.basic:
            self.drone_controller = TelloDigitalTwin()
        else:
            self.drone_controller = TelloDigitalTwinWithML()
        
        # Add mission controller
        self.drone_controller = add_mission_controller(self.drone_controller)
        
        # Add path optimizer
        self.path_optimizer = OptimalPathFinder(self.drone_controller)
        
        # Initialize GUI if specified
        self.root = None
        self.app_running = True
        
        # Attempt to connect to the drone immediately if run-drone flag is set
        if self.args.run_drone:
            try:
                print("Attempting to connect to Tello drone...")
                self.drone_controller.connect()
                print("✓ Successfully connected to Tello drone")
                print(f"Battery level: {self.drone_controller.battery}%")
            except Exception as e:
                print(f"✗ Error connecting to Tello drone: {str(e)}")
                print("Please make sure:")
                print("  - The drone is powered on")
                print("  - Your computer is connected to the Tello WiFi network")
                print("  - The djitellopy package is installed")
                if not self.args.no_gui:
                    messagebox.showerror("Connection Error", 
                                      f"Error connecting to Tello drone:\n{str(e)}\n\n" +
                                      "Please make sure the drone is powered on and\n" +
                                      "your computer is connected to the Tello WiFi network.")
        
        if not self.args.no_gui:
            self._init_gui()
        
        # Start drone control in a separate thread
        if self.args.run_drone:
            self.drone_thread = threading.Thread(target=self._run_drone_controller)
            self.drone_thread.daemon = True
            self.drone_thread.start()
    
    def _parse_arguments(self):
        """Parse command line arguments"""
        parser = argparse.ArgumentParser(description="Tello Drone Controller with ML and Path Optimization")
        parser.add_argument('--basic', action='store_true', help='Run with basic controller (no ML)')
        parser.add_argument('--no-gui', action='store_true', help='Run without GUI')
        parser.add_argument('--analyze', action='store_true', help='Launch in analysis mode')
        parser.add_argument('--run-drone', action='store_true', help='Connect and run the drone')
        parser.add_argument('--data-dir', type=str, default='tello_data', help='Data directory')
        
        args = parser.parse_args()
        
        # Default to GUI and no drone connection unless specified
        if not (args.run_drone or args.analyze):
            args.run_drone = False
        
        return args
    
    def _init_gui(self):
        """Initialize the GUI"""
        self.root = tk.Tk()
        self.root.title("Tello Drone Digital Twin Controller")
        self.root.geometry("1280x800")
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        # Create main notebook
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.control_tab = ttk.Frame(self.notebook)
        self.mission_tab = ttk.Frame(self.notebook)
        self.data_tab = ttk.Frame(self.notebook)
        self.path_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.control_tab, text="Drone Control")
        self.notebook.add(self.mission_tab, text="Missions")
        self.notebook.add(self.data_tab, text="Flight Data")
        self.notebook.add(self.path_tab, text="Path Optimization")
        
        # Initialize tabs
        self._init_control_tab()
        self._init_mission_tab()
        self._init_data_tab()
        self._init_path_tab()
        
        # Start UI update loop
        self._update_ui()
    
    def _init_control_tab(self):
        """Initialize the drone control tab"""
        # Create frames
        info_frame = ttk.LabelFrame(self.control_tab, text="Drone Information")
        info_frame.pack(fill=tk.X, expand=False, padx=10, pady=10)
        
        video_frame = ttk.LabelFrame(self.control_tab, text="Video Feed")
        video_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        control_frame = ttk.Frame(self.control_tab)
        control_frame.pack(fill=tk.X, expand=False, padx=10, pady=10)
        
        # Info frame content
        self.info_labels = {}
        info_items = [
            ("status", "Status:"),
            ("battery", "Battery:"),
            ("position", "Position:"),
            ("altitude", "Altitude:"),
            ("speed", "Speed:"),
            ("mode", "Mode:")
        ]
        
        for i, (key, text) in enumerate(info_items):
            ttk.Label(info_frame, text=text).grid(row=i//3, column=(i%3)*2, sticky=tk.W, padx=5, pady=5)
            self.info_labels[key] = ttk.Label(info_frame, text="N/A")
            self.info_labels[key].grid(row=i//3, column=(i%3)*2+1, sticky=tk.W, padx=5, pady=5)
        
        # Video frame content
        self.video_label = ttk.Label(video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add placeholder for video
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "No Video Feed", (220, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        self._update_video(placeholder)
        
        # Control frame content
        control_buttons = [
            ("Connect", self._connect_drone),
            ("Takeoff", self._takeoff),
            ("Land", self._land),
            ("Emergency", self._emergency),
            ("Toggle Video", self._toggle_video)
        ]
        
        for i, (text, command) in enumerate(control_buttons):
            ttk.Button(control_frame, text=text, command=command).grid(row=0, column=i, padx=5, pady=5)
        
        # ML controls if available
        if hasattr(self.drone_controller, 'ml'):
            ml_frame = ttk.LabelFrame(self.control_tab, text="ML Features")
            ml_frame.pack(fill=tk.X, expand=False, padx=10, pady=10)
            
            ml_buttons = [
                ("Toggle Mapping", lambda: self._toggle_ml_feature('mapping')),
                ("Toggle Feature Extraction", lambda: self._toggle_ml_feature('features')),
                ("Toggle Autonomous Mode", lambda: self._toggle_ml_feature('autonomous')),
                ("Save Models", self._save_ml_models)
            ]
            
            for i, (text, command) in enumerate(ml_buttons):
                ttk.Button(ml_frame, text=text, command=command).grid(row=0, column=i, padx=5, pady=5)
    
    def _init_mission_tab(self):
        """Initialize the mission control tab"""
        # Use the mission controller's built-in UI
        mission_frame = self.drone_controller.mission_controller.create_mission_ui(self.mission_tab)
        mission_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add mission history
        history_frame = ttk.LabelFrame(self.mission_tab, text="Mission History")
        history_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create treeview for mission history
        columns = ("timestamp", "name", "type", "duration", "success")
        self.mission_tree = ttk.Treeview(history_frame, columns=columns, show="headings")
        
        # Define headings
        self.mission_tree.heading("timestamp", text="Time")
        self.mission_tree.heading("name", text="Mission")
        self.mission_tree.heading("type", text="Type")
        self.mission_tree.heading("duration", text="Duration")
        self.mission_tree.heading("success", text="Success")
        
        # Define columns
        self.mission_tree.column("timestamp", width=150)
        self.mission_tree.column("name", width=200)
        self.mission_tree.column("type", width=100)
        self.mission_tree.column("duration", width=100)
        self.mission_tree.column("success", width=100)
        
        self.mission_tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, command=self.mission_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.mission_tree.configure(yscrollcommand=scrollbar.set)
        
        # Add button to view selected mission
        button_frame = ttk.Frame(self.mission_tab)
        button_frame.pack(fill=tk.X, expand=False, padx=10, pady=10)
        
        ttk.Button(button_frame, text="View Selected Mission", 
                  command=self._view_selected_mission).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Refresh History", 
                  command=self._refresh_mission_history).pack(side=tk.LEFT, padx=5)
        
        # Initial load of mission history
        self._refresh_mission_history()
    
    def _init_data_tab(self):
        """Initialize the flight data analysis tab"""
        # Create frames
        select_frame = ttk.Frame(self.data_tab)
        select_frame.pack(fill=tk.X, expand=False, padx=10, pady=10)
        
        graph_frame = ttk.Frame(self.data_tab)
        graph_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Select frame content
        ttk.Label(select_frame, text="Select Flight Log:").pack(side=tk.LEFT, padx=5)
        
        self.log_var = tk.StringVar()
        self.log_dropdown = ttk.Combobox(select_frame, textvariable=self.log_var, width=50)
        self.log_dropdown.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(select_frame, text="Load", command=self._load_flight_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(select_frame, text="Refresh Logs", command=self._refresh_logs).pack(side=tk.LEFT, padx=5)
        
        # Graph frame content - will be populated when data is loaded
        self.graph_canvas = None
        
        # Initial load of logs
        self._refresh_logs()
    
    def _init_path_tab(self):
        """Initialize the path optimization tab"""
        # Create frames
        config_frame = ttk.LabelFrame(self.path_tab, text="Path Configuration")
        config_frame.pack(fill=tk.X, expand=False, padx=10, pady=10)
        
        obstacle_frame = ttk.LabelFrame(self.path_tab, text="Obstacles")
        obstacle_frame.pack(fill=tk.X, expand=False, padx=10, pady=10)
        
        vis_frame = ttk.Frame(self.path_tab)
        vis_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Config frame content
        param_frame = ttk.Frame(config_frame)
        param_frame.pack(fill=tk.X, expand=True, padx=5, pady=5)
        
        # Start position
        ttk.Label(param_frame, text="Start X:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.start_x_var = tk.StringVar(value="0.0")
        ttk.Entry(param_frame, textvariable=self.start_x_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(param_frame, text="Start Y:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        self.start_y_var = tk.StringVar(value="0.0")
        ttk.Entry(param_frame, textvariable=self.start_y_var, width=10).grid(row=0, column=3, padx=5, pady=5)
        
        # Target position
        ttk.Label(param_frame, text="Target X:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.target_x_var = tk.StringVar(value="5.0")
        ttk.Entry(param_frame, textvariable=self.target_x_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(param_frame, text="Target Y:").grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)
        self.target_y_var = tk.StringVar(value="5.0")
        ttk.Entry(param_frame, textvariable=self.target_y_var, width=10).grid(row=1, column=3, padx=5, pady=5)
        
        # Learning parameters
        param_frame2 = ttk.Frame(config_frame)
        param_frame2.pack(fill=tk.X, expand=True, padx=5, pady=5)
        
        ttk.Label(param_frame2, text="Initial Learning Rate:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.lr_var = tk.StringVar(value="0.5")
        ttk.Entry(param_frame2, textvariable=self.lr_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(param_frame2, text="Momentum:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        self.momentum_var = tk.StringVar(value="0.8")
        ttk.Entry(param_frame2, textvariable=self.momentum_var, width=10).grid(row=0, column=3, padx=5, pady=5)
        
        ttk.Label(param_frame2, text="Iterations:").grid(row=0, column=4, sticky=tk.W, padx=5, pady=5)
        self.iters_var = tk.StringVar(value="100")
        ttk.Entry(param_frame2, textvariable=self.iters_var, width=10).grid(row=0, column=5, padx=5, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(config_frame)
        button_frame.pack(fill=tk.X, expand=True, padx=5, pady=5)
        
        ttk.Button(button_frame, text="Use Current Position", 
                  command=self._use_current_position).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Calculate Optimal Path", 
                  command=self._calculate_optimal_path).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Execute Path", 
                  command=self._execute_optimal_path).pack(side=tk.LEFT, padx=5)
        
        # Obstacle frame content
        self.obstacles = []
        self.obstacle_listbox = tk.Listbox(obstacle_frame, width=50, height=5)
        self.obstacle_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        obstacle_buttons = ttk.Frame(obstacle_frame)
        obstacle_buttons.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        
        ttk.Button(obstacle_buttons, text="Add Obstacle", 
                  command=self._add_obstacle).pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Button(obstacle_buttons, text="Remove Selected", 
                  command=self._remove_obstacle).pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Button(obstacle_buttons, text="Clear All", 
                  command=self._clear_obstacles).pack(fill=tk.X, padx=5, pady=2)
        
        # Path visualization - will be populated when calculated
        self.path_canvas = None
        
    def _update_ui(self):
        """Update UI elements"""
        if not self.root:
            return
            
        # Update drone info if connected
        if hasattr(self.drone_controller, 'is_flying'):
            self.info_labels["status"].config(text="Connected" if self.drone_controller.drone else "Disconnected")
            self.info_labels["battery"].config(text=f"{self.drone_controller.battery}%")
            
            # Position
            x_pos = (self.drone_controller.x - 500) / 100
            y_pos = (self.drone_controller.y - 500) / 100
            self.info_labels["position"].config(text=f"({x_pos:.2f}, {y_pos:.2f})m")
            
            # Altitude
            self.info_labels["altitude"].config(text=f"{self.drone_controller.z/100:.2f}m")
            
            # Speed
            self.info_labels["speed"].config(text=f"{self.drone_controller.speed} cm/s")
            
            # Mode
            self.info_labels["mode"].config(text=self.drone_controller.mode)
        
        # Update video feed if available
        if hasattr(self.drone_controller, 'frame') and self.drone_controller.frame is not None:
            self._update_video(self.drone_controller.frame)
        
        # Schedule next update
        if self.app_running:
            self.root.after(100, self._update_ui)
    
    def _update_video(self, frame):
        """Update the video feed display"""
        # Resize frame to fit display
        frame = cv2.resize(frame, (640, 480))
        
        # Convert to format suitable for tkinter
        cv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cv_image)
        tk_image = ImageTk.PhotoImage(image=pil_image)
        
        # Update label
        self.video_label.config(image=tk_image)
        self.video_label.image = tk_image  # Keep a reference
    
    def _connect_drone(self):
        """Connect to the drone"""
        if not hasattr(self.drone_controller, 'drone'):
            messagebox.showerror("Error", "Drone controller not initialized")
            return
            
        try:
            self.drone_controller.drone.connect()
            messagebox.showinfo("Success", "Connected to Tello drone")
            
            # Update battery
            self.drone_controller.battery = self.drone_controller.drone.get_battery()
        except Exception as e:
            messagebox.showerror("Connection Error", f"Failed to connect: {str(e)}")
    
    def _takeoff(self):
        """Command drone to take off"""
        if not hasattr(self.drone_controller, 'drone'):
            messagebox.showerror("Error", "Drone controller not initialized")
            return
            
        try:
            self.drone_controller.drone.takeoff()
            self.drone_controller.is_flying = True
            self.drone_controller.z = 100  # Approximate takeoff height
            messagebox.showinfo("Success", "Takeoff command sent")
        except Exception as e:
            messagebox.showerror("Takeoff Error", f"Failed to take off: {str(e)}")
    
    def _land(self):
        """Command drone to land"""
        if not hasattr(self.drone_controller, 'drone'):
            messagebox.showerror("Error", "Drone controller not initialized")
            return
            
        try:
            self.drone_controller.drone.land()
            self.drone_controller.is_flying = False
            self.drone_controller.z = 0
            messagebox.showinfo("Success", "Land command sent")
        except Exception as e:
            messagebox.showerror("Land Error", f"Failed to land: {str(e)}")
    
    def _emergency(self):
        """Emergency stop"""
        if not hasattr(self.drone_controller, 'drone'):
            messagebox.showerror("Error", "Drone controller not initialized")
            return
            
        try:
            self.drone_controller.drone.emergency()
            self.drone_controller.is_flying = False
            messagebox.showinfo("Emergency", "Emergency stop command sent")
        except Exception as e:
            messagebox.showerror("Emergency Error", f"Failed to execute emergency stop: {str(e)}")
    
    def _toggle_video(self):
        """Toggle video stream"""
        if not hasattr(self.drone_controller, 'drone'):
            messagebox.showerror("Error", "Drone controller not initialized")
            return
            
        try:
            if self.drone_controller.video_stream:
                self.drone_controller.drone.streamoff()
                self.drone_controller.video_stream = False
                messagebox.showinfo("Video", "Video stream turned off")
            else:
                self.drone_controller.drone.streamon()
                self.drone_controller.video_stream = True
                messagebox.showinfo("Video", "Video stream turned on")
        except Exception as e:
            messagebox.showerror("Video Error", f"Failed to toggle video: {str(e)}")
    
    def _toggle_ml_feature(self, feature):
        """Toggle an ML feature on/off"""
        if not hasattr(self.drone_controller, 'ml'):
            messagebox.showerror("Error", "ML features not available")
            return
        
        if feature == 'mapping':
            self.drone_controller.mapping_active = not self.drone_controller.mapping_active
            status = "activated" if self.drone_controller.mapping_active else "deactivated"
            messagebox.showinfo("ML Feature", f"Mapping {status}")
            
        elif feature == 'features':
            self.drone_controller.feature_extraction_active = not self.drone_controller.feature_extraction_active
            status = "activated" if self.drone_controller.feature_extraction_active else "deactivated"
            messagebox.showinfo("ML Feature", f"Feature extraction {status}")
            
        elif feature == 'autonomous':
            self.drone_controller.autonomous_mode = not self.drone_controller.autonomous_mode
            status = "activated" if self.drone_controller.autonomous_mode else "deactivated"
            messagebox.showinfo("ML Feature", f"Autonomous mode {status}")
    
    def _save_ml_models(self):
        """Save ML models to disk"""
        if not hasattr(self.drone_controller, 'ml'):
            messagebox.showerror("Error", "ML features not available")
            return
            
        try:
            success = self.drone_controller.ml.save_models()
            if success:
                messagebox.showinfo("ML Models", "Models saved successfully")
            else:
                messagebox.showerror("ML Models", "Failed to save models")
        except Exception as e:
            messagebox.showerror("ML Error", f"Error saving models: {str(e)}")
    
    def _refresh_mission_history(self):
        """Refresh the mission history tree"""
        # Clear current items
        for item in self.mission_tree.get_children():
            self.mission_tree.delete(item)
            
        # Get mission directory
        missions_dir = os.path.join(self.drone_controller.flight_dir, "missions")
        if not os.path.exists(missions_dir):
            return
            
        # Find all mission files
        mission_files = [f for f in os.listdir(missions_dir) if f.endswith('.json')]
        mission_files.sort(reverse=True)  # Newest first
        
        # Load and display missions
        for file in mission_files:
            try:
                with open(os.path.join(missions_dir, file), 'r') as f:
                    mission_data = json.load(f)
                    
                # Extract data
                timestamp = mission_data.get('timestamp', 'Unknown')
                name = mission_data.get('name', 'Unnamed')
                mission_type = mission_data.get('mission_type', 'Unknown')
                duration = f"{mission_data.get('duration', 0):.1f}s"
                success = "Yes" if mission_data.get('success', False) else "No"
                
                # Add to tree
                self.mission_tree.insert("", "end", values=(timestamp, name, mission_type, duration, success))
                
            except Exception as e:
                print(f"Error loading mission file {file}: {e}")
    
    def _view_selected_mission(self):
        """View details of the selected mission"""
        selected = self.mission_tree.selection()
        if not selected:
            messagebox.showinfo("Selection", "Please select a mission to view")
            return
            
        # Get selected item values
        item = self.mission_tree.item(selected[0])
        values = item['values']
        
        # Find mission file
        missions_dir = os.path.join(self.drone_controller.flight_dir, "missions")
        mission_files = [f for f in os.listdir(missions_dir) if f.endswith('.json')]
        
        # Search for matching timestamp
        target_timestamp = values[0]
        mission_file = None
        
        for file in mission_files:
            try:
                with open(os.path.join(missions_dir, file), 'r') as f:
                    data = json.load(f)
                    if data.get('timestamp') == target_timestamp:
                        mission_file = file
                        mission_data = data
                        break
            except:
                pass
                
        if not mission_file:
            messagebox.showerror("Error", "Could not find mission file")
            return
            
        # Create popup window to display mission details
        popup = tk.Toplevel(self.root)
        popup.title(f"Mission: {values[1]}")
        popup.geometry("800x600")
        
        # Create figure for visualization
        fig = plt.figure(figsize=(10, 8))
        
        # Extract mission data
        path = np.array(mission_data.get('path', []))
        altitude = np.array(mission_data.get('altitude', []))
        timestamps = np.array(mission_data.get('timestamps', []))
        battery = np.array(mission_data.get('battery', []))
        
        if len(path) > 1:
            # Plot 2D path
            ax1 = fig.add_subplot(2, 2, 1)
            ax1.plot(path[:, 0], path[:, 1], 'b-')
            ax1.scatter(path[0, 0], path[0, 1], color='green', s=100, marker='o', label='Start')
            ax1.scatter(path[-1, 0], path[-1, 1], color='red', s=100, marker='x', label='End')
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.set_title('Flight Path (Top View)')
            ax1.legend()
            ax1.grid(True)
            
            # Plot altitude
            if len(altitude) > 0:
                ax2 = fig.add_subplot(2, 2, 2)
                if len(timestamps) == len(altitude):
                    ax2.plot(timestamps, altitude, 'r-')
                    ax2.set_xlabel('Time (s)')
                else:
                    ax2.plot(altitude, 'r-')
                    ax2.set_xlabel('Sample')
                ax2.set_ylabel('Altitude (m)')
                ax2.set_title('Altitude Profile')
                ax2.grid(True)
                
            # Plot 3D path
            if len(altitude) == len(path):
                ax3 = fig.add_subplot(2, 2, 3, projection='3d')
                ax3.plot(path[:, 0], path[:, 1], altitude, 'b-')
                ax3.scatter(path[0, 0], path[0, 1], altitude[0], color='green', s=100, marker='o', label='Start')
                ax3.scatter(path[-1, 0], path[-1, 1], altitude[-1], color='red', s=100, marker='x', label='End')
                ax3.set_xlabel('X (m)')
                ax3.set_ylabel('Y (m)')
                ax3.set_zlabel('Z (m)')
                ax3.set_title('3D Flight Path')
                
            # Plot battery
            if len(battery) > 0 and len(timestamps) == len(battery):
                ax4 = fig.add_subplot(2, 2, 4)
                ax4.plot(timestamps, battery, 'g-')
                ax4.set_xlabel('Time (s)')
                ax4.set_ylabel('Battery (%)')
                ax4.set_title('Battery Level')
                ax4.set_ylim(0, 100)
                ax4.grid(True)
                
            plt.tight_layout()
            
            # Embed in Tkinter
            canvas = FigureCanvasTkAgg(fig, master=popup)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        else:
            ttk.Label(popup, text="No flight path data available").pack(pady=20)
    
    def _refresh_logs(self):
        """Refresh flight logs dropdown"""
        # Get flight directory
        flight_dir = self.drone_controller.flight_dir
        if not os.path.exists(flight_dir):
            return
            
        # Find all flight logs
        log_files = [f for f in os.listdir(flight_dir) if f.startswith('flight_') and f.endswith('.json')]
        log_files.sort(reverse=True)  # Newest first
        
        # Update dropdown
        self.log_dropdown['values'] = log_files
        
        if log_files:
            self.log_dropdown.current(0)
    
    def _load_flight_data(self):
        """Load and display selected flight data"""
        selected_log = self.log_var.get()
        if not selected_log:
            messagebox.showinfo("Selection", "Please select a flight log")
            return
            
        # Load flight data
        try:
            with open(os.path.join(self.drone_controller.flight_dir, selected_log), 'r') as f:
                flight_data = json.load(f)
                
            # Extract data
            path = np.array(flight_data.get('path', []))
            altitude = np.array(flight_data.get('altitude', []))
            timestamps = np.array(flight_data.get('timestamps', []))
            battery = np.array(flight_data.get('battery', []))
            
            # Create figure
            fig = plt.figure(figsize=(10, 8))
            
            # Plot 2D path
            ax1 = fig.add_subplot(2, 2, 1)
            ax1.plot(path[:, 0], path[:, 1], 'b-')
            ax1.scatter(path[0, 0], path[0, 1], color='green', s=100, marker='o', label='Start')
            ax1.scatter(path[-1, 0], path[-1, 1], color='red', s=100, marker='x', label='End')
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.set_title('Flight Path (Top View)')
            ax1.legend()
            ax1.grid(True)
            
            # Plot altitude
            if len(altitude) > 0:
                ax2 = fig.add_subplot(2, 2, 2)
                if len(timestamps) == len(altitude):
                    ax2.plot(timestamps, altitude, 'r-')
                    ax2.set_xlabel('Time (s)')
                else:
                    ax2.plot(altitude, 'r-')
                    ax2.set_xlabel('Sample')
                ax2.set_ylabel('Altitude (m)')
                ax2.set_title('Altitude Profile')
                ax2.grid(True)
                
            # Plot 3D path
            if len(altitude) == len(path):
                ax3 = fig.add_subplot(2, 2, 3, projection='3d')
                ax3.plot(path[:, 0], path[:, 1], altitude, 'b-')
                ax3.scatter(path[0, 0], path[0, 1], altitude[0], color='green', s=100, marker='o', label='Start')
                ax3.scatter(path[-1, 0], path[-1, 1], altitude[-1], color='red', s=100, marker='x', label='End')
                ax3.set_xlabel('X (m)')
                ax3.set_ylabel('Y (m)')
                ax3.set_zlabel('Z (m)')
                ax3.set_title('3D Flight Path')
                
            # Plot battery
            if len(battery) > 0 and len(timestamps) == len(battery):
                ax4 = fig.add_subplot(2, 2, 4)
                ax4.plot(timestamps, battery, 'g-')
                ax4.set_xlabel('Time (s)')
                ax4.set_ylabel('Battery (%)')
                ax4.set_title('Battery Level')
                ax4.set_ylim(0, 100)
                ax4.grid(True)
                
            plt.tight_layout()
            
            # Remove old canvas if it exists
            if self.graph_canvas:
                self.graph_canvas.get_tk_widget().destroy()
                
            # Create new canvas
            self.graph_canvas = FigureCanvasTkAgg(fig, master=self.data_tab)
            self.graph_canvas.draw()
            self.graph_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load flight data: {str(e)}")
    
    def _add_obstacle(self):
        """Add obstacle for path planning"""
        # Create popup for obstacle parameters
        popup = tk.Toplevel(self.root)
        popup.title("Add Obstacle")
        popup.geometry("300x200")
        
        # Parameters
        ttk.Label(popup, text="X Position (m):").grid(row=0, column=0, sticky=tk.W, padx=10, pady=10)
        x_var = tk.StringVar(value="2.5")
        ttk.Entry(popup, textvariable=x_var, width=10).grid(row=0, column=1, padx=10, pady=10)
        
        ttk.Label(popup, text="Y Position (m):").grid(row=1, column=0, sticky=tk.W, padx=10, pady=10)
        y_var = tk.StringVar(value="2.5")
        ttk.Entry(popup, textvariable=y_var, width=10).grid(row=1, column=1, padx=10, pady=10)
        
        ttk.Label(popup, text="Radius (m):").grid(row=2, column=0, sticky=tk.W, padx=10, pady=10)
        r_var = tk.StringVar(value="0.5")
        ttk.Entry(popup, textvariable=r_var, width=10).grid(row=2, column=1, padx=10, pady=10)
        
        # Function to add the obstacle
        def add():
            try:
                x = float(x_var.get())
                y = float(y_var.get())
                r = float(r_var.get())
                
                self.obstacles.append((x, y, r))
                self.obstacle_listbox.insert(tk.END, f"Obstacle at ({x}, {y}) with radius {r}m")
                popup.destroy()
                
            except ValueError:
                messagebox.showerror("Input Error", "Please enter valid numbers")
        
        ttk.Button(popup, text="Add", command=add).grid(row=3, column=0, columnspan=2, pady=10)
    
    def _remove_obstacle(self):
        """Remove selected obstacle"""
        selected = self.obstacle_listbox.curselection()
        if not selected:
            return
            
        # Remove from list and listbox
        index = selected[0]
        if 0 <= index < len(self.obstacles):
            self.obstacles.pop(index)
            self.obstacle_listbox.delete(index)
    
    def _clear_obstacles(self):
        """Clear all obstacles"""
        self.obstacles = []
        self.obstacle_listbox.delete(0, tk.END)
    
    def _use_current_position(self):
        """Use current drone position as start position"""
        if not hasattr(self.drone_controller, 'x'):
            messagebox.showinfo("Drone Position", "Drone position not available")
            return
            
        # Get current position
        x_pos = (self.drone_controller.x - 500) / 100
        y_pos = (self.drone_controller.y - 500) / 100
        
        # Update entry fields
        self.start_x_var.set(f"{x_pos:.2f}")
        self.start_y_var.set(f"{y_pos:.2f}")
    
    def _calculate_optimal_path(self):
        """Calculate optimal path using gradient descent"""
        try:
            # Get parameters
            start_x = float(self.start_x_var.get())
            start_y = float(self.start_y_var.get())
            target_x = float(self.target_x_var.get())
            target_y = float(self.target_y_var.get())
            
            # Update path optimizer settings
            self.path_optimizer.learning_rate = float(self.lr_var.get())
            self.path_optimizer.momentum = float(self.momentum_var.get())
            self.path_optimizer.max_iterations = int(self.iters_var.get())
            
            # Calculate path
            start_pos = (start_x, start_y)
            target_pos = (target_x, target_y)
            
            # Generate visualization
            fig = self.path_optimizer.visualize_path_finding(start_pos, target_pos, self.obstacles)
            
            # Remove old canvas if it exists
            if self.path_canvas:
                self.path_canvas.get_tk_widget().destroy()
                
            # Create new canvas
            self.path_canvas = FigureCanvasTkAgg(fig, master=self.path_tab)
            self.path_canvas.draw()
            self.path_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            messagebox.showinfo("Path Calculation", "Optimal path calculated successfully")
            
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numbers for all fields")
        except Exception as e:
            messagebox.showerror("Calculation Error", f"Error calculating path: {str(e)}")
    
    def _execute_optimal_path(self):
        """Execute the calculated optimal path on the drone"""
        if not hasattr(self.drone_controller, 'drone') or not self.drone_controller.is_flying:
            messagebox.showerror("Drone Error", "Drone is not connected or not flying")
            return
            
        if not hasattr(self.path_optimizer, 'path_history') or not self.path_optimizer.path_history:
            messagebox.showinfo("Path", "Please calculate a path first")
            return
            
        # Get path points
        path = [(entry['position'][0], entry['position'][1]) for entry in self.path_optimizer.path_history]
        
        # Create a custom mission for this path
        from TelloMissions import TelloMission
        
        class OptimalPathMission(TelloMission):
            def __init__(self, controller, path):
                super().__init__(controller, name="Optimal Path Execution")
                self.path = path
                self.mission_data["mission_type"] = "optimal_path"
                self.mission_data["parameters"] = {"path_length": len(path)}
                
            def _execute_mission(self):
                try:
                    # Record initial state
                    self._record_data_point()
                    print(f"Starting optimal path mission with {len(self.path)} waypoints")
                    
                    # Skip very close points to create a more efficient path
                    simplified_path = [self.path[0]]
                    for point in self.path[1:]:
                        # Only add point if it's significantly different from the last one
                        last_point = simplified_path[-1]
                        distance = math.sqrt((point[0] - last_point[0])**2 + (point[1] - last_point[1])**2)
                        if distance > 0.2:  # Only include points more than 20cm apart
                            simplified_path.append(point)
                    
                    print(f"Simplified path to {len(simplified_path)} waypoints")
                    
                    # Execute path
                    for i, waypoint in enumerate(simplified_path):
                        if self.abort_flag:
                            break
                            
                        print(f"Moving to waypoint {i+1}/{len(simplified_path)}: {waypoint}")
                        
                        # Convert to drone coordinates
                        target_x = waypoint[0] * 100 + 500
                        target_y = waypoint[1] * 100 + 500
                        
                        # Calculate vector to waypoint
                        current_x, current_y = self.controller.x, self.controller.y
                        dx = target_x - current_x
                        dy = target_y - current_y
                        distance = math.sqrt(dx**2 + dy**2)
                        
                        if distance < 30:  # Skip if already close
                            continue
                            
                        # Calculate angle to waypoint
                        angle_to_point = math.degrees(math.atan2(dy, dx))
                        target_angle = (90 - angle_to_point) % 360
                        
                        # Calculate yaw difference
                        yaw_diff = (target_angle - self.controller.yaw) % 360
                        if yaw_diff > 180:
                            yaw_diff -= 360
                            
                        # Rotate to face waypoint
                        if abs(yaw_diff) > 15:
                            print(f"Rotating {yaw_diff} degrees")
                            yaw_speed = min(30, max(-30, int(yaw_diff / 3)))
                            self.controller.drone.send_rc_control(0, 0, 0, yaw_speed)
                            time.sleep(1)
                            self._record_data_point()
                            self.controller.drone.send_rc_control(0, 0, 0, 0)
                            time.sleep(0.5)
                        
                        # Move towards waypoint
                        duration = distance / (self.controller.speed * 100)  # seconds
                        duration = min(2.0, max(0.5, duration))  # Limit between 0.5-2s
                        
                        print(f"Moving forward for {duration:.1f}s")
                        self.controller.drone.send_rc_control(0, min(self.controller.speed, 30), 0, 0)
                        
                        # Track during movement
                        move_start = time.time()
                        while time.time() - move_start < duration and not self.abort_flag:
                            time.sleep(0.1)
                            self._record_data_point()
                            
                        # Stop at waypoint
                        self.controller.drone.send_rc_control(0, 0, 0, 0)
                        time.sleep(0.5)
                        
                    # Final stop at end of path
                    self.controller.drone.send_rc_control(0, 0, 0, 0)
                    self._record_data_point()
                    print("Path execution completed")
                    
                    self.mission_data["success"] = True
                    
                except Exception as e:
                    print(f"Path execution error: {e}")
                    self.mission_data["error"] = str(e)
                finally:
                    # Ensure drone stops
                    try:
                        self.controller.drone.send_rc_control(0, 0, 0, 0)
                    except:
                        pass
                    self._complete_mission()
        
        # Create and start the mission
        mission = OptimalPathMission(self.drone_controller, path)
        mission.start()
        messagebox.showinfo("Mission", "Optimal path execution started")
    
    def _run_drone_controller(self):
        """Run the drone controller in a separate thread"""
        try:
            self.drone_controller.run()
        except Exception as e:
            print(f"Drone controller error: {e}")
    
    def _on_closing(self):
        """Handle window closing"""
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.app_running = False
            
            # Land drone if flying
            if hasattr(self.drone_controller, 'is_flying') and self.drone_controller.is_flying:
                try:
                    print("Landing drone before exit...")
                    self.drone_controller.drone.land()
                except:
                    pass
            
            # Close the window
            if self.root:
                self.root.destroy()
    
    def run(self):
        """Run the application"""
        if self.root:
            self.root.mainloop()
        else:
            # Run without GUI
            print("Running in headless mode...")
            try:
                self.drone_controller.run()
            except KeyboardInterrupt:
                print("Program interrupted")
            finally:
                # Land drone if flying
                if hasattr(self.drone_controller, 'is_flying') and self.drone_controller.is_flying:
                    try:
                        print("Landing drone...")
                        self.drone_controller.drone.land()
                    except:
                        pass


if __name__ == "__main__":
    controller = TelloController()
    controller.run()
