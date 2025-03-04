"""
Tello Drone Path Optimization Using Gradient Descent Principles

This module implements path optimization techniques for the Tello drone
based on principles from gradient descent optimization, similar to what's
shown in the visualization of learning rates in Stochastic Gradient Descent.
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class PathOptimizer:
    """
    Path optimizer using gradient descent principles to find optimal routes
    for drone navigation.
    
    The implementation is inspired by how gradient descent minimizes loss functions
    in machine learning, where:
    - The "loss" is the distance to the target
    - The "learning rate" controls how large steps we take
    - We use momentum to avoid "getting stuck" in local minima (obstacles)
    - Adaptive learning rates help balance between speed and stability
    """
    
    def __init__(self):
        """Initialize the path optimizer with default parameters"""
        # Learning rate parameters
        self.initial_learning_rate = 0.5  # Initial step size (higher = faster convergence)
        self.min_learning_rate = 0.01     # Minimum step size (for stability)
        self.decay_factor = 5.0           # How quickly learning rate decays
        
        # Momentum parameters
        self.momentum = 0.8               # Momentum coefficient (higher = smoother paths)
        
        # Convergence parameters
        self.max_iterations = 100         # Maximum number of steps to try
        self.convergence_threshold = 0.01 # Stop when we get this close to target
        
        # Obstacle avoidance
        self.avoidance_strength = 2.0     # Strength of obstacle repulsion
        self.avoidance_radius = 1.0       # How far to stay from obstacles
        
        # Energy efficiency
        self.energy_weight = 0.01         # Weight for energy conservation
        
        # History for analysis
        self.optimization_history = []
        
    def adaptive_learning_rate(self, iteration, remaining_distance=None):
        """
        Calculate adaptive learning rate based on iteration and distance
        
        This implements a decaying learning rate schedule:
        - Start with high learning rate for fast initial progress
        - Gradually decrease to avoid overshooting the target
        - Optionally adjust based on remaining distance
        """
        # Basic exponential decay
        progress = iteration / self.max_iterations
        lr = self.initial_learning_rate * math.exp(-self.decay_factor * progress)
        
        # Adjust based on remaining distance if provided
        if remaining_distance is not None:
            # Slow down more when we're close to the target
            distance_factor = min(1.0, remaining_distance)
            lr *= distance_factor
        
        # Ensure we don't go below minimum
        return max(self.min_learning_rate, lr)
    
    def find_optimal_path(self, start_pos, target_pos, obstacles=None, terrain=None):
        """
        Find an optimal path from start to target position
        
        Args:
            start_pos: (x,y) tuple with starting position
            target_pos: (x,y) tuple with target position
            obstacles: List of (x,y,radius) tuples representing obstacles
            terrain: Optional 2D array with terrain costs
            
        Returns:
            List of (x,y) tuples representing waypoints along the path
        """
        # Initialize current position and path
        current_pos = np.array(start_pos, dtype=float)
        path = [current_pos.copy()]
        
        # Initialize velocity (for momentum)
        velocity = np.zeros(2, dtype=float)
        
        # Reset history
        self.optimization_history = []
        
        # Main optimization loop
        for iteration in range(self.max_iterations):
            # Calculate vector to target
            target = np.array(target_pos, dtype=float)
            to_target = target - current_pos
            distance = np.linalg.norm(to_target)
            
            # Check if we've reached the target
            if distance < self.convergence_threshold:
                break
                
            # Normalize direction vector
            direction = to_target / max(distance, 1e-10)
            
            # Calculate learning rate for this step
            lr = self.adaptive_learning_rate(iteration, distance)
            
            # Apply obstacle avoidance if needed
            if obstacles:
                # Calculate repulsive forces from obstacles
                repulsive_force = np.zeros(2, dtype=float)
                
                for obstacle in obstacles:
                    obs_pos = np.array(obstacle[:2], dtype=float)
                    obs_radius = obstacle[2] if len(obstacle) > 2 else 0.5
                    
                    # Vector from current position to obstacle
                    to_obstacle = obs_pos - current_pos
                    obs_distance = np.linalg.norm(to_obstacle)
                    
                    # Only apply force if within avoidance radius
                    avoidance_distance = self.avoidance_radius + obs_radius
                    if obs_distance < avoidance_distance:
                        # Normalize
                        obs_direction = to_obstacle / max(obs_distance, 1e-10)
                        
                        # Repulsive force (stronger when closer)
                        force_magnitude = max(0, avoidance_distance - obs_distance) / avoidance_distance
                        repulsive_force -= obs_direction * force_magnitude * self.avoidance_strength
                
                # Combine with goal direction
                combined_direction = direction + repulsive_force
                # Normalize the combined direction
                combined_direction = combined_direction / max(np.linalg.norm(combined_direction), 1e-10)
                direction = combined_direction
            
            # Apply terrain costs if provided
            if terrain is not None:
                # Convert current position to terrain grid coordinates
                grid_x = int(current_pos[0])
                grid_y = int(current_pos[1])
                
                # Check boundaries
                if 0 <= grid_x < terrain.shape[1] and 0 <= grid_y < terrain.shape[0]:
                    # Get terrain cost at current position
                    terrain_cost = terrain[grid_y, grid_x]
                    
                    # Reduce learning rate in high-cost areas
                    lr *= max(0.1, 1.0 - terrain_cost)
            
            # Apply momentum
            velocity = self.momentum * velocity + lr * direction
            
            # Apply energy efficiency constraint
            # (penalize large changes in velocity to save energy)
            if len(path) >= 2:
                prev_velocity = path[-1] - path[-2]
                velocity_change = velocity - prev_velocity
                energy_penalty = self.energy_weight * np.linalg.norm(velocity_change)
                velocity_magnitude = max(0.1, np.linalg.norm(velocity))
                velocity = velocity * (1.0 - energy_penalty / velocity_magnitude)
            
            # Update position
            new_pos = current_pos + velocity
            
            # Save for path
            current_pos = new_pos.copy()
            path.append(current_pos.copy())
            
            # Save history for analysis
            self.optimization_history.append({
                'iteration': iteration,
                'position': current_pos.copy(),
                'velocity': velocity.copy(),
                'learning_rate': lr,
                'distance': distance
            })
        
        return path
    
    def visualize_optimization(self, start_pos, target_pos, obstacles=None, path=None, terrain=None):
        """
        Create visualization of the path optimization process
        
        Args:
            start_pos: (x,y) tuple with starting position
            target_pos: (x,y) tuple with target position
            obstacles: List of (x,y,radius) tuples representing obstacles
            path: Pre-computed path (or None to compute a new one)
            terrain: Optional 2D array with terrain costs
            
        Returns:
            Matplotlib figure with the visualization
        """
        if path is None:
            path = self.find_optimal_path(start_pos, target_pos, obstacles, terrain)
        
        # Convert path to numpy array for easier plotting
        path_array = np.array(path)
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(15, 8))
        
        # Main path plot
        ax1 = fig.add_subplot(1, 2, 1)
        
        # Plot path
        ax1.plot(path_array[:, 0], path_array[:, 1], 'b-', alpha=0.6)
        ax1.scatter(path_array[:, 0], path_array[:, 1], c=range(len(path)), cmap='viridis', s=30)
        
        # Plot start and target
        ax1.scatter(start_pos[0], start_pos[1], color='green', s=100, marker='o', label='Start')
        ax1.scatter(target_pos[0], target_pos[1], color='red', s=100, marker='*', label='Target')
        
        # Plot obstacles if any
        if obstacles:
            for obstacle in obstacles:
                obs_x, obs_y = obstacle[:2]
                obs_radius = obstacle[2] if len(obstacle) > 2 else 0.5
                
                circle = plt.Circle((obs_x, obs_y), obs_radius, color='red', alpha=0.3)
                ax1.add_patch(circle)
                
                # Also show avoidance radius
                avoidance_circle = plt.Circle(
                    (obs_x, obs_y), 
                    obs_radius + self.avoidance_radius, 
                    color='red', 
                    alpha=0.1,
                    linestyle='--',
                    fill=False
                )
                ax1.add_patch(avoidance_circle)
        
        # Plot terrain if provided
        if terrain is not None:
            terrain_extent = (0, terrain.shape[1], 0, terrain.shape[0])
            terrain_plot = ax1.imshow(
                terrain, 
                origin='lower', 
                extent=terrain_extent,
                alpha=0.3, 
                cmap='terrain'
            )
            plt.colorbar(terrain_plot, ax=ax1, label='Terrain Cost')
        
        ax1.set_title('Path Optimization')
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.legend()
        ax1.grid(True)
        ax1.axis('equal')
        
        # Plot optimization metrics
        if self.optimization_history:
            ax2 = fig.add_subplot(2, 2, 2)
            ax3 = fig.add_subplot(2, 2, 4)
            
            iterations = [entry['iteration'] for entry in self.optimization_history]
            distances = [entry['distance'] for entry in self.optimization_history]
            learning_rates = [entry['learning_rate'] for entry in self.optimization_history]
            velocities = [np.linalg.norm(entry['velocity']) for entry in self.optimization_history]
            
            # Distance plot (similar to loss curve in ML)
            ax2.plot(iterations, distances, 'b-', label='Distance to Target')
            ax2.set_title('Distance to Target vs. Iteration')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Distance')
            ax2.grid(True)
            
            # Learning rate and velocity plot
            ax3.plot(iterations, learning_rates, 'r-', label='Learning Rate')
            ax3.set_xlabel('Iteration')
            ax3.set_ylabel('Learning Rate', color='r')
            ax3.tick_params(axis='y', labelcolor='r')
            
            # Twin axis for velocity
            ax3_twin = ax3.twinx()
            ax3_twin.plot(iterations, velocities, 'g-', label='Velocity')
            ax3_twin.set_ylabel('Velocity', color='g')
            ax3_twin.tick_params(axis='y', labelcolor='g')
            
            ax3.set_title('Learning Rate and Velocity')
            
            # Add combined legend
            lines1, labels1 = ax3.get_legend_handles_labels()
            lines2, labels2 = ax3_twin.get_legend_handles_labels()
            ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        return fig
    
    def create_3d_terrain(self, size=50, complexity=5):
        """
        Create a procedural 3D terrain for testing path planning
        
        Args:
            size: Size of the terrain grid
            complexity: Number of random hills/valleys
            
        Returns:
            2D numpy array with terrain height/cost
        """
        terrain = np.zeros((size, size))
        
        # Add random hills and valleys
        for _ in range(complexity):
            # Random center
            cx = np.random.randint(0, size)
            cy = np.random.randint(0, size)
            
            # Random radius
            radius = np.random.randint(5, size // 2)
            
            # Random height/depth (-1 to 1)
            height = np.random.uniform(-1, 1)
            
            # Create hill or valley
            for x in range(max(0, cx - radius), min(size, cx + radius)):
                for y in range(max(0, cy - radius), min(size, cy + radius)):
                    distance = math.sqrt((x - cx)**2 + (y - cy)**2)
                    if distance < radius:
                        # Smooth falloff at edges
                        factor = (1 - distance / radius)**2
                        terrain[y, x] += height * factor
        
        # Normalize to range [0, 1]
        terrain = (terrain - np.min(terrain)) / (np.max(terrain) - np.min(terrain) + 1e-10)
        
        return terrain
    
    def execute_path_on_drone(self, drone_controller, path, speed=30):
        """
        Execute the optimized path on a Tello drone
        
        Args:
            drone_controller: TelloDigitalTwin or compatible controller
            path: List of (x,y) waypoints to follow
            speed: Speed in cm/s
            
        Returns:
            Success flag
        """
        if not hasattr(drone_controller, 'drone') or not drone_controller.is_flying:
            print("Drone is not connected or not flying")
            return False
        
        # Simplify path to avoid too many small movements
        simplified_path = [path[0]]
        min_distance = 0.3  # meters
        
        for point in path[1:]:
            last_point = simplified_path[-1]
            distance = math.sqrt((point[0] - last_point[0])**2 + (point[1] - last_point[1])**2)
            if distance >= min_distance:
                simplified_path.append(point)
        
        print(f"Simplified path from {len(path)} to {len(simplified_path)} waypoints")
        
        try:
            # Execute path
            for i, waypoint in enumerate(simplified_path):
                print(f"Moving to waypoint {i+1}/{len(simplified_path)}: {waypoint}")
                
                # Convert to drone coordinates
                target_x = waypoint[0] * 100 + 500  # Convert to cm and add offset
                target_y = waypoint[1] * 100 + 500
                
                # Calculate vector to waypoint
                current_x, current_y = drone_controller.x, drone_controller.y
                dx = target_x - current_x
                dy = target_y - current_y
                distance = math.sqrt(dx**2 + dy**2)
                
                if distance < 30:  # Skip if already close
                    continue
                    
                # Calculate angle to waypoint
                angle_to_point = math.degrees(math.atan2(dy, dx))
                target_angle = (90 - angle_to_point) % 360
                
                # Calculate yaw difference
                yaw_diff = (target_angle - drone_controller.yaw) % 360
                if yaw_diff > 180:
                    yaw_diff -= 360
                    
                # Rotate to face waypoint
                if abs(yaw_diff) > 15:
                    print(f"Rotating {yaw_diff} degrees")
                    yaw_speed = min(30, max(-30, int(yaw_diff / 3)))
                    drone_controller.drone.send_rc_control(0, 0, 0, yaw_speed)
                    time.sleep(1)
                    drone_controller.drone.send_rc_control(0, 0, 0, 0)
                    time.sleep(0.5)
                
                # Move towards waypoint
                move_speed = min(speed, int(distance / 10) + 10)  # Adaptive speed
                duration = distance / (move_speed * 100)  # seconds
                duration = min(3.0, max(0.5, duration))  # Limit between 0.5-3s
                
                print(f"Moving forward at {move_speed} cm/s for {duration:.1f}s")
                drone_controller.drone.send_rc_control(0, move_speed, 0, 0)
                time.sleep(duration)
                
                # Stop at waypoint
                drone_controller.drone.send_rc_control(0, 0, 0, 0)
                time.sleep(0.5)
                
            # Final stop
            drone_controller.drone.send_rc_control(0, 0, 0, 0)
            return True
            
        except Exception as e:
            print(f"Error executing path: {e}")
            # Emergency stop
            try:
                drone_controller.drone.send_rc_control(0, 0, 0, 0)
            except:
                pass
            return False


class GradientDescentDroneNavigator:
    """
    Advanced path planning and execution for Tello drones based on
    principles from gradient descent optimization algorithms.
    
    This class extends the PathOptimizer with drone-specific features
    and adaptive navigation based on real-time feedback.
    """
    
    def __init__(self, drone_controller):
        """Initialize with a drone controller"""
        self.drone_controller = drone_controller
        self.path_optimizer = PathOptimizer()
        self.current_path = None
        self.terrain_map = None
        self.obstacle_map = []
        self.is_mapping_terrain = False
        self.is_detecting_obstacles = False
        
    def set_optimization_parameters(self, learning_rate=None, momentum=None, 
                                  avoidance_strength=None, energy_weight=None):
        """Update optimization parameters"""
        optimizer = self.path_optimizer
        
        if learning_rate is not None:
            optimizer.initial_learning_rate = learning_rate
        
        if momentum is not None:
            optimizer.momentum = momentum
            
        if avoidance_strength is not None:
            optimizer.avoidance_strength = avoidance_strength
            
        if energy_weight is not None:
            optimizer.energy_weight = energy_weight
    
    def start_terrain_mapping(self):
        """Start mapping the terrain"""
        if not hasattr(self.drone_controller, 'ml'):
            print("ML features required for terrain mapping")
            return False
            
        self.is_mapping_terrain = True
        
        # Initialize terrain map if needed
        if self.terrain_map is None:
            # Create empty terrain map
            self.terrain_map = np.zeros((100, 100))  # 10m x 10m grid with 10cm resolution
        
        return True
    
    def start_obstacle_detection(self):
        """Start detecting obstacles"""
        if not hasattr(self.drone_controller, 'video_stream') or not self.drone_controller.video_stream:
            print("Video stream required for obstacle detection")
            return False
            
        self.is_detecting_obstacles = True
        return True
    
    def update_maps_from_sensors(self):
        """Update terrain and obstacle maps from drone sensors"""
        if not self.is_mapping_terrain and not self.is_detecting_obstacles:
            return
            
        # Get current drone position
        x_pos = (self.drone_controller.x - 500) / 100  # meters
        y_pos = (self.drone_controller.y - 500) / 100
        
        # Update terrain map if active
        if self.is_mapping_terrain and hasattr(self.drone_controller, 'ml'):
            # Convert to grid coordinates
            grid_x = int(x_pos * 10) + 50  # 10 cells per meter, offset by 50
            grid_y = int(y_pos * 10) + 50
            
            # Update terrain passability based on ML occupancy grid
            if hasattr(self.drone_controller.ml, 'occupancy_grid'):
                occupancy = self.drone_controller.ml.occupancy_grid
                
                # Copy relevant section to our terrain map
                size = min(10, occupancy.shape[0], occupancy.shape[1])
                for i in range(-size//2, size//2):
                    for j in range(-size//2, size//2):
                        # Get occupancy value
                        if 0 <= i + size//2 < occupancy.shape[0] and 0 <= j + size//2 < occupancy.shape[1]:
                            occ_value = occupancy[i + size//2, j + size//2]
                            
                            # Map to terrain cost
                            if 0 <= grid_y + i < self.terrain_map.shape[0] and 0 <= grid_x + j < self.terrain_map.shape[1]:
                                # Obstacles become high cost
                                if occ_value > 0:
                                    self.terrain_map[grid_y + i, grid_x + j] = min(1.0, occ_value)
                                # Free space becomes low cost
                                else:
                                    self.terrain_map[grid_y + i, grid_x + j] = max(0.0, occ_value)
        
        # Update obstacle map if active and video is available
        if self.is_detecting_obstacles and hasattr(self.drone_controller, 'frame'):
            frame = self.drone_controller.frame
            if frame is not None:
                # Simple obstacle detection using computer vision
                obstacles = self._detect_obstacles_from_frame(frame, x_pos, y_pos)
                
                # Add to obstacle map
                for obstacle in obstacles:
                    # Check if already in map
                    new_obstacle = True
                    for i, existing in enumerate(self.obstacle_map):
                        dist = math.sqrt((obstacle[0] - existing[0])**2 + (obstacle[1] - existing[1])**2)
                        if dist < obstacle[2] + existing[2]:  # Overlapping
                            # Update with average
                            self.obstacle_map[i] = (
                                (obstacle[0] + existing[0]) / 2,
                                (obstacle[1] + existing[1]) / 2,
                                max(obstacle[2], existing[2])
                            )
                            new_obstacle = False
                            break
                    
                    if new_obstacle:
                        self.obstacle_map.append(obstacle)
    
    def _detect_obstacles_from_frame(self, frame, drone_x, drone_y):
        """Detect obstacles from camera frame"""
        obstacles = []
        
        # Convert to HSV for easier color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Detect red objects (example - can be extended)
        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)
        
        # Alternative red range
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask, mask2)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process contours
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Min size threshold
                # Get bounding circle
                (x, y), radius = cv2.minEnclosingCircle(contour)
                
                # Estimate real-world position and size
                # This is a simplification - would need actual depth estimation
                # and camera calibration for accurate results
                
                # Simple estimation: convert pixel coordinates to world coordinates
                # based on current drone position, camera FOV, and assumed height
                
                # Example: 75° FOV, 480x640 resolution
                h, w = frame.shape[:2]
                fov_x = 75 * math.pi / 180
                
                # Calculate angle from center
                angle_x = ((x - w/2) / (w/2)) * (fov_x/2)
                
                # Estimate distance based on size
                estimated_distance = 100 / (radius + 1)  # in cm
                
                # Calculate world coordinates
                world_x = drone_x + (estimated_distance/100) * math.sin(angle_x)
                world_y = drone_y + (estimated_distance/100) * math.cos(angle_x)
                
                # Estimate real radius (assuming spherical object)
                real_radius = 0.3  # meters (default size)
                
                obstacles.append((world_x, world_y, real_radius))
        
        return obstacles
    
    def plan_path_to_target(self, target_pos, start_pos=None):
        """Plan an optimal path to the target"""
        # Use current position as start if not specified
        if start_pos is None:
            x_pos = (self.drone_controller.x - 500) / 100
            y_pos = (self.drone_controller.y - 500) / 100
            start_pos = (x_pos, y_pos)
            
        # Calculate path
        self.current_path = self.path_optimizer.find_optimal_path(
            start_pos, 
            target_pos, 
            obstacles=self.obstacle_map,
            terrain=self.terrain_map
        )
        
        return self.current_path
    
    def visualize_current_state(self):
        """Visualize current state with path, obstacles, and terrain"""
        if self.current_path is None:
            print("No path calculated yet")
            return None
            
        # Get current position
        x_pos = (self.drone_controller.x - 500) / 100
        y_pos = (self.drone_controller.y - 500) / 100
        current_pos = (x_pos, y_pos)
        
        # Target is the end of the path
        target_pos = self.current_path[-1]
        
        # Create visualization
        fig = self.path_optimizer.visualize_optimization(
            current_pos,
            target_pos,
            obstacles=self.obstacle_map,
            path=self.current_path,
            terrain=self.terrain_map
        )
        
        return fig
    
    def execute_current_path(self):
        """Execute the current path on the drone"""
        if self.current_path is None:
            print("No path calculated yet")
            return False
            
        return self.path_optimizer.execute_path_on_drone(
            self.drone_controller,
            self.current_path,
            speed=self.drone_controller.speed
        )


# Example usage in main method
if __name__ == "__main__":
    # For standalone testing
    import matplotlib.pyplot as plt
    
    # Create optimizer
    optimizer = PathOptimizer()
    
    # Define test scenario
    start = (0, 0)
    target = (8, 8)
    
    # Add obstacles
    obstacles = [
        (2, 2, 1),
        (4, 5, 1.5),
        (6, 3, 1),
    ]
    
    # Create terrain
    terrain = optimizer.create_3d_terrain(size=10, complexity=8)
    
    # Find path
    path = optimizer.find_optimal_path(start, target, obstacles, terrain)
    
    # Visualize
    fig = optimizer.visualize_optimization(start, target, obstacles, path, terrain)
    plt.show()
