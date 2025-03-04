# Tello Drone Path Optimization System

## Overview
This project implements an advanced path optimization system for DJI Tello drones using principles from gradient descent algorithms. The system creates efficient flight paths while avoiding obstacles, similar to how machine learning algorithms optimize loss functions.

## Features
- **Gradient-Descent Based Path Planning**: Uses adaptive learning rates to find optimal paths
- **Obstacle Avoidance**: Automatically routes around obstacles using repulsive force fields
- **Energy Optimization**: Creates smooth paths that minimize battery consumption
- **Digital Twin Visualization**: Real-time visualization of drone state and environment
- **Interactive GUI**: Comprehensive interface for controlling the drone and planning paths
- **Predefined Missions**: Square, circle, spiral patterns and target tracking missions
- **Data Analysis**: Records and visualizes flight data for performance analysis

## Requirements
- Python 3.7+
- DJI Tello drone
- Libraries:
  - OpenCV (`opencv-python`)
  - NumPy
  - Matplotlib
  - TKinter (usually included with Python)
  - PIL/Pillow
  - djitellopy

```bash
pip install opencv-python numpy matplotlib pillow djitellopy
```

## File Structure
- `Tello-path-optimization.py`: Core path optimization algorithm using gradient descent principles
- `Tello-main-implementation.py`: Main application with GUI and drone control integration

## Running the Code

### Path Optimization Demo
To run the standalone path optimization visualization:

```bash
python Tello-path-optimization.py
```

This will generate a visualization showing:
- Sample path finding with obstacles
- Convergence metrics (distance vs. iterations)
- Learning rate adaptation

### Full Application
To run the complete application with GUI:

```bash
python "Tello-main-implementation .py"
```

Command line options:

```bash
# Run with basic controller (no ML)
python "Tello-main-implementation .py" --basic

# Run without GUI
python "Tello-main-implementation .py" --no-gui

# Connect to a real drone
python "Tello-main-implementation .py" --run-drone

# Launch in analysis mode
python "Tello-main-implementation .py" --analyze
```

## How It Works

### Path Optimization Algorithm
The path optimization algorithm is inspired by gradient descent techniques from machine learning:

- **Adaptive Learning Rate**:
  - Higher learning rates for faster initial progress
  - Gradual reduction as drone approaches target (like learning rate decay)
  - Prevents overshooting the target

- **Momentum**:
  - Smooths path trajectories by maintaining movement direction
  - Helps avoid getting stuck near obstacles (similar to escaping local minima)
  - Results in more efficient, natural-looking paths

- **Obstacle Avoidance**:
  - Obstacles create repulsive force fields that push the path away
  - Force strength decreases with distance from obstacle
  - Combined with target attraction to find optimal routes

- **Energy Efficiency**:
  - Penalizes sharp turns and rapid velocity changes
  - Optimizes for battery conservation

### Visualization Components
The visualization shows:
- Real-time path planning progress
- Distance to target over iterations (similar to loss curves)
- Learning rate adaptation
- Obstacle avoidance behavior

## Implementation Details

### Learning Rate Adaptation
```python
def adaptive_learning_rate(iteration, remaining_distance=None):
    # Basic exponential decay
    progress = iteration / max_iterations
    lr = initial_learning_rate * math.exp(-decay_factor * progress)
    
    # Adjust based on remaining distance
    if remaining_distance is not None:
        distance_factor = min(1.0, remaining_distance)
        lr *= distance_factor
    
    # Ensure we don't go below minimum
    return max(min_learning_rate, lr)
```

## Real-World Applications
This algorithm has practical applications for:
- Automated drone delivery route planning
- Survey and mapping missions
- Search and rescue path optimization
- Photography and videography shot planning

## Future Improvements
- Reinforcement learning for path optimization
- Multi-drone coordination
- Real-time path adaptation based on moving obstacles
- Integration with computer vision for dynamic obstacle detection

## License
This project is licensed under the MIT License.
