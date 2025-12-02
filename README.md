# Mapless DRL Forest Navigation

ROS2 package for autonomous collision-free forest navigation using Deep Reinforcement Learning (PPO).

## Prerequisites

- ROS2 Humble
- Python 3.10+
- CUDA 12.4 (for GPU acceleration)

## Installation

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Build the package:
   ```bash
   colcon build --symlink-install
   source install/setup.bash
   ```

## Usage

### 1. Simulation & Training

Launch the simulation environment (Gazebo):
```bash
ros2 launch mapless_navigation forest_sim.launch.xml
```

Run the PPO training script:
```bash
ros2 run mapless_navigation train_ppo
```
This will train the agent for 2 million steps. TensorBoard logs are saved to `./ppo_forest_tensorboard/`.

### 2. Deployment

Deploy the trained policy on the rover (or in simulation):
```bash
ros2 launch mapless_navigation deploy.launch.xml
```

## Configuration

- **Training Hyperparameters**: `config/training.yaml`
- **Robot Parameters**: `config/rover.yaml`

## Structure

- `src/forest_env.py`: Gymnasium environment wrapper for ROS2.
- `src/train_ppo.py`: PPO training script.
- `src/navigation_node.py`: Inference node for deployment.
