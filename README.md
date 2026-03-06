<div align="center">

# Mapless DRL Navigation

**Autonomous collision-free robot navigation using Deep Reinforcement Learning — no maps, no encoders, no hand-coded rules.**

[![ROS2 Jazzy](https://img.shields.io/badge/ROS2-Jazzy-blue?logo=ros&logoColor=white)](https://docs.ros.org/en/jazzy/)
[![Ubuntu 24.04](https://img.shields.io/badge/Ubuntu-24.04-orange?logo=ubuntu&logoColor=white)](https://ubuntu.com/)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-yellow?logo=python&logoColor=white)](https://www.python.org/)
[![PPO / SAC](https://img.shields.io/badge/Algorithm-PPO%20%7C%20SAC-green)](https://stable-baselines3.readthedocs.io/)
[![Stable Baselines 3](https://img.shields.io/badge/Stable--Baselines3-2.x-orange)](https://stable-baselines3.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)

</div>

---

## What Is This?

Most mobile robots navigate using a pre-built SLAM map. If an obstacle moves, or the environment is unknown, they fail.

This project trains a neural network to navigate reactively from raw 360-degree lidar data — no map, no GPS, no wheel encoders. The policy maps sensor observations directly to motor velocities, trained entirely in Gazebo Harmonic simulation and deployed on a real 4WD rover via ROS 2 Jazzy.

**No map. No hand-coded rules. Just a sensor, a goal, and a trained policy.**

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      REAL ROBOT STACK                        │
│                                                              │
│  Slamtec C1M1 R2                                             │
│  Lidar (12m, 360°)  ──/scan──►  rf2o_laser_odometry         │
│                                  (scan-matching → /odom)    │
│                                         │                    │
│  MPU-6050 IMU  ─────/imu/data──►  robot_localization EKF    │
│  (gyro + accel)                   (fused → /odometry/       │
│                                          filtered)          │
│                                         │                    │
│  Lidar ─────────────/scan──────►  NavigationNode            │
│                                   PPO / SAC policy           │
│                                   obs = [scan(360)           │
│                                          + dist + θ]        │
│                                   action = predict(obs)      │
│                                         │ /cmd_vel           │
│                                   BTS7960Driver              │
│                                   (GPIO PWM → motors)        │
│                                         │                    │
│                                   4x JGB37 DC Motors         │
└──────────────────────────────────────────────────────────────┘
```

### Observation Space — 362-dimensional float32

| Index | Value | Normalisation |
| :--- | :--- | :--- |
| `[0..359]` | Lidar ranges (360°) | divided by 12.0 m (C1M1 R2 max range) |
| `[360]` | Distance to goal | clipped to [0, 1] over 10 m |
| `[361]` | Bearing to goal | divided by π → [-1, 1] |

### Action Space — 2-dimensional continuous, [-1, 1]

| Index | Meaning | Maps To |
| :--- | :--- | :--- |
| `[0]` | Linear velocity | [0, 0.26] m/s (always forward) |
| `[1]` | Angular velocity | [-1.82, 1.82] rad/s |

### Reward Function

```python
if min_lidar_range < 0.20 m:
    reward = -100                            # Collision
elif dist_to_goal < 0.50 m:
    reward = +100                            # Goal reached
else:
    reward  = (prev_dist - curr_dist) * 10   # Progress toward goal
    reward -= prox_weight * max(0, 0.5 - min_lidar)  # Wall-hugging penalty
    reward -= 0.05 * |angular_velocity|      # Turn smoothness
    reward -= 0.1                            # Per-step time penalty
```

### Policy Network — Conv1D Feature Extractor

The 360 lidar rays form a circular 1-D signal. A flat MLP treats all inputs independently. The custom `LidarConvExtractor` exploits this spatial structure:

```
Lidar branch (360 inputs):
    Conv1D(1→32, k=5) → ReLU → Conv1D(32→64, k=3) → ReLU
    → AdaptiveAvgPool1d(45) → Flatten  →  2880 units

Goal branch (2 inputs):
    Linear(2→32) → ReLU              →    32 units

Head: Linear(2912 → 256) → ReLU
```

---

## Hardware

| Component | Part | Role |
| :--- | :--- | :--- |
| **Compute** | Raspberry Pi 5 (8 GB) | ROS 2 runtime, EKF, 10 Hz policy inference |
| **Lidar** | Slamtec C1M1 R2 (DTOF, 12 m, 360°) | Obstacle detection + lidar odometry |
| **IMU** | MPU-6050 (I2C) | Fused with lidar odometry for robust pose |
| **Camera** | Raspberry Pi HQ Camera (IMX477) | Future: visual goal specification |
| **Motor Drivers** | 2x BTS7960 43A | High-current PWM for left/right motor banks |
| **Motors** | 4x JGB37 DC (no encoders) | 4WD skid-steer propulsion |
| **Power** | 12V LiPo | Motors + buck-converter for Pi 5 |

Odometry is derived from lidar scan-matching (`rf2o`) fused with IMU data via a `robot_localization` EKF. No wheel encoders required.

---

## Project Structure

```
mapless/
├── mapless_navigation/         # ROS 2 Python package
│   ├── obs_utils.py            # Shared observation pipeline (training + deployment)
│   ├── forest_env.py           # Gymnasium environment (Gazebo Harmonic)
│   ├── train_ppo.py            # PPO / SAC training with Conv1D policy
│   ├── navigation_node.py      # Real-time inference node (10 Hz)
│   ├── evaluate.py             # Quantitative evaluation script
│   ├── bts7960_driver.py       # BTS7960 GPIO/PWM motor driver (Pi 5)
│   └── sabertooth_driver.py    # Sabertooth 2x32 Packet Serial driver (Jetson)
│
├── launch/
│   ├── real_robot.launch.py    # Full hardware stack (Lidar → IMU → EKF → AI → Motors)
│   ├── forest_sim.launch.xml   # Simulation environment (Gazebo Harmonic)
│   ├── forest_nav.launch.xml   # Sim + inference (mode=sim|deploy)
│   └── deploy.launch.xml       # Lightweight model-only deploy launcher
│
├── config/
│   ├── training.yaml           # All hyperparameters + domain randomisation config
│   ├── rover.yaml              # Robot physical parameters + GPIO pin mapping
│   └── ekf.yaml                # robot_localization EKF configuration
│
├── models/
│   ├── ppo_forest_nav.zip      # Latest trained model
│   ├── best_model/             # EvalCallback best checkpoint
│   └── checkpoints/            # Periodic training checkpoints
│
├── docs/
│   ├── reference.md            # Full technical reference
│   ├── SETUP.md                # Setup guide
│   ├── CONNECTIONS.md          # Hardware wiring guide
│   └── media/                  # Images / demo videos
│
├── package.xml
├── setup.py
├── requirements.txt
└── install_dependencies.sh
```

---

## Getting Started

### Prerequisites

- **OS**: Ubuntu 24.04 LTS (Noble)
- **ROS 2**: Jazzy Jalisco
- **Python**: 3.12+
- **GPU** *(optional, for training)*: CUDA-capable, 8 GB+ VRAM recommended

### 1. Clone and Install

```bash
git clone https://github.com/mohammedryn/mapless.git
cd mapless
chmod +x install_dependencies.sh
./install_dependencies.sh
sudo reboot   # Required for USB and GPIO group permissions
```

### 2. Build

```bash
cd ~/mapless
colcon build --symlink-install
source install/setup.bash
```

Add to `~/.bashrc` to avoid sourcing each session:
```bash
echo "source ~/mapless/install/setup.bash" >> ~/.bashrc
```

---

## Usage

### Training in Simulation

Launch the Gazebo simulation:
```bash
ros2 launch mapless_navigation forest_sim.launch.xml
```

Train with Conv1D policy and PPO (default, all settings from `config/training.yaml`):
```bash
ros2 run mapless_navigation train_ppo
```

Train SAC instead (for ablation comparison):
```bash
ros2 run mapless_navigation train_ppo --algorithm sac
```

Train flat MLP baseline (to quantify Conv1D benefit):
```bash
ros2 run mapless_navigation train_ppo --policy mlp
```

Resume from checkpoint:
```bash
ros2 run mapless_navigation train_ppo --continue_training
```

Monitor with TensorBoard:
```bash
tensorboard --logdir ./ppo_forest_tensorboard
```

### Evaluate the Trained Policy

Run 100 episodes and get quantitative metrics:
```bash
ros2 run mapless_navigation evaluate_policy --episodes 100
```

Output includes success rate, collision rate, timeout rate, average steps, and average final distance.

### Deploy on Real Robot

Default goal at (5.0, 0.0) in the odom frame:
```bash
ros2 launch mapless_navigation real_robot.launch.py
```

Override goal at launch:
```bash
ros2 launch mapless_navigation real_robot.launch.py goal_x:=3.0 goal_y:=1.5
```

Update goal at runtime (no restart needed):
```bash
ros2 topic pub /goal_xy std_msgs/msg/Float64MultiArray '{data: [3.0, 1.5]}'
```

This starts the full stack: `Lidar → rf2o Odometry → IMU EKF → DRL Policy → BTS7960 Motors`

### Verify the Stack

```bash
# Check lidar is publishing
ros2 topic hz /scan

# Check raw odometry
ros2 topic echo /odom

# Check EKF-fused odometry (better than raw)
ros2 topic echo /odometry/filtered

# Check motor commands are being published
ros2 topic echo /cmd_vel
```

Manual drive test (motors spin for 1 second):
```bash
ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.1}, angular: {z: 0.0}}"
```

---

## Configuration

### Training — [`config/training.yaml`](config/training.yaml)

Key parameters:

| Parameter | Value | Notes |
| :--- | :--- | :--- |
| `algorithm` | `"ppo"` | `"ppo"` or `"sac"` |
| `policy_type` | `"conv1d"` | `"conv1d"` or `"mlp"` |
| `total_timesteps` | 2,000,000 | Increase for better generalization |
| `env_config.max_range` | 12.0 m | **C1M1 R2 real sensor range** |
| `env_config.collision_dist` | 0.20 m | Hard collision threshold |
| `domain_rand.lidar_noise_std` | 0.03 m | Gaussian noise for sim-to-real robustness |
| `domain_rand.speed_scale` | 0.85–1.15 | Motor speed variation per episode |

### Robot — [`config/rover.yaml`](config/rover.yaml)

| Parameter | Value |
| :--- | :--- |
| `max_linear_vel` | 0.26 m/s |
| `max_angular_vel` | 1.82 rad/s |
| `lidar_max_range` | 12.0 m |
| GPIO pin mapping | BTS7960 for Pi 5 |

---

## Hardware Wiring

> See [`docs/CONNECTIONS.md`](docs/CONNECTIONS.md) for detailed wiring.

**BTS7960 GPIO (default, Pi 5):**

| Signal | GPIO Pin |
|:---|:---:|
| Left Forward Enable | 22 |
| Left Reverse Enable | 23 |
| Left Forward PWM | 18 |
| Left Reverse PWM | 19 |
| Right Forward Enable | 17 |
| Right Reverse Enable | 27 |
| Right Forward PWM | 12 |
| Right Reverse PWM | 13 |

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
| :--- | :--- | :--- |
| `Permission denied` on `/dev/ttyUSB0` | Not in `dialout` group | Run install script + reboot |
| `Package not found` | Workspace not sourced | `source install/setup.bash` |
| `/odom` not publishing | Lidar not spinning / wrong port | `ls /dev/ttyUSB*`, update port in launch file |
| `/odometry/filtered` not publishing | EKF config error or IMU not running | Check `ekf.yaml`; node degrades to `/odom` |
| Robot crashes in real world | Sim-to-real gap | Increase `lidar_noise_std`, lower `max_linear_vel` |
| Wheels spin wrong direction | Motor wires reversed | Swap L_PWM and R_PWM pins in `rover.yaml` |

---

## Performance

Run `evaluate_policy` to get your actual numbers. Do not rely on estimates.

```bash
ros2 run mapless_navigation evaluate_policy --episodes 100
```

Paste your results here after running on the real robot.

---

## Roadmap

- [x] Custom Gymnasium environment with ROS 2 + Gazebo Harmonic bridge
- [x] PPO / SAC training with Stable Baselines 3
- [x] Custom Conv1D policy network for lidar spatial structure
- [x] Domain randomisation (lidar noise + speed variation) for sim-to-real robustness
- [x] Shared observation pipeline (`obs_utils.py`) — identical in training and deployment
- [x] Quantitative evaluation script (`evaluate_policy`)
- [x] Real robot deployment node with runtime goal updates via `/goal_xy`
- [x] BTS7960 GPIO/PWM motor driver (Raspberry Pi 5)
- [x] Sabertooth 2x32 Packet Serial driver (Jetson)
- [x] Lidar-only odometry via `rf2o`
- [x] IMU (MPU-6050) + rf2o fusion with `robot_localization` EKF
- [x] Hardware safety override (emergency stop when obstacle < 0.15 m)
- [x] Configurable goal via launch args and `/goal_xy` topic
- [ ] Real-world demo video
- [ ] Visual goal specification with Pi HQ Camera + ArUco markers
- [ ] SAC vs PPO ablation with quantitative comparison table

---

## Acknowledgements

- [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3) — PPO / SAC implementation
- [rf2o_laser_odometry](https://github.com/MAPIRlab/rf2o_laser_odometry) — Lidar scan-matching odometry
- [sllidar_ros2](https://github.com/Slamtec/sllidar_ros2) — Slamtec C1M1 R2 ROS 2 driver
- [robot_localization](https://github.com/cra-ros-pkg/robot_localization) — EKF sensor fusion
- [Gymnasium](https://gymnasium.farama.org/) — RL environment interface

---

<div align="center">

**Built by Mohammed Rayan**

[Star this repo](https://github.com/mohammedryn/mapless) · [Report a Bug](https://github.com/mohammedryn/mapless/issues)

</div>
