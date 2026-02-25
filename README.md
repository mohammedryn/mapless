<div align="center">

# 🌲 Mapless DRL Forest Navigation

**Autonomous collision-free robot navigation through unstructured environments using Deep Reinforcement Learning — no maps required.**

[![ROS2 Humble](https://img.shields.io/badge/ROS2-Humble-blue?logo=ros&logoColor=white)](https://docs.ros.org/en/humble/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-yellow?logo=python&logoColor=white)](https://www.python.org/)
[![PPO](https://img.shields.io/badge/Algorithm-PPO-green)](https://stable-baselines3.readthedocs.io/)
[![Stable Baselines 3](https://img.shields.io/badge/Stable--Baselines3-2.x-orange)](https://stable-baselines3.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)

</div>

---

## 🧠 What Is This?

Most robots navigate using a **pre-built map** (SLAM). If a chair moves, they get confused.

This project takes a fundamentally different approach: a ground robot that **learns to navigate** by interacting with its environment — like an animal moving through a forest. Using **Proximal Policy Optimization (PPO)**, the robot's neural network learns to read raw 360° Lidar data and output velocity commands that avoid obstacles and reach a goal.

**No map. No hand-coded rules. Just a sensor, a goal, and a trained brain.**

### Key Innovations

| Innovation | What It Means |
| :--- | :--- |
| **Mapless Navigation** | Zero dependency on SLAM or prior maps. Reactive to dynamic changes in real-time. |
| **Encoder-free Odometry** | No wheel encoders on the chassis. Pose is estimated entirely from Lidar scan-matching via `rf2o`. |
| **End-to-End Learning** | Raw sensor input → continuous velocity output. No hand-crafted waypoints or rules. |
| **Edge Deploy** | Full inference stack runs on the Jetson Orin Nano at 10 Hz. |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        REAL ROBOT STACK                         │
│                                                                 │
│  ┌──────────────┐    /scan    ┌────────────────────────────────┐│
│  │  Slamtec     │ ─────────► │   rf2o_laser_odometry          ││
│  │  Lidar C1M1  │            │   (Scan Matching → /odom)      ││
│  └──────────────┘            └──────────┬─────────────────────┘│
│                                         │ /odom                 │
│                              ┌──────────▼─────────────────────┐│
│  ┌──────────────┐    /scan   │   NavigationNode               ││
│  │  Slamtec     │ ─────────► │   (Loads PPO Model)            ││
│  │  Lidar C1M1  │            │   obs = [scan(360) + dist + θ] ││
│  └──────────────┘            │   action = model.predict(obs)  ││
│                              └──────────┬─────────────────────┘│
│                                         │ /cmd_vel              │
│                              ┌──────────▼─────────────────────┐│
│                              │   SabertoothDriver             ││
│                              │   (Packet Serial → motors)     ││
│                              └──────────┬─────────────────────┘│
│                                         │                       │
│                              ┌──────────▼─────────────────────┐│
│                              │   JGB37 Motors (x4)            ││
│                              │   via Sabertooth 2x32          ││
│                              └────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### Observation Space (362-dimensional)

| Index | Value | Range |
| :--- | :--- | :--- |
| `[0…359]` | Normalized Lidar ranges (360°) | `[0, 1]` |
| `[360]` | Normalized goal distance | `[0, 1]` |
| `[361]` | Normalized angle to goal | `[-1, 1]` |

### Action Space (2-dimensional, continuous)

| Index | Meaning | Maps To |
| :--- | :--- | :--- |
| `[0]` | Linear velocity | `[0, 0.26] m/s` |
| `[1]` | Angular velocity | `[-1.82, 1.82] rad/s` |

### Reward Function

```python
if min_lidar_range < 0.20m:
    reward = -100   # Collision
elif distance_to_goal < 0.5m:
    reward = +100   # Goal Reached!
else:
    reward = (progress_toward_goal × 10) - 0.1  # Progress - time penalty
```

---

## 🔧 Hardware

| Component | Specification / Details | Role |
| :--- | :--- | :--- |
| **Compute** | Raspberry Pi 5 (8GB) | Runs ROS2, handles Lidar Odometry, and executes the PPO inference. |
| **Lidar** | Slamtec Lidar C1M1 R2 | Primary sensor for obstacle detection and encoder-less odometry. |
| **Motor Drivers**| 2x BTS7960 43A Drivers | High-current PWM drivers for the left and right motor banks. |
| **Motors** | 4x JGB37 DC Motors | High-torque propulsion (No encoders needed). |
| **Vision** | Raspberry Pi HQ Camera | (Optional) Future integration for vision-language models. |
| **Power** | 12V LiPo Battery | Dedicated power for motors and buck-converter for Pi 5. |
Odometry is derived entirely from Lidar scan-matching using `rf2o_laser_odometry`.

---

## 📁 Project Structure

```
mapless/
├── src/                        # Core ROS2 Python nodes
│   ├── forest_env.py           # Gymnasium environment wrapper (training)
│   ├── train_ppo.py            # PPO training script
│   ├── navigation_node.py      # Deployment inference node (10 Hz)
│   └── sabertooth_driver.py    # Custom Packet Serial motor driver
│
├── launch/                     # Launch files
│   ├── real_robot.launch.py    # Full hardware stack (Lidar → Odom → AI → Motors)
│   ├── forest_sim.launch.xml   # Simulation-only (Gazebo)
│   ├── forest_nav.launch.xml   # Navigation in sim with trained model
│   └── deploy.launch.xml       # Lightweight deploy launcher
│
├── config/                     # Configuration YAML files
│   ├── training.yaml           # PPO hyperparameters
│   └── rover.yaml              # Robot physical parameters
│
├── models/                     # Trained neural network models
│   ├── ppo_forest_nav.zip      # Latest trained model
│   └── checkpoints/            # Training checkpoints
│
├── docs/                       # Documentation
│   ├── SETUP.md                # Full Jetson setup guide
│   ├── CONNECTIONS.md          # Hardware wiring & DIP switch guide
│   ├── PROGRESS.md             # Project progress & technical deep-dive
│   ├── reference.md            # Extended project reference
│   └── media/                  # Images and demo videos
│
├── package.xml
├── setup.py
├── requirements.txt
└── install_dependencies.sh
```

---

## 🚀 Getting Started

### Prerequisites

- **OS**: Ubuntu 22.04 (Jammy)
- **ROS2**: Humble Hawksbill
- **Python**: 3.10+
- **GPU** *(optional for training)*: CUDA 12.4+

### 1. Clone & Install

```bash
git clone https://github.com/mohammedryn/mapless.git
cd mapless
chmod +x install_dependencies.sh
./install_dependencies.sh
sudo reboot   # Required for USB serial permissions
```

### 2. Build

```bash
cd ~/mapless
colcon build --symlink-install
source install/setup.bash
```

> **Tip:** Add `source ~/mapless/install/setup.bash` to your `~/.bashrc` to avoid typing it every session.

---

## 🎮 Usage

### Training (Simulation)

Launch the Gazebo forest environment:
```bash
ros2 launch mapless_navigation forest_sim.launch.xml
```

Start PPO training (2 million steps by default):
```bash
ros2 run mapless_navigation train_ppo
```

Resume from a saved checkpoint:
```bash
ros2 run mapless_navigation train_ppo --continue_training
```

Monitor training with TensorBoard:
```bash
tensorboard --logdir ./ppo_forest_tensorboard
```

### Deployment (Real Robot)

> Ensure Lidar and Sabertooth are plugged in before running.

```bash
ros2 launch mapless_navigation real_robot.launch.py
```

This single command starts the full stack:
`Lidar Driver → TF Publisher → Lidar Odometry → Motor Driver → AI Navigation Node`

### Verify It's Working

In a second terminal, check odometry is flowing:
```bash
ros2 topic echo /odom
```

Send a manual test command (wheels spin for 1 second):
```bash
ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.1}, angular: {z: 0.0}}"
```

---

## ⚙️ Configuration

### Training Hyperparameters — [`config/training.yaml`](config/training.yaml)

| Parameter | Value | Notes |
| :--- | :--- | :--- |
| `total_timesteps` | 2,000,000 | Increase for better forest generalization |
| `learning_rate` | 0.0003 | Standard PPO starting point |
| `gamma` | 0.99 | High discount — robot plans long-term |
| `collision_penalty` | 100.0 | Strongly discourages hitting trees |

### Robot Parameters — [`config/rover.yaml`](config/rover.yaml)

| Parameter | Value |
| :--- | :--- |
| `max_linear_vel` | 0.26 m/s |
| `max_angular_vel` | 1.82 rad/s |
| `lidar_fov` | 360° |

---

## 🔌 Hardware Wiring

> See the full guide in [`docs/CONNECTIONS.md`](docs/CONNECTIONS.md).

**Quick Reference — Sabertooth DIP Switches (Packet Serial, Address 128):**

| SW1 | SW2 | SW3 | SW4 | SW5 | SW6 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| OFF | OFF | ON *(LiPo)* | OFF | OFF | OFF |

---

## 🛠️ Troubleshooting

| Symptom | Likely Cause | Fix |
| :--- | :--- | :--- |
| `Permission denied` on serial | USB group not set | Ensure `install_dependencies.sh` was run & rebooted |
| `Package not found` | Workspace not sourced | Run `source install/setup.bash` |
| `/odom` not updating | Lidar not spinning or wrong port | Check `ls /dev/ttyUSB*` and update `real_robot.launch.py` |
| Wheels spin wrong direction | Motor wires reversed | Swap M1A ↔ M1B (or M2A ↔ M2B) on Sabertooth |
| Robot crashes in real world | Sim-to-real gap | Tune speed scaling in `sabertooth_driver.py` |

---

## 📊 Performance

- **Estimated Success Rate**: 85–90% in forest environments
- **Inference Latency**: ~10 Hz on Jetson Orin Nano
- **Training Time**: ~4–8 hours for 2M steps (with GPU)

---

## 🗺️ Roadmap

- [x] Custom Gymnasium environment with ROS2 bridge
- [x] PPO agent training with Stable Baselines 3
- [x] Real robot deployment node (10 Hz inference)
- [x] Custom Sabertooth Packet Serial driver
- [x] Lidar-only odometry via `rf2o`
- [ ] Safety override layer (emergency stop on close obstacles)
- [ ] Domain randomization for improved sim-to-real transfer
- [ ] IMU-fused odometry for improved robustness
- [ ] RViz visualization launch file

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the repository
2. Create your branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push and open a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 🙏 Acknowledgements

- [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3) — PPO implementation
- [rf2o_laser_odometry](https://github.com/MAPIRlab/rf2o_laser_odometry) — Lidar scan-matching odometry
- [sllidar_ros2](https://github.com/Slamtec/sllidar_ros2) — Slamtec Lidar ROS2 driver
- [Gymnasium](https://gymnasium.farama.org/) — RL environment interface

---

<div align="center">

**Built with ❤️ by Mohammed Rayan**

[⭐ Star this repo](https://github.com/mohammedryn/mapless) · [🐛 Report a Bug](https://github.com/mohammedryn/mapless/issues) · [💡 Request a Feature](https://github.com/mohammedryn/mapless/issues)

</div>
