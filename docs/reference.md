# Mapless DRL Navigation — Technical Reference

**End-to-end deep reinforcement learning for autonomous rover navigation without maps.**

> This document describes the **actual, implemented system** as of the current codebase.
> Every detail — observation dimensions, reward weights, network architecture, config keys —
> matches the running code.  It is intended as a primary reference for code review,
> technical interviews, and future development.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Hardware Stack](#2-hardware-stack)
3. [Software Architecture](#3-software-architecture)
4. [Observation Space](#4-observation-space)
5. [Action Space](#5-action-space)
6. [Reward Function](#6-reward-function)
7. [Policy Network Architecture](#7-policy-network-architecture)
8. [Domain Randomisation](#8-domain-randomisation)
9. [Sensor Fusion (EKF)](#9-sensor-fusion-ekf)
10. [Training Pipeline](#10-training-pipeline)
11. [Sim-to-Real Transfer](#11-sim-to-real-transfer)
12. [Deployment Pipeline](#12-deployment-pipeline)
13. [Evaluation Methodology](#13-evaluation-methodology)
14. [Configuration Reference](#14-configuration-reference)
15. [Known Limitations and Future Work](#15-known-limitations-and-future-work)

---

## 1. System Overview

This project implements mapless navigation: a ground robot learns a reactive obstacle-avoidance
and goal-reaching policy from simulated experience, then deploys the policy on real hardware
without map construction or localization.

**What makes this approach different from standard SLAM + path-planning:**

| Property | Traditional (SLAM + Nav2) | This Project (DRL Mapless) |
|---|---|---|
| World model | Pre-built occupancy map | None — purely reactive |
| Compute at runtime | Path planner + controller | Single forward pass through MLP |
| New environments | Requires re-mapping | Zero-shot generalization |
| Dynamic obstacles | Requires re-planning | Handled implicitly by policy |
| Odometry requirement | Full localization | Relative goal bearing only |

The core design choices:
- **PPO** (or SAC) because the task has dense reward signal and a continuous action space; both are well-understood with many reference implementations.
- **Lidar-only observations** to sidestep camera calibration and lighting variance in early hardware testing.
- **rf2o scan-matching** for encoder-less odometry — avoids wheel encoders entirely on the JGB37 motors.
- **MPU-6050 EKF fusion** to compensate for rf2o drift during fast turns.

---

## 2. Hardware Stack

### Compute

| Component | Part | Notes |
|---|---|---|
| SBC | Raspberry Pi 5 (8 GB RAM) | ROS 2 Jazzy, gpiozero + lgpio backend for Pi 5 GPIO |
| OS | Ubuntu 24.04 LTS | Required for ROS 2 Jazzy |

### Sensors

| Sensor | Part | Interface | Key Spec |
|---|---|---|---|
| Lidar | Slamtec C1M1 R2 | USB → `/dev/ttyUSB0` | 360°, 12 m range, DTOF, 5000 samples/s |
| IMU | MPU-6050 | I2C bus 1, addr `0x68` | 3-axis gyro + 3-axis accel, 50 Hz |
| Camera | Pi HQ Camera (IMX477) | CSI | 12.3 MP — reserved for future visual goal work |

**Critical lidar note:** The C1M1 R2 has a 12.0 m maximum range. Earlier versions of this
codebase (and the simulation model) used 3.5 m (RPLIDAR A1 spec).
Normalising by the wrong constant clips all readings between 3.5 m and 12 m to 1.0 —
the policy receives a saturated observation it was never trained on. This has been
corrected: `LIDAR_MAX_RANGE = 12.0` in `obs_utils.py` and `max_range: 12.0` in
`config/training.yaml`.

### Actuation

| Component | Part | Interface |
|---|---|---|
| Motor drivers | 2x BTS7960 43A H-bridge | GPIO/PWM via `gpiozero` (lgpio) |
| Motors | 4x JGB37 DC gear motor | No encoders — 4WD skid-steer |
| Power | 12V LiPo | Buck converter → 5V/5A for Pi 5 |

---

## 3. Software Architecture

### ROS 2 Node Graph (real robot)

```
/scan ──────────────────────────────────────────────────────────────┐
  │                                                                  │
  ▼                                                                  ▼
rf2o_laser_odometry_node          NavigationNode (10 Hz)
  (scan-matching → /odom)           ├─ /scan          ─ LaserScan
       │                             ├─ /odometry/filtered ─ Odometry (EKF)
       ▼ /odom                       │    (fallback: /odom)
robot_localization:ekf_node         ├─ /goal_xy ─ Float64MultiArray
  (fuses /odom + /imu/data           └─ /cmd_vel ──►  Twist
   → /odometry/filtered)
       ▲ /imu/data
mpu6050_driver_node

/cmd_vel ──►  bts7960_driver  ──► GPIO PWM ──► JGB37 motors
              (also reads /scan for hardware safety override)
```

### The obs_utils contract

`obs_utils.py` is the single source of truth for observation construction.
Both `ForestEnv` (training) and `NavigationNode` (deployment) call
`obs_utils.build_observation()`. If the observation pipeline ever changes,
it changes in one place and both contexts pick it up automatically.
This eliminates silent failure where training and deployment observe different
distributions and the policy underperforms at test time.

---

## 4. Observation Space

**Dimension:** 362-dimensional `float32` vector.

| Index | Quantity | Construction | Range |
|---|---|---|---|
| `[0..359]` | 360 lidar ranges | Downsampled/padded to 360 pts; divided by `LIDAR_MAX_RANGE` (12.0 m); NaN/inf → 1.0 | [0, 1] |
| `[360]` | Distance to goal | Euclidean metres / 10.0, clipped to [0, 1] | [0, 1] |
| `[361]` | Bearing to goal | `atan2(dy, dx) − yaw`, wrapped to [−π, π], divided by π | [−1, 1] |

**Implementation:** `obs_utils.build_observation()` in `mapless_navigation/obs_utils.py`.

**Scan resampling:** If the lidar returns more than 360 points, every `floor(N/360)`-th
reading is taken. If fewer (unlikely for C1M1 R2), zero-padding at `max_range` is applied.

**Lidar noise (training only):** Per-episode Gaussian noise `N(0, σ)` is added before
normalisation. `σ` is sampled uniformly from `[0, lidar_noise_std]` at each `reset()`.
Deployment always calls `build_observation(..., lidar_noise_std=0.0)`.

---

## 5. Action Space

**Dimension:** 2-dimensional `float32` in `[−1, 1]`.

| Index | Policy output | Mapped value |
|---|---|---|
| `[0]` | Raw linear command | `(a[0] + 1) / 2 × max_linear_vel × speed_scale` → [0, 0.26] m/s |
| `[1]` | Raw angular command | `a[1] × max_angular_vel` → [−1.82, 1.82] rad/s |

The linear mapping restricts to forward-only motion. Reverse is not needed for the
current goal-reaching task and simplifies the learned policy.

`speed_scale` is a per-episode domain randomisation factor. It applies only to linear
velocity because angular velocity is a purely geometric constraint (turning circle),
not affected by motor output variance.

---

## 6. Reward Function

Implemented in `ForestEnv._calculate_reward()` in `mapless_navigation/forest_env.py`.

```
at each step:

  min_laser = min(raw, unnoised lidar ranges)

  if min_laser < collision_dist (0.20 m):
      reward = −collision_penalty (−100)
      done   = True

  elif dist_to_goal < goal_reached_dist (0.50 m):
      reward = +goal_reached_reward (+100)
      done   = True

  else:
      progress = prev_distance − curr_distance
      reward   = progress × 10.0
               − prox_weight × max(0, prox_dist − min_laser)
               − ang_penalty_weight × |action[1]|
               − 0.1                               # time penalty
```

**Design rationale:**

| Component | Purpose |
|---|---|
| Progress × 10 | Dense signal — policy gets reward every step, not only at goal |
| Proximity penalty | Discourages hugging walls even when collision threshold is not reached |
| Angular penalty | Promotes smooth, energy-efficient paths; discourages oscillatory turning |
| Time penalty −0.1 | Encourages shortest path; prevents standing still |
| Collision −100 | Hard terminal signal |
| Goal +100 | Terminal positive signal; sized to dominate episode return |

**Collision detection uses unnoised ranges.** Domain-randomisation noise is added to
the observation but not to the reward collision check. The penalty signal is accurate
regardless of noise level.

---

## 7. Policy Network Architecture

### Option A: Conv1D (default, `policy_type: conv1d`)

Implemented in `LidarConvExtractor` in `mapless_navigation/train_ppo.py`.

```
Input: 362-dim observation
         │
    ┌────┴───────────────────────┐   ┌─────────────┐
    │ Lidar branch (360 pts)     │   │ Goal branch │
    │ unsqueeze → (B, 1, 360)    │   │  (2 values) │
    │ Conv1d(1→32, k=5)          │   │ Linear(2→32)│
    │ ReLU                       │   │ ReLU        │
    │ Conv1d(32→64, k=3)         │   └─────┬───────┘
    │ ReLU                       │         │ 32
    │ AdaptiveAvgPool1d(45)      │         │
    │ Flatten → 2880             │         │
    └────────────────┬───────────┘         │
                     └─────────────────────┘
                                │ 2912
                        Linear(2912 → 256)
                        ReLU
                                │ 256
                      SB3 Actor / Critic heads
```

**Why Conv1D for lidar?** The 360 rays form a circular 1-D signal. Adjacent rays are
physically adjacent angles. A 1-D conv kernel learns local gap and edge detectors
(e.g., "gap of width W at angle θ") which are translation-equivariant across the scan
ring. A flat MLP has no such prior and must learn angle-specific features separately.

### Option B: Flat MLP (`policy_type: mlp`)

Standard `MlpPolicy` from SB3 with default hidden layers [64, 64]. Used as a baseline
to quantify the benefit of Conv1D in ablation experiments.

---

## 8. Domain Randomisation

Applied at every `ForestEnv.reset()`. All parameters are in
`config/training.yaml` under `env_config.domain_rand`.

| Parameter | Default | Effect |
|---|---|---|
| `lidar_noise_std` | 0.03 m | Maximum σ for Gaussian lidar noise. Per episode, σ ~ Uniform(0, 0.03). Applied before normalisation. |
| `speed_scale_min` | 0.85 | Lower bound for per-episode speed scale factor. |
| `speed_scale_max` | 1.15 | Upper bound. Scale ~ Uniform(0.85, 1.15) per episode; applied to `max_linear_vel`. |

**Motivation:**
1. **Lidar noise** — the simulated sensor is noiseless; the real C1M1 R2 has
   ±2–3 cm measurement uncertainty. Injecting noise trains the policy to be
   robust to small perturbations rather than overfitting to perfect range values.
2. **Speed scale** — the JGB37 motors have no encoders. Actual wheel speed varies
   with battery charge, load, and motor-to-motor manufacturing differences.
   Randomising effective speed prevents the policy from assuming precise velocity.

**To disable domain randomisation** (clean baseline run):
```yaml
domain_rand:
  lidar_noise_std: 0.0
  speed_scale_min: 1.0
  speed_scale_max: 1.0
```

---

## 9. Sensor Fusion (EKF)

### Problem: lidar-only odometry limitations

`rf2o_laser_odometry` degrades in two scenarios:
- **Featureless areas** — long straight corridors with no perpendicular features
  provide insufficient scan-to-scan constraints on yaw.
- **Fast rotations** — at 10 Hz, aggressive in-place turns can move features out
  of the field of view between scans.

### Solution: MPU-6050 + robot_localization EKF

The `robot_localization` EKF fuses `/odom` (rf2o) and `/imu/data` (MPU-6050) to
produce `/odometry/filtered`.

| Source | Fused quantities |
|---|---|
| rf2o (`/odom`) | x, y position; yaw; x velocity; yaw-rate |
| MPU-6050 (`/imu/data`) | yaw orientation; yaw angular rate; x/y linear acceleration |

The IMU gyroscope provides 50 Hz yaw estimates that bridge the 10 Hz gaps in rf2o.
The accelerometer provides forward acceleration independent of scan quality.

**Key EKF config settings** (`config/ekf.yaml`):
- `two_d_mode: true` — ground plane constraint
- `imu0_remove_gravitational_acceleration: true` — essential for MPU-6050 (z-accel = +9.8 m/s² at rest)
- `frequency: 30.0 Hz` — 3× the rf2o rate for smooth output

### Graceful fallback

`NavigationNode` subscribes to both `/odometry/filtered` and `/odom`. If the EKF
is not running, the node falls back to raw rf2o odometry automatically.

---

## 10. Training Pipeline

### Environment flow

```
ForestEnv.reset()
  ├─ Stop robot (empty TwistStamped)
  ├─ Sample domain randomisation params (lidar_noise_std, speed_scale)
  ├─ Randomise goal position within config bounds
  ├─ Teleport robot to spawn via gz service CLI
  ├─ Flush stale scan buffer (fill with max_range)
  └─ Return initial observation

ForestEnv.step(action)
  ├─ Map action → velocity (with speed_scale on linear)
  ├─ Publish TwistStamped to /cmd_vel
  ├─ Sleep 50 ms
  ├─ Build observation via obs_utils.build_observation()
  ├─ Compute reward via _calculate_reward()
  └─ Return (obs, reward, done, truncated, info)
```

### PPO hyperparameters (from `config/training.yaml`)

| Parameter | Value | Rationale |
|---|---|---|
| `n_steps` | 2048 | Steps collected before each policy update |
| `batch_size` | 64 | Mini-batch size |
| `n_epochs` | 10 | Passes over each rollout buffer |
| `gamma` | 0.99 | High discount — must plan across ~500 steps |
| `gae_lambda` | 0.95 | GAE trade-off between bias and variance |
| `clip_range` | 0.2 | PPO clipping |
| `ent_coef` | 0.005 | Entropy bonus for exploration |

### Callbacks

- `CheckpointCallback`: saves every 50,000 steps to `models/checkpoints/`
- `EvalCallback`: evaluates 20 episodes every 25,000 steps; saves best model to `models/best_model/`

---

## 11. Sim-to-Real Transfer

### Sources of the sim-to-real gap

| Gap | Issue | Mitigation |
|---|---|---|
| Lidar max range | Previously 3.5 m vs C1M1's 12.0 m | **Fixed**: `max_range: 12.0` everywhere |
| Lidar noise | Simulation noiseless; real sensor ±2–3 cm | Domain rand: `lidar_noise_std: 0.03` |
| Motor variation | ±15% across battery levels and motors | Domain rand: speed scale [0.85, 1.15] |
| Observation pipeline | Previously separate code in training vs deploy | **Fixed**: single `obs_utils.py` |
| Physics fidelity | Ideal contact vs real tire/floor | Lower `max_linear_vel` for first real tests |
| Odometry quality | Simulation perfect; rf2o drifts | IMU EKF fusion on real robot |

---

## 12. Deployment Pipeline

### Full stack launch

```bash
ros2 launch mapless_navigation real_robot.launch.py goal_x:=5.0 goal_y:=0.0
```

Starts 8 nodes: lidar driver → TF publishers → rf2o odometry → IMU driver → EKF → motor driver → navigation node.

### NavigationNode control loop

```python
def _control_loop(self):
    odom = odom_filtered or odom_raw        # EKF preferred, rf2o fallback
    obs, _ = obs_utils.build_observation(   # identical to training
        raw_scan, odom, goal_x, goal_y, lidar_noise_std=0.0)
    action, _ = model.predict(obs, deterministic=True)
    twist.linear.x  = (action[0] + 1) / 2 * max_linear_vel
    twist.angular.z = action[1] * max_angular_vel
    cmd_vel_pub.publish(twist)
```

### Runtime goal updates (no node restart)

```bash
ros2 topic pub /goal_xy std_msgs/msg/Float64MultiArray '{data: [3.0, 1.5]}'
```

### Hardware safety override

`bts7960_driver.py` subscribes to `/scan` independently. If any range falls below
`scan_min_dist` (0.15 m), all PWM outputs are set to zero immediately, overriding
any policy command. This layer is independent of the learned policy.

---

## 13. Evaluation Methodology

### Running the benchmark

```bash
# Terminal 1:
ros2 launch mapless_navigation forest_sim.launch.xml

# Terminal 2:
ros2 run mapless_navigation evaluate_policy --episodes 100
```

### Metrics

| Metric | Description |
|---|---|
| **Success rate** | Episodes where `dist_to_goal < 0.5 m` before timeout |
| **Collision rate** | Episodes where `min_lidar < 0.20 m` |
| **Timeout rate** | Episodes reaching `max_steps = 500` without outcome |
| **Avg steps** | Mean episode length |
| **Avg final dist** | Mean distance to goal at episode end |

### Ablation template

```bash
# Train both architectures
ros2 run mapless_navigation train_ppo --policy conv1d  # default
ros2 run mapless_navigation train_ppo --policy mlp

# Evaluate both
ros2 run mapless_navigation evaluate_policy --model models/ppo_forest_nav
# (repeat with mlp model path)

# Compare algorithms
ros2 run mapless_navigation train_ppo --algorithm sac
ros2 run mapless_navigation evaluate_policy --algorithm sac --model models/sac_forest_nav
```

---

## 14. Configuration Reference

### `config/training.yaml` — complete key list

| Key | Type | Description |
|---|---|---|
| `algorithm` | str | `"ppo"` or `"sac"` |
| `total_timesteps` | int | Total environment steps |
| `policy_type` | str | `"conv1d"` or `"mlp"` |
| `conv_features_dim` | int | Conv1D extractor output dimension |
| `env_config.max_range` | float | **Lidar max range — must match real sensor (12.0 m)** |
| `env_config.collision_dist` | float | Collision threshold (m) |
| `env_config.goal_reached_dist` | float | Success threshold (m) |
| `env_config.proximity_penalty_dist` | float | Wall-hugging penalty activation (m) |
| `env_config.proximity_penalty_weight` | float | Wall-hugging penalty scale |
| `env_config.angular_penalty_weight` | float | Turn smoothness penalty scale |
| `env_config.max_steps` | int | Episode truncation |
| `env_config.domain_rand.lidar_noise_std` | float | Max lidar noise σ (m) |
| `env_config.domain_rand.speed_scale_min/max` | float | Motor speed variation range |
| `eval_freq` | int | Steps between EvalCallback runs |
| `checkpoint_freq` | int | Steps between checkpoint saves |

---

## 15. Known Limitations and Future Work

### Known limitations

**1. Unsynchronised Gazebo stepping**
`ForestEnv.step()` sleeps 50 ms after publishing a command. This is not synchronised
with Gazebo's physics tick. Proper synchronisation via `/world/default/control` would
give more accurate training.

**2. Odometry drift on long missions**
Goal coordinates are in the odometry frame, which drifts over time. For missions
beyond ~20 m from the start point, fiducial re-localisation is needed.

**3. No reverse motion**
The action space maps to forward-only linear velocity. In very tight spaces the
robot may get stuck where reversing would be optimal.

**4. Camera unused**
The Pi HQ Camera is mounted but not integrated into the policy or goal interface.

### Future work (priority order)

1. **Real deployment + video** — highest impact for CV/interviews.
2. **Ablation table** — run Conv1D vs MLP, PPO vs SAC; fill in numbers from `evaluate_policy`.
3. **Visual goal specification** — use Pi HQ Camera to detect ArUco markers; replace
   hardcoded coordinates with camera-computed relative pose.
4. **Isaac Lab migration** — port to NVIDIA Isaac Lab for massively parallel training.
5. **SAC convergence comparison** — reward curves and success rate vs. PPO.
