# 📋 Project Review: Mapless DRL Forest Navigation

**Reviewer:** GitHub Copilot  
**Review Date:** February 2026  
**Repository:** [mohammedryn/mapless](https://github.com/mohammedryn/mapless)

---

## 🏆 Overall Score: 7 / 10

This is a genuinely impressive robotics project for its scope and goal, especially as a personal/academic build. It demonstrates solid understanding of ROS2, reinforcement learning, and embedded hardware. A few structural and code-quality issues hold it back from a higher score.

---

## ✅ What's Great

### 1. Architecture & Concept (Excellent)
- **Mapless DRL navigation** is a research-grade concept executed end-to-end by one person on real hardware.
- The **encoder-less odometry** idea — using `rf2o_laser_odometry` for pose estimation instead of wheel encoders — is a genuine engineering novelty and significantly reduces hardware cost and failure points.
- Clean **separation of concerns**: training (`forest_env.py`, `train_ppo.py`), deployment (`navigation_node.py`), and hardware (`sabertooth_driver.py`) are properly separated.
- The **reward function design** (collision penalty −100, goal reward +100, progress shaping) is well-thought-out and matches best practices for this type of DRL navigation task.

### 2. ROS2 Practices (Good)
- Proper use of **QoS profiles** (`BEST_EFFORT` + `KEEP_LAST` for LiDAR) — this is the correct approach for sensor topics.
- **ROS2 parameters** are declared properly with defaults in `navigation_node.py` and `sabertooth_driver.py`.
- The **launch file system** is well-structured: a Python launch file for hardware, XML files for sim, and a unified `forest_nav.launch.xml` with a `mode` argument.
- Static TF publishing (`base_link → laser`) is included in the launch file.

### 3. Documentation (Very Good)
- The `README.md` is professionally written with a clear architecture diagram, tables, code blocks, hardware BOM, and a troubleshooting guide.
- `docs/SETUP.md` and `docs/CONNECTIONS.md` are detailed and hardware-specific guides that lower the barrier to replication.
- `docs/PROGRESS.md` acts as a technical journal, explaining *why* decisions were made (e.g., choosing `rf2o` for odometry).
- Badges, emojis, and visual formatting make the README accessible.

### 4. Observation Space Design (Good)
- 360-point LiDAR + goal distance + goal angle = 362-dimensional observation is well-justified and consistent across training and inference.
- Action scaling (normalized `[−1, 1]` during training, denormalized in deployment) is consistent between `forest_env.py` and `navigation_node.py` — a common source of sim-to-real bugs.
- NaN/Inf handling in `scan_callback` (replacing with `3.5` max range) is a practical defensive measure.

### 5. Hardware Driver (Solid)
- `sabertooth_driver.py` correctly implements the **Sabertooth Packet Serial** 4-byte protocol (`[address, command, value, checksum & 0x7F]`).
- Motor stop is registered with `rclpy.get_default_context().on_shutdown` — motors won't keep spinning if the node crashes.

---

## ⚠️ Issues Found

### 🔴 Critical

#### 1. Training Config YAML Is Ignored
**File:** `src/train_ppo.py`  
The file `config/training.yaml` defines all PPO hyperparameters, but `train_ppo.py` **never reads it** — the hyperparameters are hardcoded directly in the Python file. The config file is effectively dead code.

```python
# train_ppo.py — hardcoded values that should come from config/training.yaml
model = PPO("MlpPolicy", env, learning_rate=0.0003, n_steps=2048, ...)
```

**Fix:** Load `training.yaml` with `yaml.safe_load()` and pass the values to `PPO(...)`.

---

#### 2. Model Save Path Is Overwritten
**File:** `src/train_ppo.py`, lines 44 and 62

```python
model_path = "models/ppo_forest_nav"  # Line 44 — used for loading
# ... training happens ...
model_path = "ppo_forest_nav"          # Line 62 — overwrites it, saves to wrong location!
model.save(model_path)
```

After training, the model is saved to `./ppo_forest_nav.zip` (in the current working directory), NOT to `models/` where the deployment node expects it. The `--continue_training` flag then won't find it.

---

#### 3. No LICENSE File
`README.md` and `package.xml` both mention **MIT License**, but there is **no `LICENSE` file** in the repository. This means the code has no legally enforceable open-source license.

---

### 🟡 Moderate

#### 4. `package.xml` Has Placeholder Values
```xml
<maintainer email="user@example.com">User</maintainer>
<license>TODO: License declaration</license>
```
Both `package.xml` and `setup.py` contain unfinished placeholder values that should be updated with the real author name and license.

---

#### 5. TensorBoard Logs & Model Committed to Git
The `ppo_forest_tensorboard/` directory (8 training run event files) and `models/ppo_forest_nav.zip` (binary ML model) are committed directly to the repository. Binary blobs and training artifacts should be in `.gitignore` or hosted on a release page / model registry.

**Current `.gitignore`:**
```
*.mp4
install/
build/
log/
__pycache__/
*.pyc
```
Missing: `ppo_forest_tensorboard/`, `models/*.zip`

---

#### 6. Differential Drive Mixing Ignores Track Width
**File:** `src/sabertooth_driver.py`

```python
left_speed = linear - angular   # Missing: * (track_width / 2)
right_speed = linear + angular
```

Proper differential drive kinematics requires the **track width (wheelbase)** to convert angular velocity (rad/s) to wheel speed difference. Without it, the robot's turning behavior will be incorrect and the deployment will not match the training environment.

**Correct formula:**
```python
left_speed  = linear - angular * (track_width / 2.0)
right_speed = linear + angular * (track_width / 2.0)
```

---

#### 7. No Watchdog Timer in Motor Driver
**File:** `src/sabertooth_driver.py`  
If the navigation node crashes or the topic stops publishing, the Sabertooth driver will hold the **last received velocity command indefinitely**. This means the robot will keep moving until physically stopped. A safety watchdog timer that sends zero velocity after a timeout (e.g., 0.5 s) is strongly recommended.

---

#### 8. Wrong Robot Reference in Config
**File:** `config/rover.yaml`, line 1
```yaml
robot_radius: 0.22  # meters (TurtleBot3 Waffle Pi)
```
The comment references a **TurtleBot3**, but this is a custom rover with different dimensions. This suggests the config was copied from a TurtleBot3 project and not fully updated.

---

#### 9. Model Path Is Relative in Navigation Node
**File:** `src/navigation_node.py`, line 18
```python
self.declare_parameter('model_path', 'models/ppo_forest_nav')
```
The default path is **relative**, which means it depends on the working directory when the node is launched. On a real robot (Jetson), this will fail unless the launch is performed from exactly the right directory. It should use an absolute path or resolve the path relative to the package share directory using `get_package_share_directory()`.

---

### 🟢 Minor

#### 10. No Unit Tests
There are no tests in the repository despite the `package.xml` listing `ament_pep257`, `ament_flake8`, and `python3-pytest` as test dependencies. Even basic unit tests for the reward function or observation normalization logic would improve reliability.

#### 11. Simulation Launch Uses TurtleBot3 Model
**File:** `launch/forest_sim.launch.xml`
```xml
<set_env name="TURTLEBOT3_MODEL" value="burger"/>
<include file="$(find-pkg-share turtlebot3_gazebo)/launch/turtlebot3_world.launch.py"/>
```
Training uses the **TurtleBot3 Burger** in Gazebo, but the real robot is a **custom rover with JGB37 motors**. This is a known sim-to-real gap, but the launch file doesn't document this or reference a custom URDF/world that better matches the physical robot.

#### 12. `time.sleep()` in Training Step
**File:** `src/forest_env.py`, line 113
```python
time.sleep(0.05)  # Wait a bit for action to take effect
```
Using `time.sleep()` for simulation synchronization is fragile and wastes wall-clock time. A proper ROS2 training loop should use Gazebo's `/clock` topic or a simulation step service for deterministic, reproducible training.

#### 13. Duplicate File Structure
The repository contains a nested `mapless/` folder (i.e., `mapless/mapless/`) with duplicate `src/__init__.py`, `package.xml`, `requirements.txt`, and `setup.py` files. This suggests a packaging/structure inconsistency that could confuse `colcon build`.

#### 14. `forest_sim.launch.xml` Doesn't Launch a Forest
The simulation launch file starts `turtlebot3_world.launch.py` — which is a standard TurtleBot3 demo world, **not a forest**. There is no custom Gazebo world defined in the repository for the forest environment that the training uses.

---

## 📊 Dimension Scorecard

| Dimension | Score | Notes |
|:---|:---:|:---|
| **Concept & Innovation** | 9/10 | Genuinely novel: DRL + encoder-less Lidar odometry on edge hardware |
| **Code Quality** | 6/10 | Clean structure but config not wired, critical model save bug |
| **Documentation** | 8/10 | Excellent README, setup guide, and wiring doc; missing LICENSE |
| **ROS2 Architecture** | 7/10 | Correct patterns; no watchdog, relative model path |
| **Testing** | 2/10 | No tests at all despite test dependencies declared |
| **Hardware Design** | 7/10 | Good Sabertooth driver; drive mixing needs track width |
| **Reproducibility** | 5/10 | No forest sim world; TensorBoard & model in repo; config ignored |

---

## 🗺️ Priority Fixes (Recommended Order)

1. **Fix the model save path bug** in `train_ppo.py` (line 62) — critical for the training workflow to work.
2. **Wire up `config/training.yaml`** to `train_ppo.py` — the config infrastructure is there, just not connected.
3. **Add a `LICENSE` file** (MIT) to legitimize the open-source claim.
4. **Fix differential drive kinematics** in `sabertooth_driver.py` to include track width.
5. **Add watchdog timer** in `sabertooth_driver.py` for safety.
6. **Add `ppo_forest_tensorboard/` and `models/*.zip` to `.gitignore`** — these don't belong in source control.
7. **Update `package.xml`** and `setup.py` placeholder values.
8. **Use absolute model path** in `navigation_node.py` via `get_package_share_directory()`.

---

## 💡 Conclusion

This project sits at the intersection of robotics, deep reinforcement learning, and embedded systems — a technically ambitious combination for a personal/academic project. The high-level architecture is sound, the documentation is genuinely excellent, and the end-to-end DRL + encoder-less odometry approach is innovative.

The critical issues (model save path bug, training config not wired) are fixable in under an hour. Addressing the safety concerns (watchdog timer, drive mixing) would make it physically safer to test. With those fixes applied, this is a project that stands on par with many academic robotics repositories.

**Bottom line: Impressive concept and documentation, needs a focused code-quality pass before the real-robot deployment phase.**
