import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
import os
import subprocess
import time
import yaml
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry

from mapless_navigation import obs_utils


class ForestEnv(gym.Env):
    """Gymnasium environment for DRL training in Gazebo Harmonic simulation.

    Observation (362-dimensional float32):
        [0..359]  Lidar ranges normalised by LIDAR_MAX_RANGE (C1M1 R2 = 12 m)
        [360]     Normalised distance to goal in [0, 1]
        [361]     Normalised bearing to goal in [-1, 1]

    Action (2-dimensional float32 in [-1, 1]):
        [0] → linear velocity   mapped to [0, max_linear_vel] m/s
        [1] → angular velocity  mapped to [-max_angular_vel, max_angular_vel] rad/s

    Domain randomisation (per episode, applied at reset()):
        - Gaussian lidar noise to bridge the sim-to-real sensor gap.
        - Random linear-speed scale to account for motor-to-motor variation
          and battery voltage sag on the real JGB37 motors.
    """

    def __init__(self, config_path=None):
        super().__init__()

        if not rclpy.ok():
            rclpy.init()

        self.node = rclpy.create_node('forest_env_node')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.scan_sub = self.node.create_subscription(
            LaserScan, '/scan', self._scan_cb, qos)
        self.odom_sub = self.node.create_subscription(
            Odometry, '/odom', self._odom_cb, 10)
        self.cmd_vel_pub = self.node.create_publisher(TwistStamped, '/cmd_vel', 10)

        # ── Spaces ────────────────────────────────────────────────────────────
        n_obs = obs_utils.N_SCAN + 2
        obs_low        = np.zeros(n_obs, dtype=np.float32)
        obs_low[-1]    = -1.0          # bearing can be negative
        self.observation_space = spaces.Box(
            low=obs_low, high=1.0, shape=(n_obs,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # ── State ─────────────────────────────────────────────────────────────
        self.raw_scan_data = np.full(obs_utils.N_SCAN, obs_utils.LIDAR_MAX_RANGE)
        self.current_odom  = None
        self.goal_x        = 5.0
        self.goal_y        = 0.0
        self.step_count    = 0
        self.prev_distance = 0.0
        # Per-episode domain randomisation values (updated in reset())
        self.lidar_noise_std = 0.0
        self.speed_scale     = 1.0

        self._load_config()

    # ─────────────────────────────────────────── config loading ──────────────

    def _load_config(self):
        try:
            from ament_index_python.packages import get_package_share_directory
            pkg = get_package_share_directory('mapless_navigation')
            with open(os.path.join(pkg, 'config', 'rover.yaml')) as f:
                rover = yaml.safe_load(f)
            with open(os.path.join(pkg, 'config', 'training.yaml')) as f:
                train = yaml.safe_load(f)

            ec = train.get('env_config', {})
            dr = ec.get('domain_rand', {})

            self.max_linear_vel       = float(rover.get('max_linear_vel',  0.26))
            self.max_angular_vel      = float(rover.get('max_angular_vel', 1.82))
            self.collision_dist       = float(ec.get('collision_dist',       0.20))
            self.collision_penalty    = float(ec.get('collision_penalty',   100.0))
            self.goal_reached_dist    = float(ec.get('goal_reached_dist',    0.50))
            self.goal_reached_reward  = float(ec.get('goal_reached_reward', 100.0))
            self.prox_dist            = float(ec.get('proximity_penalty_dist',   0.50))
            self.prox_weight          = float(ec.get('proximity_penalty_weight', 5.0))
            self.ang_penalty_weight   = float(ec.get('angular_penalty_weight',   0.05))
            self.max_steps            = int(ec.get('max_steps', 500))
            self.goal_x_range         = ec.get('goal_x_range', [2.0, 8.0])
            self.goal_y_range         = ec.get('goal_y_range', [-3.0, 3.0])
            self.dr_noise_std         = float(dr.get('lidar_noise_std',  0.03))
            self.dr_speed_min         = float(dr.get('speed_scale_min',  0.85))
            self.dr_speed_max         = float(dr.get('speed_scale_max',  1.15))

        except Exception as e:
            self.node.get_logger().error(f"Config load failed, using defaults: {e}")
            self.max_linear_vel = 0.26;   self.max_angular_vel = 1.82
            self.collision_dist = 0.20;   self.collision_penalty = 100.0
            self.goal_reached_dist = 0.50; self.goal_reached_reward = 100.0
            self.prox_dist = 0.50;        self.prox_weight = 5.0
            self.ang_penalty_weight = 0.05
            self.max_steps = 500
            self.goal_x_range = [2.0, 8.0]; self.goal_y_range = [-3.0, 3.0]
            self.dr_noise_std = 0.03
            self.dr_speed_min = 0.85;     self.dr_speed_max = 1.15

    # ─────────────────────────────────────────── ROS 2 callbacks ─────────────

    def _scan_cb(self, msg):
        """Store raw range data (no processing here — obs_utils handles it)."""
        self.raw_scan_data = np.array(msg.ranges, dtype=np.float64)

    def _odom_cb(self, msg):
        self.current_odom = msg

    # ─────────────────────────────────────────── Gymnasium API ───────────────

    def step(self, action):
        rclpy.spin_once(self.node, timeout_sec=0.0)
        self.step_count += 1

        # Speed scale applies only to linear; angular is kept constant
        # (turning radius is a geometric property, not a speed constraint).
        linear_vel  = (action[0] + 1.0) / 2.0 * self.max_linear_vel * self.speed_scale
        angular_vel = action[1] * self.max_angular_vel

        cmd = TwistStamped()
        cmd.twist.linear.x  = float(linear_vel)
        cmd.twist.angular.z = float(angular_vel)
        self.cmd_vel_pub.publish(cmd)

        time.sleep(0.05)

        obs, raw_dist = self._get_obs()
        reward, done  = self._calculate_reward(raw_dist, action)
        truncated     = self.step_count >= self.max_steps

        return obs, reward, done, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.cmd_vel_pub.publish(TwistStamped())   # stop robot
        self.step_count = 0

        # ── Sample domain randomisation for this episode ──────────────────────
        # Lidar noise: random σ in [0, dr_noise_std] — teaches policy to be
        # robust to sensor imprecision.
        self.lidar_noise_std = float(np.random.uniform(0.0, self.dr_noise_std))
        # Speed scale: random factor in [dr_speed_min, dr_speed_max] — mimics
        # battery-level and motor-load variation on the real robot.
        self.speed_scale = float(np.random.uniform(self.dr_speed_min, self.dr_speed_max))

        # Randomise goal position.
        self.goal_x = float(np.random.uniform(*self.goal_x_range))
        self.goal_y = float(np.random.uniform(*self.goal_y_range))

        # Teleport robot back to spawn in Gazebo Harmonic.
        try:
            req = ('name: "burger", '
                   'position: {x: -2.0, y: -0.5, z: 0.05}, '
                   'orientation: {w: 1.0, x: 0.0, y: 0.0, z: 0.0}')
            subprocess.run(
                ['gz', 'service', '-s', '/world/default/set_pose',
                 '--reqtype', 'gz.msgs.Pose', '--reptype', 'gz.msgs.Boolean',
                 '--timeout', '2000', '--req', req],
                check=True, capture_output=True, timeout=3,
            )
        except subprocess.TimeoutExpired:
            self.node.get_logger().warn('set_pose timed out, continuing anyway')
        except subprocess.CalledProcessError as e:
            self.node.get_logger().error(f'set_pose failed: {e.stderr}')

        # Flush stale scan data so we don't trigger an immediate collision
        # on the first step with data from the previous episode.
        self.raw_scan_data = np.full(obs_utils.N_SCAN, obs_utils.LIDAR_MAX_RANGE)

        time.sleep(0.5)
        rclpy.spin_once(self.node, timeout_sec=0.1)

        obs, raw_dist      = self._get_obs()
        self.prev_distance = raw_dist

        return obs, {}

    # ─────────────────────────────────────────── internals ───────────────────

    def _get_obs(self):
        """Build observation using the shared obs_utils pipeline."""
        return obs_utils.build_observation(
            self.raw_scan_data,
            self.current_odom,
            self.goal_x,
            self.goal_y,
            lidar_noise_std=self.lidar_noise_std,
        )

    def _calculate_reward(self, raw_dist: float, action) -> tuple:
        # Collision check uses the TRUE (un-noised) minimum range so that
        # domain-randomisation noise cannot mask a real collision event.
        raw = np.array(self.raw_scan_data, dtype=np.float64)
        raw = np.nan_to_num(raw, nan=obs_utils.LIDAR_MAX_RANGE,
                            posinf=obs_utils.LIDAR_MAX_RANGE, neginf=0.0)
        min_laser = float(np.min(raw))

        reward = 0.0
        done   = False

        if min_laser < self.collision_dist:
            # Terminal collision penalty.
            reward = -self.collision_penalty
            done   = True

        elif raw_dist < self.goal_reached_dist:
            # Terminal goal-reached reward.
            reward = self.goal_reached_reward
            done   = True

        else:
            # Dense progress reward: each step that moves closer to goal.
            reward += (self.prev_distance - raw_dist) * 10.0

            # Proximity penalty: discourages the policy from hugging walls
            # even when it avoids hard collision.
            if min_laser < self.prox_dist:
                reward -= self.prox_weight * (self.prox_dist - min_laser)

            # Angular velocity penalty: promotes smooth, straight-line motion.
            reward -= self.ang_penalty_weight * abs(float(action[1]))

            # Per-step time penalty to encourage efficient paths.
            reward -= 0.1

            self.prev_distance = raw_dist

        return reward, done

    def close(self):
        self.node.destroy_node()
        rclpy.shutdown()
