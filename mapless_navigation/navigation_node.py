import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
from stable_baselines3 import PPO, SAC
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from mapless_navigation import obs_utils


class NavigationNode(Node):
    """Real-time inference node for the trained DRL navigation policy.

    Runs at 10 Hz. Builds the 362-dimensional observation vector via
    obs_utils (the same pipeline used in ForestEnv during training), runs
    model.predict(), and publishes Twist commands to /cmd_vel.

    Preferred odometry source: /odometry/filtered (robot_localization EKF
    fusing MPU-6050 IMU + rf2o lidar odometry).  Falls back to /odom
    (raw rf2o) if the EKF node is not running.

    Parameters
    ----------
    model_path      : str   — path to the .zip model (omit extension)
    algorithm       : str   — "ppo" | "sac"
    max_linear_vel  : float — maximum forward speed (m/s)
    max_angular_vel : float — maximum rotation rate (rad/s)
    goal_x          : float — goal x-coordinate in the odometry frame (m)
    goal_y          : float — goal y-coordinate in the odometry frame (m)

    The goal can be updated at runtime without restarting the node:
        ros2 topic pub /goal_xy std_msgs/msg/Float64MultiArray '{data: [5.0, 2.0]}'
    """

    def __init__(self):
        super().__init__('navigation_node')

        # ── ROS 2 Parameters ──────────────────────────────────────────────────
        self.declare_parameter('model_path',      'models/ppo_forest_nav')
        self.declare_parameter('algorithm',       'ppo')
        self.declare_parameter('max_linear_vel',  0.26)
        self.declare_parameter('max_angular_vel', 1.82)
        self.declare_parameter('goal_x',          5.0)
        self.declare_parameter('goal_y',          0.0)

        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        algo       = self.get_parameter('algorithm').get_parameter_value().string_value.lower()

        AlgoClass = SAC if algo == 'sac' else PPO
        try:
            self.model = AlgoClass.load(model_path)
            self.model.set_training_mode(False)
            self.get_logger().info(f"Loaded {algo.upper()} model from '{model_path}'")
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            self.model = None

        # ── QoS profile for the lidar subscription ────────────────────────────
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── Subscriptions ─────────────────────────────────────────────────────
        self.create_subscription(LaserScan, '/scan', self._scan_cb, qos)

        # /odometry/filtered: EKF-fused pose (IMU + rf2o) — preferred.
        self.create_subscription(
            Odometry, '/odometry/filtered', self._odom_filtered_cb, 10)
        # /odom: raw rf2o output — fallback if EKF is not running.
        self.create_subscription(
            Odometry, '/odom', self._odom_raw_cb, 10)

        # /goal_xy: runtime goal updates [x, y] without node restart.
        self.create_subscription(
            Float64MultiArray, '/goal_xy', self._goal_cb, 10)

        # ── Publisher ─────────────────────────────────────────────────────────
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # ── State ─────────────────────────────────────────────────────────────
        self.raw_scan_data    = np.full(obs_utils.N_SCAN, obs_utils.LIDAR_MAX_RANGE)
        self.odom_filtered    = None   # EKF output (preferred)
        self.odom_raw         = None   # rf2o fallback

        self.goal_x = float(
            self.get_parameter('goal_x').get_parameter_value().double_value)
        self.goal_y = float(
            self.get_parameter('goal_y').get_parameter_value().double_value)

        self.create_timer(0.1, self._control_loop)   # 10 Hz

        self.get_logger().info(
            f"NavigationNode ready — goal: ({self.goal_x:.2f}, {self.goal_y:.2f})")

    # ─────────────────────────────────────────── callbacks ───────────────────

    def _scan_cb(self, msg):
        self.raw_scan_data = np.array(msg.ranges, dtype=np.float64)

    def _odom_filtered_cb(self, msg):
        """EKF-fused odometry (IMU + rf2o) — used when robot_localization runs."""
        self.odom_filtered = msg

    def _odom_raw_cb(self, msg):
        """Raw rf2o odometry — fallback when EKF node is not running."""
        self.odom_raw = msg

    def _goal_cb(self, msg):
        """Update navigation goal at runtime via /goal_xy topic."""
        if len(msg.data) >= 2:
            self.goal_x = float(msg.data[0])
            self.goal_y = float(msg.data[1])
            self.get_logger().info(
                f"Goal updated: ({self.goal_x:.2f}, {self.goal_y:.2f})")

    # ─────────────────────────────────────────── control loop ────────────────

    def _control_loop(self):
        if self.model is None:
            return

        # Prefer EKF-fused odometry; fall back to raw rf2o.
        odom = (self.odom_filtered
                if self.odom_filtered is not None
                else self.odom_raw)

        # Build observation using the SAME pipeline as training (obs_utils).
        # noise_std=0.0: no artificial noise during deployment.
        obs, _ = obs_utils.build_observation(
            self.raw_scan_data,
            odom,
            self.goal_x,
            self.goal_y,
            lidar_noise_std=0.0,
        )

        action, _ = self.model.predict(obs, deterministic=True)

        max_linear_vel  = (self.get_parameter('max_linear_vel')
                           .get_parameter_value().double_value)
        max_angular_vel = (self.get_parameter('max_angular_vel')
                           .get_parameter_value().double_value)

        twist = Twist()
        twist.linear.x  = float((action[0] + 1.0) / 2.0 * max_linear_vel)
        twist.angular.z = float(action[1] * max_angular_vel)
        self.cmd_vel_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = NavigationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
