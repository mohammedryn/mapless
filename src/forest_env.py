import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import time
import math
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_srvs.srv import Empty

class ForestEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    """
    def __init__(self, config_path=None):
        super(ForestEnv, self).__init__()

        # Initialize ROS2
        if not rclpy.ok():
            rclpy.init()
        
        self.node = rclpy.create_node('forest_env_node')
        
        # QoS for LaserScan
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscribers and Publishers
        self.scan_sub = self.node.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos_profile
        )
        self.odom_sub = self.node.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        self.cmd_vel_pub = self.node.create_publisher(Twist, '/cmd_vel', 10)
        
        # Reset Service Client
        self.reset_client = self.node.create_client(Empty, '/reset_simulation')

        # Action space: [linear_vel, angular_vel]
        # Normalized action space [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation space: LaserScan ranges (normalized) + Goal info (distance, angle)
        # Assuming 360 scan points
        self.n_scan = 360
        low = np.zeros(self.n_scan + 2)
        low[-1] = -1.0 # Angle can be negative
        
        self.observation_space = spaces.Box(
            low=low, 
            high=1.0, 
            shape=(self.n_scan + 2,), # Scan + Goal Dist + Goal Angle
            dtype=np.float32
        )

        self.scan_data = np.ones(self.n_scan) * 3.5 # Default max range
        self.current_odom = None
        self.goal_x = 5.0 # Example goal
        self.goal_y = 0.0
        
        self.max_linear_vel = 0.26
        self.max_angular_vel = 1.82
        self.min_range = 0.12
        self.collision_dist = 0.20

    def scan_callback(self, msg):
        # Process scan data: take 360 points, handle infs
        ranges = np.array(msg.ranges)
        # Resize or sample to match n_scan if needed. Assuming msg.ranges is large enough.
        # Simple downsampling or taking first 360 if it matches
        if len(ranges) >= self.n_scan:
            step = len(ranges) // self.n_scan
            self.scan_data = ranges[::step][:self.n_scan]
        else:
            # Pad if too small (unlikely for LIDAR)
            self.scan_data = np.pad(ranges, (0, self.n_scan - len(ranges)), 'constant', constant_values=3.5)
        
        # Replace inf/nan
        self.scan_data = np.nan_to_num(self.scan_data, nan=3.5, posinf=3.5, neginf=0.0)
        self.scan_data = np.clip(self.scan_data, 0.0, 3.5)

    def odom_callback(self, msg):
        self.current_odom = msg

    def step(self, action):
        # Spin ROS to get latest messages
        rclpy.spin_once(self.node, timeout_sec=0.0)

        # Execute action
        linear_vel = (action[0] + 1.0) / 2.0 * self.max_linear_vel # Map [-1, 1] to [0, max]
        angular_vel = action[1] * self.max_angular_vel

        twist = Twist()
        twist.linear.x = float(linear_vel)
        twist.angular.z = float(angular_vel)
        self.cmd_vel_pub.publish(twist)

        # Wait a bit for action to take effect (simple simulation step)
        # In real training, this might need better synchronization
        time.sleep(0.05) 

        # Get observation
        obs = self._get_obs()

        # Calculate reward
        reward, done = self._calculate_reward(obs)
        
        info = {}
        
        return obs, reward, done, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset robot (in simulation this might involve a service call to Gazebo)
        # For now, we just stop the robot
        twist = Twist()
        self.cmd_vel_pub.publish(twist)

        # Call reset service
        while not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('reset service not available, waiting again...')
        
        req = Empty.Request()
        future = self.reset_client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)
        
        # Randomize goal
        self.goal_x = np.random.uniform(2.0, 8.0)
        self.goal_y = np.random.uniform(-3.0, 3.0)
        
        # Reset scan data to safe values to avoid immediate collision detection from stale data
        self.scan_data = np.ones(self.n_scan) * 3.5
        
        # Wait for fresh observation
        # Give Gazebo time to physically reset the robot and publish new scans
        time.sleep(0.5)
        rclpy.spin_once(self.node, timeout_sec=0.1)
        
        obs = self._get_obs()
        self.prev_distance = obs[-2] * 10.0
        
        return obs, {}

    def _get_obs(self):
        # Calculate goal distance and angle
        if self.current_odom:
            pos = self.current_odom.pose.pose.position
            orient = self.current_odom.pose.pose.orientation
            
            # Quaternion to Euler (Yaw)
            # Simplified for 2D
            siny_cosp = 2 * (orient.w * orient.z + orient.x * orient.y)
            cosy_cosp = 1 - 2 * (orient.y * orient.y + orient.z * orient.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            
            dist_x = self.goal_x - pos.x
            dist_y = self.goal_y - pos.y
            distance = math.sqrt(dist_x**2 + dist_y**2)
            
            angle_to_goal = math.atan2(dist_y, dist_x) - yaw
            # Normalize angle to [-pi, pi]
            angle_to_goal = (angle_to_goal + math.pi) % (2 * math.pi) - math.pi
        else:
            distance = 0.0
            angle_to_goal = 0.0

        # Normalize scan
        norm_scan = self.scan_data / 3.5
        
        # Normalize goal info
        norm_dist = np.clip(distance / 10.0, 0.0, 1.0)
        norm_angle = angle_to_goal / math.pi
        
        obs = np.concatenate([norm_scan, [norm_dist, norm_angle]])
        return obs.astype(np.float32)

    def _calculate_reward(self, obs):
        # obs structure: [scan... , dist, angle]
        min_laser = np.min(self.scan_data)
        dist_to_goal = obs[-2] * 10.0 # Un-normalize roughly
        
        reward = 0.0
        done = False
        
        # Collision
        if min_laser < self.collision_dist:
            reward = -100.0
            done = True
        # Goal Reached
        elif dist_to_goal < 0.5:
            reward = 100.0
            done = True
        else:
            # Progress reward
            progress = self.prev_distance - dist_to_goal
            reward = (progress * 10.0) - 0.1 # Progress bonus - time penalty
            self.prev_distance = dist_to_goal
            
        return reward, done

    def close(self):
        self.node.destroy_node()
        rclpy.shutdown()
