import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from stable_baselines3 import PPO
import numpy as np
import math
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class NavigationNode(Node):
    def __init__(self):
        super().__init__('navigation_node')
        
        # Load Model
        # Assuming model is in the current directory or specified path
        # In a real package, this should be a parameter
        self.declare_parameter('model_path', 'ppo_forest_nav')
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        
        try:
            self.model = PPO.load(model_path)
            self.get_logger().info(f"Loaded model from {model_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            self.model = None

        # QoS
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos_profile
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        
        # Publisher
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # State
        self.scan_data = np.ones(360) * 3.5
        self.current_odom = None
        self.goal_x = 5.0 # Default goal
        self.goal_y = 0.0
        
        # Timer for control loop
        self.create_timer(0.1, self.control_loop) # 10 Hz

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        if len(ranges) >= 360:
            step = len(ranges) // 360
            self.scan_data = ranges[::step][:360]
        else:
            self.scan_data = np.pad(ranges, (0, 360 - len(ranges)), 'constant', constant_values=3.5)
        
        self.scan_data = np.nan_to_num(self.scan_data, nan=3.5, posinf=3.5, neginf=0.0)
        self.scan_data = np.clip(self.scan_data, 0.0, 3.5)

    def odom_callback(self, msg):
        self.current_odom = msg

    def get_obs(self):
        if self.current_odom:
            pos = self.current_odom.pose.pose.position
            orient = self.current_odom.pose.pose.orientation
            
            siny_cosp = 2 * (orient.w * orient.z + orient.x * orient.y)
            cosy_cosp = 1 - 2 * (orient.y * orient.y + orient.z * orient.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            
            dist_x = self.goal_x - pos.x
            dist_y = self.goal_y - pos.y
            distance = math.sqrt(dist_x**2 + dist_y**2)
            
            angle_to_goal = math.atan2(dist_y, dist_x) - yaw
            angle_to_goal = (angle_to_goal + math.pi) % (2 * math.pi) - math.pi
        else:
            distance = 0.0
            angle_to_goal = 0.0

        norm_scan = self.scan_data / 3.5
        norm_dist = np.clip(distance / 10.0, 0.0, 1.0)
        norm_angle = angle_to_goal / math.pi
        
        obs = np.concatenate([norm_scan, [norm_dist, norm_angle]])
        return obs.astype(np.float32)

    def control_loop(self):
        if self.model is None:
            return

        obs = self.get_obs()
        action, _ = self.model.predict(obs, deterministic=True)
        
        # Action scaling (must match environment)
        max_linear_vel = 0.26
        max_angular_vel = 1.82
        
        linear_vel = (action[0] + 1.0) / 2.0 * max_linear_vel
        angular_vel = action[1] * max_angular_vel
        
        twist = Twist()
        twist.linear.x = float(linear_vel)
        twist.angular.z = float(angular_vel)
        self.cmd_vel_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = NavigationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
