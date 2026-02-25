import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import numpy as np

# Use gpiozero for Pi 5 compatibility (uses lgpio backend)
try:
    from gpiozero import PWMOutputDevice, DigitalOutputDevice
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False


class BTS7960Driver(Node):
    def __init__(self):
        super().__init__('bts7960_driver')

        if not GPIO_AVAILABLE:
            self.get_logger().error("gpiozero library not found! Please install: pip install gpiozero")

        # Declare parameters for Left Motor (Pairs 1 & 2)
        self.declare_parameter('left_l_en', 22)  # Left Forward Enable
        self.declare_parameter('left_r_en', 23)  # Left Reverse Enable
        self.declare_parameter('left_l_pwm', 18) # Left Forward PWM (Must be PWM capable pin)
        self.declare_parameter('left_r_pwm', 19) # Left Reverse PWM (Must be PWM capable pin)

        # Declare parameters for Right Motor (Pairs 3 & 4)
        self.declare_parameter('right_l_en', 17) # Right Forward Enable
        self.declare_parameter('right_r_en', 27) # Right Reverse Enable
        self.declare_parameter('right_l_pwm', 12) # Right Forward PWM (Must be PWM capable pin)
        self.declare_parameter('right_r_pwm', 13) # Right Reverse PWM (Must be PWM capable pin)

        # Safety parameters
        self.declare_parameter('scan_min_dist', 0.15) # Safety margin in meters
        self.scan_min_dist = self.get_parameter('scan_min_dist').value
        self.emergency_stop = False

        # Initialize Pins if library available
        if GPIO_AVAILABLE:
            try:
                # Left Motor Pins
                self.l_l_en = DigitalOutputDevice(self.get_parameter('left_l_en').value)
                self.l_r_en = DigitalOutputDevice(self.get_parameter('left_r_en').value)
                self.l_l_pwm = PWMOutputDevice(self.get_parameter('left_l_pwm').value, frequency=100)
                self.l_r_pwm = PWMOutputDevice(self.get_parameter('left_r_pwm').value, frequency=100)

                # Right Motor Pins
                self.r_l_en = DigitalOutputDevice(self.get_parameter('right_l_en').value)
                self.r_r_en = DigitalOutputDevice(self.get_parameter('right_r_en').value)
                self.r_l_pwm = PWMOutputDevice(self.get_parameter('right_l_pwm').value, frequency=100)
                self.r_r_pwm = PWMOutputDevice(self.get_parameter('right_r_pwm').value, frequency=100)

                # Enable all channels
                self.l_l_en.on()
                self.l_r_en.on()
                self.r_l_en.on()
                self.r_r_en.on()

                self.get_logger().info("BTS7960 Hardware Initialized on GPIO.")
            except Exception as e:
                self.get_logger().error(f"Failed to initialize GPIO pins: {e}")
                GPIO_AVAILABLE = False

        # Subscribers
        self.cmd_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )
        
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # Stop motors on shutdown
        rclpy.get_default_context().on_shutdown(self.stop_motors)


    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        ranges = np.nan_to_num(ranges, nan=99.0, posinf=99.0, neginf=0.0)
        
        if len(ranges) > 0 and np.min(ranges) < self.scan_min_dist:
            if not self.emergency_stop:
                self.get_logger().warn("🚨 SAFETY OVERRIDE: Obstacle too close! Halting BTS7960 motors.")
            self.emergency_stop = True
            self.stop_motors() # Immediately stop on callback thread
        else:
            self.emergency_stop = False

    def cmd_vel_callback(self, msg):
        if not GPIO_AVAILABLE:
            return

        if self.emergency_stop:
            self.stop_motors()
            return

        linear = msg.linear.x
        angular = msg.angular.z
        
        # Skid-Steer Kinematics Mixing
        left_speed = linear - angular
        right_speed = linear + angular
        
        # Constrain to [-1.0, 1.0]
        left_speed = max(min(left_speed, 1.0), -1.0)
        right_speed = max(min(right_speed, 1.0), -1.0)

        # Apply to pins
        self.set_left_motor(left_speed)
        self.set_right_motor(right_speed)

    def set_left_motor(self, speed):
        # Speed is between -1.0 and 1.0
        if speed >= 0:
            self.l_r_pwm.value = 0.0      # Turn off reverse
            self.l_l_pwm.value = speed    # Turn on forward
        else:
            self.l_l_pwm.value = 0.0      # Turn off forward
            self.l_r_pwm.value = abs(speed) # Turn on reverse

    def set_right_motor(self, speed):
        # Speed is between -1.0 and 1.0
        if speed >= 0:
            self.r_r_pwm.value = 0.0      # Turn off reverse
            self.r_l_pwm.value = speed    # Turn on forward
        else:
            self.r_l_pwm.value = 0.0      # Turn off forward
            self.r_r_pwm.value = abs(speed) # Turn on reverse

    def stop_motors(self):
        if GPIO_AVAILABLE:
            self.l_l_pwm.value = 0.0
            self.l_r_pwm.value = 0.0
            self.r_l_pwm.value = 0.0
            self.r_r_pwm.value = 0.0

def main(args=None):
    rclpy.init(args=args)
    node = BTS7960Driver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
