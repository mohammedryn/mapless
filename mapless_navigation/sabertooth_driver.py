import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import serial
import time
import math
import numpy as np

class SabertoothDriver(Node):
    def __init__(self):
        super().__init__('sabertooth_driver')
        
        # Parameters
        self.declare_parameter('serial_port', '/dev/ttyTHS1') # Jetson UART
        self.declare_parameter('baud_rate', 9600)
        self.declare_parameter('address', 128)
        self.declare_parameter('max_speed_linear', 0.5) # m/s
        self.declare_parameter('max_speed_angular', 2.0) # rad/s
        
        self.declare_parameter('scan_min_dist', 0.15) # Safety margin

        self.port = self.get_parameter('serial_port').value
        self.baud = self.get_parameter('baud_rate').value
        self.address = self.get_parameter('address').value
        self.scan_min_dist = self.get_parameter('scan_min_dist').value
        
        self.emergency_stop = False

        # Connect to Serial
        try:
            self.serial = serial.Serial(self.port, self.baud, timeout=0.1)
            self.get_logger().info(f"Connected to Sabertooth on {self.port}")
        except Exception as e:
            self.get_logger().error(f"Failed to connect to serial: {e}")
            self.serial = None

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
                self.get_logger().warn("🚨 SAFETY OVERRIDE: Obstacle too close! Halting motors.")
            self.emergency_stop = True
        else:
            self.emergency_stop = False

    def cmd_vel_callback(self, msg):
        if not self.serial:
            return

        if self.emergency_stop:
            self.send_motor_command(1, 0.0)
            self.send_motor_command(2, 0.0)
            return

        # Differential Drive Kinematics
        # We need to map linear/angular velocity to motor values [-127, 127]
        # This is a simplified mapping. You might need to tune 'width' (track width).
        
        linear = msg.linear.x
        angular = msg.angular.z
        
        # Simple mixing
        left_speed = linear - angular
        right_speed = linear + angular
        
        # Scale to [-1.0, 1.0] roughly
        # Assuming max input is around 1.0
        left_speed = max(min(left_speed, 1.0), -1.0)
        right_speed = max(min(right_speed, 1.0), -1.0)
        
        # Map to Sabertooth command [0-127] for each direction
        # Packet Serial Mode 1:
        # Command 0: Drive Forward Motor 1 (0-127)
        # Command 1: Drive Backward Motor 1 (0-127)
        # Command 4: Drive Forward Motor 2 (0-127)
        # Command 5: Drive Backward Motor 2 (0-127)
        
        self.send_motor_command(1, left_speed)
        self.send_motor_command(2, right_speed)

    def send_motor_command(self, motor_id, speed):
        # speed is [-1.0, 1.0]
        val = int(abs(speed) * 127)
        
        if motor_id == 1: # Left Motor
            command = 0 if speed >= 0 else 1
        else: # Right Motor
            command = 4 if speed >= 0 else 5
            
        self.send_packet(self.address, command, val)

    def send_packet(self, address, command, value):
        checksum = (address + command + value) & 0x7F
        packet = bytearray([address, command, value, checksum])
        self.serial.write(packet)

    def stop_motors(self):
        if self.serial:
            self.send_packet(self.address, 0, 0)
            self.send_packet(self.address, 4, 0)
            self.serial.close()

def main(args=None):
    rclpy.init(args=args)
    node = SabertoothDriver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
