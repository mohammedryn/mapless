import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    pkg_name = 'mapless_navigation'
    
    # Lidar Driver (Slamtec)
    # Assuming sllidar_ros2 is installed
    lidar_launch = Node(
        package='sllidar_ros2',
        executable='sllidar_node',
        name='sllidar_node',
        parameters=[{
            'channel_type': 'serial',
            'serial_port': '/dev/ttyUSB0', # Check this!
            'serial_baudrate': 115200,
            'frame_id': 'laser',
            'inverted': False,
            'angle_compensate': True,
        }],
        output='screen'
    )

    # Lidar Odometry (rf2o)
    # Generates /odom from /scan
    rf2o_node = Node(
        package='rf2o_laser_odometry',
        executable='rf2o_laser_odometry_node',
        name='rf2o_laser_odometry',
        output='screen',
        parameters=[{
            'laser_scan_topic': '/scan',
            'odom_topic': '/odom',
            'publish_tf': True,
            'base_frame_id': 'base_link',
            'odom_frame_id': 'odom',
            'init_pose_from_topic': '',
            'freq': 10.0
        }]
    )

    # Static TF: base_link -> laser
    # Adjust x, y, z, yaw, pitch, roll based on actual mounting
    tf_base_laser = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0.1', '0', '0.2', '0', '0', '0', 'base_link', 'laser'],
        output='screen'
    )

    rover_config = os.path.join(
        get_package_share_directory(pkg_name),
        'config',
        'rover.yaml'
    )

    # Motor Driver (Custom Node)
    motor_driver_node = Node(
        package=pkg_name,
        executable='bts7960_driver', 
        name='bts7960_driver',
        output='screen',
        parameters=[rover_config] # Load safety and speed limits
    )

    # Navigation Node (DRL Agent)
    nav_node = Node(
        package=pkg_name,
        executable='navigation_node',
        name='navigation_node',
        output='screen',
        parameters=[
            {'model_path': 'models/ppo_forest_nav'},
            rover_config # Load max speed limits
        ]
    )

    return LaunchDescription([
        lidar_launch,
        tf_base_laser,
        rf2o_node,
        motor_driver_node,
        nav_node
    ])
