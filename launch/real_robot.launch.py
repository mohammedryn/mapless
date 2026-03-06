import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = 'mapless_navigation'
    pkg_share = get_package_share_directory(pkg)

    rover_config = os.path.join(pkg_share, 'config', 'rover.yaml')
    ekf_config   = os.path.join(pkg_share, 'config', 'ekf.yaml')
    model_path   = os.path.join(pkg_share, 'models', 'ppo_forest_nav')

    # ── Launch arguments ───────────────────────────────────────────────────────
    # Goal coordinates can be overridden at launch time:
    #   ros2 launch mapless_navigation real_robot.launch.py goal_x:=3.0 goal_y:=1.5
    goal_x_arg = DeclareLaunchArgument(
        'goal_x', default_value='5.0',
        description='Navigation goal x-coordinate in the odom frame (metres)')
    goal_y_arg = DeclareLaunchArgument(
        'goal_y', default_value='0.0',
        description='Navigation goal y-coordinate in the odom frame (metres)')
    goal_x = LaunchConfiguration('goal_x')
    goal_y = LaunchConfiguration('goal_y')

    # ── 1. Slamtec C1M1 R2 Lidar ──────────────────────────────────────────────
    lidar_node = Node(
        package='sllidar_ros2',
        executable='sllidar_node',
        name='sllidar_node',
        parameters=[{
            'channel_type':     'serial',
            'serial_port':      '/dev/ttyUSB0',   # verify with: ls /dev/ttyUSB*
            'serial_baudrate':  115200,
            'frame_id':         'laser',
            'inverted':         False,
            'angle_compensate': True,
        }],
        output='screen',
    )

    # ── 2. Static TF: base_link -> laser ──────────────────────────────────────
    # Adjust x/z offsets to match your actual lidar mounting position.
    tf_base_laser = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0.1', '0', '0.2', '0', '0', '0', 'base_link', 'laser'],
        output='screen',
    )

    # ── 3. Static TF: base_link -> imu_link ───────────────────────────────────
    # Adjust if the MPU-6050 is not mounted at the robot's centre of mass.
    tf_base_imu = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0.05', '0', '0', '0', 'base_link', 'imu_link'],
        output='screen',
    )

    # ── 4. rf2o Lidar Odometry ─────────────────────────────────────────────────
    # Generates /odom from /scan via scan-matching (no wheel encoders needed).
    rf2o_node = Node(
        package='rf2o_laser_odometry',
        executable='rf2o_laser_odometry_node',
        name='rf2o_laser_odometry',
        parameters=[{
            'laser_scan_topic': '/scan',
            'odom_topic':       '/odom',
            'publish_tf':       True,
            'base_frame_id':    'base_link',
            'odom_frame_id':    'odom',
            'init_pose_from_topic': '',
            'freq':             10.0,
        }],
        output='screen',
    )

    # ── 5. MPU-6050 IMU Driver ─────────────────────────────────────────────────
    # Publishes /imu/data  (sensor_msgs/Imu) via I2C on the Raspberry Pi 5.
    #
    # Install driver: pip install mpu6050-raspberrypi
    #   or use:  ros-jazzy-imu-tools with a custom I2C reader node.
    #
    # If you have a custom driver node, replace the package/executable below.
    # The EKF node (step 6) will simply not receive IMU data until this node
    # is running — it degrades gracefully back to rf2o-only odometry.
    imu_node = Node(
        package='imu_tools',            # adjust to your actual IMU driver package
        executable='imu_node',          # adjust to your actual executable name
        name='mpu6050_node',
        parameters=[{
            'i2c_bus':    1,
            'i2c_address': 0x68,        # default MPU-6050 (AD0 low)
            'frame_id':   'imu_link',
            'publish_rate': 50.0,
        }],
        output='screen',
    )

    # ── 6. robot_localization EKF (IMU + rf2o fusion) ─────────────────────────
    # Produces /odometry/filtered — a more robust pose estimate than raw
    # lidar odometry alone, especially during fast turns and on slippery floors.
    # NavigationNode subscribes to /odometry/filtered with /odom as fallback.
    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        parameters=[ekf_config],
        remappings=[('odometry/filtered', '/odometry/filtered')],
        output='screen',
    )

    # ── 7. BTS7960 Motor Driver ────────────────────────────────────────────────
    motor_driver_node = Node(
        package=pkg,
        executable='bts7960_driver',
        name='bts7960_driver',
        parameters=[rover_config],
        output='screen',
    )

    # ── 8. DRL Navigation Node ─────────────────────────────────────────────────
    nav_node = Node(
        package=pkg,
        executable='navigation_node',
        name='navigation_node',
        parameters=[
            rover_config,
            {
                'model_path': model_path,
                'algorithm':  'ppo',
                'goal_x':     goal_x,
                'goal_y':     goal_y,
            },
        ],
        output='screen',
    )

    return LaunchDescription([
        goal_x_arg,
        goal_y_arg,
        lidar_node,
        tf_base_laser,
        tf_base_imu,
        rf2o_node,
        imu_node,
        ekf_node,
        motor_driver_node,
        nav_node,
    ])
