# Jetson Setup & Launch Guide

**Goal:** Go from a fresh Jetson to a moving robot.

---

## Part 1: Download & Install (Run once)

1.  **Open a Terminal** on your Jetson.

2.  **Clone the Repository**:
    ```bash
    cd ~
    git clone https://github.com/mohammedryn/mapless.git
    cd mapless
    ```

3.  **Run the Install Script**:
    This script installs all the ROS packages and Python libraries you need.
    ```bash
    chmod +x install_dependencies.sh
    ./install_dependencies.sh
    ```

4.  **Reboot**:
    You MUST reboot (or log out/in) for the USB permissions to work.
    ```bash
    sudo reboot
    ```

---

## Part 2: Build the Code (Run after changing code)

1.  **Go to the workspace**:
    ```bash
    cd ~/mapless
    ```

2.  **Build**:
    ```bash
    colcon build --symlink-install
    ```

3.  **Source the environment**:
    ```bash
    source install/setup.bash
    ```
    *(Tip: Add this line to your `~/.bashrc` so you don't have to type it every time)*

---

## Part 3: Launch the Robot (Run to drive)

1.  **Plug in Hardware**:
    *   Lidar -> USB
    *   Sabertooth -> USB (or UART)
    *   Turn on Battery.

2.  **Launch**:
    ```bash
    ros2 launch mapless_navigation real_robot.launch.py
    ```

3.  **Verify**:
    *   **Lidar**: Is spinning?
    *   **Odometry**: Open a new terminal and check `ros2 topic echo /odom`. Push the robot; numbers should change.
    *   **Move**: Open a new terminal and run:
        ```bash
        ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.1}, angular: {z: 0.0}}"
        ```

---

## Troubleshooting

*   **"Serial Exception: Permission denied"**:
    *   Did you run `./install_dependencies.sh`?
    *   Did you reboot?
    *   Try `sudo chmod 666 /dev/ttyUSB0` (Temporary fix).

*   **"Package not found"**:
    *   Did you run `source install/setup.bash`?
