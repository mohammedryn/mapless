#!/bin/bash

echo "Starting Mapless Navigation Setup..."

# 1. Update System
echo "Updating apt repositories..."
sudo apt update

# 2. Install ROS 2 Dependencies
echo "Installing ROS 2 packages..."
sudo apt install -y ros-humble-sllidar-ros2 \
                    ros-humble-rf2o-laser-odometry \
                    ros-humble-navigation2 \
                    ros-humble-nav2-bringup \
                    ros-humble-xacro \
                    python3-pip

# 3. Install Python Dependencies
echo "Installing Python libraries..."
pip3 install pyserial stable-baselines3 shimmy gymnasium

# 4. Permissions
echo "Setting up USB permissions..."
# Add user to dialout group to access USB ports without sudo
sudo usermod -a -G dialout $USER

echo "Setup Complete! Please reboot your Jetson or log out/in for USB permissions to take effect."
