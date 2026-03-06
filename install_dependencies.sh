#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Mapless Navigation — Dependency Installer
# Target: Ubuntu 24.04 LTS + ROS 2 Jazzy Jalisco + Raspberry Pi 5
# ─────────────────────────────────────────────────────────────────────────────

set -e   # Exit immediately on any error

echo "============================================"
echo " Mapless Navigation — Dependency Installer"
echo " ROS 2 Jazzy | Ubuntu 24.04 | Raspberry Pi 5"
echo "============================================"

# 1. Update System
echo ""
echo "[1/5] Updating apt repositories..."
sudo apt update

# 2. Install ROS 2 Jazzy Packages
echo ""
echo "[2/5] Installing ROS 2 Jazzy packages..."
sudo apt install -y \
    ros-jazzy-sllidar-ros2 \
    ros-jazzy-rf2o-laser-odometry \
    ros-jazzy-robot-localization \
    ros-jazzy-imu-tools \
    ros-jazzy-tf2-ros \
    ros-jazzy-tf2-tools \
    ros-jazzy-xacro \
    ros-jazzy-ros-gz-sim \
    python3-pip

# 3. Install Python ML / RL Dependencies
echo ""
echo "[3/5] Installing Python libraries..."
pip3 install --upgrade \
    pyserial \
    "stable-baselines3[extra]" \
    shimmy \
    gymnasium \
    torch torchvision torchaudio \
    numpy \
    pyyaml \
    tensorboard

# 4. GPIO (Raspberry Pi 5 — uses lgpio backend via gpiozero)
echo ""
echo "[4/5] Installing GPIO support for Raspberry Pi 5..."
pip3 install gpiozero lgpio

# 5. Permissions for serial and GPIO
echo ""
echo "[5/5] Setting up device permissions..."
# Serial port access (Slamtec lidar on /dev/ttyUSB0)
sudo usermod -a -G dialout "$USER"
# GPIO access on Pi 5 (lgpio uses /dev/gpiochip*)
sudo usermod -a -G gpio "$USER" 2>/dev/null || true

echo ""
echo "============================================"
echo " Setup complete!"
echo " ACTION REQUIRED: reboot (or log out/in)"
echo " for group permissions to take effect."
echo "============================================"
