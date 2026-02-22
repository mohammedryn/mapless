# Robot Wiring & First Run Guide

**Goal:** Connect your Jetson Orin Nano, Sabertooth Motor Driver, Battery, and Motors, and make the robot move for the first time.

---

## 1. The Components
You should have:
1.  **Jetson Orin Nano** (The Brain)
2.  **Sabertooth 2x32** (The Muscle - Motor Driver)
3.  **JGB37 Motors** (x4)
4.  **Battery** (LiPo or Lead Acid, 12V or 24V)
5.  **USB-to-TTL Adapter** (To talk to Sabertooth) OR Jumper Wires for GPIO.
6.  **Slamtec Lidar** (The Eyes)

---

## 2. Wiring Diagram

### A. Power Connections (The Dangerous Part - Be Careful!)
**WARNING:** Red is Positive (+), Black is Negative (-). **NEVER** reverse these, or you will fry the Sabertooth instantly.

1.  **Battery to Sabertooth**:
    *   Connect Battery **(+)** to Sabertooth **B+**.
    *   Connect Battery **(-)** to Sabertooth **B-**.
2.  **Motors to Sabertooth**:
    *   **Left Side Motors**: Connect both Left motors together.
        *   Red wires -> Sabertooth **M1A**
        *   Black wires -> Sabertooth **M1B**
    *   **Right Side Motors**: Connect both Right motors together.
        *   Red wires -> Sabertooth **M2A**
        *   Black wires -> Sabertooth **M2B**
    *   *Note: If a wheel spins backwards later, just swap the Red/Black wires for that motor.*

### B. Signal Connections (The Brain to The Muscle)
We need to send commands from the Jetson to the Sabertooth. We will use **Packet Serial** mode.

**Option 1: Using a USB-to-TTL Adapter (Recommended & Easiest)**
1.  Plug the USB adapter into the Jetson.
2.  Connect Adapter **TX** (Transmit) -> Sabertooth **S1**.
3.  Connect Adapter **GND** (Ground) -> Sabertooth **0V**.
4.  *Do not connect the 5V pin.*

**Option 2: Using Jetson GPIO Pins (Advanced)**
1.  Jetson Pin 8 (UART TX) -> Sabertooth **S1**.
2.  Jetson Pin 6 (GND) -> Sabertooth **0V**.

---

## 3. Sabertooth Configuration (DIP Switches)
The Sabertooth has little switches on it. They tell it how to behave. For **Packet Serial Mode** (Address 128), set them exactly like this:

| Switch | Position | Meaning |
| :--- | :--- | :--- |
| **1** | **OFF (Down)** | Packet Serial Address 128 |
| **2** | **OFF (Down)** | Packet Serial Address 128 |
| **3** | **ON (Up)** | Lithium Cutoff (Protects battery) - *Set OFF if using Lead Acid* |
| **4** | **OFF (Down)** | Serial Mode |
| **5** | **OFF (Down)** | Serial Mode |
| **6** | **OFF (Down)** | Standard Mode |

**Summary:** **DOWN, DOWN, UP, DOWN, DOWN, DOWN** (Assuming LiPo battery).

---

## 4. Lidar Connection
1.  Plug the Slamtec Lidar USB cable into the Jetson.
2.  That's it!

---

## 5. First Run: "Hello World" Drive
Now we test if it works.

### Step 1: Check Connections
1.  Lift the robot off the ground (put it on a box) so wheels can spin freely.
2.  Turn on the Battery. The Sabertooth Status LED should light up (Blue/Green).

### Step 2: Find the USB Port
On the Jetson terminal, run:
```bash
ls /dev/ttyUSB*
```
*   If you see `/dev/ttyUSB0`, that's likely your Lidar or Motor adapter.
*   If you have two USB devices, you might see `/dev/ttyUSB0` and `/dev/ttyUSB1`.
*   Unplug one to see which is which.
    *   **Lidar Port**: Update `launch/real_robot.launch.py` (line 19).
    *   **Motor Port**: Update `src/sabertooth_driver.py` (line 12).

### Step 3: Launch the Driver
Open a terminal on the Jetson:
```bash
cd ~/mapless
source install/setup.bash
ros2 run mapless_navigation sabertooth_driver --ros-args -p serial_port:=/dev/ttyUSB0
```
*(Replace `/dev/ttyUSB0` with your actual Motor Adapter port)*.

### Step 4: Send a Move Command
Open a **second terminal**:
```bash
ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.2}, angular: {z: 0.0}}"
```
*   **What should happen:** The wheels should spin forward slowly.
*   **If they spin backward:** Swap the Red/Black wires on the Sabertooth (M1A/M1B or M2A/M2B).
*   **If they spin opposite directions (one forward, one back):** Swap the wires for just the backward side.

### Step 5: Stop
```bash
ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0}, angular: {z: 0.0}}"
```

---

## Troubleshooting
*   **Sabertooth LED is Red:** Error/Overcurrent. Check wiring. Battery might be too low.
*   **Nothing happens:**
    *   Check DIP switches.
    *   Check if you are using the correct `/dev/ttyUSB` port.
    *   Check if TX is connected to S1 (not S2).
