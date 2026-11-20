# HW01 – Robotics Assignment Report

## 1. Objective

The objective is to detect a red marker using RGB and depth images, reconstruct its 3D position, and move the robotic arm to the required target point above the marker without touching it.

## 2. Solution

### Marker detection

The RGB and depth images are synchronized by their timestamps (within 0.05 seconds). The RGB image is converted to HSV, and red pixels are selected using two hue ranges. Morphological opening removes small noise. The largest connected red region (with at least 10 pixels) is selected, and its centroid gives the marker's image coordinates `(u, v)`.

### Depth reconstruction

The marker's depth `Z` is estimated as the median of the finite, positive depth values inside the detected red region. Using the **pinhole camera model** and the camera calibration parameters, the 3D point in `camera_optical_frame` is calculated as:

\[
X = \frac{(u-c_x)Z}{f_x}, \qquad
Y = \frac{(v-c_y)Z}{f_y}.
\]

Here, `f_x`, `f_y` are focal lengths and `c_x`, `c_y` are the principal-point coordinates.

### TF direction and timing

TF2 transforms the reconstructed point **from `camera_optical_frame` to `base_link`**. The transform is requested at the RGB image's timestamp rather than at the latest available time, so it corresponds to the observation. Before transformation, the code checks that the transform is available at that timestamp.

The marker position is published on `/hw01/target_camera` and `/hw01/target_base`, before applying the hover offset. The target received from `/hw01/target_base` is then raised by **0.08 m along the `base_link` Z-axis**. This is exactly the fixed 8 cm height difference specified in the assignment. The resulting point is sent to the arm controller when it is ready.

### Failure handling

The code skips processing if camera calibration or the required TF transform is unavailable, or if the marker cannot be detected. It rejects tiny red regions and invalid depth measurements; a failed TF lookup prevents publication of the base-frame target. A motion command is sent only when the arm is ready. The program has a 100-second wall-clock deadline, reports whether camera data is recent and the motion succeeded, and cancels outstanding motion during shutdown.

## 3. Code structure

- **`my_solution/student_solution.py`** handles subscriptions, image synchronization, target publication, and arm movement.
- **`my_solution/utils.py`** handles image conversion, marker detection, pinhole-camera reconstruction, TF conversion, and the 8 cm offset.

## 4. Build and run commands

The simulation environment must be built and sourced according to the environment setup guide. The student solution itself consists of Python scripts and **does not have a separate build step**.

From the assignment directory (`assignments/hw01`), source the existing ROS 2 environment and launch the simulation in one terminal:

```bash
source /opt/ros/jazzy/setup.bash
source environment/install/setup.bash
ros2 launch hw01_sim simulation.launch.py preset:=1 gui:=false
```

In a second terminal, run the public checker for the matching preset:

```bash
source /opt/ros/jazzy/setup.bash
source environment/install/setup.bash
ros2 run hw01_sim check_reach --preset 1
```

In a third terminal, run the solution:

```bash
source /opt/ros/jazzy/setup.bash
source environment/install/setup.bash
python3 my_solution/student_solution.py
```

Repeat the procedure for presets **2** and **3**, restarting the simulation and changing both preset arguments accordingly. No separate `colcon build` command is needed for `my_solution`.

## 5. Results

All three presets **passed**. The calculated arm target points below are in meters, in the `base_link` frame, **after** adding the 8 cm hover offset.

| Preset | Target X (m) | Target Y (m) | Target Z (m) | Result |
|---|---:|---:|---:|---|
| 1 | 0.352228 | -0.147511 | 0.140000 | PASS |
| 2 | 0.401450 | 0.001696 | 0.140000 | PASS |
| 3 | 0.352228 | 0.150902 | 0.140000 | PASS |

## 6. Summary

The solution detects the red marker, estimates its 3D location from RGB-D data, transforms it into the robot's base frame, and moves the arm to the specified hover point. All three tested presets passed.
