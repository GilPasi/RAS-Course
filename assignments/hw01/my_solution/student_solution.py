#!/usr/bin/env python3
import time
import rclpy

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time

from sensor_msgs.msg import Image, CameraInfo
from tf2_ros import Buffer, TransformListener
from message_filters import Subscriber, ApproximateTimeSynchronizer

from hw01_sim.arm_client import ArmClient
from utils import (
    depth_to_numpy,
    rgb_to_numpy,
    detect_marker,
    pixel_to_camera,
    camera_to_base,
    elevate_point,
)


def main():
    rclpy.init()

    node = Node('solution_hw1')
    arm = ArmClient(node)

    tf_buffer = Buffer()
    tf_listener = TransformListener(tf_buffer, node)

    camera_info = None
    marker_position = None
    last_frame_received = None

    def on_camera_info(info):
        nonlocal camera_info
        camera_info = info

    def on_synchronized(rgb_msg, depth_msg):
        nonlocal marker_position, last_frame_received

        last_frame_received = time.monotonic()

        # Stop processing after successful detection
        if marker_position is not None or camera_info is None:
            return

        stamp = rgb_msg.header.stamp

        # Check TF availability at the image timestamp
        if not tf_buffer.can_transform(
            'base_link',
            'camera_optical_frame',
            Time.from_msg(stamp)
        ):
            return

        rgb_matrix = rgb_to_numpy(rgb_msg)
        depth_matrix = depth_to_numpy(depth_msg)

        print("[DEBUG] Synchronized images received")
        print(f"[DEBUG] RGB shape: {rgb_matrix.shape}")
        print(f"[DEBUG] Depth shape: {depth_matrix.shape}")

        marker = detect_marker(rgb_matrix, depth_matrix)

        if marker is None:
            print("[DEBUG] Marker not detected")
            return

        print(f"[DEBUG] Marker pixel and depth: {marker}")

        camera_point = pixel_to_camera(marker, camera_info)
        print(f"[DEBUG] Camera coordinates: {camera_point}")

        base_point = camera_to_base(camera_point, tf_buffer, stamp)

        if base_point is None:
            print("[DEBUG] Transformation failed, retrying...")
            return

        print(f"[DEBUG] Base coordinates: {base_point.point}")

        marker_position = elevate_point(base_point)

        print(f"[DEBUG] Target position: {marker_position.point}")
        print(f"[DEBUG] Target frame: {marker_position.header.frame_id}")

    # Subscriptions
    rgb_sub = Subscriber(
        node, Image, '/camera/image_raw',
        qos_profile=qos_profile_sensor_data
    )

    depth_sub = Subscriber(
        node, Image, '/camera/depth/image_raw',
        qos_profile=qos_profile_sensor_data
    )

    info_sub = Subscriber(
        node, CameraInfo, '/camera/camera_info',
        qos_profile=qos_profile_sensor_data
    )
    info_sub.registerCallback(on_camera_info)

    sync = ApproximateTimeSynchronizer(
        [rgb_sub, depth_sub],
        queue_size=10,
        slop=0.05
    )
    sync.registerCallback(on_synchronized)

    motion = None
    deadline = time.monotonic() + 100.0

    try:
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)

            if motion is None and marker_position is not None and arm.ready():
                print("[DEBUG] Sending movement command")
                motion = arm.move_to_point(marker_position)

            if motion is not None and motion.done():
                break

        image_ok = (
            last_frame_received is not None
            and time.monotonic() - last_frame_received < 2.0
        )

        motion_ok = (
            motion is not None
            and motion.done()
            and motion.result().success
        )

        print(f'{"PASS" if image_ok else "FAIL"}: fresh camera image reception')
        print(f'{"PASS" if motion_ok else "FAIL"}: marker-target arm movement')

        if motion is not None and motion.done():
            print(motion.result().message)

        return 0 if image_ok and motion_ok else 1

    except KeyboardInterrupt:
        return 1

    finally:
        arm.cancel()

        for _ in range(5):
            if rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.05)

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
