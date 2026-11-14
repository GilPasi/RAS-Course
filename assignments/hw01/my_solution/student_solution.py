#!/usr/bin/env python3
"""Run after launching HW1: receive RGB images and move to one fixed point."""
import time

import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from tf2_ros import Buffer, TransformListener
from rclpy.time import Time
from message_filters import Subscriber, ApproximateTimeSynchronizer

from hw01_sim.arm_client import ArmClient

import numpy as np

from utils import depth_to_numpy, rgb_to_numpy

def main():
    rclpy.init()
    node = Node('solution_hw1')
    arm = ArmClient(node)
    camera_info_received = None
    last_frame_received = None

    def on_camera_info(info):
        nonlocal camera_info_received
        if camera_info_received is None:
            # print(f'Camera info: {info}', flush=True)
            camera_info_received = time.monotonic()

    def on_synchronized(rgb_msg, depth_msg):
        rgb_matrix = rgb_to_numpy(rgb_msg)
        depth_matrix = depth_to_numpy(depth_msg)
        nonlocal last_frame_received
        if last_frame_received is None:
            last_frame_received = time.monotonic()
            print("Synchronized images received!")
            print("RGB shape:", rgb_matrix.shape)
            print("Depth shape:", depth_matrix.shape)

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
            if motion is None and last_frame_received is not None and arm.ready():
                # Fixed demo point, unrelated to assignment targets. Units are metres.
                point = PointStamped()
                point.header.frame_id = 'base_link'
                point.point.x, point.point.y, point.point.z = 0.20, -0.20, 0.14
                if arm.ready():
                    motion = arm.move_to_point(point)
                else:
                    motion = None
                    print("Arm is not ready yet")
            if motion is not None and motion.done():
                break
        image_ok = last_frame_received is not None and time.monotonic() - last_frame_received < 2.0
        motion_ok = motion is not None and motion.done() and motion.result().success
        print(f'{"PASS" if image_ok else "FAIL"}: fresh camera image reception')
        print(f'{"PASS" if motion_ok else "FAIL"}: fixed-point arm movement')
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
