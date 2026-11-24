#!/usr/bin/env python3
"""Setup example: two clear go-to moves. No map processing or planner."""
import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from hw02_sim.robot_client import RobotClient
from std_msgs.msg import Bool
from nav_msgs.msg import OccupancyGrid
from hw02_sim.map_server import MAP_QOS
from prm import probabilistic_roadmap
from a_star import a_star
from utils import view_configuration_space


def main():
    rclpy.init(); node = rclpy.create_node('hw02_solution'); robot = RobotClient(node)
    pose, pose_time, contact_time, collision = None, 0.0, 0.0, False
    RESOLUTION = 0.025
    future = None
    sink:PoseStamped | None = None
    hole:PoseStamped | None = None
    map: np.ndarray | None = None

    def on_map(message):
        nonlocal map
        map = np.asarray(
            message.data, dtype=np.int8
        ).reshape(message.info.height, message.info.width)

    def on_start(message:PoseStamped)->None:
        nonlocal sink
        sink = message

    def on_goal(message:PoseStamped)->None:
        nonlocal hole
        hole = message

    def on_pose(message):
        nonlocal pose, pose_time
        pose, pose_time = message.pose.position, time.monotonic()

    def on_contact(message):
        nonlocal collision, contact_time
        collision |= message.data; contact_time = time.monotonic()

    node.create_subscription(PoseStamped, '/hw02/robot_pose', on_pose, 10)
    node.create_subscription(Bool, '/hw02/collision', on_contact, 10)
    node.create_subscription(PoseStamped, '/hw02/start', on_start, 10)
    node.create_subscription(PoseStamped, '/hw02/goal', on_goal, 10)
    node.create_subscription(OccupancyGrid, '/map', on_map, MAP_QOS)

    def wait(condition, seconds):
        deadline = time.monotonic()+seconds
        while rclpy.ok() and time.monotonic()<deadline:
            rclpy.spin_once(node, timeout_sec=.05)
            if condition(): return True
        return False
    try:
        available = wait(lambda: robot.ready()
                          and pose is not None
                          and time.monotonic()-pose_time < 1
                          and time.monotonic()-contact_time < 1
                          and sink is not None
                          and hole is not None
                          and map is not None
                         , 30)
        print(f'{"PASS" if available else "FAIL"}: fresh localization and helper', flush=True)
        if not available: return 1
        if math.hypot(pose.x-1, pose.y-3) > .10:
            print('FAIL: restart the simulation at its initial position'); return 1
        cspace, samples, graph, sink_index, hole_index = probabilistic_roadmap(map, RESOLUTION, sink, hole)
        route = a_star(graph, samples, sink_index, hole_index)
        view_configuration_space(map, cspace, samples, RESOLUTION, route)


        for label, target in [('outward movement', (1.1, 3.0)), ('return to start', (1.0, 3.0))]:
            # Each call moves directly to ONE point. Keep spinning for its result.
            break
            future = robot.go_to(*target)
            done = wait(future.done, 135)
            ok = (done and future.result().success and not collision and
                  time.monotonic()-pose_time < 1 and math.hypot(pose.x-target[0],pose.y-target[1]) <= .03)
            print(f'{"PASS" if ok else "FAIL"}: {label}', flush=True)
            if not ok:
                if done: print(future.result().message)
                return 1
        return 0
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException): return 1
    finally:
        robot.cancel()
        wait(lambda: future is None or future.done(), 2)
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()


if __name__ == '__main__': raise SystemExit(main())
