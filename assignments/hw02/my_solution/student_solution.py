
#!/usr/bin/env python3
"""HW02 student solution: PRM, A* and ROS2 navigation."""

import math
import time
import numpy as np
import rclpy

from geometry_msgs.msg import PoseStamped, Point
from hw02_sim.robot_client import RobotClient
from std_msgs.msg import Bool, String
from nav_msgs.msg import OccupancyGrid, Path
from visualization_msgs.msg import Marker, MarkerArray
from hw02_sim.map_server import MAP_QOS

from prm import probabilistic_roadmap, grid_to_world
from a_star import a_star
from utils import view_configuration_space

def publish_cspace(publisher, cspace, raw_map_msg):
    message = OccupancyGrid()
    message.header = raw_map_msg.header
    message.info = raw_map_msg.info
    message.data = (cspace.astype(np.int8) * 100).flatten().tolist()

    publisher.publish(message)


def publish_roadmap(publisher, samples, graph, raw_map_msg):
    message = MarkerArray()

    resolution = raw_map_msg.info.resolution
    origin = raw_map_msg.info.origin.position

    physical_points = [
        grid_to_world(sample, resolution, origin.x, origin.y)
        for sample in samples
    ]

    # Vertices
    vertices = Marker()
    vertices.header = raw_map_msg.header
    vertices.ns = 'prm_vertices'
    vertices.id = 0
    vertices.type = Marker.POINTS
    vertices.action = Marker.ADD
    vertices.pose.orientation.w = 1.0

    vertices.scale.x = 0.04
    vertices.scale.y = 0.04
    vertices.color.b = 1.0
    vertices.color.a = 1.0

    for x, y in physical_points:
        vertices.points.append(
            Point(x=float(x), y=float(y), z=0.02)
        )

    # Edges
    edges = Marker()
    edges.header = raw_map_msg.header
    edges.ns = 'prm_edges'
    edges.id = 1
    edges.type = Marker.LINE_LIST
    edges.action = Marker.ADD
    edges.pose.orientation.w = 1.0

    edges.scale.x = 0.01
    edges.color.g = 1.0
    edges.color.a = 0.5

    for i, neighbors in graph.items():
        for j in neighbors:

            # Avoid duplicate undirected edges
            if i >= j:
                continue

            x1, y1 = physical_points[i]
            x2, y2 = physical_points[j]

            edges.points.append(
                Point(x=float(x1), y=float(y1), z=0.02)
            )
            edges.points.append(
                Point(x=float(x2), y=float(y2), z=0.02)
            )

    message.markers = [vertices, edges]
    publisher.publish(message)


def publish_path(publisher, route, samples, raw_map_msg):
    message = Path()
    message.header = raw_map_msg.header

    resolution = raw_map_msg.info.resolution
    origin = raw_map_msg.info.origin.position

    if route is not None:
        for index in route:
            x, y = grid_to_world(
                samples[index],
                resolution,
                origin.x,
                origin.y
            )

            waypoint = PoseStamped()
            waypoint.header = message.header
            waypoint.pose.position.x = float(x)
            waypoint.pose.position.y = float(y)
            waypoint.pose.orientation.w = 1.0

            message.poses.append(waypoint)

    publisher.publish(message)


def publish_status(publisher, status):
    message = String()
    message.data = status
    publisher.publish(message)


def main():
    rclpy.init()
    node = rclpy.create_node('hw02_solution')
    robot = RobotClient(node)
    node.set_parameters([ #Use simulated seconds
        rclpy.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)
    ])

    pose, pose_time = None, 0.0
    contact_time, collision = 0.0, False

    RESOLUTION = 0.025

    future = None
    sink: PoseStamped | None = None
    hole: PoseStamped | None = None
    grid: np.ndarray | None = None
    raw_map_msg: OccupancyGrid | None = None

    def wait_simulated(seconds):
        start = node.get_clock().now()

        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)

            elapsed = node.get_clock().now() - start

            if elapsed.nanoseconds >= seconds * 1e9:
                return

    # Publishers
    cspace_pub = node.create_publisher(
        OccupancyGrid, '/hw02/cspace', MAP_QOS
    )

    roadmap_pub = node.create_publisher(
        MarkerArray, '/hw02/roadmap', MAP_QOS
    )

    path_pub = node.create_publisher(
        Path, '/hw02/planned_path', MAP_QOS
    )

    status_pub = node.create_publisher(
        String, '/hw02/planner_status', MAP_QOS
    )

    # Callbacks
    def on_map(message):
        nonlocal grid, raw_map_msg

        raw_map_msg = message

        grid = np.asarray(
            message.data, dtype=np.int8
        ).reshape(message.info.height, message.info.width)

    def on_start(message: PoseStamped):
        nonlocal sink
        sink = message

    def on_goal(message: PoseStamped):
        nonlocal hole
        hole = message

    def on_pose(message):
        nonlocal pose, pose_time
        pose = message.pose.position
        pose_time = time.monotonic()

    def on_contact(message):
        nonlocal collision, contact_time
        collision |= message.data
        contact_time = time.monotonic()

    # Subscribers
    node.create_subscription(
        PoseStamped, '/hw02/robot_pose', on_pose, 10
    )

    node.create_subscription(
        Bool, '/hw02/collision', on_contact, 10
    )

    node.create_subscription(
        PoseStamped, '/hw02/start', on_start, MAP_QOS
    )

    node.create_subscription(
        PoseStamped, '/hw02/goal', on_goal, MAP_QOS
    )

    node.create_subscription(
        OccupancyGrid, '/map', on_map, MAP_QOS
    )

    def wait(condition, seconds):
        deadline = time.monotonic() + seconds

        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.05)

            if condition():
                return True

        return False

    try:
        # Wait for all required information
        available = wait(
            lambda: robot.ready()
            and pose is not None
            and time.monotonic() - pose_time < 1
            and time.monotonic() - contact_time < 1
            and sink is not None
            and hole is not None
            and grid is not None,
            30
        )

        print(
            f'{"PASS" if available else "FAIL"}: '
            f'fresh localization and helper',
            flush=True
        )

        if not available:
            publish_status(status_pub, 'failed')
            return 1

        if math.hypot(pose.x - 1, pose.y - 3) > .10:
            publish_status(status_pub, 'failed')
            print('FAIL: restart the simulation at its initial position')
            return 1

        # PRM and A* planning
        try:
            publish_status(status_pub, 'planning')

            cspace, samples, graph, sink_index, hole_index = (
                probabilistic_roadmap(
                    grid, RESOLUTION, sink, hole
                )
            )

            publish_cspace(cspace_pub, cspace, raw_map_msg)

            publish_roadmap(
                roadmap_pub, samples, graph, raw_map_msg
            )

            route = a_star(
                graph, samples, sink_index, hole_index
            )

            publish_path(
                path_pub, route, samples, raw_map_msg
            )

            view_configuration_space(grid,cspace,samples,RESOLUTION, route)

            if route is None or len(route) == 0:
                publish_status(status_pub, 'no_path_found')
                wait_simulated(3.0)
                return 2
            
            publish_status(status_pub, 'path_found')
        except RuntimeError as ex:
            publish_status(status_pub, 'no_path_found')
            print(f"[ERROR]: {ex}")
            return 2
        except (RuntimeError, ValueError) as ex:
            publish_status(status_pub, 'failed')
            print(f"[ERROR]: {ex}")
            return 3


        # Convert grid coordinates to physical coordinates
        origin = raw_map_msg.info.origin.position

        physical_points = [
            grid_to_world(
                sample, RESOLUTION, origin.x, origin.y
            )
            for sample in samples
        ]

        # Verify fresh localization after planning
        available = wait(
            lambda: time.monotonic() - pose_time < 1
            and time.monotonic() - contact_time < 1,
            5
        )

        if not available or collision:
            publish_status(status_pub, 'failed')
            return 1

        # Execute the planned route
        publish_status(status_pub, 'executing')

        for point in route:
            physical_point = physical_points[point]

            future = robot.go_to(*physical_point)

            done = wait(
                lambda: future.done() or collision,
                135
            )

            ok = (
                done
                and future.done()
                and future.result().success
                and not collision
                and time.monotonic() - pose_time < 1
                and math.hypot(
                    pose.x - physical_point[0],
                    pose.y - physical_point[1]
                ) <= .03
            )

            print(
                f'{"PASS" if ok else "FAIL"}: {physical_point}',
                flush=True
            )

            if not ok:
                publish_status(status_pub, 'failed')

                if future.done():
                    print(future.result().message)

                return 1

        publish_status(status_pub, 'completed')
        return 0

    except (KeyboardInterrupt,
            rclpy.executors.ExternalShutdownException):

        if rclpy.ok():
            publish_status(status_pub, 'failed')

        return 1

    finally:
        robot.cancel()
        wait(lambda: future is None or future.done(), 2)

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
