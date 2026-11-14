import numpy as np
import cv2
from geometry_msgs.msg import PointStamped
from tf2_geometry_msgs import do_transform_point
from tf2_ros import Buffer, TransformListener, TransformException
from rclpy.time import Time

def depth_to_numpy(image):
    if image.encoding != '32FC1':
        raise ValueError(f"Unsupported encoding: {image.encoding}")

    dtype = np.dtype('float32').newbyteorder(
        '>' if image.is_bigendian else '<'
    )

    depth = np.ndarray(
        shape=(image.height, image.width),
        dtype=dtype,
        buffer=image.data,
        strides=(image.step, 4)
    )

    return depth.copy()

def rgb_to_numpy(image):
    if image.encoding not in ('rgb8', 'bgr8'):
        raise ValueError(f"Unsupported encoding: {image.encoding}")

    rgb = np.ndarray(
        shape=(image.height, image.width, 3),
        dtype=np.uint8,
        buffer=image.data,
        strides=(image.step, 3, 1)
    )

    if image.encoding == 'bgr8':
        rgb = rgb[:, :, ::-1]

    return rgb.copy()

def detect_marker(rgb_matrix, depth_matrix):
    # Convert RGB to HSV
    hsv = cv2.cvtColor(rgb_matrix, cv2.COLOR_RGB2HSV)

    # Example: detect a red marker
    mask1 = cv2.inRange(hsv, (0, 100, 70), (10, 255, 255))
    mask2 = cv2.inRange(hsv, (170, 100, 70), (179, 255, 255))
    mask = mask1 | mask2

    # Remove small noise
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find connected components
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    if count <= 1:
        return None

    # Largest detected component
    marker_id = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    # Marker centroid in image coordinates
    x, y = centroids[marker_id]

    # Find valid depths inside marker region
    marker_pixels = labels == marker_id
    valid = marker_pixels & np.isfinite(depth_matrix) & (depth_matrix > 0)

    if not np.any(valid):
        return None

    # Robust depth estimate
    z = np.median(depth_matrix[valid])

    return x, y, z

def pixel_to_camera(marker, camera_info):
    if marker is None:
        return None

    u, v, z = marker

    fx = camera_info.k[0]
    fy = camera_info.k[4]
    cx = camera_info.k[2]
    cy = camera_info.k[5]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    return x, y, z

def camera_to_base(camera_point, tf_buffer, stamp):
    if camera_point is None:
        return None

    x, y, z = camera_point

    point = PointStamped()
    point.header.frame_id = 'camera_optical_frame'
    point.header.stamp = stamp

    point.point.x = float(x)
    point.point.y = float(y)
    point.point.z = float(z)

    try:
        transform = tf_buffer.lookup_transform(
            'base_link',
            'camera_optical_frame',
            Time.from_msg(stamp)
        )

        return do_transform_point(point, transform)

    except TransformException as ex:
        print(f"Transformation unavailable: {ex}")
        return None