
import numpy as np
import cv2

from geometry_msgs.msg import PointStamped
from tf2_geometry_msgs import do_transform_point
from tf2_ros import TransformException
from rclpy.time import Time


# -------------------- Image Conversion --------------------

def depth_to_numpy(image):
    """Convert a ROS 32FC1 depth image into a 2D NumPy array."""
    if image.encoding != '32FC1':
        raise ValueError(f"Unsupported depth encoding: {image.encoding}")

    dtype = np.dtype('float32').newbyteorder(
        '>' if image.is_bigendian else '<'
    )

    return np.ndarray(
        shape=(image.height, image.width),
        dtype=dtype,
        buffer=image.data,
        strides=(image.step, 4)
    ).copy()


def rgb_to_numpy(image):
    """Convert a ROS RGB/BGR image into an RGB NumPy array."""
    if image.encoding not in ('rgb8', 'bgr8'):
        raise ValueError(f"Unsupported RGB encoding: {image.encoding}")

    rgb = np.ndarray(
        shape=(image.height, image.width, 3),
        dtype=np.uint8,
        buffer=image.data,
        strides=(image.step, 3, 1)
    )

    if image.encoding == 'bgr8':
        rgb = rgb[:, :, ::-1]

    return rgb.copy()


# -------------------- Marker Detection --------------------

def detect_marker(rgb_matrix, depth_matrix):
    """Find the largest red marker and estimate its median depth."""

    if rgb_matrix.shape[:2] != depth_matrix.shape:
        raise ValueError("RGB and depth image dimensions do not match")

    hsv = cv2.cvtColor(rgb_matrix, cv2.COLOR_RGB2HSV)

    # Red occupies two regions in HSV hue space.
    mask = (
        cv2.inRange(hsv, (0, 100, 70), (10, 255, 255))
        | cv2.inRange(hsv, (170, 100, 70), (179, 255, 255))
    )

    # Remove isolated noisy pixels.
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    if count <= 1:
        return None

    # Ignore the background and very small components.
    min_area = 10
    candidates = np.where(
        stats[1:, cv2.CC_STAT_AREA] >= min_area
    )[0]

    if candidates.size == 0:
        return None

    marker_id = 1 + candidates[
        np.argmax(stats[1 + candidates, cv2.CC_STAT_AREA])
    ]

    u, v = centroids[marker_id]

    # Use only valid depths belonging to the marker.
    marker_mask = labels == marker_id
    valid = (
        marker_mask
        & np.isfinite(depth_matrix)
        & (depth_matrix > 0)
    )

    if not np.any(valid):
        return None

    z = float(np.median(depth_matrix[valid]))

    return float(u), float(v), z


# -------------------- Coordinate Conversion --------------------

def pixel_to_camera(marker, camera_info):
    """Convert pixel coordinates and depth to camera optical coordinates."""
    if marker is None:
        return None

    u, v, z = marker

    fx = camera_info.k[0]
    fy = camera_info.k[4]
    cx = camera_info.k[2]
    cy = camera_info.k[5]

    if fx <= 0 or fy <= 0:
        raise ValueError("Invalid camera focal lengths")

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    return float(x), float(y), float(z)


def camera_to_base(camera_point, tf_buffer, stamp):
    """Transform a camera optical point into base_link coordinates."""
    if camera_point is None:
        return None

    point = PointStamped()
    point.header.frame_id = 'camera_optical_frame'
    point.header.stamp = stamp

    point.point.x, point.point.y, point.point.z = map(
        float, camera_point
    )

    try:
        transform = tf_buffer.lookup_transform(
            'base_link',
            'camera_optical_frame',
            Time.from_msg(stamp)
        )

        return do_transform_point(point, transform)

    except TransformException as ex:
        print(f"[DEBUG] Transformation unavailable: {ex}")
        return None


# -------------------- Target Adjustment --------------------

def elevate_point(stamped_point, height=0.08):
    """Raise a target along the base_link Z-axis."""
    if stamped_point is None:
        return None

    elevated = PointStamped()

    # Preserve the original coordinate frame and timestamp.
    elevated.header = stamped_point.header

    elevated.point.x = stamped_point.point.x
    elevated.point.y = stamped_point.point.y
    elevated.point.z = stamped_point.point.z + height

    return elevated
