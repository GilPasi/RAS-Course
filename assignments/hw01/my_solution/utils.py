import numpy as np
import cv2

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

# def detect_marker(depth_matrix, threshold=0.02):
#     # Ignore invalid depth values
#     valid = np.isfinite(depth_matrix) & (depth_matrix > 0)
#
#     if not np.any(valid):
#         return None
#
#     # Estimate background depth
#     background = np.median(depth_matrix[valid])
#
#     # Replace invalid pixels before filtering
#     clean = np.where(valid, depth_matrix, background).astype(np.float32)
#
#     # Reduce noise
#     filtered = cv2.medianBlur(clean, 5)
#
#     # Identify pixels deviating from background
#     mask = (
#         (np.abs(filtered - background) > threshold)
#         & valid
#     ).astype(np.uint8)
#
#     # Find connected components
#     count, labels, stats, centroids = (
#         cv2.connectedComponentsWithStats(mask, connectivity=8)
#     )
#
#     if count <= 1:
#         return None
#
#     # Find largest component, excluding background
#     marker_id = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
#
#     x, y = centroids[marker_id]
#
#     return x, y