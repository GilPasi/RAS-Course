import numpy as np
import math
from scipy.ndimage import distance_transform_edt
from geometry_msgs.msg import PoseStamped

def probabilistic_roadmap(
        grid,
        resolution: float,
        sink: PoseStamped,
        hole: PoseStamped,
        n=300,
        k=10,
        retries=3):

    for i in range(retries):
        try:
            return _probabilistic_roadmap(
                grid,
                resolution,
                sink,
                hole,
                n=n + i * 100,
                k=k
            )

        except RuntimeError as ex:
            print(f"[WARNING] Attempt {i + 1}/{retries} failed: {ex}")

            if i == retries - 1:
                raise RuntimeError(
                    f"PRM failed after {retries} attempts."
                ) from ex



def _probabilistic_roadmap(grid, resolution, sink:PoseStamped, hole:PoseStamped, n=300, k=10):
    cspace = _build_configuration_space(grid, resolution)
    height, width = cspace.shape

    sink = pose_to_grid(sink, resolution)
    hole = pose_to_grid(hole, resolution)

    for point in (sink, hole):
        row, col = point

        if not (0 <= row < height and 0 <= col < width):
            raise ValueError("Point is outside the map")

        if cspace[row, col]:
            raise ValueError("Point is inside an obstacle")

    samples = _sample_points(cspace, n)

    samples = np.vstack([sink, hole, samples])

    sink_index = 0
    hole_index = 1

    graph = _connect_neighbors(samples, cspace, k)
    _validate_reachability(graph, sink_index, hole_index)

    return cspace, samples, graph, sink_index, hole_index

def _build_configuration_space(grid, resolution):
    RADIUS = 0.3
    free = grid == 0
    padded = np.pad(free, 1, constant_values=False) # Walls padding

    distances = (
        distance_transform_edt(padded)[1:-1, 1:-1] # Remove walls padding
        * resolution
    )

    return distances <= RADIUS + math.sqrt(2) * resolution


def _sample_points(configuration_space, n=300):
    free_cells = np.argwhere(configuration_space == False)
    rng = np.random.default_rng()

    indices = rng.choice(
        len(free_cells),
        size=min(n, len(free_cells)),
        replace=False
    )

    return free_cells[indices]


def is_collision_free(p1, p2, configuration_space):
    distance = np.linalg.norm(p2 - p1)

    num_steps = int(np.ceil(distance * 2)) + 1

    points = np.linspace(p1, p2, num_steps)

    height, width = configuration_space.shape

    for point in points:
        row, col = np.round(point).astype(int)

        if not (0 <= row < height and 0 <= col < width): #Assert point in bounds
            return False

        if configuration_space[row, col]:
            return False

    return True

def _connect_neighbors(samples, configuration_space, k=10):
    graph = {i: {} for i in range(len(samples))}
    connected = 0
    currently_tested = 0

    for i, q in enumerate(samples):
        currently_tested = i
        distances = np.linalg.norm(
            samples - q, axis=1
        )

        sorted_indices = np.argsort(distances)

        connected = 0

        for j in sorted_indices:

            if i == j:
                continue

            if not is_collision_free(q, samples[j], configuration_space):
                continue

            weight = distances[j]

            graph[i][j] = weight
            graph[j][i] = weight

            connected += 1

            if connected >= k:
                break

    if connected < k:
        raise RuntimeError(
            f"Vertex {currently_tested}: found only {connected} "
            f"collision-free neighbors out of {k} required."
        )
    return graph


def pose_to_grid(pose_stamped, resolution, origin_x=0.0, origin_y=0.0):
    x = pose_stamped.pose.position.x
    y = pose_stamped.pose.position.y

    col = math.floor((x - origin_x) / resolution)
    row = math.floor((y - origin_y) / resolution)

    return np.array([row, col], dtype=int)

def grid_to_world(point, resolution=0.025,
                  origin_x=0.0, origin_y=0.0):
    row, col = point

    x = origin_x + (col + 0.5) * resolution
    y = origin_y + (row + 0.5) * resolution

    return x, y


def _validate_reachability(graph, sink_index, hole_index):
    # Simply a DFS algorithm
    visited = set()
    stack = [sink_index]

    while stack:
        current = stack.pop()

        if current == hole_index:
            return

        if current in visited:
            continue

        visited.add(current)

        for neighbor in graph[current]:
            if neighbor not in visited:
                stack.append(neighbor)

    raise RuntimeError("No path exists between sink and hole.")