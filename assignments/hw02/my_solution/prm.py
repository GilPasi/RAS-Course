import numpy as np
import math
from scipy.ndimage import distance_transform_edt

def probabilistic_roadmap(grid, resolution, n=300, k=10):
    configuration_space= _build_configuration_space(grid, resolution)
    samples = _sample_points(configuration_space, n)
    graph = _connect_neighbors(samples, configuration_space, k)
    

    return configuration_space, samples, graph

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

    for point in points:
        row, col = np.round(point).astype(int)

        if configuration_space[row, col]:
            return False

    return True

def _connect_neighbors(samples, configuration_space, k=10):
    graph = {i: {} for i in range(len(samples))}

    for i, q in enumerate(samples):

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

    return graph