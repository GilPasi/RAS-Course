import heapq
import numpy as np


def euclidean_heuristic(q, goal, graph):
    return np.linalg.norm(q - goal)


def a_star(graph, samples, start, goal,
           heuristic=euclidean_heuristic):

    open_set = []
    came_from = {}

    g_score = {node: float('inf') for node in graph}
    g_score[start] = 0.0

    h = heuristic(samples[start], samples[goal], graph)

    heapq.heappush(open_set, (h, start))

    while open_set:

        _, current = heapq.heappop(open_set)

        if current == goal:
            return reconstruct_path(came_from, current)

        for neighbor, weight in graph[current].items():

            tentative_g = g_score[current] + weight

            if tentative_g < g_score[neighbor]:

                came_from[neighbor] = current
                g_score[neighbor] = tentative_g

                h = heuristic(samples[neighbor], samples[goal], graph)
                f = tentative_g + h

                heapq.heappush(open_set, (f, neighbor))

    return None


def reconstruct_path(came_from, current):
    path = [current]

    while current in came_from:
        current = came_from[current]
        path.append(current)

    return path[::-1]