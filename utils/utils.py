import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from numpy.random import default_rng
rng = default_rng()


def generate_inital_swarm(width, height, min_distance, num_robots=30, max_tries=100000):
    minimum_coverage = num_robots * min_distance ** 2
    assert minimum_coverage <= width*height, "Number of robots cover the given width and height"
    n_sample_points = int(max(num_robots ** 2 * (minimum_coverage / (width * height)), 10))
    ixs = width * rng.random(n_sample_points)
    iys = height * rng.random(n_sample_points)

    x_diff = ixs.reshape((1, n_sample_points)) - ixs.reshape((n_sample_points, 1))
    y_diff = iys.reshape((1, n_sample_points)) - iys.reshape((n_sample_points, 1))
    dist = np.hypot(x_diff, y_diff)

    neighbours_idx = (dist < min_distance)
    random_points = np.argsort(np.random.rand(max_tries, n_sample_points), axis=1)[:, :num_robots]
    neighbour_points = neighbours_idx[random_points.reshape(-1, num_robots, 1), random_points.reshape(-1, 1, num_robots)]
    sampled_neighbours = np.sum(neighbour_points, axis=(1, 2))
    samples = random_points[sampled_neighbours.argmin()]
    points_x, points_y = ixs[samples], iys[samples]
    # fig, ax = plt.subplots()
    #
    # _ = ax.scatter(ixs, iys)
    # _ = ax.scatter(points_x, points_y)
    #
    # circles = [plt.Circle((xi, yi), radius=min_distance / 2, fill=False) for xi, yi in zip(points_x, points_y)]
    # collection = PatchCollection(circles, match_original=True)
    # ax.add_collection(collection)
    # _ = ax.set(aspect='equal', xlabel=r'$x_1$', ylabel=r'$x_2$',
    #            xlim=[0, width], ylim=[0, height])
    # plt.show()
    return points_x, points_y


def generate_poisson_disk_samples(width, height, min_distance, max_attempts=30):
    cell_size = min_distance / np.sqrt(2)
    grid_width = int(np.ceil(width / cell_size))
    grid_height = int(np.ceil(height / cell_size))

    grid = np.full((grid_width, grid_height), -1)
    active_points = []
    samples = []

    def is_valid_point(point):
        x, y = point
        if x < 0 or x >= width or y < 0 or y >= height:
            return False

        cell_x = int(x / cell_size)
        cell_y = int(y / cell_size)

        start_x = max(0, cell_x - 2)
        end_x = min(grid_width, cell_x + 3)
        start_y = max(0, cell_y - 2)
        end_y = min(grid_height, cell_y + 3)

        for i in range(start_x, end_x):
            for j in range(start_y, end_y):
                neighbor = grid[i, j]
                if neighbor != -1 and np.linalg.norm(point - active_points[neighbor]) < min_distance:
                    return False
        return True

    def get_random_point_around(point):
        for _ in range(max_attempts):
            angle = 2 * np.pi * np.random.rand()
            distance = np.random.uniform(min_distance, 2 * min_distance)
            new_point = point + np.array([np.cos(angle), np.sin(angle)]) * distance

            if is_valid_point(new_point):
                return new_point

        return None

    def insert_point(point):
        index = len(active_points)
        active_points.append(point)

        cell_x = int(point[0] / cell_size)
        cell_y = int(point[1] / cell_size)
        grid[cell_x, cell_y] = index

        samples.append(point)

    initial_point = np.array([np.random.uniform(0, width), np.random.uniform(0, height)])
    insert_point(initial_point)

    while active_points:
        current_point_index = np.random.choice(len(active_points))
        current_point = active_points[current_point_index]

        new_point = get_random_point_around(current_point)

        if new_point is not None:
            insert_point(new_point)

    fig, ax = plt.subplots()

    points_x, points_y = np.array(samples)[:, 0], np.array(samples)[:, 1]
    _ = ax.scatter(points_x, points_y)

    circles = [plt.Circle((xi, yi), radius=min_distance / 2, fill=False) for xi, yi in zip(points_x, points_y)]
    collection = PatchCollection(circles, match_original=True)
    ax.add_collection(collection)
    _ = ax.set(aspect='equal', xlabel=r'$x_1$', ylabel=r'$x_2$',
               xlim=[0, width], ylim=[0, height])
    plt.show()
    generate_poisson_disk_samples(width, height, min_distance)
    return np.array(samples)

# def latin_hypersquare_sampling(width, height, min_distance, num_samples=30):
#     samples = np.random.rand(num_samples, 2)
#
#     for i in range(num_dimensions):
#         indices = np.arange(num_samples)
#         np.random.shuffle(indices)
#         samples[:, i] = (indices + samples[:, i]) / num_samples
#
#     return samples
#
#     return lhs