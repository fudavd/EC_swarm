import numpy as np


class Sensors:
    # Different sensor types for robots in the swarm. Each sensor is a "device" located in "each robot". This class put
    # all sensor outputs of all robots in a single matrix for convenience.
    def __init__(self):
        pass

    def four_dir_sensor(self, positions, headings):
        # The sensor model used in "Evolving flocking in embodied agents based on local and global application of
        # Reynolds"

        # Inputs:
        # positions : np.array(2,number_of_robots), positions[0][:] --> x positions of all robots, positions[1][:] --> y
        # positions of all robots
        # headings: np.array(1, number_of_robots), heading angles of all robots, in radians

        # Outputs:
        # np.array(4, num_robots) : 4 distances that sensors of each robot measures, are located in
        # corresponding row in this matrix. In case of no neighbor, output matrix is a zeros matrix.

        robot_num = np.shape(positions[0])[0]
        sensing_range = 1.0  # Sensing range of sensors, 25 cm (0.25) in the paper
        xx1, xx2 = np.meshgrid(positions[0], positions[0])
        yy1, yy2 = np.meshgrid(positions[1], positions[1])
        d_ij_x = xx1 - xx2
        d_ij_y = yy1 - yy2
        d_ij = np.sqrt(np.multiply(d_ij_x, d_ij_x) + np.multiply(d_ij_y, d_ij_y))
        d_ij[d_ij > sensing_range] = 0.0

        ij_ang = np.arctan2(d_ij_y, d_ij_x)
        ij_ang = ij_ang - headings
        ij_ang[d_ij == 0] = 0.0

        output = np.zeros([robot_num, 4])

        for i in range(robot_num):
            for j in range(robot_num):
                ij_ang[i, j] = self.wraptopi(ij_ang[i, j])

                if (ij_ang[i, j] < 0.7854) and (ij_ang[i, j] >= -0.7854):
                    if (d_ij[i, j] != 0) and ((d_ij[i, j] < output[i, 0]) or (output[i, 0] == 0.0)):
                        output[i, 0] = 1 - (d_ij[i, j] / sensing_range)
                elif (ij_ang[i, j] < 2.3562) and (ij_ang[i, j] >= 0.7854):
                    if (d_ij[i, j] != 0) and ((d_ij[i, j] < output[i, 1]) or (output[i, 1] == 0.0)):
                        output[i, 1] = 1 - (d_ij[i, j] / sensing_range)
                elif ((ij_ang[i, j] < 3.1416) and (ij_ang[i, j] >= 2.3562)) or (
                        (ij_ang[i, j] < -2.3562) and (ij_ang[i, j] >= -3.1416)):
                    if (d_ij[i, j] != 0) and ((d_ij[i, j] < output[i, 2]) or (output[i, 2] == 0.0)):
                        output[i, 2] = 1 - (d_ij[i, j] / sensing_range)
                elif (ij_ang[i, j] < -0.7854) and (ij_ang[i, j] >= -2.3562):
                    if (d_ij[i, j] != 0) and ((d_ij[i, j] < output[i, 3]) or (output[i, 3] == 0.0)):
                        output[i, 3] = 1 - (d_ij[i, j] / sensing_range)
        return output

    def k_nearest_sensor(self, positions, headings):
        # The sensor model for giving distance and bearing for k-nearest neighbor of each robot.

        # Inputs:
        # positions : np.array(2,number_of_robots), positions[0][:] --> x positions of all robots, positions[1][:] --> y
        # positions of all robots
        # headings: np.array(1, number_of_robots), heading angles of all robots, in radians

        # Outputs:
        # output_distances: np.array(k, num_robots) : k distances that sensors of each robot measures, are located in
        # corresponding row in this matrix.
        # output_bearings: np.array(k, num_robots) : k bearing angles that sensors of each robot measures, are located
        # in corresponding row in this matrix.
        # In case of neighbors < k, matrices are filled with zeros.

        robot_num = np.shape(positions[0])[0]
        sensing_range = 2.0  # Sensing range of sensors
        k = 4  # The k value
        num_neigh = np.zeros([1, robot_num])

        xx1, xx2 = np.meshgrid(positions[0], positions[0])
        yy1, yy2 = np.meshgrid(positions[1], positions[1])
        d_ij_x = xx1 - xx2
        d_ij_y = yy1 - yy2
        d_ij = np.sqrt(np.multiply(d_ij_x, d_ij_x) + np.multiply(d_ij_y, d_ij_y))
        d_ij[d_ij > sensing_range] = 0.0

        ij_ang = np.arctan2(d_ij_y, d_ij_x)
        ij_ang = ij_ang - headings
        ij_ang[d_ij == 0] = 0.0

        output_distances = np.zeros([robot_num, 4])
        output_bearings = np.zeros([robot_num, 4])

        for i in range(robot_num):

            for j in range(robot_num):
                ij_ang[i, j] = self.wraptopi(ij_ang[i, j])

                if (d_ij[i, j] != 0):
                    num_neigh[0, i] = num_neigh[0, i] + 1

            if num_neigh[0, i] >= k:
                temp_distances = np.argsort(d_ij[i, :])
                idx = temp_distances[d_ij[i, :][temp_distances] != 0][:k]
                output_distances[i, :] = d_ij[i, :][idx]
                output_bearings[i, :] = ij_ang[i, :][idx]
            elif num_neigh[0, i] <= k and num_neigh[0, i] > 0:
                temp_distances = np.argsort(d_ij[i, :])
                idx = temp_distances[d_ij[i, :][temp_distances] != 0][:int(num_neigh[0, i])]
                output_distances[i, 0:int(num_neigh[0, i])] = d_ij[i, :][idx]
                output_bearings[i, 0:int(num_neigh[0, i])] = ij_ang[i, :][idx]

        return output_distances, output_bearings

    def omni_dir_sensor(self, positions, headings):
        # The sensor model for giving distance and bearing for every neighbor within a certain radius.

        # Inputs:
        # positions : np.array(2,number_of_robots), positions[0][:] --> x positions of all robots, positions[1][:] --> y
        # positions of all robots
        # headings: np.array(1, number_of_robots), heading angles of all robots, in radians

        # Outputs:
        # list: output_distances[output_distances_robot1, output_distances_robot2, ..., output_distances_robotn]
        # list: output_bearings[output_bearings_robot1, output_bearings_robot2, ..., output_bearings_robotn]
        # output_distances_robot_n: np.array(neighbor_number, 1) : Distances that sensor of n'th robot
        # measures, are located in corresponding row in this matrix.
        # output_bearings_robot_n: np.array(neighbor_number, 1) : Bearing angles that sensor of n'th robot measures,
        # are located in corresponding row of this matrix.
        # In case of no neighbors, output_distances_robot_n and output_bearings_robot_n is [1,1] zero matrix.

        robot_num = np.shape(positions[0])[0]
        sensing_range = 2.0  # The radius for omni-directional sensing
        xx1, xx2 = np.meshgrid(positions[0], positions[0])
        yy1, yy2 = np.meshgrid(positions[1], positions[1])
        d_ij_x = xx1 - xx2
        d_ij_y = yy1 - yy2
        d_ij = np.sqrt(np.multiply(d_ij_x, d_ij_x) + np.multiply(d_ij_y, d_ij_y))
        d_ij[d_ij > sensing_range] = 0.0

        ij_ang = np.arctan2(d_ij_y, d_ij_x)
        ij_ang = ij_ang - headings
        ij_ang[d_ij == 0] = 0.0

        output_distances = []
        output_angles = []
        num_neigh = np.zeros([1, robot_num])

        for i in range(robot_num):
            distances_to_neighbors = []
            bearing_of_neighbors = []

            for j in range(robot_num):
                ij_ang[i, j] = self.wraptopi(ij_ang[i, j])

                if (d_ij[i, j] != 0):
                    distances_to_neighbors.append(d_ij[i, j])
                    bearing_of_neighbors.append(ij_ang[i, j])
                    num_neigh[0, i] = num_neigh[0, i] + 1

            if num_neigh[0, i] > 0:
                distances_to_neighbors = np.array(distances_to_neighbors)
                bearing_of_neighbors = np.array(bearing_of_neighbors)
                output_distances.append(distances_to_neighbors)
                output_angles.append(bearing_of_neighbors)
            else:
                distances_to_neighbors = np.array(0.00)
                bearing_of_neighbors = np.array(0.00)
                output_distances.append(distances_to_neighbors)
                output_angles.append(bearing_of_neighbors)

        return output_distances, output_angles

    def heading_sensor(self, positions, headings):
        # The sensor model for giving a value between 0 and 1 according to the relative angle difference between focal
        # robot and average heading angle of neighbors. As defined in "Evolving flocking in embodied agents based on
        # local and global application of Reynolds".

        # Inputs:
        # positions : np.array(2,number_of_robots), positions[0][:] --> x positions of all robots, positions[1][:] --> y
        # positions of all robots
        # headings: np.array(1, number_of_robots), heading angles of all robots, in radians

        # Outputs:
        # avg_rel_neg_heading: np.array(1, number_of_robots) --> the value calculated for each robot. In case of no
        # neighbors, the value is 0.5 (the same with perfectly matching heading angles).

        robot_num = np.shape(positions[0])[0]
        sensing_range = 2.0  # The range which a heading average is calculated in.

        xx1, xx2 = np.meshgrid(positions[0], positions[0])
        yy1, yy2 = np.meshgrid(positions[1], positions[1])
        d_ij_x = xx1 - xx2
        d_ij_y = yy1 - yy2
        d_ij = np.sqrt(np.multiply(d_ij_x, d_ij_x) + np.multiply(d_ij_y, d_ij_y))

        sum_cosh = np.zeros((1, robot_num))
        sum_sinh = np.zeros((1, robot_num))

        num_neigh = np.zeros((1, robot_num))

        for i in range(0, robot_num):
            for j in range(0, robot_num):
                if d_ij[i, j] < sensing_range and i != j:
                    sum_cosh[0, i] = sum_cosh[0, i] + np.cos(headings[j])
                    sum_sinh[0, i] = sum_sinh[0, i] + np.sin(headings[j])
                    num_neigh[0, i] = num_neigh[0, i] + 1

        avg_abs_neg_heading_vec_x = np.divide(sum_cosh, np.sqrt(np.square(sum_cosh) + np.square(sum_sinh)))
        avg_abs_neg_heading_vec_y = np.divide(sum_sinh, np.sqrt(np.square(sum_cosh) + np.square(sum_sinh)))
        avg_abs_neg_heading = np.arctan2(avg_abs_neg_heading_vec_y, avg_abs_neg_heading_vec_x)
        avg_abs_neg_heading = avg_abs_neg_heading[0]
        avg_rel_neg_heading = avg_abs_neg_heading - headings

        for i in range(0, robot_num):
            avg_rel_neg_heading[i] = self.wraptopi(avg_rel_neg_heading[i])

        for i in range(0, robot_num):
            if num_neigh[0, i] > 0:
                if (avg_rel_neg_heading[i] > -3.1416) and (avg_rel_neg_heading[i] < 0.00):
                    avg_rel_neg_heading[i] = 0.5 + (avg_rel_neg_heading[i] / (-3.1416)) * 0.5
                elif (avg_rel_neg_heading[i] < 3.1416) and (avg_rel_neg_heading[i] > 0.00):
                    avg_rel_neg_heading[i] = 0.5 - (avg_rel_neg_heading[i] / (3.1416)) * 0.5
                else:
                    avg_rel_neg_heading[i] = 0.5
            else:
                avg_rel_neg_heading[i] = 0.5

        return avg_rel_neg_heading

    def wraptopi(self, x):
        x = x % (3.1415926 * 2)
        x = (x + (3.1415926 * 2)) % (3.1415926 * 2)

        if (x > 3.1415926):
            x = x - (3.1415926 * 2)

        return x
