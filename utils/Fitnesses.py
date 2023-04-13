from typing import List

import numpy as np
from copy import deepcopy
import re
import scipy.io as sio

from utils.Individual import Individual


class FitnessCalculator:
    def __init__(self, swarm: List[Individual], initial_positions: np.ndarray, desired_movement: float,
                 arena: str = "circle_30x30", objectives: List = ['gradient']):
        """
        A calculator class for all fitness functions/elements.

        :param swarm: Swarm description
        :param initial_positions: initial (x,y) positions of all robots
        :param desired_movement: The desired distance to be traveled by the swarm
        :param arena: Type of arena
        """

        self.objectives = objectives
        self.current_cohesion = 0
        self.current_separation = 0
        self.current_alignment = 0
        self.current_movement = 0
        self.current_grad = 0

        self.sub_group_map = []
        self.unique_individuals = list(dict.fromkeys(swarm))
        self.n_sub_groups = len(self.unique_individuals)
        self.current_subs_grad = [0] * self.n_sub_groups
        self.current_subs_alignment = [0] * self.n_sub_groups
        for member in self.unique_individuals:
            self.sub_group_map.append([i for i, x in enumerate(swarm) if x == member])

        self.map = sio.loadmat(f'./utils/Gradient Maps/{arena}.mat')
        self.map = self.map['I']
        self.size_x = int(re.findall('\d+', arena)[-1])
        self.size_y = int(re.findall('\d+', arena)[-1])

        max_ind = np.unravel_index(self.map.argmax(), self.map.shape)
        relative_distance = np.divide(np.array(max_ind) + 1, self.map.shape)
        self.source_pos = relative_distance * self.size_x
        self.grad_constant_x = (len(np.arange(start=0.00, stop=self.size_x, step=0.04))) / self.size_x
        self.grad_constant_y = (len(np.arange(start=0.00, stop=self.size_y, step=0.04))) / self.size_y

        self.num_robots = len(swarm)
        self.initial_positions = initial_positions
        self.update_grad_vals(self.initial_positions)
        self.cohesion_range = 2.0  # The range to accept as a "neighborhood" of the focal robot
        self.desired_movement = desired_movement

        self.dij = np.zeros((self.num_robots, self.num_robots))

    def obtain_fitnesses(self, positions, headings):
        xx1, xx2 = np.meshgrid(positions[0], positions[0])
        yy1, yy2 = np.meshgrid(positions[1], positions[1])
        d_ij_x = xx1 - xx2
        d_ij_y = yy1 - yy2
        d_ij = np.sqrt(np.multiply(d_ij_x, d_ij_x) + np.multiply(d_ij_y, d_ij_y))
        self.dij = d_ij
        if self.objectives is None:
            return 0
        else:
            fitnesses = []
            for objective in self.objectives:
                if objective == 'gradient':
                    fitnesses.append(self.calculate_grad(positions))
                if objective == 'alignment':
                    fitnesses.append(self.calculate_alignment(headings))
                if objective == 'movement':
                    fitnesses.append(self.calculate_movement(positions))
                if objective == 'n_groups':
                    fitnesses.append(self.calculate_number_of_groups(positions)[:])
                if objective == 'cohesion_and_separation' or objective == 'coh_sep':
                    fitnesses.append(self.calculate_cohesion_and_separation(positions))
                if objective == 'alignment_sub':
                    fitnesses.append(self.calculate_subgroup_alignment(headings))
                if objective == 'gradient_sub':
                    fitnesses.append(self.calculate_subgroup_grad(positions))
            return np.hstack(fitnesses)

    def get_fitness_size(self):
        length = 0
        for objective in self.objectives:
            if objective == 'gradient' or objective == 'alignment' or objective == 'movement' or objective == 'n_groups':
                length += 1
            if objective == 'cohesion_and_separation' or objective == 'coh_sep':
                length += 2
            if objective == 'alignment_sub' or objective == 'gradient_sub':
                length += self.n_sub_groups
        return length

    def update_grad_vals(self, positions):
        grad_y = np.ceil(np.multiply(positions[0], self.grad_constant_x)).astype(int)
        grad_x = np.ceil(np.multiply(positions[1], self.grad_constant_y)).astype(int)

        grad_x[grad_x < 0] = 0
        grad_x[grad_x >= self.size_x / 0.04] = 0
        grad_y[grad_y < 0] = 0
        grad_y[grad_y >= self.size_y / 0.04] = 0
        self.grad_vals = self.map[grad_x, grad_y]

    def calculate_grad(self, positions):
        self.update_grad_vals(positions)
        self.current_grad = self.current_grad + (np.sum(self.grad_vals) / self.num_robots) / 255.0
        return self.current_grad

    def calculate_subgroup_grad(self, positions) -> np.array:
        self.update_grad_vals(positions)
        for i_sub in range(self.n_sub_groups):
            indices = self.sub_group_map[i_sub]
            sub_group_size = len(indices)
            self.current_subs_grad[i_sub] += (np.sum(self.grad_vals[indices]) / max(1, sub_group_size)) / 255.0

        return np.array(self.current_subs_grad)

    def calculate_cohesion_and_separation(self, positions):
        """
        The "cohesion" and "separation", fitness function elements, as in "Evolving flocking in embodied agents based
        on local and global application of Reynolds".

        :param positions: (x,y) positions of all robots

        :return: cohesion and separation values [0, 1]
        In case of no neighbors, "cohesion" is calculated as zero for that robot, separation is naturally zero.
        """

        distance_to_com_of_neighbors = np.zeros((1, self.num_robots))
        cohesion_metric = np.zeros((1, self.num_robots))
        collision_counter = np.zeros((1, self.num_robots))
        collision_assumption = 0.2

        xx1, xx2 = np.meshgrid(positions[0], positions[0])
        yy1, yy2 = np.meshgrid(positions[1], positions[1])
        d_ij_x = xx1 - xx2
        d_ij_y = yy1 - yy2
        d_ij = np.sqrt(np.multiply(d_ij_x, d_ij_x) + np.multiply(d_ij_y, d_ij_y))
        self.dij = d_ij  # This is a shared variable
        neighbors = deepcopy(d_ij)
        collisions = deepcopy(d_ij)

        collisions[collisions > collision_assumption] = 0.0
        collision_counter = np.count_nonzero(collisions, axis=1)
        collision_at_t = np.sum(collision_counter) / self.num_robots
        self.current_separation = self.current_separation + collision_at_t

        neighbors[neighbors > self.cohesion_range] = 0.0
        neighbors[neighbors != 0] = 1

        cohesion_at_t = 0

        for i in range(self.num_robots):
            nsx = np.multiply(d_ij_x, neighbors[i][:])
            nsx = nsx[np.nonzero(nsx)]
            nsy = np.multiply(d_ij_y, neighbors[i][:])
            nsy = nsy[np.nonzero(nsy)]

            if len(nsx) != 0:
                distance_to_com_of_neighbors_x = np.mean(nsx)
                distance_to_com_of_neighbors_y = np.mean(nsy)

                distance_to_com_of_neighbors[0][i] = np.sqrt(
                    np.power(distance_to_com_of_neighbors_x, 2) + np.power(distance_to_com_of_neighbors_y, 2))
                cohesion_at_t = cohesion_at_t + (1 - (distance_to_com_of_neighbors[0][i] / self.cohesion_range))

        self.current_cohesion = self.current_cohesion + cohesion_at_t / self.num_robots

        return np.array([self.current_cohesion, self.current_separation])

    def calculate_alignment(self, headings):
        """
        The "alignment", fitness function element, as in "Evolving flocking in embodied agents based
        on local and global application of Reynolds". Requires calculate_cohesion_and_separation() method to work
        alongside, this method use some information from there.

        :param headings: heading angles of all robots, in radians

        :return: alignment value [0, 1]
        In case of no neighbors, "alignment" is calculated as zero for that robot, separation is naturally zero.
        """
        sum_cosh = np.zeros((1, self.num_robots))
        sum_sinh = np.zeros((1, self.num_robots))
        number_of_neighbors = np.zeros((1, self.num_robots))
        alignment_at_t = 0

        for i in range(0, self.num_robots):
            sum_cosh[0, i] = np.cos(headings[i])
            sum_sinh[0, i] = np.sin(headings[i])

            for j in range(0, self.num_robots):
                if self.dij[i, j] < self.cohesion_range and i != j:
                    sum_cosh[0, i] = sum_cosh[0, i] + np.cos(headings[j])
                    sum_sinh[0, i] = sum_sinh[0, i] + np.sin(headings[j])
                    number_of_neighbors[0][i] = number_of_neighbors[0][i] + 1

            if number_of_neighbors[0][i] > 0:
                alignment_at_t = alignment_at_t + np.sqrt(np.power(sum_cosh[0, i], 2) + np.power(sum_sinh[0, i], 2)) / (
                            number_of_neighbors[0][i] + 1)
        self.current_alignment = self.current_alignment + alignment_at_t / self.num_robots

        return np.array([self.current_alignment])

    def calculate_subgroup_alignment(self, headings):
        """
        The "alignment", fitness function element, as in "Evolving flocking in embodied agents based
        on local and global application of Reynolds". Requires calculate_cohesion_and_separation() method to work
        alongside, this method use some information from there.

        :param headings: heading angles of all robots, in radians

        :return: alignment value [0, 1]
        In case of no neighbors, "alignment" is calculated as zero for that robot, separation is naturally zero.
        """
        for i_sub in range(self.n_sub_groups):
            indices = self.sub_group_map[i_sub]
            sub_group_size = len(indices)

            sum_cosh = np.zeros((1, self.num_robots))
            sum_sinh = np.zeros((1, self.num_robots))
            number_of_neighbors = np.zeros((1, self.num_robots))
            alignment_at_t = 0

            for i in indices:
                sum_cosh[0, i] = np.cos(headings[i])
                sum_sinh[0, i] = np.sin(headings[i])

                for j in range(0, self.num_robots):
                    if self.dij[i, j] < self.cohesion_range and i != j:
                        sum_cosh[0, i] = sum_cosh[0, i] + np.cos(headings[j])
                        sum_sinh[0, i] = sum_sinh[0, i] + np.sin(headings[j])
                        number_of_neighbors[0][i] = number_of_neighbors[0][i] + 1

                if number_of_neighbors[0][i] > 0:
                    alignment_at_t = alignment_at_t + np.sqrt(
                        np.power(sum_cosh[0, i], 2) + np.power(sum_sinh[0, i], 2)) / (number_of_neighbors[0][i] + 1)
            self.current_subs_alignment[i_sub] += alignment_at_t / sub_group_size

        return np.array(self.current_subs_alignment)

    def calculate_movement(self, positions):
        """
        The "movement", fitness function element, as in "Evolving flocking in embodied agents based
        on local and global application of Reynolds".

        :param positions: (x, y) positions of all robots
        :return: movement value [0, 1]
        """

        total_movement = 0

        for i in range(0, self.num_robots):
            movement_percentage = np.sqrt(np.square(self.initial_positions[0][i] - positions[0][i]) + np.square(
                self.initial_positions[1][i] - positions[1][i])) / self.desired_movement

            if movement_percentage >= 1.0:
                movement_percentage = 1.0

            total_movement = total_movement + movement_percentage

        total_movement = total_movement / self.num_robots

        return np.array([total_movement])

    def calculate_number_of_groups(self, positions) -> int:
        """
        The number of groups, calculated based on "equivalence classes".

        :param positions: (x, y) positions of all robots
        :return: number_of_groups (integer)
        """

        sensing_range = 2.0  # The range to assume robots can sense each other.
        member_number = np.shape(positions[0])[0]
        xx1, xx2 = np.meshgrid(positions[0], positions[0])
        yy1, yy2 = np.meshgrid(positions[1], positions[1])
        d_ij_x = xx1 - xx2
        d_ij_y = yy1 - yy2
        d_ij = np.sqrt(np.multiply(d_ij_x, d_ij_x) + np.multiply(d_ij_y, d_ij_y))
        d_ij = np.triu(d_ij)
        mm = 0

        list1 = []
        list2 = []

        for j in range(member_number):
            for k in range(member_number):
                if (d_ij[j, k] < sensing_range) and (j != k) and (d_ij[j, k] != 0):
                    mm = mm + 1
                    list1.append(j)
                    list2.append(k)

        list1 = np.array(list1)
        list2 = np.array(list2)

        m = mm
        n = member_number
        nf = []

        for k in range(n):
            nf.append(k)

        nf = np.array(nf)

        for l in range(m):
            j = list1[l]

            while nf[j] != j:
                j = nf[j]

            k = list2[l]

            while nf[k] != k:
                k = nf[k]

            if j != k:
                nf[j] = k

        for j in range(n):
            while nf[j] != nf[nf[j]]:
                nf[j] = nf[nf[j]]

        uniques = np.unique(nf)
        number_of_groups = uniques.shape[0]

        return number_of_groups


def Calculate_fitness_size(swarm: List[Individual], env_params):
    positions = np.zeros((2, len(swarm)))
    fitness_size = FitnessCalculator(swarm, positions, 1.0,
                                     objectives=env_params['objectives']).get_fitness_size()
    return fitness_size
