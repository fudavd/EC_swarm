import numpy as np
from copy import deepcopy
import scipy.io as sio

class FitnessCalculator:

    def __init__(self, num_robots, initial_positions, desired_movement):
        # A calculator class for all fitness functions/elements.

        # Arguments:
        # num_robots (integer) : Number of robots in the swarm
        # initial_positions : np.array(2,number_of_robots) --> initial x positions of all robots,  positions[1][:] -->
        # initial y positions of all robots
        # desired_movement (float) : The desired distance to be traveled by the swarm. Defined in "Evolving flocking in
        # embodied agents based on local and global application of Reynolds"

        self.current_cohesion = 0
        self.current_separation = 0
        self.current_alignment = 0
        self.current_movement = 0
        self.current_grad = 0

        self.map = sio.loadmat('./utils/Gradient Maps/circle_30x30.mat')
        self.map = self.map['I']
        self.size_x = 30
        self.size_y = 30
        self.grad_constant_x = (len(np.arange(start=0.00, stop=self.size_x, step=0.04))) / self.size_x
        self.grad_constant_y = (len(np.arange(start=0.00, stop=self.size_y, step=0.04))) / self.size_y

        self.num_robots = num_robots
        self.initial_positions = initial_positions
        self.cohesion_range = 2.0  # The range to accept as a "neighborhood" of the focal robot
        self.desired_movement = desired_movement

        self.dij = np.zeros((self.num_robots,self.num_robots))

    def calculate_grad(self, positions):
        self.grad_y = np.ceil(np.multiply(positions[0], self.grad_constant_x))
        self.grad_x = np.ceil(np.multiply(positions[1], self.grad_constant_y))
        self.grad_x = self.grad_x.astype(int)
        self.grad_y = self.grad_y.astype(int)

        self.grad_x[self.grad_x < 0] = 0
        self.grad_x[self.grad_x >= self.size_x/0.04] = 0
        self.grad_y[self.grad_y < 0] = 0
        self.grad_y[self.grad_y >= self.size_y/0.04] = 0

        self.grad_vals = self.map[self.grad_x, self.grad_y]

        self.current_grad = self.current_grad + (np.sum(self.grad_vals) / self.num_robots) / 255.0

        return self.current_grad

    def calculate_cohesion_and_separation(self,positions):
        # The "cohesion" and "separation", fitness function elements, as in "Evolving flocking in embodied agents based
        # on local and global application of Reynolds".

        # Input:
        # positions : np.array(2,number_of_robots), positions[0][:] --> x positions of all robots, positions[1][:] --> y
        # positions of all robots

        # Output:
        # np.array([current_cohesion], [current_separation]) --> In each call of this function, the "cohesion" and
        # "separation" values are returned. In order to get a value between 0-1 for each (time average), the value
        # returned by this function should be divided by the number which this function is called (time step).

        # In case of no neighbors, "cohesion" is calculated as zero for that robot, separation is naturally zero.

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
        # The "alignment", fitness function element, as in "Evolving flocking in embodied agents based
        # on local and global application of Reynolds".

        # Requires calculate_cohesion_and_separation() method to work alongside, this method use some information from
        # there.

        # Input:
        # headings : np.array(1, number_of_robots), heading angles of all robots, in radians

        # Output:
        # np.array([current_alignment] --> In each call of this function, the "alignment"
        # value is returned. In order to get a value between 0-1 for "alignment" (time average), the value
        # returned by this function should be divided by the number which this function is called (time step).

        # In case of no neighbors, "alignment" is calculated as zero for that robot, separation is naturally zero.

        sum_cosh = np.zeros((1, self.num_robots))
        sum_sinh = np.zeros((1, self.num_robots))
        number_of_neighbors = np.zeros((1, self.num_robots))
        alignment_at_t = 0

        for i in range(0,self.num_robots):
            sum_cosh[0,i] = np.cos(headings[i])
            sum_sinh[0,i] = np.sin(headings[i])

            for j in range(0,self.num_robots):
                if self.dij[i,j] < self.cohesion_range and i != j:

                    sum_cosh[0,i] = sum_cosh[0,i] + np.cos(headings[j])
                    sum_sinh[0,i] = sum_sinh[0,i] + np.sin(headings[j])
                    number_of_neighbors[0][i] = number_of_neighbors[0][i] + 1

            if number_of_neighbors[0][i]>0:
                alignment_at_t = alignment_at_t + np.sqrt(np.power(sum_cosh[0,i],2)+np.power(sum_sinh[0,i],2))/(number_of_neighbors[0][i]+1)
        self.current_alignment = self.current_alignment + alignment_at_t/self.num_robots

        return np.array([self.current_alignment])

    def calculate_movement(self,positions):
        # The "movement", fitness function element, as in "Evolving flocking in embodied agents based
        # on local and global application of Reynolds".

        # Input:
        # positions : np.array(2,number_of_robots), positions[0][:] --> x positions of all robots, positions[1][:] --> y
        # positions of all robots

        # Output:
        # np.array([total_movement] --> In each call of this function, the "movement"
        # value is returned. In order to get a value between 0-1 for "movement" (time average), the value
        # returned by this function should be divided by the number which this function is called (time step).

        total_movement = 0

        for i in range(0, self.num_robots):
            movement_percentage = np.sqrt(np.square(self.initial_positions[0][i]-positions[0][i])+np.square(self.initial_positions[1][i]-positions[1][i]))/self.desired_movement

            if movement_percentage>=1.0:
                movement_percentage = 1.0

            total_movement = total_movement + movement_percentage

        total_movement = total_movement/self.num_robots

        return np.array([total_movement])

    def calculate_number_of_groups(self,positions):
        # The number of groups, calculated based on "equivalence classes".

        # Input:
        # positions : np.array(2,number_of_robots), positions[0][:] --> x positions of all robots, positions[1][:] --> y
        # positions of all robots

        # Output:
        # number_of_groups (integer)

        sensing_range = 2.0  # The range to assume robots can sense each other.
        member_number = np.shape(positions[0])[0]
        xx1, xx2 = np.meshgrid(positions[0] , positions[0])
        yy1, yy2 = np.meshgrid(positions[1] , positions[1])
        d_ij_x = xx1 - xx2
        d_ij_y = yy1 - yy2
        d_ij = np.sqrt(np.multiply(d_ij_x, d_ij_x) + np.multiply(d_ij_y,d_ij_y))
        d_ij = np.triu(d_ij)
        mm=0

        list1 = []
        list2 = []

        for j in range(member_number):
            for k in range(member_number):
                if (d_ij[j,k]<sensing_range) and (j!=k) and (d_ij[j,k]!=0):
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

            while nf[k] !=k:
                k = nf[k]

            if j != k:
                nf[j] = k

        for j in range(n):
            while nf[j] != nf[nf[j]]:
                nf[j] = nf[nf[j]]

        uniques = np.unique(nf)
        number_of_groups = uniques.shape[0]

        return number_of_groups