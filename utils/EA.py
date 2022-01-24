import numpy as np
import random as rand
import copy
import math


class EA(object):
    def __init__(self, params, output_dir="./results"):
        self.x_new = None
        self.x_best_so_far = None
        self.f_best_so_far = None
        self.x = None
        self.f = None
        self.directory_name = output_dir

    def save_results(self):
        np.save(self.directory_name + '/' + 'fitnesses', np.array(self.f))
        np.save(self.directory_name + '/' + 'genomes', np.array(self.x))

        np.save(self.directory_name + '/' + 'f_best', np.array(self.f_best_so_far))
        np.save(self.directory_name + '/' + 'x_best', np.array(self.x_best_so_far))

    def save_checkpoint(self):
        self.save_results()
        np.save(self.directory_name + '/' + 'last_x_new', np.array(self.x_new))
        np.save(self.directory_name + '/' + 'last_x', np.array(self.x))
        np.save(self.directory_name + '/' + 'last_f', np.array(self.f))

    @staticmethod
    def get_new_genome(self):
        return


class DE(EA):
    def __init__(self, params, output_dir="./results"):
        super().__init__(params, output_dir)
        self.Np = params['pop_size']  # Number of individuals
        self.D = params['D']  # Dimension
        self.Cr = params['CR']  # Crossover probability
        self.F = params['F']  # Mutation factor
        self.bounds = params['bounds']  # lower/upper bound on design variables

        # Initialize the new population array
        self.x_new = np.random.uniform(self.bounds[0], self.bounds[1], (self.Np, self.D))/5
        self.f_new = np.empty(self.Np)

        # Initialize the current population array
        self.x = copy.deepcopy(self.x_new)
        self.f = np.ones(self.Np) * np.inf

        self.f_best_so_far = []
        self.x_best_so_far = []

        self.directory_name = output_dir
        self.fitnesses = []
        self.genomes = []

    def get_new_genome(self):
        pop_best = np.min(self.f_new)  # some book keeping
        if self.f_best_so_far == [] or pop_best < self.f_best_so_far[-1]:
            self.f_best_so_far.append(pop_best)
            self.x_best_so_far.append(copy.deepcopy(self.x_new[np.argmin(self.f_new), :].squeeze()))
        else:
            self.f_best_so_far.append(self.f_best_so_far[-1])
            self.x_best_so_far.append(self.x_best_so_far[-1])

        for i in range(self.Np):
            if self.f_new[i] < self.f[i]:  # update population
                self.f[i] = self.f_new[i]
                self.x[i] = copy.deepcopy(self.x_new[i])

            r0 = i
            while (r0 == i):
                r0 = math.floor(rand.random() * self.Np)
            r1 = r0
            while (r1 == r0 or r1 == i):
                r1 = math.floor(rand.random() * self.Np)
            r2 = r1
            while (r2 == r1 or r2 == r0 or r2 == i):
                r2 = math.floor(rand.random() * self.Np)

            jrand = math.floor(rand.random() * self.D)

            for j in range(self.D):
                if (rand.random() <= self.Cr or j == jrand):
                    # Mutation
                    self.x_new[i][j] = copy.deepcopy(self.x[r0][j] + self.F * (self.x[r1][j] - self.x[r2][j]))
                else:
                    self.x_new[i][j] = copy.deepcopy(self.x[i][j])

        self.x_new = np.clip(self.x_new, self.bounds[0], self.bounds[1])
        return self.x_new
