import copy

from utils import EA
import cma
import numpy as np

class CMAes(EA.EA):
    es: cma.CMAEvolutionStrategy
    def __init__(self, params, output_dir='./results'):
        super().__init__(params, output_dir)
        self.Np = params['pop_size']  # Number of individuals
        self.D = params['D']  # Dimension
        # self.Cr = params['CR']  # Crossover probability
        # self.F = params['F']  # Mutation factor
        self.bounds = params['bounds']  # lower/upper bound on design variables

        solution_seed = np.random.uniform(self.bounds[0], self.bounds[1], self.D) / 5
        self.f_new = np.empty(self.Np)

        opts = {
            'popsize': params['pop_size'],
            'bounds': (
                [self.bounds[0]] * self.D, # lower bounds per dimension
                [self.bounds[1]] * self.D, # upper bounds per dimension
	        ),
        }
        self.es = cma.CMAEvolutionStrategy(solution_seed, params['sigma0'], inopts=opts)
        self.x_new = np.array(self.es.ask())

        self.x = copy.deepcopy(self.x_new)
        self.f = np.ones(self.Np) * np.inf

        self.f_best_so_far = []
        self.x_best_so_far = []

    def get_new_genome(self):
        ## CMA Documentation Example:
        # while not self.es.stop():
        #     solutions = self.es.ask()
        #     fitness = [cma.ff.rosen(x) for x in solutions]
        #     self.es.tell(solutions, fitness)
        #     self.es.logger.add() # write data to disc to be plotted
        #     self.es.disp()
        # self.es.result.pretty()

        pop_best = np.min(self.f_new)  # some book keeping
        if self.f_best_so_far == [] or pop_best < self.f_best_so_far[-1]:
            self.f_best_so_far.append(pop_best)
            self.x_best_so_far.append(copy.deepcopy(self.x_new[np.argmin(self.f_new), :].squeeze()))
        else:
            self.f_best_so_far.append(self.f_best_so_far[-1])
            self.x_best_so_far.append(self.x_best_so_far[-1])

        self.es.tell(self.x_new, self.f_new)
        # self.es.disp()
        self.x = copy.deepcopy(self.x_new)
        self.f = copy.deepcopy(self.f_new)
        self.x_new = np.array(self.es.ask())
        # clipped = np.clip(self.x_new, self.bounds[0], self.bounds[1])
        # assert (self.x_new == clipped).all()
        return self.x_new

