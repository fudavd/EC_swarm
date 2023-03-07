from pathlib import Path
import sys
import os
import unittest
from typing import AnyStr

import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
print('Python %s on %s' % (sys.version, sys.platform))
sys.path.append(Path(os.path.abspath(__file__)).parents[2].__str__())
from utils import EA, CMAES

# matplotlib.use('module://backend_interagg')
# matplotlib.use('TkAgg')


def fitness_fun(genome):
    # rosenbrock_fitness
    # x, y = genome
    # return 100 * ((y - (x ** 2)) ** 2) + ((1 - (x ** 2)) ** 2)
    import cma
    return cma.ff.rosen(np.array(genome) + np.array([0.0, 0.0]))


class TestEA(unittest.TestCase):
    def test_DE(self, verbose=True):
        params = {}
        params['bounds'] = (-10, 10)
        params['D'] = 2
        params['evaluate_objective_type'] = 'full'
        params['pop_size'] = 250
        params['CR'] = 0.7
        params['F'] = 0.3

        learner = EA.DE(params, output_dir=None)
        self.EA(learner, params, verbose, 'de')

    def test_CMA(self, verbose=True):
        params = {}
        params['bounds'] = (-10, 10)
        params['D'] = 2
        params['evaluate_objective_type'] = 'full'
        params['pop_size'] = 250
        params['sigma0'] = 0.5

        learner = CMAES.CMAes(params, output_dir=None)
        self.EA(learner, params, verbose, 'CMAes')

    def EA(self, learner, params, verbose: bool, EA_type: AnyStr):
        if verbose:
            plt.ion()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            cont_x = np.arange(params['bounds'][0], params['bounds'][1]+0.1, 0.1)
            cont_y = np.arange(params['bounds'][0], params['bounds'][1]+0.1, 0.1)
            contour_pos = np.meshgrid(cont_x, cont_y)
            Z = []
            for x in cont_x:
                for y in cont_y:
                    Z.append(fitness_fun([x, y]))
            ax.grid(True)

        # setting number of:
        n_generations = 100  # generations

        population = learner.x_new

        for gen in range(n_generations):  # loop over generations
            fitnesses = []
            for individual in learner.x_new:  # loop over individuals
                fitness = fitness_fun(individual)
                fitnesses.append(fitness)

            learner.f_new = np.array(fitnesses)
            learner.x_new = learner.get_new_genome()

            if verbose:
                if gen % 10 == 0:
                    line1 = ax.plot(learner.x_new[:, 0], learner.x_new[:, 1], 'b+', alpha=0.15 + gen / (n_generations*1.2))
                    line2 = ax.plot(learner.x[:, 0], learner.x[:, 1], 'g*', alpha=0.15 + gen / (n_generations*1.2))
                    ax.set_xlim(params['bounds'][0], params['bounds'][1])
                    ax.set_ylim(params['bounds'][0], params['bounds'][1])
                    fig.canvas.draw()
                    ax.grid(True)
                print(f'number of evaluations: {gen}')

        if verbose:
            print('Best genome: ')
            print(learner.x_best_so_far[-1].flatten(), learner.f_best_so_far[-1])

            line1 = ax.plot(learner.x_new[:, 0], learner.x_new[:, 1], 'k+')
            line2 = ax.plot(learner.x[:, 0], learner.x[:, 1], 'k*')
            plt.contourf(contour_pos[0], contour_pos[1], np.array(Z).reshape(contour_pos[0].shape).T,
                         locator=ticker.LogLocator(),
                         levels=[0.1, 1, 100, 1000, 10_000, 100_000, 1000_000, 10_000_000],
                         cmap='Reds'
                         )
            ax.set_title(EA_type)
            plt.show()
            fig.savefig(f"./results/TEST_{EA_type.upper()}.pdf")
        self.assertAlmostEqual(fitness_fun(learner.x_best_so_far[-1].flatten()), 0.0, delta=0.01)  # add assertion here


if __name__ == '__main__':
    unittest.main()
