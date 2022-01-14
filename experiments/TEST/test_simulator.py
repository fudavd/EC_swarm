import math
import unittest
import sys
import os
import time

import numpy as np

print('Python %s on %s' % (sys.version, sys.platform))
from pathlib import Path
sys.path.append(Path(os.path.abspath(__file__)).parents[2].__str__())

from utils.Simulate_swarm import simulate_swarm_with_restart
from utils import Individual


class TestSim(unittest.TestCase):
    def test_sim(self, controller_type="Rand", headless=False):
        genotype = Individual.thymio_genotype("NN", 7, 2)
        # genotype = Individual.thymio_genotype("4dir", 7, 2)
        # genotype = Individual.thymio_genotype(controller_type, 7, 2)
        x_best_data = np.load("./results/NN/0/x_best.npy")[-1]
        # x_best_data = np.array([0.960515, 2.74798901, 3.57167516, 0.55376596])
        genotype['controller']["encoding"] = x_best_data
        individual = Individual.Individual(genotype, 0)
        tic = time.perf_counter()
        try:
            t_avg = 0
            for _ in range(10):
                fitness = simulate_swarm_with_restart(30, individual, True, [1, 1, 1, 1, 1])
                toc = time.perf_counter()
                t_avg += (toc - tic)/10
                tic = toc
        except:
            raise Exception("Could not calculate fitness")
        print(f"Average Simulate_Swarm running time: {t_avg:0.4f} seconds")
        self.assertEqual(fitness > 0, True)

if __name__ == '__main__':
    print("STARTING")
    unittest.main()
    print("FINISHED")
