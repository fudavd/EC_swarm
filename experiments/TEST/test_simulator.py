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
        # x_best_data = np.load("./results/NN/0/x_best.npy")[-1]
        # x_best_data = np.array([-2.01726279, 6.30453255, 2.0527406, 3.04765784, -5.6516686, 1.37751445,
        #                         3.08969845, -9.09071622, 7.58943147, -7.26425468, -3.32725735, -6.55240763,
        #                         3.19336503, -6.96198762, 1.51649782, 7.77321661, 0.6312846, 6.08140945,
        #                         -7.62144621, 7.98073233, -8.27071306, -9.5645343, -7.28719315, 5.62086136,
        #                         -1.81314204, 1.60956186, 5.3963316, 2.63793128, -3.10520065, 7.42890612,
        #                         8.70913579, 4.02015165, 5.8084021, 5.47545099, -2.18094337, 2.90340204,
        #                         0.01553369, -8.03005755, 6.83006695, -7.742819, 5.2081103, -7.82327787,
        #                         0.15770026, 8.63430149, -3.39332106, -1.53456719, -8.09988099, -9.60153121,
        #                         2.04077271, -9.37981625, -7.72498987, -9.40744374, 5.66737034, -8.80126302,
        #                         4.67719079, 6.37541599, -3.34162634, -1.73790739, 2.87770606, -5.3657794,
        #                         -4.21654982, -2.59173481, -3.45108313, -6.31337429, 4.10077297, -4.21639661,
        #                         -0.4427585, 3.43910915, 0.6438242, 8.93780549, -0.87891166, 2.46293976,
        #                         -4.65282899, 2.64042385, -1.26370578, -8.96288862, -9.02543338, 1.18550862,
        #                         6.70087616, 0.93965397])
        x_best_data = np.random.uniform(-5, 5, 100)
        genotype['controller']["encoding"] = x_best_data
        genotype['controller']["params"]['torch'] = False
        individual = Individual.Individual(genotype, 0)
        tic = time.perf_counter()
        try:
            t_avg = 0
            for _ in range(10):
                fitness = simulate_swarm_with_restart(60, individual, False, [1, 1, 1, 1, 1])
                toc = time.perf_counter()
                t_avg += (toc - tic) / 10
                tic = toc
        except:
            raise Exception("Could not calculate fitness")
        print(f"Average Simulate_Swarm running time: {t_avg:0.4f} seconds")
        self.assertEqual(fitness > 0, True)


if __name__ == '__main__':
    print("STARTING")
    unittest.main()
    print("FINISHED")
