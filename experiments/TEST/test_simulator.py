import unittest
import sys
import os

print('Python %s on %s' % (sys.version, sys.platform))
from pathlib import Path

sys.path.append(Path(os.path.abspath(__file__)).parents[2].__str__())

from utils.Simulate_swarm_population import simulate_swarm_with_restart_population
from utils import Individual


class TestSim(unittest.TestCase):
    def test_sim(self, controller_type="Rand", headless=False):
        genotype = Individual.thymio_genotype(controller_type, 9, 2)

        individuals = []
        swarm_size = 20
        for i in range(swarm_size):
            genotype['morphology']['rgb'] = [float(i < 10), float(i < 10), float(i < 10)]
            individual = Individual.Individual(genotype, 0)
            individuals.append(individual)
        try:
            fitness = simulate_swarm_with_restart_population(600, [individuals], swarm_size, headless, [0, 0, 0, 0, 1])
        except:
            raise Exception("Could not calculate fitness")
        self.assertEqual(fitness > 0, True)


if __name__ == '__main__':
    print("STARTING")
    unittest.main()
    print("FINISHED")
