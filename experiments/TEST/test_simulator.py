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
        individual = Individual.Individual(genotype, 0)
        try:
            fitness = simulate_swarm_with_restart_population(600, [individual], 14, headless, [0, 0, 0, 0, 1])
        except:
            raise Exception("Could not calculate fitness")
        self.assertEqual(fitness > 0, True)


if __name__ == '__main__':
    print("STARTING")
    unittest.main()
    print("FINISHED")
