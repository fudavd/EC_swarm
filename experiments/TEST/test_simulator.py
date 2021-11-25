import math
import unittest

from utils.Simulate_swarm import simulate_swarm
from utils import Individual


class TestSim(unittest.TestCase):
    def test_sim(self, controller_type="Rand", verbose=False):
        genotype = Individual.thymio_genotype(controller_type)
        individual = Individual.Individual(genotype, 0)
        try:
            fitness = simulate_swarm(30, individual, not verbose)
        except:
            raise Exception("Could not calculate fitness")

        self.assertEqual(fitness > 0, True)


if __name__ == '__main__':
    print("STARTING")
    unittest.main()
    print("FINISHED")
