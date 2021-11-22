import math
import unittest

from utils.Simulate_swarm import simulate_swarm
from utils import Controllers


class TestSim(unittest.TestCase):
    def test_sim(self, controller_type="rand", verbose=False):
        if controller_type == "nn":
            controller = Controllers.NNController(5, 2)
        elif controller_type == "rand":
            controller = Controllers.RandomWalk(5, 2)
        else:
            raise Exception(f"Undefined type: {controller_type}")

        try:
            fitness = simulate_swarm(30, controller, not verbose)
        except:
            raise Exception("Could not calculate fitness")

        self.assertEqual(fitness > 0, True)


if __name__ == '__main__':
    print("STARTING")
    unittest.main()
    print("FINISHED")
