import math
import numpy as np
import time
import os

from utils.Simulate_swarm import simulate_swarm
from utils.Individual import Individual
from utils.Controllers import RandomWalk

def main():
    # brain = {'type': 'Rand',
    #          'params': {'output_space': 2}}
    # body = 'models/thymio/model.urdf'
    # genotype = {'morphology': body,
    #             'controller': brain}
    #
    # robot = Individual('models/thymio/model.urdf', genotype, 420)

    controller = RandomWalk(10, 2)
    simulate_swarm(30, controller, False)


if __name__ == '__main__':
    print("STARTING")
    main()
    print("FINISHED")
