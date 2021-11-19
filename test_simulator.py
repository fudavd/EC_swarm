import math
import numpy as np
import time
import os

from utils.Simulate_swarm import simulate_swarm
from utils.Controllers import NNController

def main():
    controller = NNController(5, 2)
    fitness = simulate_swarm(30, controller, False)
    print(fitness)


if __name__ == '__main__':
    print("STARTING")
    main()
    print("FINISHED")
