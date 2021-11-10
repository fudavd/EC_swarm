import math
import numpy as np
import time
import os

from utils.Simulate_swarm import simulate_swarm
from utils.Controllers import RandomWalk

def main():
    controller = RandomWalk(10, 2)
    simulate_swarm(30, controller, False)


if __name__ == '__main__':
    print("STARTING")
    main()
    print("FINISHED")
