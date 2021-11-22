from utils.Simulate_swarm import simulate_swarm
from utils.Controllers import NNController, ActiveElastic

def main():
    controller = ActiveElastic(11, 2)
    fitness = simulate_swarm(60, controller, False)
    print(fitness)


if __name__ == '__main__':
    print("STARTING")
    main()
    print("FINISHED")
