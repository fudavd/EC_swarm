from utils.Simulate_swarm import simulate_swarm
from utils.Controllers import ActiveElastic_omni, ActiveElastic_k_near, ActiveElastic_4dir

def main(sensor, k=None):
    if sensor == "omni":
        controller = ActiveElastic_omni(4, 2)
    elif sensor == "k_nearest":
        controller = ActiveElastic_k_near(2*k+3, 2)
    elif sensor == "4dir":
        controller = ActiveElastic_4dir(7, 2)

    fitness = simulate_swarm(600, controller, False)
    print(fitness)


if __name__ == '__main__':
    print("STARTING")
    sensor = "4dir"
    k = 4
    main(sensor, k)
    print("FINISHED")
