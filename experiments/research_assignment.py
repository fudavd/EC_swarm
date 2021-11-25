from utils.Simulate_swarm import simulate_swarm
import os

import pickle
import numpy as np

# get fitness of individual, but apply learner algorithm revdeknn first
from utils.EA import DE
from utils.Individual import Individual, thymio_genotype


def main():
    genotype = thymio_genotype("4dir", 7, 2)

    experiment_name = "ActiviveElastic4dir"
    simulation_time = 10
    # setting number of:
    n_runs = 1  # runs/repetitions
    n_generations = 10  # generations

    params = {}
    params['bounds'] = (0, 4)
    params['D'] = 4
    params['pop_size'] = 10
    params['CR'] = 0.7
    params['F'] = 0.3

    experiment = []
    for run in range(n_runs):
        learner_res_dir = os.path.join("./results", experiment_name, str(run))
        if not os.path.exists(learner_res_dir):
            os.makedirs(learner_res_dir)

        learner = DE(params, output_dir=learner_res_dir)
        swarms = []
        for gen in range(n_generations):  # loop over generations
            fitnesses = []
            population = []
            for (individual, x) in enumerate(learner.x_new):  # loop over individuals
                genotype['controller']["encoding"] = []
                swarm = Individual(genotype, individual + params['pop_size'] * gen)
                fitness = simulate_swarm(simulation_time, swarm, True)
                swarm.set_fitness(fitness)
                population.append(swarm)
                fitnesses.append(fitness)

            learner.f = np.array(fitnesses)
            learner.x = learner.x_new
            _ = learner.get_new_genome()
            swarms.append(population)
            print(f"Experiment {experiment_name}: {run}/{n_runs}\n"
                  f"Finished gen: {gen}/{n_generations}\n"
                  f"\tBest gen: {learner.x_best_so_far[-1]}"
                  f"\tBest fit: {learner.f_best_so_far[-1]}")
        learner.save_results()
        experiment.append(swarms)
    file = open(f"{learner_res_dir}/DATA.pkl")
    pickle.dump(experiment, file)


if __name__ == '__main__':
    print("STARTING evolutionary experiment")
    main()
    print("FINISHED")
