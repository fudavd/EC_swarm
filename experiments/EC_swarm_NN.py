#!/usr/bin/env python3
import os
import sys

from utils.Simulate_swarm_population import simulate_swarm_with_restart_population

print('Python %s on %s' % (sys.version, sys.platform))
from pathlib import Path

print("Experiment root: ", Path(os.path.abspath(__file__)).parents[1].__str__())
sys.path.append(Path(os.path.abspath(__file__)).parents[1].__str__())

import numpy as np
# from utils.Simulate_swarm import simulate_swarm_with_restart
from utils.EA import DE
from utils.Individual import Individual, thymio_genotype


def main():
    n_input = 9
    n_output = 2
    genotype = thymio_genotype("NN", n_input, n_output)
    genotype['controller']["params"]['torch'] = False

    experiment_name = "TEST"
    simulation_time = 300
    # setting number of:
    n_runs = 10  # runs/repetitions
    n_generations = 100  # generations
    reps = 1

    params = {}
    params['bounds'] = (-10, 10)
    params['D'] = (n_input + n_output) * n_input
    params['pop_size'] = 25
    params['CR'] = 0.9
    params['F'] = 0.5

    experiment = []
    for run in range(n_runs):
        genomes = []
        fitnesses = []
        learner_res_dir = os.path.join("./results", experiment_name, str(run))
        if not os.path.exists(learner_res_dir):
            os.makedirs(learner_res_dir)

        learner = DE(params, output_dir=learner_res_dir)
        for gen in range(n_generations):  # loop over generations
            fitnesses_gen = []
            population = []
            for (individual, x) in enumerate(learner.x_new):  # loop over individuals
                genotype['controller']["encoding"] = x
                swarm = Individual(genotype, individual + params['pop_size'] * gen)
                # fitness = simulate_swarm_with_restart(simulation_time, swarm, True, [0, 0, 0, 0, 1])
                # fitness_retest = simulate_swarm_with_restart(simulation_time, swarm, True, [0, 0, 0, 0, 1])
                # fitness = min(fitness, fitness_retest)
                # swarm.set_fitness(fitness)
                # fitnesses_gen.append(fitness)
                population.append(swarm)

            fitnesses_gen = simulate_swarm_with_restart_population(simulation_time, population, True, [0, 0, 0, 0, 1])
            for _ in range(reps):
                fitnesses_gen_rep = simulate_swarm_with_restart_population(simulation_time, population, True, [0, 0, 0, 0, 1])
                fitnesses_gen = np.min((fitnesses_gen, fitnesses_gen_rep), axis=0)
            # %% Some bookkeeping
            genomes.append(learner.x_new.tolist())
            print("\n\n", genomes.__len__(), "\n\n")
            fitnesses.append(fitnesses_gen)

            learner.f_new = -np.array(fitnesses_gen)
            learner.x_new = learner.get_new_genome()
            learner.save_checkpoint()
            print(f"Experiment {experiment_name}: {run}/{n_runs}\n"
                  f"Finished gen: {gen}/{n_generations}\n"
                  f"\tBest gen: {learner.x_best_so_far[-1]}\n"
                  f"\tBest fit: {-learner.f_best_so_far[-1]}\n"
                  f"\tMean fit: {np.mean(-learner.f)}\n")

        learner.save_results()

        np.save(f"{learner_res_dir}/genomes.npy", genomes)
        np.save(f"{learner_res_dir}/fitnesses.npy", fitnesses)


if __name__ == '__main__':
    print("STARTING evolutionary experiment")
    main()
    print("FINISHED")
