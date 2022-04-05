#!/usr/bin/env python3
import copy
import os
import sys
print('Python %s on %s' % (sys.version, sys.platform))
from pathlib import Path
print("Experiment root: ", Path(os.path.abspath(__file__)).parents[1].__str__())
sys.path.append(Path(os.path.abspath(__file__)).parents[1].__str__())
import numpy as np
from utils.Simulate_swarm_population import simulate_swarm_with_restart_population
from utils.Individual import Individual, thymio_genotype


def Robustness():
    n_input = 9
    n_output = 2
    genotype = thymio_genotype("NN", n_input, n_output)
    genotype['controller']["params"]['torch'] = False

    simulation_time = 600
    n_retest = 30

    conditions = [10, 30, 45]
    best_runs = [18, 29, 2]
    for exp_nr, condition in enumerate(conditions):
        robustness_res_dir = os.path.join("DATA/Robustness", f"evo{condition}/")
        if not os.path.exists(robustness_res_dir):
            os.mkdir(robustness_res_dir)
        best_genome_dir = os.path.join("DATA/DE-results", f"{condition}x{condition}", f"{best_runs[exp_nr]}/")

        genotype['controller']["encoding"] = np.ones(n_output * n_input)
        swarm = Individual(genotype, 0)
        swarm.controller.load_geno(best_genome_dir)
        swarm.geno2pheno(np.load(best_genome_dir + "x_best.npy", allow_pickle=True)[-1, :])
        for arena_size in conditions:
            print(f"STARTING retest best for {condition}x{condition} experiment: arena circle {arena_size}x{arena_size}")
            fitnesses = simulate_swarm_with_restart_population(simulation_time, [swarm]*n_retest, 14,
                                                               True,
                                                               [0, 0, 0, 0, 1],
                                                               f"circle_{arena_size}x{arena_size}")
            print(f"\t Mean fitness: {round(fitnesses.mean(), 3)} \t +- {round(fitnesses.std(), 3)}")
            np.save(f"{robustness_res_dir}/{arena_size}x{arena_size}.npy", fitnesses)


def Scalability():
    n_input = 9
    n_output = 2
    genotype = thymio_genotype("NN", n_input, n_output)
    genotype['controller']["params"]['torch'] = False

    simulation_time = 600
    n_retest = 30

    conditions = [10, 30, 45]
    swarm_size = [5, 14, 50]
    best_runs = [18, 28, 2]
    for exp_nr, condition in enumerate(conditions):
        scalability_res_dir = os.path.join("DATA/Scalability", f"evo{condition}/")
        if not os.path.exists(scalability_res_dir):
            os.mkdir(scalability_res_dir)
        best_genome_dir = os.path.join("DATA/DE-results", f"{condition}x{condition}", f"{best_runs[exp_nr]}/")

        genotype['controller']["encoding"] = np.ones(n_output * n_input)
        swarm = Individual(genotype, 0)
        swarm.controller.load_geno(best_genome_dir)
        swarm.geno2pheno(np.load(best_genome_dir + "x_best.npy", allow_pickle=True)[-1, :])
        for retest in swarm_size:
            print(f"STARTING retest best for {condition}x{condition} experiment: {retest} agents")
            fitnesses = simulate_swarm_with_restart_population(simulation_time, [swarm]*n_retest, 14,
                                                               headless=True,
                                                               objectives=[0, 0, 0, 0, 1],
                                                               arena=f"circle_{condition}x{condition}")
            print(f"\t Mean fitness: {round(fitnesses.mean(), 3)} \t +- {round(fitnesses.std(), 3)}")
            np.save(f"{scalability_res_dir}/{swarm_size}_agents.npy", fitnesses)


if __name__ == '__main__':
    print("STARTING retesting best controller experiments")
    Robustness()
    Scalability()
    print("FINISHED")
