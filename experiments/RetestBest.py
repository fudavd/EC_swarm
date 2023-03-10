#!/usr/bin/env python3
import copy
import os
import sys
print('Python %s on %s' % (sys.version, sys.platform))
from pathlib import Path
print("Experiment root: ", Path(os.path.abspath(__file__)).parents[1].__str__())
sys.path.append(Path(os.path.abspath(__file__)).parents[1].__str__())
import numpy as np
from utils.Simulate_swarm_population import simulate_swarm_with_restart_population_split
from utils.Individual import Individual, thymio_genotype


def search_file_list(rootname, file_name):
    file_list = []
    for root, dirs, files in os.walk(rootname):
        for file in files:
            if file_name in file:
                file_list.append(os.path.join(root, file))
    return file_list


def Ratios():
    n_input = 9
    n_output = 2
    arena = 30
    pop_size = 30
    genotype = thymio_genotype("NN", 9, 2)
    run = 4
    experiment_folder = f"./results/{arena}x{arena}_pop{pop_size}_final/{run}"
    reservoir_list = search_file_list(experiment_folder, "reservoir.npy")
    n_subs = len(reservoir_list)
    genotype['controller']["params"]['torch'] = False

    individuals = []
    swarm_size = 20
    simulation_time = 600
    repetitions = 30

    ratios = [0.0, 0.25, 0.5, 0.75, 1.0]
    for ratio in ratios:
        for n_sub in range(n_subs):
            genotype['controller']["encoding"] = np.ones(n_output * n_input)
            genotype['morphology']['rgb'] = [2 * n_sub / n_subs, 1 - 2 * n_sub / n_subs, 0]
            # genotype['morphology']['rgb'] = [2 * n_sub / n_subs, 2 * n_sub / n_subs, 2 * n_sub / n_subs]
            sub_swarm = Individual(genotype, 0)
            learner_res_dir = reservoir_list[n_sub]
            sub_swarm.controller.load_geno(learner_res_dir[:-14])
            x = np.load(f"{experiment_folder}/x_best.npy")
            sub_swarm.geno2pheno(x[-1][n_sub * n_input * n_output:(1 + n_sub) * n_input * n_output])
            individuals += [sub_swarm] * int(swarm_size * np.abs(n_sub-ratio))

        print(f"STARTING retest best for subswarm ratio = {ratio} experiment: arena circle")
        fitnesses = simulate_swarm_with_restart_population_split(simulation_time, [individuals]*repetitions, swarm_size,
                                                           True,
                                                           [0, 0, 0, 0, 1],
                                                           15,)
        print(f"\t Mean fitness: {round(fitnesses.mean(), 3)} \t +- {round(fitnesses.std(), 3)}")
        np.save(f"./Validation/ratios/r_1:{ratio}.npy", fitnesses)

#
# def Scalability():
#     n_input = 9
#     n_output = 2
#     genotype = thymio_genotype("NN", n_input, n_output)
#     genotype['controller']["params"]['torch'] = False
#
#     simulation_time = 600
#     n_retest = 30
#
#     conditions = [10, 30, 45]
#     swarm_size = [5, 14, 50]
#     best_runs = [18, 28, 2]
#     for exp_nr, condition in enumerate(conditions):
#         scalability_res_dir = os.path.join("DATA/Scalability", f"evo{condition}/")
#         if not os.path.exists(scalability_res_dir):
#             os.mkdir(scalability_res_dir)
#         best_genome_dir = os.path.join("DATA/DE-results", f"{condition}x{condition}", f"{best_runs[exp_nr]}/")
#
#         genotype['controller']["encoding"] = np.ones(n_output * n_input)
#         swarm = Individual(genotype, 0)
#         swarm.controller.load_geno(best_genome_dir)
#         swarm.geno2pheno(np.load(best_genome_dir + "x_best.npy", allow_pickle=True)[-1, :])
#         for retest in swarm_size:
#             print(f"STARTING retest best for {condition}x{condition} experiment: {retest} agents")
#             fitnesses = simulate_swarm_with_restart_population(simulation_time, [swarm]*n_retest, 14,
#                                                                headless=True,
#                                                                objectives=[0, 0, 0, 0, 1],
#                                                                arena=f"circle_{condition}x{condition}")
#             print(f"\t Mean fitness: {round(fitnesses.mean(), 3)} \t +- {round(fitnesses.std(), 3)}")
#             np.save(f"{scalability_res_dir}/{swarm_size}_agents.npy", fitnesses)


if __name__ == '__main__':
    print("STARTING retesting best controller experiments")
    Ratios()
    # Scalability()
    print("FINISHED")
