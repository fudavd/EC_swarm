#!/usr/bin/env python3
import copy
import os
import sys

import scipy
from matplotlib import pyplot as plt


print('Python %s on %s' % (sys.version, sys.platform))
from distutils.dir_util import copy_tree
import numpy as np
from utils.Simulate_swarm_population import simulate_swarm_with_restart_population_split
from utils.Simulate_swarm_population import EnvSettings
from utils.Fitnesses import Calculate_fitness_size
from utils.Individual import Individual, thymio_genotype


def search_file_list(rootname, file_name):
    file_list = []
    for root, dirs, files in os.walk(rootname):
        for file in files:
            if file_name in file:
                file_list.append(os.path.join(root, file))
    return file_list

def video():
    n_input = 9
    n_output = 2

    simulation_time = 600
    swarm_size = 20
    arena = 30
    simulator_settings = EnvSettings
    simulator_settings['arena_type'] = f"circle_{arena}x{arena}"
    simulator_settings['objectives'] = ['gradient']
    simulator_settings['record_video'] = True
    experiment_names = ["Hebbian"]#, "Baseline"]

    for experiment_name in experiment_names:
        genotype = thymio_genotype((experiment_name == 'Hebbian') * 'h' + "NN", n_input, n_output)
        genotype['controller']["params"]['torch'] = False
        genotype['morphology']['rgb'] = [1, 0.5, 0]
        swarm = Individual(genotype, 0)
        results_dir = os.path.join("./results", experiment_name)
        filenames_fit = search_file_list(results_dir, 'fitnesses.npy')
        best_fitness = -np.inf
        best_genome = None
        best_folder = None
        for filename in filenames_fit:
            fitnesses = np.load(filename)
            if fitnesses.max() > best_fitness:
                best_fitness = fitnesses.max()
                best_folder = filename.replace('fitnesses.npy','')
                best_genome = np.load(best_folder + 'x_best.npy')[-1]
        if best_genome is None:
            print("No best genome found")
            break
        print(best_folder)
        swarm.controller.load_geno(best_folder)
        swarm.geno2pheno(best_genome)
        swarm.controller.log_weights = True
        swarm_members = []

        for _ in range(swarm_size):
            swarm_members += [copy.deepcopy(swarm)]
        simulator_settings['fitness_size'] = Calculate_fitness_size(swarm_members, simulator_settings)
        simulate_swarm_with_restart_population_split(simulation_time, [swarm_members],
                                                     headless=False,
                                                     env_params=simulator_settings,
                                                     splits=1)
        for end_dir in ['plot', 'viewer']:
            source = os.path.join("./results", 'images', end_dir)
            destination = os.path.join("./results", 'images', experiment_name, end_dir)
            copy_tree(source, destination)

if __name__ == '__main__':
    print("STARTING retesting best controller experiments")
    ## Video
    video()

    print("FINISHED")
