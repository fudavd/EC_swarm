#!/usr/bin/env python3
import copy
import os
import sys
from matplotlib import pyplot as plt

print('Python %s on %s' % (sys.version, sys.platform))
from pathlib import Path
print("Experiment root: ", Path(os.path.abspath(__file__)).parents[1].__str__())
sys.path.append(Path(os.path.abspath(__file__)).parents[1].__str__())
import numpy as np
from utils.Simulate_swarm_population import simulate_swarm_with_restart_population_split
from utils import Simulate_swarm_population
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
    experiment_folder = f"./results/{arena}x{arena}_pop{pop_size}/{run}"
    reservoir_list = search_file_list(experiment_folder, "reservoir.npy")
    n_subs = len(reservoir_list)
    genotype['controller']["params"]['torch'] = False

    swarm_size = 20
    simulation_time = 600
    repetitions = 1

    # ratios = [0.0, 0.25, 0.5, 0.75, 1.0] # sub-group ratios
    ratios = [0.0, 1.0]

    for r_length in [0.0, 0.25, 0.5]: # distance away from source
        Simulate_swarm_population.radius_spawn = r_length
        print(r_length)
        print(Simulate_swarm_population.radius_spawn)

        for ratio in ratios:
            individuals = []
            for n_sub in range(n_subs):
                genotype['controller']["encoding"] = np.ones(n_output * n_input)
                genotype['morphology']['rgb'] = [2 * n_sub / n_subs, 1 - 2 * n_sub / n_subs, 0]
                # genotype['morphology']['rgb'] = [2 * n_sub / n_subs, 2 * n_sub / n_subs, 2 * n_sub / n_subs]
                sub_swarm = Individual(genotype, 0)
                learner_res_dir = reservoir_list[n_sub]
                sub_swarm.controller.load_geno(learner_res_dir[:-14])
                x = np.load(f"{experiment_folder}/x_best.npy")
                sub_swarm.geno2pheno(x[-1][n_sub * n_input * n_output:(1 + n_sub) * n_input * n_output])
                subswarmsize = int(swarm_size * np.abs(n_sub - ratio))
                print(f'subswarm {n_sub} with size {subswarmsize}')
                individuals += [sub_swarm] * subswarmsize

            print(f"STARTING retest best for subswarm ratio = {ratio} experiment: arena circle")
            print(f'swarm_size={swarm_size}')
            print(f'individuals={individuals}')
            print(f'reservoir_list={reservoir_list}')
            fitnesses = simulate_swarm_with_restart_population_split(simulation_time, [individuals] * repetitions,
                                                                     swarm_size,
                                                                     False,
                                                                     [0, 0, 0, 1, 0, 0],
                                                                     1, )
            print(f"\t Mean fitness: {round(fitnesses.mean(), 3)} \t +- {round(fitnesses.std(), 3)}")
            plt.save_fig("/tmp/fig.png")
            # prerun = np.load(f"./Validation/ratios_r{r_length}/r_1:{ratio}.npy")
            # fitnesses_stack = np.hstack((prerun,fitnesses))
            # np.save(f"./results/Validation/ratios/r_{r_length}:{ratio}2.npy", fitnesses)
            # np.save(f"./results/Validation/ratios/r_{r_length}:{ratio}_stack.npy", fitnesses_stack)


def Arenas():
    n_input = 9
    n_output = 2
    arena_size = 30
    pop_size = 30
    genotype = thymio_genotype("NN", 9, 2)
    run = 4
    experiment_folder = f"./results/{arena_size}x{arena_size}_pop{pop_size}/{run}"
    reservoir_list = search_file_list(experiment_folder, "reservoir.npy")
    n_subs = len(reservoir_list)
    genotype['controller']["params"]['torch'] = False

    swarm_size = 20
    simulation_time = 600
    repetitions = 1

    arenas = ['circle_corner']
    for arena in arenas:
        individuals = []
        for n_sub in range(n_subs):
            genotype['controller']["encoding"] = np.ones(n_output * n_input)
            genotype['morphology']['rgb'] = [2 * n_sub / n_subs, 1 - 2 * n_sub / n_subs, 0]
            # genotype['morphology']['rgb'] = [2 * n_sub / n_subs, 2 * n_sub / n_subs, 2 * n_sub / n_subs]
            sub_swarm = Individual(genotype, 0)
            learner_res_dir = reservoir_list[n_sub]
            sub_swarm.controller.load_geno(learner_res_dir[:-14])
            x = np.load(f"{experiment_folder}/x_best.npy")
            sub_swarm.geno2pheno(x[-1][n_sub * n_input * n_output:(1 + n_sub) * n_input * n_output])
            subswarmsize = int(swarm_size * 0.5)
            print(f'subswarm {n_sub} with size {subswarmsize}')
            individuals += [sub_swarm] * subswarmsize

        print(f"STARTING retest best for arena = {arena}")
        fitnesses = simulate_swarm_with_restart_population_split(simulation_time, [individuals]*repetitions, swarm_size,
                                                           False,
                                                           [0, 0, 0, 1, 0, 0],
                                                           1,
                                                           f'{arena}_{arena_size}x{arena_size}')
        print(f"\t Mean fitness: {round(fitnesses.mean(), 3)} \t +- {round(fitnesses.std(), 3)}")
        np.save(f"./results/Validation/arenas/{arena}.npy", fitnesses)


def Alignment():
    n_input = 9
    n_output = 2
    arena = 30
    pop_size = 30
    genotype = thymio_genotype("NN", 9, 2)
    run = 4
    experiment_folder = f"./results/{arena}x{arena}_pop{pop_size}/{run}"
    reservoir_list = search_file_list(experiment_folder, "reservoir.npy")
    n_subs = len(reservoir_list)
    genotype['controller']["params"]['torch'] = False

    individuals = []
    swarm_size = 20
    Simulate_swarm_population.record_video = True
    for n_sub in range(n_subs):
        genotype['controller']["encoding"] = np.ones(n_output * n_input)
        genotype['morphology']['rgb'] = [2 * n_sub / n_subs, 1 - 2 * n_sub / n_subs, 0]
        # genotype['morphology']['rgb'] = [2 * n_sub / n_subs, 2 * n_sub / n_subs, 2 * n_sub / n_subs]
        sub_swarm = Individual(genotype, 0)
        learner_res_dir = reservoir_list[n_sub]
        sub_swarm.controller.load_geno(learner_res_dir[:-14])
        x = np.load(f"{experiment_folder}/x_best.npy")
        sub_swarm.geno2pheno(x[-1][n_sub * n_input * n_output:(1 + n_sub) * n_input * n_output])
        individuals += [sub_swarm] * int(swarm_size/n_subs)
    simulation_time = 600
    _ = simulate_swarm_with_restart_population_split(simulation_time,
                                                     [individuals],
                                                     swarm_size,
                                                     False,
                                                     [1, 1, 1, 1, 1, 1],
                                                     1)


def Plot_Alignment():
    alignment = np.load("./results/Validation/alignment.npy")
    time_stamps = [0, 45, 130, 300, 360, 600]
    time = np.arange(0, len(alignment))*0.1
    # plt.rcParams['text.usetex'] = True
    fig, ax = plt.subplots(2)
    ax[0].plot(time, alignment[:, 3].squeeze(), 'k', label='Swarm', zorder=4)
    ax[0].plot(time, alignment[:, 4].squeeze(), label='reservoir 1', color='g')
    ax[0].plot(time, alignment[:, 5].squeeze(), label='reservoir 2', color='r')
    ax[0].vlines(time_stamps, 0, 0.6, colors=['k']*len(time_stamps), linestyles='dotted')#, [':k']*len(time_stamps) )
    ax[0].set_title('Retest best run: Fitness')
    ax[0].set_ylabel('Performance')
    ax[0].set_ylim(0.1, 0.4)
    ax[0].legend(loc='lower right')
    ax[1].set_title('Retest best run: Alignment')
    ax[1].plot(time, alignment[:, 2].squeeze(), '-k', label='Swarm', zorder=4)
    ax[1].plot(time, alignment[:, 0].squeeze(), label='reservoir 1', color='g')
    ax[1].plot(time, alignment[:, 1].squeeze(), label='reservoir 2', color='r')
    ax[1].vlines(time_stamps, 0, 0.6, colors=['k']*len(time_stamps), linestyles='dotted')#, [':k']*len(time_stamps))
    ax[1].set_xlabel('Time (minutes)')
    ax[1].set_ylabel('Alignment:' + r' $\Phi$')
    ax[1].set_ylim(0, 0.6)
    ax[1].legend(loc='upper right')
    fig.tight_layout()
    plt.show()
    fig.savefig('./results/Validation/align/Retest_best.pdf')
    print(alignment[-1, 4])
    print("FINISHED")

if __name__ == '__main__':
    print("STARTING retesting best controller experiments")
    Ratios()
    Arenas()
    Alignment()
    print("FINISHED")
