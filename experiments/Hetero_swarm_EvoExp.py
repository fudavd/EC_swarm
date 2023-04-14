#!/usr/bin/env python3
import copy
import os
import sys
import time

print('Python %s on %s' % (sys.version, sys.platform))
from pathlib import Path

print("Experiment root: ", Path(os.path.abspath(__file__)).parents[1].__str__())
sys.path.append(Path(os.path.abspath(__file__)).parents[1].__str__())

import numpy as np
from utils.Simulate_swarm_population import simulate_swarm_with_restart_population_split
from utils.Simulate_swarm_population import EnvSettings
from utils.EA import CMAes
from utils.Individual import Individual, thymio_genotype
from utils.Fitnesses import Calculate_fitness_size


def main():
    n_input = 9
    n_output = 2
    genotype = thymio_genotype("NN", n_input, n_output)
    genotype['controller']["params"]['torch'] = False

    simulation_time = 600
    # setting number of:
    n_subs = 2  # number of subgroups
    n_runs = 10  # runs
    n_generations = 100  # generations
    pop_size = 30  # number of individuals
    swarm_size = 20
    reps = 3  # repetitions per individual
    arenas = [30]

    params = {}
    params['bounds'] = (-5, 5)
    params['D'] = n_output * n_input * n_subs
    params['pop_size'] = pop_size
    params['sigma0'] = 1

    run_start = 0
    for arena in arenas:
        experiment_name = f"{arena}x{arena}_pop{pop_size}"
        arena_type = f"circle_{arena}x{arena}"
        simulator_settings = EnvSettings
        simulator_settings['arena_type'] = arena_type
        simulator_settings['objectives'] = ['gradient']

        for run in range(run_start, run_start + n_runs):
            gen_start = 0
            genomes = []
            fitnesses = []
            swarms = []
            # prepare learners (/load from checkpoint)
            experiment_dir = os.path.join("./results", experiment_name, str(run))
            learner = CMAes(params, output_dir=experiment_dir)
            if os.path.exists(f"{experiment_dir}/genomes.npy"):
                try:
                    learner.load_checkpoint()
                    genomes = np.load(f"{experiment_dir}/genomes.npy", allow_pickle=True).tolist()
                    fitnesses = np.load(f"{experiment_dir}/fitnesses.npy", allow_pickle=True).tolist()
                    gen_start = len(np.load(f"{experiment_dir}/x_best.npy", allow_pickle=True))
                    if gen_start == 1 or (not genomes.__len__() == gen_start):
                        print("Generation buffer and genomes size does not match!", file=sys.stderr)
                        genomes = [genomes]
                        fitnesses = [fitnesses]

                    print(f"### Starting experiment from checkpoint ###\n"
                          f"Generation:\t{gen_start}/{n_generations}\n"
                          f"Best genome: {learner.x_best_so_far[-1]}\n"
                          f"\tBest fit: {-learner.f_best_so_far[-1]}\n"
                          f"\tMean fit: {np.mean(-learner.f)}\n"
                          )
                except Exception as e:
                    print(e)
                    sys.exit(e)

            for n_sub in range(n_subs):
                swarm_res_dir = os.path.join(experiment_dir, f"subgroup_{n_sub}")
                if not os.path.exists(swarm_res_dir):
                    os.makedirs(swarm_res_dir)

                genotype['controller']["encoding"] = np.ones(n_output * n_input)
                genotype['morphology']['rgb'] = [2 * n_sub / n_subs, 2 * n_sub / n_subs, 2 * n_sub / n_subs]
                sub_swarm = Individual(genotype, 0)

                if not os.path.exists(f"{swarm_res_dir}/reservoir.npy"):
                    print("Could not find corresponding genomes restart experiment from gen 0!", file=sys.stderr)
                    sub_swarm.controller.save_geno(swarm_res_dir)
                else:
                    sub_swarm.controller.load_geno(swarm_res_dir)
                swarms.append(sub_swarm)

            # Run evolution
            start_t = time.time()
            for gen in range(gen_start, n_generations):  # loop over generations
                population = [[] for _ in range(pop_size)]

                for (individual, x) in enumerate(learner.x_new):  # loop over individuals
                    for n_sub in range(n_subs):
                        sub_swarm = copy.deepcopy(swarms[n_sub])
                        sub_swarm.geno2pheno(x[n_sub * n_input * n_output:(1 + n_sub) * n_input * n_output])
                        population[individual] += [sub_swarm] * int(swarm_size / n_subs)

                simulator_settings['fitness_size'] = Calculate_fitness_size(population[0], simulator_settings)
                fitnesses_gen = np.zeros((pop_size, simulator_settings['fitness_size'], reps))
                for r in range(reps):
                    fitnesses_gen[:, :, r] = simulate_swarm_with_restart_population_split(simulation_time, population,
                                                                                       headless=True,
                                                                                       env_params=simulator_settings,
                                                                                       splits=2)
                fitnesses_gen = np.median(fitnesses_gen, axis=-1).squeeze()

                # %% Some bookkeeping
                avg_time = (time.time()-start_t)/(gen+1-gen_start)
                genomes.append(learner.x_new.tolist())
                fitnesses.append(fitnesses_gen)
                learner.f_new = -np.array(fitnesses_gen)
                learner.x_new = learner.get_new_genome()
                learner.save_checkpoint()
                np.save(f"{learner.directory_name}/genomes.npy", genomes)
                np.save(f"{learner.directory_name}/fitnesses.npy", fitnesses)
                print(f"Experiment {experiment_name}: {run}/{run_start + n_runs} | {learner.directory_name}\n"
                      f"Finished gen: {fitnesses.__len__()}/{n_generations} - {avg_time.__round__(2)}s\n"
                      f"\tBest gen: {learner.x_best_so_far[-1]}\n"
                      f"\tBest fit: {-learner.f_best_so_far[-1]}\n"
                      f"\tMean fit: {np.mean(-learner.f)}\n")

            learner.save_results()

            np.save(f"{learner.directory_name}/genomes.npy", genomes)
            np.save(f"{learner.directory_name}/fitnesses.npy", fitnesses)


if __name__ == '__main__':
    print("STARTING evolutionary experiment")
    main()
    print("FINISHED")
