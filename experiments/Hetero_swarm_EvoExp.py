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

# from utils.Simulate_swarm import simulate_swarm_with_restart
from utils.EA import DE
from utils.Individual import Individual, thymio_genotype


def main():
    n_input = 9
    n_output = 2
    genotype = thymio_genotype("NN", n_input, n_output)
    genotype['controller']["params"]['torch'] = False

    simulation_time = 60
    # setting number of:
    n_subs = 2  # number of subgroups
    n_runs = 30  # runs
    n_generations = 100  # generations
    pop_size = 25  # number of individuals
    swarm_size = 20
    reps = 2  # repetitions per individual
    arenas = [30]

    params = {}
    params['bounds'] = (-10, 10)
    params['D'] = n_output * n_input
    params['pop_size'] = pop_size
    params['CR'] = 0.9
    params['F'] = 0.5

    experiment = []
    for arena in arenas:
        for run in range(n_runs):
            experiment_name = f"{arena}x{arena}"
            arena_type = f"circle_{arena}x{arena}"
            genomes = []
            fitnesses = []
            learners = []
            swarms = []
            # prepare learners (/load from checkpoint)
            experiment_dir = os.path.join("./results", experiment_name, str(run))
            for n_sub in range(n_subs):
                learner_res_dir = os.path.join(experiment_dir, f"EA_{n_sub}")
                if not os.path.exists(learner_res_dir):
                    os.makedirs(learner_res_dir)

                learner = DE(params, output_dir=learner_res_dir)
                genotype['controller']["encoding"] = np.ones(n_output * n_input)
                genotype['morphology']['rgb'] = [2*n_sub/n_subs, 2*n_sub/n_subs, 2*n_sub/n_subs]
                sub_swarm = Individual(genotype, 0)
                gen_start = 0
                if not os.path.exists(f"{learner_res_dir}/reservoir.npy"):
                    sub_swarm.controller.save_geno(learner_res_dir)
                else:
                    if os.path.exists(f"{learner_res_dir}/genomes.npy"):
                        try:
                            learner.load_checkpoint()
                            genomes = np.load(f"{learner_res_dir}/genomes.npy", allow_pickle=True).tolist()
                            fitnesses = np.load(f"{learner_res_dir}/fitnesses.npy", allow_pickle=True).tolist()
                            gen_start = len(np.load(f"{learner_res_dir}/x_best.npy", allow_pickle=True))
                            sub_swarm.controller.load_geno(learner_res_dir)
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
                    else:
                        sub_swarm.controller.load_geno(learner_res_dir)
                        print("Could not find corresponding genomes restart experiment from gen 0!", file=sys.stderr)
                learners.append(learner)
                swarms.append(sub_swarm)

            # Run evolution
            for gen in range(gen_start, n_generations):  # loop over generations
                population = [[] for _ in range(pop_size)]
                for (n_sub, learner) in enumerate(learners):
                    sub_swarm = copy.deepcopy(swarms[n_sub])
                    for (individual, x) in enumerate(learner.x_new):  # loop over individuals
                        sub_swarm.geno2pheno(x)
                        population[individual] += [sub_swarm]*int(swarm_size/n_subs)

                fitnesses_gen = [np.inf]*pop_size
                for _ in range(reps):
                    fitnesses_gen_rep = simulate_swarm_with_restart_population(simulation_time, population, swarm_size,
                                                                               headless=True,
                                                                               objectives=[0, 0, 0, 0, 1],
                                                                               arena=arena_type)
                    fitnesses_gen = np.min((fitnesses_gen, fitnesses_gen_rep), axis=0)
                genomes.append(learner.x_new.tolist())
                fitnesses.append(fitnesses_gen)

                # %% Some bookkeeping
                for learner in learners:
                    learner.f_new = -np.array(fitnesses_gen)
                    learner.x_new = learner.get_new_genome()
                    learner.save_checkpoint()
                    np.save(f"{learner.directory_name}/genomes.npy", genomes)
                    np.save(f"{learner.directory_name}/fitnesses.npy", fitnesses)
                    print(f"Experiment {experiment_name}: {run}/{n_runs}\n"
                          f"Finished gen: {fitnesses.__len__()}/{n_generations}\n"
                          f"\tBest gen: {learner.x_best_so_far[-1]}\n"
                          f"\tBest fit: {-learner.f_best_so_far[-1]}\n"
                          f"\tMean fit: {np.mean(-learner.f)}\n")
            for learner in learners:
                learner.save_results()

                np.save(f"{learner.directory_name}/genomes.npy", genomes)
                np.save(f"{learner.directory_name}/fitnesses.npy", fitnesses)


if __name__ == '__main__':
    print("STARTING evolutionary experiment")
    main()
    print("FINISHED")
