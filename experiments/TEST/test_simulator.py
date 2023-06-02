import unittest
import sys
import os

import pandas as pd
import ribs
from ribs.archives import ArchiveDataFrame

import numpy as np

print('Python %s on %s' % (sys.version, sys.platform))
from pathlib import Path

sys.path.append(Path(os.path.abspath(__file__)).parents[2].__str__())

from utils.Simulate_swarm_population import simulate_swarm_population, EnvSettings
from utils.Individual import Individual, thymio_genotype


class TestSim(unittest.TestCase):
    def test_sim(self, controller_type="NN", headless=False):
        best_run = 0
        learner_res_dir = f'./results/MAP/{best_run}'
        env_params = EnvSettings
        env_params['objectives'] = ['alignment_sub', 'alignment', 'gradient_sub', 'gradient']
        n_input = 9
        n_output = 2
        n_subs = 2
        individuals = []
        swarm_size = 20
        df = ArchiveDataFrame(pd.read_csv(f"{learner_res_dir}/MAP_archive.csv"))
        best_objective = 0
        for elite in df.iterelites():
            if best_objective < elite.objective:
                best_objective = elite.objective
                best_elite = elite.solution
                best_measures = elite.measures
                best_index = elite.index
                print(f'Elite {best_index}: {best_objective.round(6)}  \t| {best_measures}')
        for n_sub in range(n_subs):
            genotype = thymio_genotype(controller_type, n_input, n_output)
            genotype['controller']["params"]['torch'] = False
            genotype['controller']["encoding"] = np.ones(n_output * n_input)
            genotype['morphology']['rgb'] = [2 * n_sub / n_subs, 2 * n_sub / n_subs, 2 * n_sub / n_subs]
            sub_swarm = Individual(genotype, 0)
            sub_swarm_dir = f"{learner_res_dir}/subgroup_{n_sub}"
            sub_swarm.controller.load_geno(sub_swarm_dir)
            x = best_elite
            sub_swarm.geno2pheno(x[n_sub*n_input*n_output:(1 + n_sub)*n_input*n_output])

            individuals += [sub_swarm]*int(swarm_size/n_subs)

        # for i in range(swarm_size):
        #     genotype['morphology']['rgb'] = [float(i < 10), float(i < 10), float(i < 10)]
        #     individual = Individual.Individual(genotype, 0)
        #     individuals.append(individual)
        try:
            fitness = simulate_swarm_population(600, [individuals], headless, env_params)
            print(fitness)
        except:
            raise Exception("Could not calculate fitness")
        self.assertEqual((fitness > 0).all(), True)


if __name__ == '__main__':
    print("STARTING")
    unittest.main()
    print("FINISHED")
