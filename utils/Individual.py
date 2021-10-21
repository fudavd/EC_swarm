import numpy as np
import torch
from typing import AnyStr, Dict
from . import Controllers
import numpy as py


class Individual:
    def __init__(self, robot_file, genotype, id: int):
        self.urdf = robot_file
        self.genotype = genotype
        self.phenotype = {"body": genotype['morphology'],
                          "brain": self.brain(genotype['controller']),
                          "id": id,
                          "fitness": -np.inf}
        self.fitness = []
        self.id = id
        self.controller = self.brain(genotype['controller'])

    def geno2pheno(self, genotype):
        self.genotype = genotype
        self.body = genotype['morphology']['string']
        self.brain = self.brain(genotype['controller'])

        self.phenotype["body"] = genotype['morphology']
        self.phenotype["brain"] = self.brain(genotype['controller'])

    def get_phenotype(self):
        return self.phenotype

    def brain(self, controller: Dict):
        controller_type = controller['type']
        params = controller['params']
        if controller_type == 'NN':
            controller = Controllers.NNController(params['input_space'], params['output_space'])
        elif controller_type == 'Rand':
            controller = Controllers.RandomWalk(params['output_space'])
        else:
            print("ERROR: controller type not found")
            controller = None
        return controller

    def set_fitness(self, fitness: float):
        self.fitness = fitness
        self.phenotype["fitness"] = fitness
