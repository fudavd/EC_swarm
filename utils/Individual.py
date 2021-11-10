import numpy as np
import torch
from typing import AnyStr, Dict
from . import Controllers
import numpy as py


def default_genotype():
    brain = {'type': 'Rand',
             'params': {'output_space': 2},
             'encoding': []}
    body = {'model_file': 'models/thymio/model.urdf'}
    genotype = {'morphology': body,
                'controller': brain}
    return genotype



class Individual:
    def __init__(self, robot_file, genotype, id: int):
        self.urdf = robot_file
        self.genotype = genotype
        self.body = genotype['morphology']
        self.controller = self.set_brain(genotype['controller'])

        self.phenotype = {"body": self.body,
                          "brain": self.controller,
                          "id": id,
                          "fitness": -np.inf}
        self.fitness = []
        self.id = id

    def geno2pheno(self, genotype):
        self.genotype = genotype
        self.body = genotype['morphology']
        self.controller = self.set_brain(genotype['controller'])
        self.controller.geno2pheno(genotype['controller']["encoding"])

        self.phenotype["body"] = self.body
        self.phenotype["brain"] = self.controller

    def get_phenotype(self):
        return self.phenotype

    def set_brain(self, controller: Dict):
        controller_type = controller['type']
        params = controller['params']
        if controller_type == 'NN':
            controller = Controllers.NNController(params['input_space'], params['output_space'])
            controller.geno2pheno(controller["encoding"])
        elif controller_type == 'Rand':
            controller = Controllers.RandomWalk(params['output_space'])
        else:
            print("ERROR: controller type not found")
            controller = None
        return controller

    def set_fitness(self, fitness: float):
        self.fitness = fitness
        self.phenotype["fitness"] = fitness
