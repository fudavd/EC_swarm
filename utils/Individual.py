import numpy as np
import torch
from typing import AnyStr, Dict
from . import Controllers
import numpy as py


def thymio_genotype(controller_type: AnyStr = "Rand", n_input: int = 5, n_output: int = 2) -> Dict:
    """
    Default thymio genotype with empty encoding

    Inputs:
        :param controller_type : Type of Controller <Default: Rand>
        :param n_input : Number of inputs of Controller <Default: 5>
        :param n_output : Number of outputs of Controller <Default: 2>

    Outputs:
        <Dict> genotype : A genotype
    """
    brain = {'type': controller_type,
             'params': {'output_space': n_output,
                        'input_space': n_input},
             'encoding': []}
    body = {'robot_name': 'thymio',
            'model_file': 'models/thymio/model.urdf',
            'rgb': None}
    genotype = {'morphology': body,
                'controller': brain}
    return genotype


class Individual:
    def __init__(self, genotype, id: int):
        self.genotype = genotype
        self.body = genotype['morphology']["model_file"]
        self.controller = self.set_brain(genotype['controller'])

        self.phenotype = {"color": genotype['morphology']['rgb'],
                          "brain": self.controller,
                          "id": id,
                          "fitness": -np.inf}
        self.fitness = []
        self.id = id

    def geno2pheno(self, genotype):
        self.genotype = genotype
        self.controller.geno2pheno(genotype)
        self.phenotype["brain"] = self.controller

    def get_phenotype(self):
        return self.phenotype

    def set_brain(self, brain_description: Dict):
        controller_type = brain_description['type']
        params = brain_description['params']
        if not np.isreal((params['input_space'], params['output_space'])).any() and not controller_type == 'Rand':
            print("Invalid input/output space: set brain to Default (RandomWalk)")
            controller_type = "Rand"
            params['input_space'] = 0
            params['output_space'] = 2

        if controller_type == 'NN':
            if "torch" not in params:
                controller = Controllers.NNController(params['input_space'], params['output_space'], False)
            else:
                controller = Controllers.NNController(params['input_space'], params['output_space'], params['torch'])
        elif controller_type == 'aNN':
            controller = Controllers.adaptiveNNController(params['input_space'], params['output_space'])
        elif controller_type == "GNN":
            controller = Controllers.GNNController(params['input_space'], params['output_space'])
        elif controller_type == 'Rand':
            controller = Controllers.RandomWalk(params['input_space'], params['output_space'])
        elif controller_type == '4dir':
            controller = Controllers.ActiveElastic_4dir(params['input_space'], params['output_space'])
        elif controller_type == 'omni':
            controller = Controllers.ActiveElastic_omni(params['input_space'], params['output_space'])
        elif controller_type == 'k_nearest':
            controller = Controllers.ActiveElastic_k_near(params['input_space'], params['output_space'])
        else:
            raise ValueError("ERROR: controller type not found")
        return controller

    def set_fitness(self, fitness: float):
        self.fitness = fitness
        self.phenotype["fitness"] = fitness
