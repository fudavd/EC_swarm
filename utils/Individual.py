import numpy as np
import torch
from typing import AnyStr, Dict
from . import Controllers
import numpy as py


def thymio_genotype(controller_type: AnyStr = "Rand", n_input: int = 5, n_output: int = 2) -> Dict:
    """
    Default thymio genotype with empty encoding

    Inputs:
        <AnyStr> controller_type : Type of Controller <Default: Rand>
        <int> n_input : Number of inputs of Controller <Default: 5>
        <int> n_output : Number of outputs of Controller <Default: 2>

    Outputs:
        <Dict> genotype : A genotype
    """
    brain = {'type': controller_type,
             'params': {'output_space': n_output,
                        'input_space': n_input},
             'encoding': []}
    body = {'robot_name': 'thymio',
            'model_file': 'models/thymio/model.urdf'}
    genotype = {'morphology': body,
                'controller': brain}
    return genotype


class Individual:
    def __init__(self, genotype, id: int):
        self.genotype = genotype
        self.body = self.set_body(genotype['morphology'])
        self.controller = self.set_brain(genotype['controller'])

        self.phenotype = {"body": self.body,
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

    def set_body(self, body_description: Dict):
        return body_description["model_file"]

    def set_brain(self, brain_description: Dict):
        controller_type = brain_description['type']
        params = brain_description['params']
        if not np.any(brain_description["encoding"]):
            print("Invalid encoding: set brain to Default (RandomWalk)")
            controller_type = "Rand"
            params['input_space'] = 5
            params['output_space'] = 2

        if controller_type == 'NN':
            if "torch" not in params:
                controller = Controllers.NNController(params['input_space'], params['output_space'])
            else:
                controller = Controllers.NNController(params['input_space'], params['output_space'], params['torch'])
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
        controller.geno2pheno(brain_description["encoding"])
        return controller

    def set_fitness(self, fitness: float):
        self.fitness = fitness
        self.phenotype["fitness"] = fitness
