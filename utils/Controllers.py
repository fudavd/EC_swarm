import time
from typing import List

import numpy
import numpy as np
import torch
from torch import nn

from .graph_network.GCN_layer import GCNLayerNumpy

torch.set_grad_enabled(False)
rng = numpy.random.default_rng()

class Controller(object):
    def __init__(self, n_states, n_actions, gcn_output_dim=0):
        self.n_input = n_states
        self.n_output = n_actions
        self.gcn_output_dim = gcn_output_dim
        self.controller_type = "default"
        self.umax_const = 0.1
        self.wmax = 1.5708 / 2.5

    @staticmethod
    def velocity_commands(state: np.ndarray) -> np.ndarray:
        return np.array([0])

    @staticmethod
    def geno2pheno(genotype: np.array):
        return

    @staticmethod
    def save_geno(path: str):
        return

    @staticmethod
    def load_geno(path: str):
        return


class NeuralNetwork(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(NeuralNetwork, self).__init__()
        self.NN = nn.Sequential(
            nn.Linear(n_input, n_hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(n_hidden, n_output, bias=False),
            nn.Tanh()
        )
        self.n_con1 = n_input * n_hidden
        self.n_con2 = n_hidden * n_output

    def set_weights(self, weights: numpy.array):
        """
        Set the weights of the Neural Network controller
        """
        weights = torch.tensor(weights)
        assert (len(weights) == self.n_con1 + self.n_con2)
        weight_matrix1 = weights[:self.n_con1].reshape(self.NN[0].weight.shape)
        weight_matrix2 = weights[-self.n_con2:].reshape(self.NN[2].weight.shape)
        self.NN[0].weight = nn.Parameter(weight_matrix1)
        self.NN[2].weight = nn.Parameter(weight_matrix2)

    def forward(self, state):
        return self.NN(torch.tensor(state, dtype=torch.float)).numpy()


class NumpyNetwork:
    def __init__(self, n_input, n_hidden, n_output, reservoir=True):
        self.reservoir = reservoir
        self.n_con1 = n_input * n_hidden
        self.n_con2 = n_hidden * n_output
        self.lin1 = np.random.uniform(-1, 1, (n_hidden, n_input))
        if reservoir:
            self.lin2 = np.random.uniform(-1, 1, (n_hidden, n_input))
        self.output = np.random.uniform(-1, 1, (n_output, n_hidden))

    def set_weights(self, weights: np.array):
        """
        Set the weights of the Neural Network controller
        """
        if self.reservoir:
            assert len(weights) == self.n_con2, f"Got {len(weights)} but expected {self.n_con2}"
            weight_matrix = weights[-self.n_con2:].reshape(self.output.shape)
            self.output = weight_matrix
        else:
            assert len(
                weights) == self.n_con1 + self.n_con2, f"Got {len(weights)} but expected {self.n_con1 + self.n_con2}"
            weight_matrix1 = weights[:self.n_con1].reshape(self.lin1.shape)
            weight_matrix2 = weights[-self.n_con2:].reshape(self.output.shape)
            self.lin1 = weight_matrix1
            self.output = weight_matrix2

    def forward(self, state: numpy.array):
        # hid_l = np.maximum(np.dot(self.lin1, state)*0.01, np.dot(self.lin1, state))
        hid_l = np.log(1 + np.exp(np.dot(self.lin1, state)))
        if self.reservoir:
            hid_l = np.log(1 + np.exp(np.dot(self.lin2, state)))
        output_l = 1 / (1 + np.exp(-np.dot(self.output, hid_l)))
        output_l[1] = output_l[1] * 2 - 1
        return output_l


class NNController(Controller):
    def __init__(self, n_states, n_actions, torch_=True):
        super().__init__(n_states, n_actions)
        self.controller_type = "NN"
        if torch_:
            self.model = NeuralNetwork(n_states, n_states, n_actions)
        else:
            self.model = NumpyNetwork(n_states, n_states, n_actions)

    def geno2pheno(self, genotype: np.array):
        self.model.set_weights(genotype)

    def map_state(self, min_from, max_from, min_to, max_to, state_portion):
        return min_to + np.multiply((max_to - min_to), np.divide((state_portion - min_from), (max_from - min_from)))

    def velocity_commands(self, state: np.ndarray) -> np.ndarray:
        """
        Given a state, give an appropriate action

        :param <np.array> state: A single observation of the current state, dimension is (state_dim)
        :return: <np.array> action: A vector of motor inputs
        """

        assert (len(state) == self.n_input), "State does not correspond with expected input size"
        state[:4] = self.map_state(0, 2, -1, 1, state[:4])
        state[4:8] = self.map_state(-np.pi, np.pi, -1, 1, state[4:8])  # Assumed distance sensing range is 2.0 meters. If not, check!
        state[-1] = self.map_state(0, 255.0, -1, 1, state[-1])  # Gradient value, [0, 255]

        action = self.model.forward(state)
        control_input = action * np.array([self.umax_const, self.wmax])
        return control_input

    def save_geno(self, path: str):
        if self.model.reservoir:
            np.save(path + "/reservoir", [self.model.lin1, self.model.lin2, self.model.output], allow_pickle=True)

    def load_geno(self, path: str):
        if self.model.reservoir:
            self.model.lin1, self.model.lin2, self.model.output = np.load(path + "/reservoir.npy", allow_pickle=True)


class adaptiveNNController(Controller):
    def __init__(self, n_states, n_actions):
        super().__init__(n_states, n_actions)
        self.controller_type = "aNN"
        self.rnn1 = NNController(n_states, n_actions, False)
        self.rnn1.controller_type = "rnn1"
        self.rnn2 = NNController(n_states, n_actions, False)
        self.rnn2.controller_type = "rnn2"
        self.probabilities = np.array([1., 0.75, 0.75, 0.5, 0.5])
        self.intensity_thr = np.array([229.14699, 178.0845, 127.02098, 75.957306, 0])
        self.current_controller = None
        self.refract_time = 10
        self.refract_n = 0

    def velocity_commands(self, state: np.ndarray) -> np.ndarray:
        """
        Given a state, give an appropriate action

        :param <np.array> state: A single observation of the current state, dimension is (state_dim)
        :return: <np.array> action: A vector of motor inputs
        """
        assert (len(state) == self.n_input), "State does not correspond with expected input size"
        local_intensity = state[-1]  # Gradient value, [0, 255]

        if (self.refract_n % self.refract_time) == 0:
            prob = self.probabilities[np.argmax(self.intensity_thr <= local_intensity)]
            if rng.random() < prob:
                self.current_controller = self.rnn1
            else:
                self.current_controller = self.rnn2
            self.refract_n = 0
        self.refract_n += 50
        control_input = self.current_controller.velocity_commands(state)
        return control_input

    def geno2pheno(self, genotype: List[np.array]):
        self.rnn1.geno2pheno(genotype[0])
        self.rnn2.geno2pheno(genotype[1])

    def load_geno(self, path: List[str]):
        self.rnn1.load_geno(path[0])
        self.rnn2.load_geno(path[1])


class GNNController(Controller):
    def __init__(self, n_states: int, n_actions: int, gcn_output_dim: int = 1, torch_=False):
        super().__init__(n_states, n_actions, gcn_output_dim)
        """
        The GNN controller is a graph neural network controller.
        First, uses message passing to update the node states.
        This node states are encoded into a latent space.
        The latent space is concatenated to the state and fed into a linear layer.

        1. Obs <- env.Step(actions_[t-1])
        2. NodeStates_latent <- GCN(Obs)
        3. Action <- LinearLayer(NodeStates_latent + Obs)

        :param n_states: Number of states
        :param n_actions: Number of actions
        :param gcn_output_dim: Dimension of the latent space
        :param torch_: If true, uses torch, else uses numpy
        """
        self.controller_type = "GNN"
        input_dim = n_states + gcn_output_dim
        hidden_dim = n_states

        self.gcn = GCNLayerNumpy(n_states, gcn_output_dim, n_actions)
        if torch_:
            self.LinearLayer = NeuralNetwork(input_dim, hidden_dim, n_actions)
        else:
            self.LinearLayer = NumpyNetwork(input_dim, hidden_dim, n_actions)

    def geno2pheno(self, genotype: np.array):
        assert len(genotype) == 4, "Genotype must be a list of 4 elements [GCN(W), LinearLayer(lin1, lin2, output)]]"
        self.LinearLayer.set_weights(genotype[1:])
        self.gcn.set_weights(genotype[0])

    def map_state(self, min_from, max_from, min_to, max_to, state_portion):
        return min_to + np.multiply((max_to - min_to), np.divide((state_portion - min_from), (max_from - min_from)))

    def velocity_commands(self, state: np.ndarray) -> np.ndarray:
        """
        Given a state, give an appropriate action

        :param <np.array> state: A single observation of the current state, dimension is (state_dim)
        :return: <np.array> action: A vector of motor inputs
        """

        assert (len(state) == self.n_input), "State does not correspond with expected input size"
        state[:4] = self.map_state(0, 2, -1, 1, state[:4])
        state[4:8] = self.map_state(-np.pi, np.pi, -1, 1,
                                    state[4:8])  # Assumed distance sensing range is 2.0 meters. If not, check!
        # state[4] = self.map_state(-3.1416, 3.1416, -1, 1, state[4]) # Heading average, already converted
        # state[5] = self.map_state(-3.1416, 3.1416, -1, 1, state[5])  # Own heading, [-pi, pi]
        state[-1] = self.map_state(0, 255.0, -1, 1, state[-1])  # Gradient value, [0, 255]

        node_states = self.gcn.forward(state)
        state = np.concatenate((node_states, state))
        action = self.LinearLayer.forward(state)
        control_input = action * np.array([self.umax_const, self.wmax])
        return control_input

    def save_geno(self, path: str):
        if self.LinearLayer.reservoir:
            np.save(path + "/reservoir_linear", [self.LinearLayer.lin1, self.LinearLayer.lin2, self.LinearLayer.output],
                    allow_pickle=True)
        if self.gcn.reservoir:
            np.save(path + "/reservoir_gcn", [self.gcn.W], allow_pickle=True)

    def load_geno(self, path: str):
        if self.LinearLayer.reservoir:
            self.LinearLayer.lin1, self.LinearLayer.lin2, self.LinearLayer.output = np.load(
                path + "/reservoir_linear.npy", allow_pickle=True)
        if self.gcn.reservoir:
            self.gcn.W = np.load(path + "/reservoir_gcn.npy", allow_pickle=True)


class RandomWalk(Controller):
    def velocity_commands(self, state: np.ndarray) -> np.ndarray:
        """
        Given a state, give an appropriate action.

        :param state: A single observation of the current state, dimension is (state_dim)
        :return: A vector of random motor inputs [-1, 1]^n_output
        """
        return np.random.uniform(-1, 1, self.n_output).astype('f')


class ActiveElastic_4dir(Controller):
    def __init__(self, n_states, n_actions):
        super().__init__(n_states, n_actions)
        self.controller_type = "4dir"

        #  Parameters to be optimized
        self.alpha = 1.0
        self.beta = 2.0
        self.K1 = 0.04
        self.K2 = 0.6

        self.epsilon = 12.0
        self.sigma_const = 0.4

    def velocity_commands(self, state: np.ndarray) -> np.ndarray:
        """
        Given a state, give an appropriate action.

        :param state: A single observation of the current state, dimension is (state_dim)
        :return: A vector of motor inputs
        """
        assert (len(state) == self.n_input), "State does not correspond with expected input size"
        self.bearings = np.array([0.0, 1.571, 3.14, -1.571])
        k = 4

        self.distances = state[0:k]
        self.distances[self.distances == 0] = np.inf
        self.distances = 2 * (1 - self.distances)
        self.headings = state[k:k + 2]
        self.own_heading = state[k + 2]

        # Calculate proximal forces
        pi_s = -self.epsilon * (2 * (np.divide(np.power(self.sigma_const, 4), np.power(self.distances, 5))) - (
            np.divide(np.power(self.sigma_const, 2), np.power(self.distances, 3))))
        px_s = np.multiply(pi_s, np.cos(np.array(self.bearings)))
        py_s = np.multiply(pi_s, np.sin(np.array(self.bearings)))
        pbar_xs = np.sum(px_s, axis=0)
        pbar_ys = np.sum(py_s, axis=0)

        # Calculate alignment control forces
        hbar_x = np.cos(
            np.arctan2(self.headings[0] / np.sqrt(np.power(self.headings[0], 2) + np.power(self.headings[0], 2)),
                       self.headings[1] / np.sqrt(np.power(self.headings[0], 2) + np.power(self.headings[0], 2)))
            - self.own_heading)

        hbar_y = np.sin(
            np.arctan2(self.headings[0] / np.sqrt(np.power(self.headings[0], 2) + np.power(self.headings[0], 2)),
                       self.headings[1] / np.sqrt(np.power(self.headings[0], 2) + np.power(self.headings[0], 2)))
            - self.own_heading)

        f_x = self.alpha * pbar_xs + self.beta * hbar_x
        f_y = self.alpha * pbar_ys + self.beta * hbar_y

        f_mag = np.sqrt(np.square(f_x) + np.square(f_y))
        glob_ang = np.arctan2(f_y, f_x)

        u = self.K1 * np.multiply(f_mag, np.cos(glob_ang)) + 0.05
        if u > self.umax_const:
            u = self.umax_const
        elif u < 0:
            u = 0.0

        w = self.K2 * np.multiply(f_mag, np.sin(glob_ang))
        if w > self.wmax:
            w = self.wmax
        elif w < -self.wmax:
            w = -self.wmax

        if np.isnan([u, w]).max():
            print("ERROR: output is NAN")
        return np.array([u, w])

    def geno2pheno(self, genotype: np.array):
        assert (len(genotype) == 4)
        self.alpha = genotype[0]
        self.beta = genotype[1]
        self.K1 = genotype[2]
        self.K2 = genotype[3]
        return


class ActiveElastic_k_near(Controller):
    def __init__(self, n_states, n_actions):
        super().__init__(n_states, n_actions)
        self.controller_type = "k_nearest"

        #  Parameters to be optimized
        self.alpha = 1.0
        self.beta = 2.0
        self.K1 = 0.04
        self.K2 = 0.6

        self.epsilon = 12.0
        self.sigma_const = 0.4

    def velocity_commands(self, state: np.array) -> np.array:
        assert (len(state) == self.n_input), "State does not correspond with expected input size"
        k = 4

        self.distances = state[0:k]
        self.bearings = state[k:2 * k]
        self.headings = state[2 * k:2 * k + 2]
        self.own_heading = state[2 * k + 2]

        # Calculate proximal forces
        pi_s = -self.epsilon * (2 * (np.divide(np.power(self.sigma_const, 4), np.power(self.distances, 5))) - (
            np.divide(np.power(self.sigma_const, 2), np.power(self.distances, 3))))
        px_s = np.multiply(pi_s, np.cos(np.array(self.bearings)))
        py_s = np.multiply(pi_s, np.sin(np.array(self.bearings)))
        pbar_xs = np.sum(px_s, axis=0)
        pbar_ys = np.sum(py_s, axis=0)

        # Calculate alignment control forces
        hbar_x = np.cos(
            np.arctan2(self.headings[0] / np.sqrt(np.power(self.headings[0], 2) + np.power(self.headings[0], 2)),
                       self.headings[1] / np.sqrt(np.power(self.headings[0], 2) + np.power(self.headings[0], 2)))
            - self.own_heading)

        hbar_y = np.sin(
            np.arctan2(self.headings[0] / np.sqrt(np.power(self.headings[0], 2) + np.power(self.headings[0], 2)),
                       self.headings[1] / np.sqrt(np.power(self.headings[0], 2) + np.power(self.headings[0], 2)))
            - self.own_heading)

        f_x = self.alpha * pbar_xs + self.beta * hbar_x
        f_y = self.alpha * pbar_ys + self.beta * hbar_y

        f_mag = np.sqrt(np.square(f_x) + np.square(f_y))
        glob_ang = np.arctan2(f_y, f_x)

        u = self.K1 * np.multiply(f_mag, np.cos(glob_ang)) + 0.05
        if u > self.umax_const:
            u = self.umax_const
        elif u < 0:
            u = 0.0

        w = self.K2 * np.multiply(f_mag, np.sin(glob_ang))
        if w > self.wmax:
            w = self.wmax
        elif w < -self.wmax:
            w = -self.wmax

        return np.array([u, w])


class ActiveElastic_omni(Controller):
    def __init__(self, n_states, n_actions):
        super().__init__(n_states, n_actions)
        self.controller_type = "omni"

        #  Parameters to be optimized
        self.alpha = 1.0
        self.beta = 2.0
        self.K1 = 0.04
        self.K2 = 0.6

        self.epsilon = 12.0
        self.sigma_const = 0.4

    def velocity_commands(self, state: np.array) -> np.array:
        assert (len(state) == self.n_input), "State does not correspond with expected input size"
        self.distances = state[0]
        self.distances[self.distances == 0] = np.inf
        self.bearings = state[1]
        self.headings = state[2:4]
        self.own_heading = state[4]

        # Calculate proximal forces
        pi_s = -self.epsilon * (2 * (np.divide(np.power(self.sigma_const, 4), np.power(self.distances, 5))) - (
            np.divide(np.power(self.sigma_const, 2), np.power(self.distances, 3))))
        px_s = np.multiply(pi_s, np.cos(np.array(self.bearings)))
        py_s = np.multiply(pi_s, np.sin(np.array(self.bearings)))
        pbar_xs = np.sum(px_s, axis=0)
        pbar_ys = np.sum(py_s, axis=0)

        # Calculate alignment control forces
        hbar_x = np.cos(
            np.arctan2(self.headings[0] / np.sqrt(np.power(self.headings[0], 2) + np.power(self.headings[0], 2)),
                       self.headings[1] / np.sqrt(np.power(self.headings[0], 2) + np.power(self.headings[0], 2)))
            - self.own_heading)

        hbar_y = np.sin(
            np.arctan2(self.headings[0] / np.sqrt(np.power(self.headings[0], 2) + np.power(self.headings[0], 2)),
                       self.headings[1] / np.sqrt(np.power(self.headings[0], 2) + np.power(self.headings[0], 2)))
            - self.own_heading)

        f_x = self.alpha * pbar_xs + self.beta * hbar_x
        f_y = self.alpha * pbar_ys + self.beta * hbar_y

        f_mag = np.sqrt(np.square(f_x) + np.square(f_y))
        glob_ang = np.arctan2(f_y, f_x)

        u = self.K1 * np.multiply(f_mag, np.cos(glob_ang)) + 0.05
        if u > self.umax_const:
            u = self.umax_const
        elif u < 0:
            u = 0.0

        w = self.K2 * np.multiply(f_mag, np.sin(glob_ang))
        if w > self.wmax:
            w = self.wmax
        elif w < -self.wmax:
            w = -self.wmax

        return np.array([u, w])

    def geno2pheno(self, genotype: np.array):
        assert (len(genotype) == 4)
        self.alpha = genotype[0]
        self.beta = genotype[1]
        self.K1 = genotype[2]
        self.K2 = genotype[3]
        return
