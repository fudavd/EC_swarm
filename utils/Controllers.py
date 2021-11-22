import numpy as np
import torch
from torch import nn


class Controller(object):
    def __init__(self, n_states, n_actions):
        self.n_input = n_states
        self.n_output = n_actions

    @staticmethod
    def velocity_commands(state: np.array) -> np.array:
        return np.array([0])

    @staticmethod
    def geno2pheno(genotype: np.array):
        return


class NeuralNetwork(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(NeuralNetwork, self).__init__()
        self.NN = nn.Sequential(
            nn.Linear(n_input, n_hidden, bias=False),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output, bias=False),
            nn.Tanh()
        )
        self.n_con1 = n_input * n_hidden
        self.n_con2 = n_hidden * n_output
        for p in self.NN.parameters():
            p.requires_grad = False

    def set_weights(self, weights: torch.Tensor):
        """
        Set the weights of the Neural Network controller
        """
        assert (len(weights) == self.n_con1 + self.n_con2)
        with torch.no_grad():
            weight_matrix1 = weights[:self.n_con1].reshape(self.NN[0].weight.shape)
            weight_matrix2 = weights[-self.n_con2:].reshape(self.NN[1].weight.shape)
            self.NN[0].weight = weight_matrix1
            self.NN[1].weight = weight_matrix2

    def forward(self, state):
        return self.NN(state)


class NNController(Controller):
    def __init__(self, n_states, n_actions):
        super().__init__(n_states, n_actions)
        self.model = NeuralNetwork(n_states, n_states, n_actions)

    def geno2pheno(self, genotype: np.array):
        weights = torch.Tensor(genotype).requires_grad(False)
        self.model.set_weights(weights)

    def velocity_commands(self, state: np.array) -> np.array:
        """
        Given a state, give an appropriate action

        Inputs:
        <np.array> state : A single observation of the current state, dimension is (state_dim)
        Outputs:
        <np.array> action : A vector of motor inputs
        """
        action = self.model.forward(torch.Tensor(state))
        return action.numpy()


class RandomWalk(Controller):
    def velocity_commands(self, state: np.array) -> np.array:
        """
        Outputs:
        <np.array> action : A vector of motor inputs
        """
        return np.random.uniform(-1, 1, self.n_output).astype('f')

class ActiveElastic(Controller):
    def __init__(self, n_states, n_actions):
        super().__init__(n_states, n_actions)

        #  Parameters to be optimized
        self.alpha = 1.0
        self.beta = 2.0
        self.K1 = 0.04
        self.K2 = 0.6

        self.epsilon = 12.0
        self.sigma_const = 0.4
        self.umax_const = 0.1
        self.wmax = 1.5708 / 2.5

    def velocity_commands(self, state: np.array) -> np.array:
        k = 4

        self.distances = state[0:k]
        self.bearings = state[k:2*k]
        self.headings = state[2*k:2*k+2]
        self.own_heading = state[2*k+2]

        self.distances[self.distances == 0] = np.inf

        # Calculate proximal forces
        pi_s = -self.epsilon * (2 * (np.divide(np.power(self.sigma_const, 4), np.power(self.distances, 5))) - (
            np.divide(np.power(self.sigma_const, 2), np.power(self.distances, 3))))
        px_s = np.multiply(pi_s, np.cos(np.array(self.bearings)))
        py_s = np.multiply(pi_s, np.sin(np.array(self.bearings)))
        pbar_xs = np.sum(px_s, axis=0)
        pbar_ys = np.sum(py_s, axis=0)

        # Calculate alignment control forces
        hbar_x = self.headings[0]
        hbar_y = self.headings[1]

        f_x = self.alpha * pbar_xs + self.beta * hbar_x  # + self.gama * r_x
        f_y = self.alpha * pbar_ys + self.beta * hbar_y  # + self.gama * r_y

        f_mag = np.sqrt(np.square(f_x) + np.square(f_y))
        glob_ang = np.arctan2(f_y, f_x) - self.own_heading

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
