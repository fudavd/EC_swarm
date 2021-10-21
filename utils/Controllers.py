import numpy as np
import torch
from torch import nn


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
        for p in self.model.NN.parameters():
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


class NNController:
    def __init__(self, n_state, n_actions):
        self.model = NeuralNetwork(n_state, n_state, n_actions)
        self.n_input = n_state
        self.n_output = n_actions

    def controller_target(self, state: np.array) -> np.array:
        """
        Given a state, give an appropriate action

        Inputs:
        <np.array> state : A single observation of the current state, dimension is (state_dim)
        Outputs:
        <np.array> action : A vector of motor inputs
        """
        action = self.model.forward(state.float())
        return action.astype('f')


class RandomWalk:
    def __init__(self, n_actions):
        self.n_output = n_actions

    def controller(self) -> np.array:
        """
        Outputs:
        <np.array> action : A vector of motor inputs
        """
        return np.random.uniform(-1, 1, self.n_output).astype('f')
