from tensor.tensor import Tensor
from nn.module import Module
import numpy as np


class Linear(Module):
    def __init__(self, n_inputs, n_neurons, use_bias=True):
        super().__init__()
        self.use_bias = use_bias
        self.weights = Tensor(np.random.randn(n_neurons, n_inputs), requires_grad=True)
        if self.use_bias:
            self.bias = Tensor(np.zeros((1, n_neurons)), requires_grad=True)
        else:
            self.bias = None

    def forward(self, x):
        out = x @ self.weights.transpose((1, 0))
        if self.use_bias:
            out = out + self.bias
        return out
