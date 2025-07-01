from .optimizer import Optimizer
import numpy as np

class SGDOptimizer(Optimizer):
    def __init__(self, params, lr: float) -> None:
        """Constructor for Stochastic Gradient Descent (SGD) Optimizer.
        Args:
            params: Parameters to update each step. You don't need to do anything with them.
                They are properly initialize through the super call.
            lr (float): Learning Rate of the gradient descent.
        """
        if lr < 0.0:
            raise ValueError(f"Invalid Learning rate: {lr}")
        super().__init__([{'params': params}], {"lr": lr})

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                p.data = p.data - (lr * p.grad.data)
