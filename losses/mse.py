from tensor.tensor import Tensor
from nn.module import Module
import math


class MSELoss(Module):

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        error = (y_pred - y_true) ** 2
        num_elements = math.prod(error.shape)
        total_error = error.sum_axis(axis=None, keepdims=False)
        mse = total_error / num_elements
        return mse
