from tensor.tensor import Tensor
from nn.module import Module


class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x.relu()
