from importlib.metadata import requires

from tensor.tensor import Tensor
from nn.module import Module
import numpy as np


class Conv2D(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,padding=0,dilation=1,bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, int) else kernel_size
        self.stride = stride if isinstance(stride, int) else stride
        self.padding = padding
        self.dilation = dilation

        if isinstance(kernel_size, int):
            ky = kx = kernel_size
        else:
            ky, kx = kernel_size

        self.weight = Tensor(np.random.randn(out_channels, in_channels, ky, kx).astype(np.float64), requires_grad=True)
        if bias:
            self.bias = Tensor(np.zeros(out_channels), requires_grad=True)
        else:
            self.bias = None

    def forward(self, x):
        return x.conv2d(self.weight, self.stride, self.padding, self.dilation, self.bias)
