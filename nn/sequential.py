from .module import Module


class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = layers

        for i, layer in enumerate(layers):
            self._modules[str(i)] = layer

    def forward(self, x):
        """
        Defines the forward pass of the sequential model.
        The input is passed through each layer in order.
        """
        for layer in self.layers:
            x = layer(x)
        return x
