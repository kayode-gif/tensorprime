from tensor.tensor import Tensor

class Module:
    def __init__(self):
        self._modules = {}
        self._parameters = {}
        self.training = True

    def __setattr__(self, name, value):
        if isinstance(value, Module):
            self._modules[name] = value
        elif isinstance(value, Tensor):
            self._parameters[name] = value
        super().__setattr__(name, value)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def parameters(self):
        params = list(self._parameters.values())
        for module in self._modules.values():
            if hasattr(module, 'parameters'):
                params.extend(module.parameters())
        return params

    def zero_grad(self):
        for p in self.parameters():
            if p.grad is not None:
                # use np.fill to insert the data in place
                p.grad.data.fill(0)

