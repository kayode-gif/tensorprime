
class Optimizer:
    def __init__(self, params, defaults):
        self.defaults = defaults
        self.param_groups = []

        if not isinstance(params, list):
            params = [{'params': params}]

        for param_group in params:

            # A param_group is a dict that can override defaults
            # e.g., {'params': model.layer1.parameters(), 'lr': 0.001}
            group = defaults.copy()
            group.update(param_group)
            print(group.update(param_group))
            self.param_groups.append(group)

    def zero_grad(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    # modify in place the gradient of tensor's data
                    p.grad.data.fill(0)


    def step(self):
        raise NotImplementedError("Optimizer subclasses must implement this method")
