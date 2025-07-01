import numpy as np

def flatten_tensor(tensor):
    """Recursively flatten a nested list or ndarray into a flat list."""
    output = []
    for ele in tensor:
        if isinstance(ele, (list, np.ndarray)):
            output.extend(flatten_tensor(ele))
        else:
            output.append(ele)
    return output

def compute_strides(shape):
    """Compute strides for a given shape."""
    strides = [1] * len(shape)
    for i in range(len(strides) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return strides
