# tensorprime

A minimal, extensible deep learning framework with custom tensor operations, autograd, neural network layers, and a built-in performance profiler (inspired by Karapthy)

---

## Features

- **Custom Tensor Engine:**
  Supports broadcasting, operator overloading, and automatic differentiation.

- **Neural Network Layers:**  
  Modular layers (Linear, Conv2d, etc.) and sequential models.

- **Losses & Optimizers:**  
  Includes MSE loss and SGD optimizer; easy to extend.

- ** Custom Performance Profiler:**  
  Analyze per-operation CPU time, memory usage, and call frequency for both ops and layers.

---

## Installation

It is recommended to use a Python virtual environment (venv):

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Quick Start

```python
from tensor.tensor import Tensor
from tensor.ops import AddOp, MulOp

# initialize tensors
a = Tensor([1, 2, 3])
b = Tensor([4, 5, 6])

# use AddOp and MulOp directly
add_result = AddOp.forward(a, b)
mul_result = MulOp.forward(a, b)
print("Add result:", add_result)
print("Mul result:", mul_result)

# example backward operation (if using autograd)
# suppose u and t are Tensors with requires_grad=True
t = Tensor([1, 2, 3], requires_grad=True)
u = Tensor([4, 5, 6], requires_grad=True)
res = u + t
res.backward() # To simulate backward, you would typically call res.backward() if res is a Tensor
print("gradient t :", t.grad) # return the respective gradients
print("gradient u:", u.grad)
```

### Run XOR Dataset Example

```bash
python -m examples.train_xor
```

---

## Profiling Example

```python
from profiler.setup_profiler import setup_profiler
from tensor.tensor import Tensor
from tensor.ops import AddOp, MulOp

profiler = setup_profiler(profile_memory=True)

a = Tensor([1, 2, 3])
b = Tensor([4, 5, 6])

# profiled AddOp and MulOp
add_result = AddOp.forward(a, b)
mul_result = MulOp.forward(a, b)

# example backward profiling (if using autograd and backward is patched)
# if you have a Tensor with requires_grad=True, you can call .backward()
# for demonstration, let's assume backward is patched and called like:
# AddOp.backward(out_grad, a, b)
# (Replace out_grad with an appropriate gradient Tensor)
profiler.report() # returns the performance analysis of the operation (sorted by time)
```

---

## Sample Subset of Profiler Output (XOR Dataset)

```
Function               Calls  Avg Wall (ms)  Avg CPU (ms)  Avg Mem (MB)  Shapes
------------------------------------------------------------------------------------------
PowOp.backward         2000   0.786          0.798         0.017         [(4, 1), (4, 1), ()]
MatMulOp.backward      4000   0.335          0.349         0.007         [(4, 1), (4, 4), (4, 1)]
DivOp.backward         2000   0.322          0.337         0.007         [(), (), ()]
Linear.forward         4002   0.285          0.297         0.005         [(4, 2)]
SigmoidOp.backward     2000   0.281          0.294         0.006         [(4, 1), (4, 1), (4, 1)]
```

---

## Project Structure

```
tensorprime/
  tensor/         # Tensor engine and ops
  layers/         # Neural network layers
  nn/             # Model and sequential containers
  losses/         # Loss functions
  optimizers/     # Optimizers (SGD, etc.)
  profiler/       # Performance profiling tools
  examples/       # Example scripts and notebooks
  tests/          # Unit tests
```

---

## Why tensorprime?

- **Educational:** Understand deep learning from the ground up.
- **Research-Friendly:** Prototype new ops, layers, or training techniques.
- **Performance-Aware:** Built-in profiling to help you optimize and debug.

---

## License
MIT License 
