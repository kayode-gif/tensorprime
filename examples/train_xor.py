
import numpy as np
from layers.sigmoid import Sigmoid
from nn.sequential import Sequential
from layers.linear import Linear
from layers.relu import ReLU
from losses.mse import MSELoss
from optimizers.sgd import SGDOptimizer
from profiler.setup_profiler import setup_profiler
from tensor.tensor import Tensor

# import prodi
profiler = setup_profiler(profile_memory=True)
# add the linear layer to be profiled
Linear.forward = profiler.profile_op(Linear.forward)

# Data
X_np = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
y_np = np.array([[0], [1], [1], [0]], dtype=np.float64)

X_custom = Tensor(X_np)
y_custom = Tensor(y_np)

custom_model = Sequential(
    Linear(2, 4),
    ReLU(),
    Linear(4, 1),
    Sigmoid(),
)

custom_criterion = MSELoss()
custom_optimizer = SGDOptimizer(custom_model.parameters(), lr=0.5)

epochs = 2000
for epoch in range(epochs):
    y_pred_custom = custom_model(X_custom)
    loss_custom = custom_criterion(y_pred_custom, y_custom)
    custom_optimizer.zero_grad()
    loss_custom.backward()
    custom_optimizer.step()

    if epoch % 200 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch}")
        print(f"  Custom Loss: {loss_custom.data:.6f}")
        print(f"  Custom preds: {y_pred_custom.data.flatten()}")
        print("-" * 40)

print("Final Custom predictions:", custom_model(X_custom).data.flatten())
profiler.report()
