from tensor.tensor import Tensor
from tensor.broadcast_utils import Broadcast
import numpy as np
import torch
import pytest
import numpy as np
import torch
import torch.nn as nn
from tensor.ops import MaxPool2dOp
from tensor.ops import Conv2dOp


# ============================================================================
# ELEMENTWISE OPERATIONS
# ============================================================================

class TestElementwiseOps:
    def test_add_same_shape(self):
        # Your implementation
        a = Tensor([[1, 2], [3, 4]], requires_grad=True)
        b = Tensor([[5, 6], [7, 8]], requires_grad=True)
        out = a + b
        out.backward()

        # PyTorch reference
        a_torch = torch.tensor([[1, 2], [3, 4]], dtype=torch.float64, requires_grad=True)
        b_torch = torch.tensor([[5, 6], [7, 8]], dtype=torch.float64, requires_grad=True)
        out_torch = a_torch + b_torch
        out_torch.backward(torch.ones_like(out_torch))

        # Compare results
        assert np.allclose(out.data, out_torch.detach().numpy(), atol=1e-10)
        assert np.allclose(a.grad.data, a_torch.grad.numpy(), atol=1e-10)
        assert np.allclose(b.grad.data, b_torch.grad.numpy(), atol=1e-10)

    def test_add_broadcasting(self):
        a = Tensor([[1, 2], [3, 4]], requires_grad=True)
        b = Tensor([10, 20], requires_grad=True)
        out = a + b
        out.backward()

        a_torch = torch.tensor([[1, 2], [3, 4]], dtype=torch.float64, requires_grad=True)
        b_torch = torch.tensor([10, 20], dtype=torch.float64, requires_grad=True)
        out_torch = a_torch + b_torch
        out_torch.backward(torch.ones_like(out_torch))

        assert np.allclose(out.data, out_torch.detach().numpy(), atol=1e-10)
        assert np.allclose(a.grad.data, a_torch.grad.numpy(), atol=1e-10)
        assert np.allclose(b.grad.data, b_torch.grad.numpy(), atol=1e-10)

    def test_complex_broadcasting(self):
        a = Tensor([[1.], [2.]], requires_grad=True)
        b = Tensor([[5., 6., 7.], [8., 9., 10.]], requires_grad=True)
        out = a + b
        out.backward()

        a_torch = torch.tensor([[1.], [2.]], dtype=torch.float64, requires_grad=True)
        b_torch = torch.tensor([[5., 6., 7.], [8., 9., 10.]], dtype=torch.float64, requires_grad=True)
        out_torch = a_torch + b_torch
        out_torch.backward(torch.ones_like(out_torch))

        assert np.allclose(out.data, out_torch.detach().numpy(), atol=1e-10)
        assert np.allclose(a.grad.data, a_torch.grad.numpy(), atol=1e-10)
        assert np.allclose(b.grad.data, b_torch.grad.numpy(), atol=1e-10)

    def test_sub_same_shape(self):
        a = Tensor([[5, 7], [9, 11]], requires_grad=True)
        b = Tensor([[2, 3], [4, 5]], requires_grad=True)
        out = a - b
        out.backward()

        a_torch = torch.tensor([[5, 7], [9, 11]], dtype=torch.float64, requires_grad=True)
        b_torch = torch.tensor([[2, 3], [4, 5]], dtype=torch.float64, requires_grad=True)
        out_torch = a_torch - b_torch
        out_torch.backward(torch.ones_like(out_torch))

        assert np.allclose(out.data, out_torch.detach().numpy(), atol=1e-10)
        assert np.allclose(a.grad.data, a_torch.grad.numpy(), atol=1e-10)
        assert np.allclose(b.grad.data, b_torch.grad.numpy(), atol=1e-10)

    def test_mul_same_shape(self):
        a = Tensor([[2, 3], [4, 5]], requires_grad=True)
        b = Tensor([[6, 7], [8, 9]], requires_grad=True)
        out = a * b
        out.backward()

        a_torch = torch.tensor([[2, 3], [4, 5]], dtype=torch.float64, requires_grad=True)
        b_torch = torch.tensor([[6, 7], [8, 9]], dtype=torch.float64, requires_grad=True)
        out_torch = a_torch * b_torch
        out_torch.backward(torch.ones_like(out_torch))

        assert np.allclose(out.data, out_torch.detach().numpy(), atol=1e-10)
        assert np.allclose(a.grad.data, a_torch.grad.numpy(), atol=1e-10)
        assert np.allclose(b.grad.data, b_torch.grad.numpy(), atol=1e-10)

    def test_chained_operations(self):
        a = Tensor([[2., 3.], [4., 5.]], requires_grad=True)
        b = Tensor([1., 2.], requires_grad=True)
        c = Tensor([[3.], [4.]], requires_grad=True)
        out = a * b * c
        out.backward()

        a_torch = torch.tensor([[2., 3.], [4., 5.]], dtype=torch.float64, requires_grad=True)
        b_torch = torch.tensor([1., 2.], dtype=torch.float64, requires_grad=True)
        c_torch = torch.tensor([[3.], [4.]], dtype=torch.float64, requires_grad=True)
        out_torch = a_torch * b_torch * c_torch
        out_torch.backward(torch.ones_like(out_torch))

        assert np.allclose(out.data, out_torch.detach().numpy(), atol=1e-10)
        assert np.allclose(a.grad.data, a_torch.grad.numpy(), atol=1e-10)
        assert np.allclose(b.grad.data, b_torch.grad.numpy(), atol=1e-10)
        assert np.allclose(c.grad.data, c_torch.grad.numpy(), atol=1e-10)

    def test_div_same_shape(self):
        a = Tensor([[6, 8], [10, 12]], requires_grad=True)
        b = Tensor([[2, 2], [5, 3]], requires_grad=True)
        out = a / b
        out.backward()

        a_torch = torch.tensor([[6, 8], [10, 12]], dtype=torch.float64, requires_grad=True)
        b_torch = torch.tensor([[2, 2], [5, 3]], dtype=torch.float64, requires_grad=True)
        out_torch = a_torch / b_torch
        out_torch.backward(torch.ones_like(out_torch))

        assert np.allclose(out.data, out_torch.detach().numpy(), atol=1e-10)
        assert np.allclose(a.grad.data, a_torch.grad.numpy(), atol=1e-10)
        assert np.allclose(b.grad.data, b_torch.grad.numpy(), atol=1e-10)

    def test_pow_same_shape(self):
        a = Tensor([[2, 3]], requires_grad=True)
        b = Tensor([[3, 2]], requires_grad=True)
        out = a ** b
        out.backward()

        a_torch = torch.tensor([[2, 3]], dtype=torch.float64, requires_grad=True)
        b_torch = torch.tensor([[3, 2]], dtype=torch.float64, requires_grad=True)
        out_torch = a_torch ** b_torch
        out_torch.backward(torch.ones_like(out_torch))

        assert np.allclose(out.data, out_torch.detach().numpy(), atol=1e-10)
        assert np.allclose(a.grad.data, a_torch.grad.numpy(), atol=1e-10)
        assert np.allclose(b.grad.data, b_torch.grad.numpy(), atol=1e-10)


# ============================================================================
# UNARY OPERATIONS
# ============================================================================

class TestUnaryOps:
    def test_neg(self):
        a = Tensor([[1, -2], [3, -4]], requires_grad=True)
        out = -a
        out.backward()

        a_torch = torch.tensor([[1, -2], [3, -4]], dtype=torch.float64, requires_grad=True)
        out_torch = -a_torch
        out_torch.backward(torch.ones_like(out_torch))

        assert np.allclose(out.data, out_torch.detach().numpy(), atol=1e-10)
        assert np.allclose(a.grad.data, a_torch.grad.numpy(), atol=1e-10)

    def test_tanh(self):
        a = Tensor([[0.0, 1.0], [2.0, 3.0]], requires_grad=True)
        out = a.tanh()
        out.backward()

        a_torch = torch.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=torch.float64, requires_grad=True)
        out_torch = torch.tanh(a_torch)
        out_torch.backward(torch.ones_like(out_torch))

        assert np.allclose(out.data, out_torch.detach().numpy(), atol=1e-10)
        assert np.allclose(a.grad.data, a_torch.grad.numpy(), atol=1e-10)

    def test_exp(self):
        a = Tensor([[0.0, 1.0], [2.0, 3.0]], requires_grad=True)
        out = a.exp()
        out.backward()

        a_torch = torch.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=torch.float64, requires_grad=True)
        out_torch = torch.exp(a_torch)
        out_torch.backward(torch.ones_like(out_torch))

        assert np.allclose(out.data, out_torch.detach().numpy(), atol=1e-10)
        assert np.allclose(a.grad.data, a_torch.grad.numpy(), atol=1e-10)

    def test_log(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        out = a.log()
        out.backward()

        a_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64, requires_grad=True)
        out_torch = torch.log(a_torch)
        out_torch.backward(torch.ones_like(out_torch))

        assert np.allclose(out.data, out_torch.detach().numpy(), atol=1e-10)
        assert np.allclose(a.grad.data, a_torch.grad.numpy(), atol=1e-10)

    def test_relu(self):
        a = Tensor([[1.0, -2.0], [3.0, -4.0]], requires_grad=True)
        out = a.relu()
        out.backward()

        a_torch = torch.tensor([[1.0, -2.0], [3.0, -4.0]], dtype=torch.float64, requires_grad=True)
        out_torch = torch.relu(a_torch)
        out_torch.backward(torch.ones_like(out_torch))

        assert np.allclose(out.data, out_torch.detach().numpy(), atol=1e-10)
        assert np.allclose(a.grad.data, a_torch.grad.numpy(), atol=1e-10)

    def test_sigmoid(self):
        a = Tensor([[1.0, -2.0], [3.0, -4.0]], requires_grad=True)
        out = a.sigmoid()
        out.backward()

        a_torch = torch.tensor([[1.0, -2.0], [3.0, -4.0]], dtype=torch.float64, requires_grad=True)
        out_torch = torch.sigmoid(a_torch)
        out_torch.backward(torch.ones_like(out_torch))

        assert np.allclose(out.data, out_torch.detach().numpy(), atol=1e-10)
        assert np.allclose(a.grad.data, a_torch.grad.numpy(), atol=1e-10)


# ============================================================================
# MATRIX OPERATIONS
# ============================================================================

class TestMatrixOps:
    def test_matmul(self):
        a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        b = Tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], requires_grad=True)
        out = a @ b
        out.backward()

        a_torch = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64, requires_grad=True)
        b_torch = torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], dtype=torch.float64, requires_grad=True)
        out_torch = a_torch @ b_torch
        out_torch.backward(torch.ones_like(out_torch))

        assert np.allclose(out.data, out_torch.detach().numpy(), atol=1e-10)
        assert np.allclose(a.grad.data, a_torch.grad.numpy(), atol=1e-10)
        assert np.allclose(b.grad.data, b_torch.grad.numpy(), atol=1e-10)

    def test_dot(self):
        a = Tensor([1, 2, 3, 4], requires_grad=True)
        b = Tensor([1, 1, 1, 1], requires_grad=True)
        out = a.dot(b)
        out.backward()

        a_torch = torch.tensor([1, 2, 3, 4], dtype=torch.float64, requires_grad=True)
        b_torch = torch.tensor([1, 1, 1, 1], dtype=torch.float64, requires_grad=True)
        out_torch = torch.dot(a_torch, b_torch)
        out_torch.backward(torch.ones_like(out_torch))

        assert np.allclose(out.data, out_torch.detach().numpy(), atol=1e-10)
        assert np.allclose(a.grad.data, a_torch.grad.numpy(), atol=1e-10)
        assert np.allclose(b.grad.data, b_torch.grad.numpy(), atol=1e-10)

    def test_batch_matmul_broadcasting(self):
        a = Tensor(np.random.randn(2, 1, 3), requires_grad=True)
        b = Tensor(np.random.randn(1, 3, 4), requires_grad=True)
        out = a.batch_matmul(b)
        out.backward()

        a_torch = torch.tensor(a.data, dtype=torch.float64, requires_grad=True)
        b_torch = torch.tensor(b.data, dtype=torch.float64, requires_grad=True)
        out_torch = torch.matmul(a_torch, b_torch)
        out_torch.backward(torch.ones_like(out_torch))

        assert np.allclose(out.data, out_torch.detach().numpy(), atol=1e-10)
        assert np.allclose(a.grad.data, a_torch.grad.numpy(), atol=1e-10)
        assert np.allclose(b.grad.data, b_torch.grad.numpy(), atol=1e-10)


# ============================================================================
# SHAPE OPERATIONS
# ============================================================================

class TestShapeOps:
    def test_reshape(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        out = a.reshape((4,))
        out.backward()

        a_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64, requires_grad=True)
        out_torch = a_torch.reshape(4)
        out_torch.backward(torch.ones_like(out_torch))

        assert np.allclose(out.data, out_torch.detach().numpy(), atol=1e-10)
        assert np.allclose(a.grad.data, a_torch.grad.numpy(), atol=1e-10)

    def test_transpose(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        out = a.transpose((1, 0))
        out.backward()

        a_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64, requires_grad=True)
        out_torch = a_torch.transpose(0, 1)
        out_torch.backward(torch.ones_like(out_torch))

        assert np.allclose(out.data, out_torch.detach().numpy(), atol=1e-10)
        assert np.allclose(a.grad.data, a_torch.grad.numpy(), atol=1e-10)

    def test_expand_dims(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        out = a.expand_dims(1)
        out.backward()

        a_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64, requires_grad=True)
        out_torch = a_torch.unsqueeze(1)
        out_torch.backward(torch.ones_like(out_torch))

        assert np.allclose(out.data, out_torch.detach().numpy(), atol=1e-10)
        assert np.allclose(a.grad.data, a_torch.grad.numpy(), atol=1e-10)


# ============================================================================
# REDUCTION OPERATIONS
# ============================================================================

class TestReductionOps:
    def test_sum_axis_none(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        out = a.sum_axis()
        out.backward()

        a_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64, requires_grad=True)
        out_torch = a_torch.sum()
        out_torch.backward(torch.ones_like(out_torch))

        assert np.allclose(out.data, out_torch.detach().numpy(), atol=1e-10)
        assert np.allclose(a.grad.data, a_torch.grad.numpy(), atol=1e-10)

    def test_sum_axis_0(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        out = a.sum_axis(axis=0)
        out.backward()

        a_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64, requires_grad=True)
        out_torch = a_torch.sum(dim=0)
        out_torch.backward(torch.ones_like(out_torch))

        assert np.allclose(out.data, out_torch.detach().numpy(), atol=1e-10)
        assert np.allclose(a.grad.data, a_torch.grad.numpy(), atol=1e-10)

    def test_sum_axis_keepdims(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        out = a.sum_axis(axis=0, keepdims=True)
        out.backward()

        a_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64, requires_grad=True)
        out_torch = a_torch.sum(dim=0, keepdim=True)
        out_torch.backward(torch.ones_like(out_torch))

        assert np.allclose(out.data, out_torch.detach().numpy(), atol=1e-10)
        assert np.allclose(a.grad.data, a_torch.grad.numpy(), atol=1e-10)

    def test_sum_axis_negative_indices(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        out = a.sum_axis(axis=-1)
        out.backward()

        a_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64, requires_grad=True)
        out_torch = a_torch.sum(dim=-1)
        out_torch.backward(torch.ones_like(out_torch))

        assert np.allclose(out.data, out_torch.detach().numpy(), atol=1e-10)
        assert np.allclose(a.grad.data, a_torch.grad.numpy(), atol=1e-10)

    def test_sum_axis_multiple_axes(self):
        a = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], requires_grad=True)
        out = a.sum_axis(axis=(0, 1))
        out.backward()

        a_torch = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float64, requires_grad=True)
        out_torch = a_torch.sum(dim=(0, 1))
        out_torch.backward(torch.ones_like(out_torch))

        assert np.allclose(out.data, out_torch.detach().numpy(), atol=1e-10)
        assert np.allclose(a.grad.data, a_torch.grad.numpy(), atol=1e-10)

class TestMaxPool2D:

    def test_maxpool2d_forward(self):
        A = np.array([[1, 1, 2, 4],
                      [5, 6, 7, 8],
                      [3, 2, 1, 0],
                      [1, 2, 3, 4]])
        A_4d = A.reshape(1, 1, 4, 4)

        A_torch = torch.tensor(A_4d, dtype=torch.float32)
        maxpool_torch = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        output_torch = maxpool_torch(A_torch).numpy()

        A_custom = Tensor(A_4d)
        output_custom = MaxPool2dOp.forward(A_custom, kernel_size=2, stride=2, padding=0)

        assert np.allclose(output_custom, output_torch, atol=1e-10)

    def test_maxpool2d_backward(self):
        A = np.array([[1, 1, 2, 4],
                      [5, 6, 7, 8],
                      [3, 2, 1, 0],
                      [1, 2, 3, 4]])
        A_4d = A.reshape(1, 1, 4, 4)

        input_torch = torch.tensor(A_4d, dtype=torch.float32, requires_grad=True)
        maxpool_torch = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        output_torch = maxpool_torch(input_torch)
        grad_torch = torch.ones_like(output_torch)
        output_torch.backward(grad_torch)
        grad_torch_result = input_torch.grad.numpy()

        input_custom = Tensor(A_4d)
        grad_custom_tensor = Tensor(1.0)
        grad_custom = MaxPool2dOp.backward(
            grad_custom_tensor,
            input_custom,
            kernel_size=2,
            stride=2,
            padding=0
        )

        assert np.allclose(grad_custom, grad_torch_result, atol=1e-10)


class TestConv2D:
    def test_conv2d_forward(self):
        """Test Conv2D forward pass"""

        batch_size, in_channels, height, width = 2, 3, 4, 4
        out_channels, kernel_size = 2, 3

        x = Tensor(np.random.randn(batch_size, in_channels, height, width))
        weight = Tensor(np.random.randn(out_channels, in_channels, kernel_size, kernel_size))

        output_custom = Conv2dOp.forward(x, weight, stride=1, padding=0)

        x_torch = torch.tensor(x.data, dtype=torch.float32)
        weight_torch = torch.tensor(weight.data, dtype=torch.float32)
        conv_torch = torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=False)
        conv_torch.weight.data = weight_torch
        output_torch = conv_torch(x_torch).detach().numpy()

        assert np.allclose(output_custom, output_torch, atol=1e-5)

    def test_conv2d_forward_with_bias(self):
        """Test Conv2D forward pass with bias"""

        batch_size, in_channels, height, width = 2, 3, 4, 4
        out_channels, kernel_size = 2, 3

        x = Tensor(np.random.randn(batch_size, in_channels, height, width))
        weight = Tensor(np.random.randn(out_channels, in_channels, kernel_size, kernel_size))
        bias = Tensor(np.random.randn(out_channels))

        output_custom = Conv2dOp.forward(x, weight, stride=1, padding=0, bias=bias)

        x_torch = torch.tensor(x.data, dtype=torch.float32)
        weight_torch = torch.tensor(weight.data, dtype=torch.float32)
        bias_torch = torch.tensor(bias.data, dtype=torch.float32)
        conv_torch = torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=True)
        conv_torch.weight.data = weight_torch
        conv_torch.bias.data = bias_torch
        output_torch = conv_torch(x_torch).detach().numpy()

        assert np.allclose(output_custom, output_torch, atol=1e-5)

    def test_conv2d_backward(self):
        """Test Conv2D backward pass"""

        batch_size, in_channels, height, width = 2, 3, 4, 4
        out_channels, kernel_size = 2, 3

        x = Tensor(np.random.randn(batch_size, in_channels, height, width))
        weight = Tensor(np.random.randn(out_channels, in_channels, kernel_size, kernel_size))

        output_custom = Conv2dOp.forward(x, weight, stride=1, padding=0)

        out_grad = Tensor(np.ones_like(output_custom))

        grad_x, grad_weight, grad_bias = Conv2dOp.backward(out_grad, x, weight, stride=1, padding=0)

        x_torch = torch.tensor(x.data, dtype=torch.float32, requires_grad=True)
        weight_torch = torch.tensor(weight.data, dtype=torch.float32, requires_grad=True)
        conv_torch = torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=False)
        conv_torch.weight.data = weight_torch
        output_torch = conv_torch(x_torch)
        output_torch.backward(torch.ones_like(output_torch))

        assert np.allclose(grad_x, x_torch.grad.numpy(), atol=1e-5)
        assert np.allclose(grad_weight, conv_torch.weight.grad.numpy(), atol=1e-5)
        assert grad_bias is None

    def test_conv2d_backward_with_bias(self):
        """Test Conv2D backward pass with bias"""

        batch_size, in_channels, height, width = 2, 3, 4, 4
        out_channels, kernel_size = 2, 3

        x = Tensor(np.random.randn(batch_size, in_channels, height, width))
        weight = Tensor(np.random.randn(out_channels, in_channels, kernel_size, kernel_size))
        bias = Tensor(np.random.randn(out_channels))

        output_custom = Conv2dOp.forward(x, weight, stride=1, padding=0, bias=bias)

        out_grad = Tensor(np.ones_like(output_custom))

        grad_x, grad_weight, grad_bias = Conv2dOp.backward(out_grad, x, weight, stride=1, padding=0, bias=bias)

        x_torch = torch.tensor(x.data, dtype=torch.float32, requires_grad=True)
        weight_torch = torch.tensor(weight.data, dtype=torch.float32, requires_grad=True)
        bias_torch = torch.tensor(bias.data, dtype=torch.float32, requires_grad=True)
        conv_torch = torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=True)
        conv_torch.weight.data = weight_torch
        conv_torch.bias.data = bias_torch
        output_torch = conv_torch(x_torch)
        output_torch.backward(torch.ones_like(output_torch))

        assert np.allclose(grad_x, x_torch.grad.numpy(), atol=1e-5)
        assert np.allclose(grad_weight, conv_torch.weight.grad.numpy(), atol=1e-5)
        assert np.allclose(grad_bias, conv_torch.bias.grad.numpy(), atol=1e-5)

    def test_conv2d_autograd(self):

        batch_size, in_channels, height, width = 2, 3, 4, 4
        out_channels, kernel_size = 2, 3

        x = Tensor(np.random.randn(batch_size, in_channels, height, width), requires_grad=True)
        weight = Tensor(np.random.randn(out_channels, in_channels, kernel_size, kernel_size), requires_grad=True)

        output_tensor = x.conv2d(weight, stride=1, padding=0)
        output_tensor.backward()

        # PyTorch reference
        x_torch = torch.tensor(x.data, dtype=torch.float32, requires_grad=True)
        weight_torch = torch.tensor(weight.data, dtype=torch.float32, requires_grad=True)
        conv_torch = torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=False)
        conv_torch.weight.data = weight_torch
        output_torch = conv_torch(x_torch)
        output_torch.backward(torch.ones_like(output_torch))

        assert np.allclose(output_tensor.data, output_torch.detach().numpy(), atol=1e-5)
        assert np.allclose(x.grad.data, x_torch.grad.numpy(), atol=1e-5)
        assert np.allclose(weight.grad.data, conv_torch.weight.grad.numpy(), atol=1e-5)
