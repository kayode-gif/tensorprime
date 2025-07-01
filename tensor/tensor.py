import numpy as np
from .ops import *


class TensorBase:
    def __init__(self, data):
        self.data = np.array(data, dtype=np.float64)
        self.shape = self.data.shape

    @staticmethod
    def validate_operand(other):
        return other if isinstance(other, TensorBase) else Tensor(other, requires_grad=False)


class Tensor(TensorBase):
    def __init__(self, data, _children=(), _op='', requires_grad=False):
        super().__init__(data)
        self.requires_grad = requires_grad
        self._prev = set(_children)
        self._op = _op
        self.grad = None if not requires_grad else Tensor(np.zeros_like(self.data, dtype=np.float64))
        self._backward = lambda: None

    # patch issue (fix)
    # this resets the gradients to false after computation for some ops
    # add this for mul, div and power
    def detach(self):
        return Tensor(self.data, requires_grad=False)

    def __repr__(self):
        return f"Tensor(data = {self.data}, requires_grad = {self.requires_grad})"

    def __add__(self, other):
        other = TensorBase.validate_operand(other)
        result = AddOp.forward(self, other)
        out = Tensor(result, (self, other), _op='add', requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            grad_x, grad_y = AddOp.backward(out.grad, self, other)
            if self.requires_grad:
                self.grad = self.grad + grad_x
            if other.requires_grad:
                other.grad = other.grad + grad_y

        out._backward = _backward
        return out

    def __sub__(self, other):
        other = TensorBase.validate_operand(other)
        result = SubOp.forward(self, other)
        out = Tensor(result, (self, other), _op='sub', requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            grad_x, grad_y = SubOp.backward(out.grad, self, other)
            if self.requires_grad:
                self.grad = self.grad + grad_x
            if other.requires_grad:
                other.grad = other.grad - grad_y

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = TensorBase.validate_operand(other)
        result = MulOp.forward(self, other)
        out = Tensor(result, (self, other), _op='mul', requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            grad_x, grad_y = MulOp.backward(out.grad, self, other)
            if self.requires_grad:
                self.grad = self.grad + grad_x
            if other.requires_grad:
                other.grad = other.grad + grad_y

        out._backward = _backward
        return out

    def __truediv__(self, other):
        other = TensorBase.validate_operand(other)
        result = DivOp.forward(self, other)
        out = Tensor(result, (self, other), _op='div', requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            grad_x, grad_y = DivOp.backward(out.grad, self, other)
            if self.requires_grad:
                self.grad = self.grad + grad_x
            if other.requires_grad:
                other.grad = other.grad + grad_y

        out._backward = _backward
        return out

    def __pow__(self, other):
        other = TensorBase.validate_operand(other)
        result = PowOp.forward(self, other)
        out = Tensor(result, (self, other), _op='pow', requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            grad_x, grad_y = PowOp.backward(out.grad, self, other)
            if self.requires_grad:
                self.grad = self.grad + grad_x
            if other.requires_grad:
                other.grad = other.grad + grad_y

        out._backward = _backward
        return out

    def __neg__(self):
        result = NegOp.forward(self)
        out = Tensor(result, (self,), _op='neg', requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + NegOp.backward(out.grad, self, out)

        out._backward = _backward
        return out

    def tanh(self):
        result = TanhOp.forward(self)
        out = Tensor(result, (self,), _op='tanh', requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + TanhOp.backward(out.grad, self, out)

        out._backward = _backward
        return out

    def exp(self):
        result = ExpOp.forward(self)
        out = Tensor(result, (self,), _op='exp', requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + Tensor(ExpOp.backward(out.grad, self, out))

        out._backward = _backward
        return out

    def log(self):
        result = LogOp.forward(self)
        out = Tensor(result, (self,), _op='log', requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + LogOp.backward(out.grad, self, out)

        out._backward = _backward
        return out

    def relu(self):
        result = ReluOp.forward(self)
        out = Tensor(result, (self,), _op='relu', requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + ReluOp.backward(out.grad, self, out)

        out._backward = _backward
        return out

    def sigmoid(self):
        result = SigmoidOp.forward(self)
        out = Tensor(result, (self,), _op='sigmoid', requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + SigmoidOp.backward(out.grad, self, out)

        out._backward = _backward
        return out

    def dot(self, other):
        other = TensorBase.validate_operand(other)
        result = DotOp.forward(self, other)
        out = Tensor(result, (self, other), _op='dot', requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            grad_x, grad_y = DotOp.backward(out.grad, self, other)
            if self.requires_grad:
                self.grad = self.grad + Tensor(grad_x)
            if other.requires_grad:
                other.grad = other.grad + Tensor(grad_y)

        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = TensorBase.validate_operand(other)
        result = MatMulOp.forward(self, other)
        out = Tensor(result, (self, other), _op='matmul', requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            grad_x, grad_y = MatMulOp.backward(out.grad, self, other)
            if self.requires_grad:
                self.grad = self.grad + grad_x
            if other.requires_grad:
                other.grad = other.grad + grad_y

        out._backward = _backward
        return out

    def batch_matmul(self, other):
        other = TensorBase.validate_operand(other)
        result = BatchMatMulOp.forward(self, other)
        out = Tensor(result, (self, other), _op='batch_matmul', requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            grad_x, grad_y = BatchMatMulOp.backward(out.grad, self, other)
            if self.requires_grad:
                self.grad = self.grad + Tensor(grad_x)
            if other.requires_grad:
                other.grad = other.grad + Tensor(grad_y)

        out._backward = _backward
        return out

    def transpose(self, dims):
        result = TransposeOp.forward(self, dims)
        out = Tensor(result, (self,), _op='transpose', requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + TransposeOp.backward(out.grad, dims)

        out._backward = _backward
        return out

    def reshape(self, new_shape):
        result = ReshapeOp.forward(self, new_shape).data
        out = Tensor(result, (self,), _op='reshape', requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + ReshapeOp.backward(out.grad, self, new_shape)

        out._backward = _backward
        return out

    # torch.unsqueeze (basically)
    def expand_dims(self, axis):
        result = ExpandDimsOp.forward(self, axis).data
        out = Tensor(result, (self,), _op='expand_dims', requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + ExpandDimsOp.backward(out.grad, self)

        out._backward = _backward
        return out

    def sum_axis(self, axis=None, keepdims=False):
        result = SumAxisOp.forward(self, axis, keepdims)
        out = Tensor(result, (self,), _op='sum_axis', requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + Tensor(SumAxisOp.backward(out.grad, self, axis, keepdims))

        out._backward = _backward
        return out


    def maxpool2d(self, kernel_size, stride=None, padding=0, dilation=1):
        result = MaxPool2dOp.forward(self, kernel_size, stride, padding, dilation)
        out = Tensor(result, (self,), _op='max_pool2d', requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + Tensor(
                    MaxPool2dOp.backward(out.grad, self, kernel_size, stride=None, padding=0, dilation=1))

        out._backward = _backward
        return out

    def conv2d(self, weight, stride=1, padding=0, dilation=1, bias=None):
        result = Conv2dOp.forward(self, weight, stride, padding, dilation, bias)
        children = (self, weight)
        if bias is not None:
            children += (bias,)
        out = Tensor(result, children, _op='conv2d', requires_grad=self.requires_grad or weight.requires_grad or (bias and bias.requires_grad))

        def _backward():
            grad_x, grad_weight, grad_bias = Conv2dOp.backward(out.grad, self, weight, stride, padding, dilation, bias)
            if self.requires_grad:
                self.grad = self.grad + Tensor(grad_x)
            if weight.requires_grad:
                weight.grad = weight.grad + Tensor(grad_weight)
            if bias and bias.requires_grad:
                bias.grad = bias.grad + Tensor(grad_bias)

        out._backward = _backward
        return out

    def backward(self):
        if not self.requires_grad:
            return
        result = []
        visited = set()
        if self.grad is not None:
            self.grad = Tensor(np.ones_like(self.data, dtype=np.float64))

        def dfs(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    dfs(child)
                result.append(node)

        dfs(self)
        for node in reversed(result):
            node._backward()
