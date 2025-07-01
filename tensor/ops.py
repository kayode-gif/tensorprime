import numpy as np
import warnings
from .broadcast_utils import Broadcast
from .utils import flatten_tensor, compute_strides


class AddOp:
    @staticmethod
    def forward(x, y):
        output_shape = Broadcast.check_broadcastable(x, y)
        result = np.empty(output_shape, dtype=x.data.dtype)
        for idx in np.ndindex(output_shape):
            lhs_idx = Broadcast.broadcast_index(idx, x.shape)
            rhs_idx = Broadcast.broadcast_index(idx, y.shape)
            result[idx] = x.data[lhs_idx] + y.data[rhs_idx]
        return result

    @staticmethod
    def backward(out_grad, x, y):
        axes_lhs = Broadcast.get_axes_broadcasted(x.shape, out_grad.shape)
        axes_rhs = Broadcast.get_axes_broadcasted(y.shape, out_grad.shape)
        grad_x = out_grad.sum_axis(axis=axes_lhs, keepdims=True) if axes_lhs else out_grad
        grad_y = out_grad.sum_axis(axis=axes_rhs, keepdims=True) if axes_rhs else out_grad
        return grad_x, grad_y


class SubOp:
    @staticmethod
    def forward(x, y):
        output_shape = Broadcast.check_broadcastable(x, y)
        result = np.empty(output_shape, dtype=x.data.dtype)
        for idx in np.ndindex(output_shape):
            lhs_idx = Broadcast.broadcast_index(idx, x.shape)
            rhs_idx = Broadcast.broadcast_index(idx, y.shape)
            result[idx] = x.data[lhs_idx] - y.data[rhs_idx]
        return result

    @staticmethod
    def backward(out_grad, x, y):
        axes_lhs = Broadcast.get_axes_broadcasted(x.shape, out_grad.shape)
        axes_rhs = Broadcast.get_axes_broadcasted(y.shape, out_grad.shape)
        grad_x = out_grad.sum_axis(axis=axes_lhs, keepdims=True) if axes_lhs else out_grad
        grad_y = out_grad.sum_axis(axis=axes_rhs, keepdims=True) if axes_rhs else out_grad
        return grad_x, grad_y


class MulOp:
    @staticmethod
    def forward(x, y):
        output_shape = Broadcast.check_broadcastable(x, y)
        result = np.empty(output_shape, dtype=x.data.dtype)
        for idx in np.ndindex(output_shape):
            lhs_idx = Broadcast.broadcast_index(idx, x.shape)
            rhs_idx = Broadcast.broadcast_index(idx, y.shape)
            result[idx] = x.data[lhs_idx] * y.data[rhs_idx]
        return result

    @staticmethod
    def backward(out_grad, x, y):
        grad_x = out_grad * y
        grad_y = out_grad * x
        axes_lhs = Broadcast.get_axes_broadcasted(x.shape, out_grad.shape)
        axes_rhs = Broadcast.get_axes_broadcasted(y.shape, out_grad.shape)
        if axes_lhs:
            grad_x = grad_x.sum_axis(axis=axes_lhs, keepdims=True)
        if axes_rhs:
            grad_y = grad_y.sum_axis(axis=axes_rhs, keepdims=True)
        return grad_x, grad_y


class DivOp:
    @staticmethod
    def forward(x, y):
        output_shape = Broadcast.check_broadcastable(x, y)
        result = np.empty(output_shape, dtype=x.data.dtype)
        for idx in np.ndindex(output_shape):
            lhs_idx = Broadcast.broadcast_index(idx, x.shape)
            rhs_idx = Broadcast.broadcast_index(idx, y.shape)
            result[idx] = x.data[lhs_idx] / y.data[rhs_idx]
        return result

    @staticmethod
    def backward(out_grad, x, y):
        grad_x = (out_grad / y)
        grad_y = (((-x) / (y * y)) * out_grad)
        axes_lhs = Broadcast.get_axes_broadcasted(x.shape, out_grad.shape)
        axes_rhs = Broadcast.get_axes_broadcasted(y.shape, out_grad.shape)
        if axes_lhs:
            grad_x = grad_x.sum_axis(axis=axes_lhs, keepdims=True)
        if axes_rhs:
            grad_y = grad_y.sum_axis(axis=axes_rhs, keepdims=True)
        return grad_x, grad_y


class PowOp:
    @staticmethod
    def forward(x, y):
        output_shape = Broadcast.check_broadcastable(x, y)
        result = np.empty(output_shape, dtype=x.data.dtype)
        for idx in np.ndindex(output_shape):
            lhs_idx = Broadcast.broadcast_index(idx, x.shape)
            rhs_idx = Broadcast.broadcast_index(idx, y.shape)
            base = x.data[lhs_idx]
            exponent = y.data[rhs_idx]
            if base < 0 and exponent % 1 != 0:
                warnings.warn("Negative base with fractional exponent may produce NaNs", RuntimeWarning)
            result[idx] = base ** exponent
        return result

    @staticmethod
    def backward(out_grad, x, y):
        grad_x = ((y * (x ** (y - 1))) * out_grad)
        grad_y = (((x ** y) * x.log()) * out_grad)
        axes_lhs = Broadcast.get_axes_broadcasted(x.shape, out_grad.shape)
        axes_rhs = Broadcast.get_axes_broadcasted(y.shape, out_grad.shape)
        if axes_lhs:
            grad_x = grad_x.sum_axis(axis=axes_lhs, keepdims=True)
        if axes_rhs:
            grad_y = grad_y.sum_axis(axis=axes_rhs, keepdims=True)
        return grad_x, grad_y


class NegOp:
    @staticmethod
    def forward(x):
        return -x.data

    @staticmethod
    def backward(out_grad, x, y):
        return -out_grad


class TanhOp:
    @staticmethod
    def forward(x):
        return np.tanh(x.data)

    @staticmethod
    def backward(out_grad, x, y):
        from tensor.tensor import Tensor
        # y is the output tensor, out = tanh(x)
        # The gradient is (1 - tanh(x)^2) = (1 - y^2)
        return (Tensor(1) - y ** 2) * out_grad


class ExpOp:
    @staticmethod
    def forward(x):
        return np.exp(x.data)

    @staticmethod
    def backward(out_grad, x, y):
        grad = out_grad.data * y.data
        return grad


class LogOp:
    @staticmethod
    def forward(x):
        safe_data = x.data.copy()
        safe_data[safe_data <= 0] = 1e-10  # Replace zeros with a small value to avoid log(0)
        return np.log(safe_data)

    @staticmethod
    def backward(out_grad, x, y):
        # The gradient of log(x) is 1/x
        return out_grad / x


class ReluOp:
    @staticmethod
    def forward(x):
        return x.data * (x.data > 0)

    @staticmethod
    def backward(out_grad, x, y):
        from tensor.tensor import Tensor
        # The gradient is 1 if x > 0, else 0
        return out_grad * Tensor(x.data > 0)


class SigmoidOp:
    @staticmethod
    def forward(x):
        return 1 / (1 + np.exp(-x.data))

    @staticmethod
    def backward(out_grad, x, y):
        from tensor.tensor import Tensor
        # y = sigmoid(x)
        return out_grad * y * (Tensor(1) - y)


class DotOp:
    @staticmethod
    def forward(x, y):
        if x.data.ndim != 1 or y.data.ndim != 1:
            raise ValueError("dot() only supports 1D tensors")

        if x.shape[0] != y.shape[0]:
            raise ValueError(f"Incompatible shapes for dot(): {x.shape} and {y.shape}")

        dot_sum = 0.0
        for i, j in zip(x.data, y.data):
            dot_sum += (i * j)
        return np.array(dot_sum)

    @staticmethod
    def backward(out_grad, x, y):
        grad_x = out_grad.data * y.data
        grad_y = out_grad.data * x.data
        return grad_x, grad_y


class MatMulOp:
    @staticmethod
    def forward(x, y):
        if x.data.ndim != 2 or y.data.ndim != 2:
            raise ValueError("matmul() only supports 2D tensors currently")

        if x.shape[-1] != y.shape[-2]:
            raise ValueError(f"Incompatible shapes for matrix multiplication: {x.shape} and {y.shape}")

        m, k = x.shape
        _, n = y.shape
        matrix = np.zeros((m, n))

        for i in range(m):
            for j in range(n):
                for t in range(k):
                    matrix[i, j] += x.data[i, t] * y.data[t, j]

        return matrix

    @staticmethod
    def backward(out_grad, x, y):
        grad_x = out_grad @ y.transpose((1, 0))
        grad_y = x.transpose((1, 0)) @ out_grad
        return grad_x, grad_y


class BatchMatMulOp:

    @staticmethod
    def _transpose_last_two_dims(shape):
        _dims = list(range(len(shape)))
        _dims[-2], _dims[-1] = _dims[-1], _dims[-2]
        return _dims

    @staticmethod
    def forward(x, y):
        if x.shape[-1] != y.shape[-2]:
            raise ValueError(f"Incompatible shapes for matrix multiplication: {x.shape} and {y.shape}")

        matrix_a_batch_dims = np.empty(x.shape[:-2])
        matrix_b_batch_dims = np.empty(y.shape[:-2])
        batch_shape = Broadcast.undirectional_broadcast_shape(matrix_a_batch_dims, matrix_b_batch_dims)
        if batch_shape is None:
            raise ValueError(f"Incompatible batch shapes: {x.shape[:-2]} and {y.shape[:-2]}")

        m, n = x.shape[-2:]
        _, p = y.shape[-2:]

        output_shape = batch_shape + (m, p)
        result = np.zeros(output_shape, dtype=x.data.dtype)

        for idx in np.ndindex(batch_shape):
            a_idx = Broadcast.broadcast_index(idx, x.shape[:-2])
            b_idx = Broadcast.broadcast_index(idx, y.shape[:-2])

            A_mat = x.data[a_idx]
            B_mat = y.data[b_idx]

            # use @ (efficiency :) )
            result[idx] = A_mat @ B_mat

        return result

    @staticmethod
    def backward(out_grad, x, y):
        dims_other = BatchMatMulOp._transpose_last_two_dims(y.shape)
        grad_x = out_grad.batch_matmul(y.transpose(dims_other))

        if x.shape != grad_x.shape:
            axes = Broadcast.get_axes_broadcasted(x.shape, grad_x.shape)
            if axes is not None:
                grad_x = grad_x.sum_axis(axis=axes, keepdims=True)

        dims_self = BatchMatMulOp._transpose_last_two_dims(x.shape)
        grad_y = x.transpose(dims_self).batch_matmul(out_grad)

        if y.shape != grad_y.shape:
            axes = Broadcast.get_axes_broadcasted(y.shape, grad_y.shape)
            if axes is not None:
                grad_y = grad_y.sum_axis(axis=axes, keepdims=True)

        return grad_x.data, grad_y.data


class TransposeOp:
    @staticmethod
    def forward(x, dims):
        input_shape = x.shape
        input_rank = len(input_shape)

        if len(dims) != input_rank:
            raise ValueError(f"Permutation length {len(dims)} must match input rank {input_rank}")

        output_shape = tuple(input_shape[d] for d in dims)
        transpose_matrix = np.zeros(output_shape, dtype=x.data.dtype)

        for new_index in np.ndindex(*output_shape):
            og_indexes = [0] * input_rank
            for i, dim in enumerate(dims):
                og_indexes[dim] = new_index[i]
            transpose_matrix[new_index] = x.data[tuple(og_indexes)]
        return transpose_matrix

    @staticmethod
    def backward(out_grad, dims):
        input_rank = len(dims)
        inv = tuple(dims.index(i) for i in range(input_rank))
        return out_grad.transpose(inv)


class ReshapeOp:
    @staticmethod
    def forward(x, new_shape):
        og_size = np.prod(x.shape)
        result_size = np.prod(new_shape)
        if og_size != result_size:
            raise ValueError(
                f"Cannot reshape array of size {og_size} into shape {new_shape}"
            )

        flatten_data = flatten_tensor(x.data)
        strides = compute_strides(new_shape)
        output = np.zeros(new_shape, dtype=x.data.dtype)
        for flat_index in range(len(flatten_data)):
            temp_index = flat_index
            multi_index = []
            for strides_k in strides:
                index = temp_index // strides_k
                temp_index %= strides_k
                multi_index.append(index)
            output[tuple(multi_index)] = flatten_data[flat_index]
        return output

    @staticmethod
    def backward(out_grad, x, new_shape):
        return out_grad.reshape(x.shape)


class ExpandDimsOp:
    @staticmethod
    def forward(x, axis):
        if isinstance(axis, int):
            axis = (axis,)
        else:
            axis = tuple(axis)
            # axis represents the shape index position to insert a 1
        total_len = len(x.shape) + len(axis)
        output_shape = []
        shape_pointer = 0
        for i in range(total_len):
            if i in axis:
                output_shape.append(1)
            else:
                output_shape.append(x.shape[shape_pointer])
                shape_pointer += 1

        # convert back to tuple
        new_shape = tuple(output_shape)
        # call .reshape
        out = x.reshape(new_shape)
        out._op = 'expand_dims'
        out._prev = (x,)
        out.requires_grad = x.requires_grad
        return out

    @staticmethod
    def backward(out_grad, x):
        return out_grad.reshape(x.shape)


class SumAxisOp:
    @staticmethod
    def forward(x, axis=None, keepdims=False):
        # case: axis undefined (we summ)
        if axis is None:
            total = 0
            for val in flatten_tensor(x.data):
                total += val
            return total

        if isinstance(axis, int):
            axis = (axis,)
        elif not isinstance(axis, (list, tuple)):
            axis = tuple(axis)

        axis = tuple(a if a >= 0 else a + len(x.shape) for a in axis)

        # we define an output shape depending on dims
        # if keepdims reduce to singleton
        # else removed from output shape (skip)
        output_shape = []
        for index, dim in enumerate(x.shape):
            if index in axis:
                if keepdims:
                    output_shape.append(1)
            else:
                output_shape.append(dim)

        output_data = np.zeros(output_shape, dtype=x.data.dtype)

        # loop over the original shape
        # map the input indexes to the output data shape as such
        for index in np.ndindex(x.shape):
            output_indexes = []
            for i, ind in enumerate(index):
                if i not in axis:
                    output_indexes.append(ind)
                elif keepdims:
                    output_indexes.append(0)
            output_data[tuple(output_indexes)] += x.data[index]
        return output_data

    @staticmethod
    def backward(out_grad, x, axis=None, keepdims=False):
        if axis is None:
            result = np.zeros(x.shape, dtype=x.data.dtype)
            fill_value = out_grad.data
            for i in np.ndindex(x.shape):
                result[i] = fill_value
            return result

        if isinstance(axis, int):
            axis = (axis,)
        elif not isinstance(axis, (list, tuple)):
            axis = tuple(axis)

        axis = tuple(a if a >= 0 else a + len(x.shape) for a in axis)

        grad = out_grad
        if not keepdims:
            for ax in sorted(axis):
                grad = grad.expand_dims(ax)

        # (sanity - that broadcasting is possible)
        Broadcast.check_broadcastable(grad, x)
        return Broadcast.broadcast_to(grad.data, x.shape)



class MaxPool2dOp:
    @staticmethod
    def forward(x, kernel_size, stride=None, padding=0, dilation=1):
        if padding > 0:
            raise NotImplementedError

        batch_size, channels, m_height, m_width = x.data.shape

        if isinstance(kernel_size, int):
            kx = ky = kernel_size
        else:
            kx, ky = kernel_size
        if stride is None:
            sx, sy = kx, ky
        elif isinstance(stride, int):
            sx = sy = stride
        else:
            sx, sy = stride

        h_out = ((m_height + 2 * padding - dilation * (ky - 1) - 1) // sy) + 1
        w_out = ((m_width + 2 * padding - dilation * (kx - 1) - 1) // sx) + 1

        output = np.zeros((batch_size, channels, h_out, w_out), dtype=x.data.dtype)

        for b in range(batch_size):
            for c in range(channels):
                for k in range(0, m_height - ky + 1, sy):
                    for l in range(0, m_width - kx + 1, sx):
                        block_max_value = float("-inf")
                        for i in range(k, k + ky):
                            for j in range(l, l + kx):
                                block_max_value = max(block_max_value, x.data[b, c, i, j])
                        output[b, c, k // sy, l // sx] = block_max_value

        if output.size == 1:
            output = output.item()
        return output

    @staticmethod
    def backward(out_grad, x, kernel_size, stride=None, padding=0, dilation=1):
        #
        if padding > 0:
            raise NotImplementedError

        batch_size, channels, m_height, m_width = x.data.shape

        if isinstance(kernel_size, int):
            kx = ky = kernel_size
        else:
            kx, ky = kernel_size
        if stride is None:
            sx, sy = kx, ky
        elif isinstance(stride, int):
            sx = sy = stride
        else:
            sx, sy = stride

        if isinstance(out_grad.data, (int, float)) or out_grad.data.ndim == 0:
            h_out = ((m_height + 2 * padding - dilation * (ky - 1) - 1) // sy) + 1
            w_out = ((m_width + 2 * padding - dilation * (kx - 1) - 1) // sx) + 1
            out_grad_data = np.full((batch_size, channels, h_out, w_out), out_grad.data)
        else:
            out_grad_data = out_grad.data

        grad_x = np.zeros_like(x.data)

        for b in range(batch_size):
            for c in range(channels):
                for k in range(0, m_height - ky + 1, sy):
                    for l in range(0, m_width - kx + 1, sx):
                        block_max_value = float("-inf")
                        max_i, max_j = k, l
                        for i in range(k, k + ky):
                            for j in range(l, l + kx):
                                if x.data[b, c, i, j] > block_max_value:
                                    block_max_value = x.data[b, c, i, j]
                                    max_i, max_j = i, j

                        out_row = k // sy
                        out_col = l // sx
                        grad_x[b, c, max_i, max_j] += out_grad_data[b, c, out_row, out_col]

        return grad_x


class Conv2dOp:
    @staticmethod
    def forward(x, weight, stride=1, padding=0, dilation=1, bias=None):
        if padding > 0:
            raise NotImplementedError

        batch_size, in_channels, m_height, m_width = x.data.shape
        out_channels, in_channels_kernel, ky, kx = weight.data.shape
        if in_channels != in_channels_kernel:
            raise ValueError(f"Input channels {in_channels} does not match kernel input channels {in_channels_kernel}")

        if isinstance(stride, int):
            sy = sx = stride
        else:
            sy, sx = stride

        h_out = ((m_height + 2 * padding - dilation * (ky - 1) - 1) // sy) + 1
        w_out = ((m_width + 2 * padding - dilation * (kx - 1) - 1) // sx) + 1

        output = np.zeros((batch_size, out_channels, h_out, w_out), dtype=x.data.dtype)

        for n in range(batch_size):
            for c in range(out_channels):
                for h in range(h_out):
                    for w in range(w_out):
                        conv_value = 0.0
                        h_start = h * sy
                        w_start = w * sx
                        for i in range(ky):
                            for j in range(kx):
                                if h_start + i < m_height and w_start + j < m_width:
                                    for ic in range(in_channels):
                                        conv_value += x.data[n, ic, h_start + i, w_start + j] * weight.data[c, ic, i, j]
                        output[n, c, h, w] = conv_value

                        if bias is not None:
                            output[n, c, h, w] += bias.data[c]

        return output

    @staticmethod
    def backward(out_grad, x, weight, stride=1, padding=0, dilation=1, bias=None):
        if padding > 0:
            raise NotImplementedError

        batch_size, in_channels, m_height, m_width = x.data.shape
        out_channels, in_channels_kernel, ky, kx = weight.data.shape

        if isinstance(stride, int):
            sy = sx = stride
        else:
            sy, sx = stride

        h_out = ((m_height + 2 * padding - dilation * (ky - 1) - 1) // sy) + 1
        w_out = ((m_width + 2 * padding - dilation * (kx - 1) - 1) // sx) + 1

        grad_x = np.zeros_like(x.data)
        grad_weight = np.zeros_like(weight.data)
        grad_bias = np.zeros_like(bias.data) if bias is not None else None

        for n in range(batch_size):
            for c in range(out_channels):
                for h in range(h_out):
                    for w in range(w_out):
                        h_start = h * sy
                        w_start = w * sx

                        # input gradient calc
                        for i in range(ky):
                            for j in range(kx):
                                if h_start + i < m_height and w_start + j < m_width:
                                    for ic in range(in_channels):
                                        grad_x[n, ic, h_start + i, w_start + j] += (
                                                weight.data[c, ic, i, j] * out_grad.data[n, c, h, w]
                                        )
                        # kernel gradient calc
                        for i in range(ky):
                            for j in range(kx):
                                if h_start + i < m_height and w_start + j < m_width:
                                    for ic in range(in_channels):
                                        grad_weight[c, ic, i, j] += (
                                                x.data[n, ic, h_start + i, w_start + j] * out_grad.data[n, c, h, w]
                                        )
                        if bias is not None:
                            grad_bias[c] += out_grad.data[n, c, h, w]

        return grad_x, grad_weight, grad_bias
