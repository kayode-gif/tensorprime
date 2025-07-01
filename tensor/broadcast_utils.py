import numpy as np
class Broadcast:
    @staticmethod
    def undirectional_broadcast_shape(lhs, rhs):
        """Determine broadcasted shape between two tensor-like objects."""
        if lhs.shape == rhs.shape:
            return lhs.shape

        lhs_len = len(lhs.shape)
        rhs_len = len(rhs.shape)
        max_len = max(lhs_len, rhs_len)

        lhs_padded = [1] * (max_len - lhs_len) + list(lhs.shape)
        rhs_padded = [1] * (max_len - rhs_len) + list(rhs.shape)

        res_shape = []
        for dl, dr in zip(lhs_padded, rhs_padded):
            if dl == dr or dr == 1 or dl == 1:
                res_shape.append(max(dl, dr))
            else:
                return None
        return tuple(res_shape)

    @staticmethod
    def check_broadcastable(lhs, rhs):
        """Verify broadcasted shape between two tensor-like objects."""
        broadcastable = Broadcast.undirectional_broadcast_shape(lhs, rhs)
        if broadcastable is None:
            raise ValueError(f"Incompatible shapes for broadcasting: {lhs.shape} and {rhs.shape}")
        return broadcastable

    @staticmethod
    def broadcast_index(idx, shape):
        """Map multi-dimensional index to broadcasted shape index."""
        offset = len(idx) - len(shape)
        idx_mapped = []
        for i, s in enumerate(shape):
            idx_mapped.append(idx[offset + i] if s > 1 else 0)
        return tuple(idx_mapped)

    @staticmethod
    def get_axes_broadcasted(original_shape, output_shape):
        """Identify which axes were broadcasted in the operation."""
        og_padded = [1] * (len(output_shape) - len(original_shape)) + list(original_shape)
        return tuple(
            i for i, (p, o) in enumerate(zip(og_padded, output_shape)) if p == 1 and o > 1
        ) or None

    @staticmethod
    def broadcast_to(input_data, output_shape):
        """broadcast an input tensor-like object to a desired output shape."""
        output_data = np.empty(output_shape, dtype=input_data.dtype)
        for index in np.ndindex(output_shape):
            input_index = Broadcast.broadcast_index(index, input_data.shape)
            output_data[index] = input_data[input_index]
        return output_data

