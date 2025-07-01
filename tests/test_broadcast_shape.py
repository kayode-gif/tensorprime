from tensor.tensor import Tensor
from tensor.broadcast_utils import Broadcast

def test_broadcast_shape():
    a = Tensor([[1, 2, 3],
                [4, 5, 6]])
    b = Tensor([[7],
                [8]])
    c = Tensor([10, 20, 30])
    d = Tensor(5)

    assert Broadcast.undirectional_broadcast_shape(a, b) == (2, 3)
    assert Broadcast.undirectional_broadcast_shape(b, c) == (2, 3)
    assert Broadcast.undirectional_broadcast_shape(c, d) == (3,)
    assert Broadcast.undirectional_broadcast_shape(d, a) == (2, 3)


def test_broadcast_shape_fail():
    a = Tensor([1, 2, 4, 4])
    b = Tensor([1, 2])

    c = Tensor([[1, 2], [3, 4], [5, 6]])
    d = Tensor([[1, 2, 3], [4, 5, 6]])
    assert Broadcast.undirectional_broadcast_shape(a, b) is None
    assert Broadcast.undirectional_broadcast_shape(c, d) is None


