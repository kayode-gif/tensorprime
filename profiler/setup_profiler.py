from .profiler import Profiler
from tensor.ops import (
    AddOp, SubOp, MulOp, DivOp, PowOp, NegOp, TanhOp, ExpOp, LogOp, ReluOp, SigmoidOp,
    DotOp, MatMulOp, BatchMatMulOp, TransposeOp, ReshapeOp, ExpandDimsOp, SumAxisOp,
    MaxPool2dOp, Conv2dOp
)


def setup_profiler(profile_memory=True):
    profiler = Profiler(profile_memory=profile_memory)
    op_classes = [
        AddOp, SubOp, MulOp, DivOp, PowOp, NegOp, TanhOp, ExpOp, LogOp, ReluOp, SigmoidOp,
        DotOp, MatMulOp, BatchMatMulOp, TransposeOp, ReshapeOp, ExpandDimsOp, SumAxisOp,
        MaxPool2dOp, Conv2dOp
    ]
    for op in op_classes:
        if hasattr(op, 'forward'):
            op.forward = profiler.profile_op(op.forward)
        if hasattr(op, 'backward'):
            op.backward = profiler.profile_op(op.backward)
    return profiler
