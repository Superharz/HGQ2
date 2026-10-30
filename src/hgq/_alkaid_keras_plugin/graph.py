"""Alkaid replay of :class:`hgq.layers.QGraphConv` and :class:`hgq.layers.QGraphPool`.

The propagation matrix is applied with :func:`alkaid.trace.ops.sparse_matmul`,
i.e. every destination node is built from its neighbours only. The node bias
(bias plus folded static node features) is a constant per node.
"""

import numpy as np
from alkaid.converter.builtin.keras.layers._base import ReplayOperationBase, to_np_arr
from alkaid.trace import FVArray
from alkaid.trace.ops import sparse_matmul

from hgq.layers import QGraphConv, QGraphPool

from ._base import QLayerMixin, mirror_quantizer


class _QGraphConv(QLayerMixin, ReplayOperationBase):
    handles = (QGraphConv,)

    def call(self, inputs: FVArray) -> FVArray:
        op: QGraphConv = self.op
        kernel = to_np_arr(op.qkernel)
        p = op.propagation
        if op.aggregate_first:
            m = mirror_quantizer(op.mq, sparse_matmul(p, inputs, axis=-2))
            y = np.einsum('...nc,cu->...nu', m, kernel)
        else:
            z = np.einsum('...nc,cu->...nu', inputs, kernel)
            m = mirror_quantizer(op.mq, z)
            y = sparse_matmul(p, m, axis=-2)
        b = op.qnode_bias
        if b is not None:
            y = y + to_np_arr(b)
        return y


class _QGraphPool(QLayerMixin, ReplayOperationBase):
    handles = (QGraphPool,)

    def call(self, inputs: FVArray) -> FVArray:
        return sparse_matmul(self.op.propagation, inputs, axis=-2)
