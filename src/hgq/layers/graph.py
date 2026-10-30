"""Graph convolution over a fixed graph.

:class:`QGraphConv` is a GCN layer (Kipf & Welling) whose graph is a
constant: the propagation matrix ``P`` (``n_dst x n_src``, e.g. the
normalised ``A + I`` of a detector-cell graph) is fixed when the layer is
created and never trained. That is what makes the layer synthesisable as a
static circuit: aggregation becomes a constant sparse matrix-vector product
(adders only when the coefficients are powers of two).

For inputs ``x`` of shape ``(batch, n_src, c_in)`` the layer computes, with
``aggregate_first=False``::

    m = mq(x_q @ W_q)
    y = P @ m + node_bias

or, with ``aggregate_first=True`` (cheaper when ``c_in`` is small, e.g. a
single energy per cell)::

    m = mq(P @ x_q)
    y = m @ W_q + node_bias

followed by the activation and the output quantizer. ``mq`` is a trainable
data-lane quantizer on the intermediate messages, so the bit width of every
message, and with it the size of the adder trees, is learned by HGQ.

``node_features`` (``n_src x s``, constant) are optional static per-node
descriptors (layer, radius, cell size, ...). They enter linearly, through a
trainable kernel ``V``, as ``P @ (S @ V)``; that is a per-node constant, so it
is folded together with the bias into ``node_bias`` and costs no multiplier
on the FPGA.

``P`` may be rectangular: choosing a subset of rows as destinations lets the
last layer of a network compute only the nodes whose outputs are needed.
"""

from __future__ import annotations

import numpy as np
from keras import activations, initializers, ops

from ..quantizer import Quantizer
from ..quantizer.config import QuantizerConfig
from .core.base import QLayerBaseSingleInput


def _as_matrix(value, name: str) -> np.ndarray:
    if isinstance(value, dict):  # serialised COO form, see _coo
        m = np.zeros(tuple(value['shape']), dtype=np.float64)
        m[np.asarray(value['row'], dtype=np.int64), np.asarray(value['col'], dtype=np.int64)] = np.asarray(value['val'])
        return m
    m = np.asarray(value, dtype=np.float64)
    if m.ndim != 2:
        raise ValueError(f'{name} must be a 2D matrix, got shape {m.shape}')
    return m


def _coo(m: np.ndarray) -> dict:
    r, c = np.nonzero(m)
    return {'shape': list(m.shape), 'row': r.tolist(), 'col': c.tolist(), 'val': m[r, c].tolist()}


def coefficient_bits(m: np.ndarray) -> np.ndarray:
    """Effective bit width of every constant coefficient (0 for zeros).

    A coefficient ``q * 2**e`` with odd integer ``q`` costs as many bits as
    ``q`` has: a power of two is a free shift (1 bit), ``0.375 = 3 / 8``
    needs 2 and ``0.3125 = 5 / 16`` needs 3.
    """
    m = np.abs(np.asarray(m, dtype=np.float64))
    bits = np.zeros(m.shape, dtype=np.float64)
    nz = m > 0
    q = m[nz] / 2.0 ** np.floor(np.log2(m[nz]))  # in [1, 2)
    k = np.zeros_like(q)
    for _ in range(53):
        frac = q != np.floor(q)
        if not np.any(frac):
            break
        q = np.where(frac, q * 2, q)
        k = k + frac
    bits[nz] = k + 1
    return bits


class QGraphConv(QLayerBaseSingleInput):
    """Quantized graph convolution with a constant propagation matrix.

    Parameters
    ----------
    units
        Number of output features per node.
    propagation
        Constant ``(n_dst, n_src)`` propagation matrix ``P``. Its entries
        should be exact fixed-point numbers (the firmware uses them as they
        are); powers of two give an adder-only aggregation.
    activation
        Activation applied after aggregation (e.g. ``'relu'``).
    use_bias
        Add a trainable bias (per unit, or per node and unit when
        ``node_features`` are given).
    aggregate_first
        Aggregate the inputs before the feature transform (``P x W``) instead
        of after (``P (x W)``). Cheaper when ``c_in < units``.
    node_features
        Optional constant ``(n_src, s)`` static node descriptors.
    kq_conf, bq_conf, nq_conf, mq_conf, iq_conf
        Quantizer configurations of the kernel, the (node) bias, the node
        feature kernel, the messages and the input.
    """

    def __init__(
        self,
        units: int,
        propagation,
        activation=None,
        use_bias: bool = True,
        aggregate_first: bool = False,
        node_features=None,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        node_kernel_initializer='glorot_uniform',
        kq_conf: QuantizerConfig | None = None,
        bq_conf: QuantizerConfig | None = None,
        nq_conf: QuantizerConfig | None = None,
        mq_conf: QuantizerConfig | None = None,
        iq_conf: QuantizerConfig | None = None,
        **kwargs,
    ):
        super().__init__(iq_conf=iq_conf, **kwargs)
        self.units = int(units)
        self.propagation = _as_matrix(propagation, 'propagation')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.aggregate_first = aggregate_first
        self.node_features = None if node_features is None else np.asarray(node_features, dtype=np.float64)
        n_dst, n_src = self.propagation.shape
        if self.node_features is not None and self.node_features.shape[0] != n_src:
            raise ValueError(f'node_features has {self.node_features.shape[0]} rows, propagation has {n_src} columns')
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.node_kernel_initializer = initializers.get(node_kernel_initializer)

        self._kq = Quantizer(kq_conf or QuantizerConfig('default', 'weight'), name=f'{self.name}_kq')
        self._mq = Quantizer(mq_conf or QuantizerConfig('default', 'datalane'), name=f'{self.name}_mq')
        has_node_bias = use_bias or self.node_features is not None
        self._bq = Quantizer(bq_conf or QuantizerConfig('default', 'bias'), name=f'{self.name}_bq') if has_node_bias else None
        self._nq = (
            Quantizer(nq_conf or QuantizerConfig('default', 'weight'), name=f'{self.name}_nq')
            if self.node_features is not None
            else None
        )
        self._p_bits = coefficient_bits(self.propagation)

    # ------------------------------------------------------------- properties
    @property
    def kq(self):
        return self._kq

    @property
    def mq(self):
        return self._mq

    @property
    def bq(self):
        return self._bq

    @property
    def nq(self):
        return self._nq

    @property
    def n_dst(self) -> int:
        return self.propagation.shape[0]

    @property
    def n_src(self) -> int:
        return self.propagation.shape[1]

    # ------------------------------------------------------------------ build
    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(f'{self.name}: expected input (batch, n_src, c_in), got {input_shape}')
        if input_shape[1] != self.n_src:
            raise ValueError(f'{self.name}: input has {input_shape[1]} nodes, propagation expects {self.n_src}')
        c_in = int(input_shape[2])
        self.c_in = c_in
        self.kernel = self.add_weight(name='kernel', shape=(c_in, self.units), initializer=self.kernel_initializer)
        self.bias = (
            self.add_weight(name='bias', shape=(self.units,), initializer=self.bias_initializer) if self.use_bias else None
        )
        if self.node_features is not None:
            self.node_kernel = self.add_weight(
                name='node_kernel', shape=(self.node_features.shape[1], self.units), initializer=self.node_kernel_initializer
            )
        else:
            self.node_kernel = None

        self._p = ops.convert_to_tensor(self.propagation, dtype=self.dtype)
        self._s = None if self.node_features is None else ops.convert_to_tensor(self.node_features, dtype=self.dtype)

        self.kq.build((c_in, self.units))
        if self.aggregate_first:
            self.mq.build((None, self.n_dst, c_in))
        else:
            self.mq.build((None, self.n_src, self.units))
        if self.bq is not None:
            self.bq.build(self._node_bias_shape)
        if self.nq is not None:
            self.nq.build((self.node_features.shape[1], self.units))
        super().build(input_shape)

    @property
    def _node_bias_shape(self):
        return (self.n_dst, self.units) if self.node_features is not None else (self.units,)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_dst, self.units)

    # --------------------------------------------------------------- forward
    def _raw_node_bias(self, training=None):
        b = None
        if self.node_kernel is not None:
            assert self.nq is not None and self._s is not None
            static = ops.matmul(self._s, self.nq(self.node_kernel, training=training))  # (n_src, units)
            b = ops.matmul(self._p, static)  # (n_dst, units)
        if self.bias is not None:
            b = self.bias if b is None else b + self.bias
        return b

    def node_bias(self, training=None):
        """The quantized per-node (or per-unit) constant added after aggregation."""
        b = self._raw_node_bias(training)
        if b is None:
            return None
        assert self.bq is not None
        return self.bq(b, training=training)

    @property
    def qkernel(self):
        return self.kq(self.kernel)

    @property
    def qnode_bias(self):
        return self.node_bias()

    def aggregate(self, x):
        """``P @ x`` over the node axis of ``x`` (batch, n_src, c)."""
        return ops.einsum('nm,bmc->bnc', self._p, x)

    def call(self, inputs, training=None):
        if self.enable_iq:
            inputs = self.iq(inputs, training=training)
        qkernel = self.kq(self.kernel, training=training)
        if self.aggregate_first:
            m = self.mq(self.aggregate(inputs), training=training)
            y = ops.matmul(m, qkernel)
        else:
            m = self.mq(ops.matmul(inputs, qkernel), training=training)
            y = self.aggregate(m)
        b = self.node_bias(training=training)
        if b is not None:
            y = y + b
        if self.activation is not None:
            y = self.activation(y)
        return y

    # ----------------------------------------------------------------- ebops
    def _compute_ebops(self, shape):
        p_bits = ops.convert_to_tensor(self._p_bits, dtype=self.dtype)
        bw_ker = self.kq.bits_((self.c_in, self.units))
        bw_inp = self.iq.bits_(shape)[0]  # (n_src, c_in)
        if self.aggregate_first:
            bw_msg = self.mq.bits_((1, self.n_dst, self.c_in))[0]  # (n_dst, c_in)
            ebops = ops.sum(ops.matmul(p_bits, bw_inp)) + ops.sum(ops.matmul(bw_msg, bw_ker))
        else:
            bw_msg = self.mq.bits_((1, self.n_src, self.units))[0]  # (n_src, units)
            ebops = ops.sum(ops.matmul(bw_inp, bw_ker)) + ops.sum(ops.matmul(p_bits, bw_msg))
        if self.bq is not None:
            bw_b = self.bq.bits_(self._node_bias_shape)
            ebops = ebops + ops.mean(bw_b) * ops.cast(self.n_dst * self.units, self.dtype)  # type: ignore
        return ebops

    # ---------------------------------------------------------------- config
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                'units': self.units,
                'propagation': _coo(self.propagation),
                'activation': activations.serialize(self.activation),
                'use_bias': self.use_bias,
                'aggregate_first': self.aggregate_first,
                'node_features': None if self.node_features is None else self.node_features.tolist(),
                'kernel_initializer': initializers.serialize(self.kernel_initializer),
                'bias_initializer': initializers.serialize(self.bias_initializer),
                'node_kernel_initializer': initializers.serialize(self.node_kernel_initializer),
                'kq_conf': self.kq.config,
                'bq_conf': self.bq.config if self.bq is not None else None,
                'nq_conf': self.nq.config if self.nq is not None else None,
                'mq_conf': self.mq.config,
            }
        )
        return config


class QGraphPool(QLayerBaseSingleInput):
    """Constant linear map between two node sets: ``y = P @ iq(x)``.

    Pools node features onto a coarser set of nodes (e.g. detector cells onto
    eta-phi towers, or 2 x 2 towers onto one), or scatters them through any
    other fixed sparse matrix. ``P`` (``n_dst x n_src``) is a constant with
    exact fixed-point entries; there are no trainable weights. The input
    quantizer is trainable, so HGQ can learn how many bits of each input the
    pooling needs.

    Parameters
    ----------
    propagation
        Constant ``(n_dst, n_src)`` matrix.
    iq_conf
        Input quantizer configuration.
    """

    def __init__(self, propagation, iq_conf: QuantizerConfig | None = None, **kwargs):
        super().__init__(iq_conf=iq_conf, **kwargs)
        self.propagation = _as_matrix(propagation, 'propagation')
        self._p_bits = coefficient_bits(self.propagation)

    @property
    def n_dst(self) -> int:
        return self.propagation.shape[0]

    @property
    def n_src(self) -> int:
        return self.propagation.shape[1]

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(f'{self.name}: expected input (batch, n_src, c), got {input_shape}')
        if input_shape[1] != self.n_src:
            raise ValueError(f'{self.name}: input has {input_shape[1]} nodes, propagation expects {self.n_src}')
        self.c = int(input_shape[2])
        self._p = ops.convert_to_tensor(self.propagation, dtype=self.dtype)
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_dst, input_shape[2])

    def call(self, inputs, training=None):
        if self.enable_iq:
            inputs = self.iq(inputs, training=training)
        return ops.einsum('nm,bmc->bnc', self._p, inputs)

    def _compute_ebops(self, shape):
        p_bits = ops.convert_to_tensor(self._p_bits, dtype=self.dtype)
        bw_inp = self.iq.bits_(shape)[0]  # (n_src, c)
        return ops.sum(ops.matmul(p_bits, bw_inp))

    def get_config(self):
        config = super().get_config()
        config['propagation'] = _coo(self.propagation)
        return config


__all__ = ['QGraphConv', 'QGraphPool', 'coefficient_bits']
