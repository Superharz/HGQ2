import keras
import numpy as np
import pytest
from keras import ops

from hgq.config import QuantizerConfigScope
from hgq.layers import QGraphConv, QGraphPool
from hgq.layers.graph import coefficient_bits

from .base import LayerTestBase

N_NODES = 12


def _graph(normalisation: str) -> np.ndarray:
    rng = np.random.default_rng(7)
    a = np.triu((rng.random((N_NODES, N_NODES)) < 0.25).astype(float), 1)
    a = a + a.T + np.eye(N_NODES)
    deg = a.sum(1)
    if normalisation == 'pow2_mean':
        return a * (2.0 ** -np.ceil(np.log2(deg)))[:, None]
    d = 1 / np.sqrt(deg)
    return np.round(d[:, None] * a * d[None, :] * 64) / 64  # 'sym', 6 fractional bits


class TestGraphConv(LayerTestBase):
    layer_cls = QGraphConv
    hls4ml_not_supported = True  # alkaid only

    @pytest.fixture(params=['pow2_mean', 'sym'])
    def propagation(self, request):
        return _graph(request.param)

    @pytest.fixture(params=[False, True])
    def aggregate_first(self, request):
        return request.param

    @pytest.fixture(params=[False, True])
    def with_node_features(self, request):
        return request.param

    @pytest.fixture(params=[None, 5])
    def n_dst(self, request):
        return request.param

    @pytest.fixture(params=[(N_NODES, 3)])
    def input_shapes(self, request):
        return request.param

    @pytest.fixture
    def layer_kwargs(self, propagation, aggregate_first, with_node_features, n_dst):
        p = propagation if n_dst is None else propagation[:n_dst]
        s = np.round(np.random.default_rng(3).random((N_NODES, 2)) * 16) / 16 if with_node_features else None
        return {
            'units': 4,
            'propagation': p,
            'activation': 'relu',
            'aggregate_first': aggregate_first,
            'node_features': s,
        }

    @pytest.fixture(params=[True])
    def use_parallel_io(self, request) -> bool:
        return request.param

    def test_behavior(self, layer_kwargs):
        """With dummy quantizers the layer is a plain GCN: relu(P x W + P S V + b)."""
        with QuantizerConfigScope(default_q_type='dummy'):
            layer = QGraphConv(**layer_kwargs, enable_ebops=False)
        x = np.random.randn(16, N_NODES, 3).astype(np.float32)
        layer.build(x.shape)
        layer.bias.assign(np.random.randn(4).astype(np.float32))
        p = layer.propagation
        w = ops.convert_to_numpy(layer.kernel)
        ref = np.einsum('nm,bmc->bnc', p, x @ w) + ops.convert_to_numpy(layer.bias)
        if layer.node_features is not None:
            ref = ref + p @ (layer.node_features @ ops.convert_to_numpy(layer.node_kernel))
        ref = np.maximum(ref, 0)
        out = ops.convert_to_numpy(layer(x))
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)

    def test_output_shape(self, layer_kwargs):
        layer = QGraphConv(**layer_kwargs)
        out = layer(np.zeros((2, N_NODES, 3), dtype=np.float32))
        assert tuple(out.shape) == (2, layer_kwargs['propagation'].shape[0], 4)


class TestGraphPool(LayerTestBase):
    layer_cls = QGraphPool
    hls4ml_not_supported = True  # alkaid only

    @pytest.fixture(params=['sum', 'fractions'])
    def propagation(self, request):
        rng = np.random.default_rng(11)
        p = np.zeros((5, N_NODES))
        for j in range(N_NODES):  # every source node feeds one or two destinations
            dst = rng.choice(5, rng.integers(1, 3), replace=False)
            p[dst, j] = 1.0 if request.param == 'sum' else rng.choice([0.5, 0.375, 0.125, 1.0], len(dst))
        return p

    @pytest.fixture(params=[(N_NODES, 3)])
    def input_shapes(self, request):
        return request.param

    @pytest.fixture
    def layer_kwargs(self, propagation):
        return {'propagation': propagation}

    @pytest.fixture(params=[True])
    def use_parallel_io(self, request) -> bool:
        return request.param

    def test_behavior(self, layer_kwargs):
        with QuantizerConfigScope(default_q_type='dummy'):
            layer = QGraphPool(**layer_kwargs, enable_ebops=False)
        x = np.random.randn(16, N_NODES, 3).astype(np.float32)
        out = ops.convert_to_numpy(layer(x))
        np.testing.assert_allclose(out, np.einsum('nm,bmc->bnc', layer.propagation, x), rtol=1e-5, atol=1e-5)

    def test_training(self, *args, **kwargs):
        pytest.skip('QGraphPool has no trainable weights besides its input quantizer')


def test_coefficient_bits():
    bits = coefficient_bits(np.array([0.0, 1.0, 0.25, -0.5, 0.375, 0.3125]))
    np.testing.assert_array_equal(bits, [0, 1, 1, 1, 2, 3])


def test_wrong_node_count():
    layer = QGraphConv(2, np.eye(4))
    with pytest.raises(ValueError):
        layer.build((None, 5, 1))


def test_ebops_counts_aggregation():
    """A pow2 propagation matrix costs one message bit width per non-zero."""
    p = _graph('pow2_mean')
    with QuantizerConfigScope(place='all', default_q_type='kif', i0=2, f0=2, trainable=False):
        layer = QGraphConv(2, p, use_bias=False)
    x = np.random.randn(8, N_NODES, 1).astype(np.float32)
    layer(x, training=True)
    ebops_full = float(ops.convert_to_numpy(layer.ebops))
    with QuantizerConfigScope(place='all', default_q_type='kif', i0=2, f0=2, trainable=False):
        dense_only = QGraphConv(2, np.eye(N_NODES), use_bias=False)
    dense_only(x, training=True)
    assert ebops_full > float(ops.convert_to_numpy(dense_only.ebops))
    _ = keras  # imported for parity with other test modules
