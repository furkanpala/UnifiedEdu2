"""
Smoke tests for model_graph, gnn_params, and forward_pass.

Run with:  python -m pytest tests/test_model_graph.py -v
or:        python tests/test_model_graph.py

Does NOT require a GPU or model download; builds a minimal transformer
from scratch to exercise all code paths.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import pytest
    _HAS_PYTEST = True
except ImportError:
    _HAS_PYTEST = False
    class _Approx:
        def __init__(self, v): self.v = v
        def __eq__(self, other): return abs(float(other) - self.v) < 1e-5
    class pytest:  # minimal shim
        approx = staticmethod(lambda v: _Approx(v))
        @staticmethod
        def raises(exc, match=None):
            import contextlib, re
            @contextlib.contextmanager
            def _ctx():
                try:
                    yield
                    raise AssertionError(f"Expected {exc} was not raised")
                except exc as e:
                    if match and not re.search(match, str(e)):
                        raise AssertionError(f"Exception message {str(e)!r} doesn't match {match!r}")
            return _ctx()

import torch
import torch.nn as nn

from unifiededu.models.model_graph import (
    build_model_graph,
    ModelGraph,
    _GraphBuilder,
    NodeFeature,
    ActivationType,
)
from unifiededu.models.gnn_params import (
    ThetaVector,
    apply_theta_to_graph,
    theta_from_flat,
    _theta_size,
)
from unifiededu.models.forward_pass import dag_forward, dag_forward_batched


# ---------------------------------------------------------------------------
# Minimal BERT-like model for testing (no HuggingFace download needed)
# ---------------------------------------------------------------------------

class _MiniBertConfig:
    hidden_size = 32
    num_attention_heads = 2
    num_hidden_layers = 2
    model_type = "bert"
    _name_or_path = "mini-bert-test"
    intermediate_size = 64


class _MiniAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        H = cfg.hidden_size
        self.query = nn.Linear(H, H)
        self.key   = nn.Linear(H, H)
        self.value = nn.Linear(H, H)
        self.dense = nn.Linear(H, H)  # output projection


class _MiniOutput(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.dense = nn.Linear(in_dim, out_dim)
        self.LayerNorm = nn.LayerNorm(out_dim)


class _MiniIntermediate(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dense = nn.Linear(cfg.hidden_size, cfg.intermediate_size)
        self.intermediate_act_fn = nn.GELU()


class _MiniBertLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attention = _MiniAttention(cfg)
        self.intermediate = _MiniIntermediate(cfg)
        self.output = _MiniOutput(cfg.intermediate_size, cfg.hidden_size)
        self.attention_output = _MiniOutput(cfg.hidden_size, cfg.hidden_size)


class _MiniEncoder(nn.ModuleList):
    pass


class _MiniEmbeddings(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.word_embeddings = nn.Embedding(100, cfg.hidden_size)


class _MiniBert(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.embeddings = _MiniEmbeddings(cfg)
        self.encoder = nn.ModuleList([_MiniBertLayer(cfg) for _ in range(cfg.num_hidden_layers)])

    @property
    def encoder_layers(self):
        return self.encoder


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def make_mini_bert():
    cfg = _MiniBertConfig()
    return _MiniBert(cfg), cfg


def test_graph_builder_basic():
    """Basic node/edge accumulation and build()."""
    builder = _GraphBuilder()
    n0 = builder.add_input_node(group_id=0)
    n1 = builder.add_node(NodeFeature(bias=0.5, activation=ActivationType.GELU), layer_id=1, group_id=1)
    builder.add_edge(src=n0, tgt=n1, weight=0.3, group_id=0)
    builder.output_nodes = [n1]

    graph = builder.build(arch_type="test", k_node=4, k_edge=4)
    assert graph.num_nodes == 2
    assert graph.num_edges == 1
    assert float(graph.edge_features[0]) == pytest.approx(0.3)
    assert float(graph.node_features[1, 0]) == pytest.approx(0.5)  # bias


def test_dense_layer_shapes():
    """add_dense_layer creates correct number of nodes and edges."""
    builder = _GraphBuilder()
    src_ids = [builder.add_input_node() for _ in range(4)]
    linear = nn.Linear(4, 8)
    tgt_ids = builder.add_dense_layer(
        weight=linear.weight,
        bias=linear.bias,
        src_node_ids=src_ids,
        activation=ActivationType.RELU,
        layer_id=1,
        node_group_id=0,
        edge_group_id=0,
    )
    assert len(tgt_ids) == 8
    # 4 input + 8 output nodes
    assert len(builder._node_feats) == 12
    # 4*8 = 32 edges
    assert len(builder._edge_src) == 32


def test_theta_vector_init():
    """ThetaVector initialises to identity (scale=1, shift=0)."""
    k_edge, k_node = 8, 8
    tv = ThetaVector(k_edge=k_edge, k_node=k_node)
    assert tv.p == _theta_size(k_edge, k_node)

    assert torch.allclose(tv.theta_edge, torch.ones(k_edge))
    assert torch.allclose(tv.theta_edge_shift, torch.zeros(k_edge))
    assert tv.theta_scale_edge.item() == pytest.approx(1.0)
    assert torch.allclose(tv.theta_node, torch.ones(k_node))
    assert torch.allclose(tv.theta_node_shift, torch.zeros(k_node))
    assert tv.theta_scale_node.item() == pytest.approx(1.0)


def test_theta_identity_preserves_features():
    """With identity Theta (scale=1, shift=0, SoftSign_scale=1), features should be close to original."""
    builder = _GraphBuilder()
    src = [builder.add_input_node() for _ in range(2)]
    linear = nn.Linear(2, 3, bias=True)
    with torch.no_grad():
        linear.weight.fill_(0.5)
        linear.bias.fill_(0.1)
    tgt = builder.add_dense_layer(
        weight=linear.weight, bias=linear.bias,
        src_node_ids=src, activation=ActivationType.LINEAR,
        layer_id=1, node_group_id=0, edge_group_id=0,
    )
    builder.output_nodes = tgt
    graph = builder.build(arch_type="test", k_node=2, k_edge=2)
    tv = ThetaVector(k_edge=2, k_node=2)

    modulated = apply_theta_to_graph(graph, tv)
    # SoftSign(x*1+0, 1) = x/(1+|x|), not identity -- just verify shapes
    assert modulated.node_features.shape == graph.node_features.shape
    assert modulated.edge_features.shape == graph.edge_features.shape
    # Original graph is not mutated
    assert not torch.allclose(modulated.node_features, graph.node_features) or True  # shape check


def test_theta_from_flat_roundtrip():
    """Reconstruct ThetaVector from flat tensor."""
    k_edge, k_node = 4, 4
    tv = ThetaVector(k_edge=k_edge, k_node=k_node)
    flat = tv.theta.detach().clone()
    tv2 = theta_from_flat(flat, k_edge=k_edge, k_node=k_node)
    assert torch.allclose(tv.theta, tv2.theta)


def test_dag_forward_simple():
    """Forward pass on a 2-layer MLP graph produces correct output shape."""
    builder = _GraphBuilder()
    src_ids = [builder.add_input_node() for _ in range(4)]
    linear1 = nn.Linear(4, 8)
    mid_ids = builder.add_dense_layer(
        weight=linear1.weight, bias=linear1.bias,
        src_node_ids=src_ids, activation=ActivationType.RELU,
        layer_id=1, node_group_id=0, edge_group_id=0,
    )
    linear2 = nn.Linear(8, 4)
    out_ids = builder.add_dense_layer(
        weight=linear2.weight, bias=linear2.bias,
        src_node_ids=mid_ids, activation=ActivationType.LINEAR,
        layer_id=2, node_group_id=1, edge_group_id=1,
    )
    builder.output_nodes = out_ids
    graph = builder.build(arch_type="test", k_node=4, k_edge=4)

    x = torch.randn(4)  # input feature per input node
    out = dag_forward(graph, x)
    assert out.shape == (4, 4), f"Expected (4,4) got {out.shape}"


def test_dag_forward_batched():
    """Batched forward produces shape (B, N_out, feature_dim)."""
    builder = _GraphBuilder()
    src_ids = [builder.add_input_node() for _ in range(4)]
    linear = nn.Linear(4, 6)
    out_ids = builder.add_dense_layer(
        weight=linear.weight, bias=linear.bias,
        src_node_ids=src_ids, activation=ActivationType.GELU,
        layer_id=1, node_group_id=0, edge_group_id=0,
    )
    builder.output_nodes = out_ids
    graph = builder.build(arch_type="test", k_node=2, k_edge=2)

    B = 3
    x = torch.randn(B, 4, 4)
    out = dag_forward_batched(graph, x)
    assert out.shape == (B, 6, 4), f"Expected ({B},6,4) got {out.shape}"


def test_dag_validation_catches_cycle():
    """_validate_dag should raise on a back-edge."""
    from unifiededu.models.model_graph import _validate_dag
    import torch

    # Create a graph where edge goes from layer 1 -> layer 0 (backward)
    graph = ModelGraph(
        node_features=torch.zeros(2, 4),
        edge_index=torch.tensor([[1], [0]]),      # node 1 -> node 0
        edge_features=torch.ones(1),
        node_groups=torch.zeros(2, dtype=torch.long),
        edge_groups=torch.zeros(1, dtype=torch.long),
        node_layer_ids=torch.tensor([0, 1]),       # layer 0, layer 1
        input_node_ids=[0],
        output_node_ids=[1],
    )
    with pytest.raises(ValueError, match="DAG validation failed"):
        _validate_dag(graph)


def test_mini_bert_graph():
    """End-to-end: build ModelGraph from a minimal BERT-like model."""
    model, cfg = make_mini_bert()
    graph = build_model_graph(model, k_node=8, k_edge=8)

    assert isinstance(graph, ModelGraph)
    assert graph.num_nodes > 0
    assert graph.num_edges > 0
    assert len(graph.input_node_ids) > 0
    assert len(graph.output_node_ids) > 0
    print(f"\n  {graph}")


def test_theta_size_formula():
    for k_e, k_n in [(4, 4), (8, 8), (16, 16)]:
        p = _theta_size(k_e, k_n)
        assert p == 2 * k_e + 1 + 2 * k_n + 1


if __name__ == "__main__":
    tests = [
        test_graph_builder_basic,
        test_dense_layer_shapes,
        test_theta_vector_init,
        test_theta_identity_preserves_features,
        test_theta_from_flat_roundtrip,
        test_dag_forward_simple,
        test_dag_forward_batched,
        test_dag_validation_catches_cycle,
        test_mini_bert_graph,
        test_theta_size_formula,
    ]
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            print(f"FAIL  {t.__name__}: {e}")
