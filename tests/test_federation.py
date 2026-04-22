"""
Integration tests for federation/client.py, server.py, fedavg.py.

Uses a tiny 2-layer MLP as the 'backbone' so no HuggingFace download
is needed, and a synthetic DataLoader that yields fixed batches.

Run: /c/Users/fp223/AppData/Local/anaconda3/envs/ML/python.exe tests/test_federation.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Iterator

from unifiededu.models.gnn_params import ThetaVector, _theta_size
from unifiededu.federation.client import (
    FederationClient,
    assign_layer_groups,
    modulate_params,
)
from unifiededu.federation.server import FederationServer
from unifiededu.federation.fedavg import FedAvgServer
from unifiededu.config import FederationConfig, ModelGraphConfig, TrainingConfig


def _assert(cond, msg=""):
    if not cond:
        raise AssertionError(msg)


# ---------------------------------------------------------------------------
# Minimal stub backbone (two linear layers with an LM-style loss output)
# ---------------------------------------------------------------------------

class _Output:
    def __init__(self, loss, logits):
        self.loss   = loss
        self.logits = logits


class _TinyLM(nn.Module):
    """
    Minimal causal-LM-style model: embed -> linear -> logits.
    Accepts input_ids and labels; returns _Output with .loss.
    """
    VOCAB = 32

    def __init__(self, hidden=16):
        super().__init__()
        self.embed  = nn.Embedding(self.VOCAB, hidden)
        self.fc1    = nn.Linear(hidden, hidden)
        self.fc2    = nn.Linear(hidden, self.VOCAB)

    def forward(self, input_ids, attention_mask=None, labels=None, **_):
        h = self.embed(input_ids)       # (B, T, H)
        h = torch.relu(self.fc1(h))     # (B, T, H)
        logits = self.fc2(h)            # (B, T, V)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.VOCAB),
                labels.view(-1),
                ignore_index=-100,
            )
        return _Output(loss, logits)


# ---------------------------------------------------------------------------
# Synthetic DataLoader
# ---------------------------------------------------------------------------

class _SyntheticLoader:
    """Yields `num_batches` fixed batches of shape (batch_size, seq_len)."""

    def __init__(self, num_batches=4, batch_size=2, seq_len=8):
        self.num_batches = num_batches
        self.batch_size  = batch_size
        self.seq_len     = seq_len

    def __iter__(self) -> Iterator[dict]:
        for _ in range(self.num_batches):
            ids = torch.randint(0, _TinyLM.VOCAB, (self.batch_size, self.seq_len))
            yield {
                "input_ids":      ids,
                "attention_mask": torch.ones_like(ids),
                "labels":         ids.clone(),
            }


# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------

K_EDGE, K_NODE = 4, 4
P = _theta_size(K_EDGE, K_NODE)

FED   = FederationConfig(num_clients=3, t_ic=2, t_bc=6, T_init=4, t_update=4)
MG    = ModelGraphConfig(k_edge=K_EDGE, k_node=K_NODE)
TRAIN = TrainingConfig(lr=1e-3, batch_size=2, gradient_accumulation_steps=2)


def _make_client(cid=0):
    model  = _TinyLM()
    loader = _SyntheticLoader(num_batches=4)
    return FederationClient(
        client_id=cid,
        model=model,
        dataloader=loader,
        k_edge=K_EDGE,
        k_node=K_NODE,
        fed_config=FED,
        train_config=TRAIN,
        device="cpu",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_assign_layer_groups_deterministic():
    m1, m2 = _TinyLM(), _TinyLM()
    g1 = assign_layer_groups(m1, K_EDGE, K_NODE)
    g2 = assign_layer_groups(m2, K_EDGE, K_NODE)
    _assert(g1 == g2, "Layer group assignment is not deterministic")
    # fc1 and fc2 should be in the groups dict
    _assert(any("fc1" in k for k in g1), "fc1 not found in groups")
    _assert(any("fc2" in k for k in g1), "fc2 not found in groups")


def test_assign_layer_groups_range():
    m = _TinyLM()
    groups = assign_layer_groups(m, K_EDGE, K_NODE)
    for name, (eg, ng) in groups.items():
        _assert(0 <= eg < K_EDGE, f"{name}: edge group {eg} out of range")
        _assert(0 <= ng < K_NODE, f"{name}: node group {ng} out of range")


def test_modulate_params_shapes():
    model  = _TinyLM()
    theta  = ThetaVector(K_EDGE, K_NODE)
    groups = assign_layer_groups(model, K_EDGE, K_NODE)
    params = modulate_params(model, theta, groups)

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name in groups:
            key_w = name + ".weight"
            _assert(key_w in params, f"{key_w} missing from modulated params")
            _assert(params[key_w].shape == module.weight.shape,
                    f"{key_w}: shape mismatch {params[key_w].shape} vs {module.weight.shape}")


def test_modulate_params_requires_grad_through_theta():
    """Gradient must flow from loss to theta through modulated params."""
    model  = _TinyLM()
    theta  = ThetaVector(K_EDGE, K_NODE)
    groups = assign_layer_groups(model, K_EDGE, K_NODE)

    # Modulated params should be in the theta computation graph
    params = modulate_params(model, theta, groups)
    dummy_loss = sum(p.sum() for p in params.values())
    dummy_loss.backward()
    _assert(theta.theta.grad is not None, "No gradient in theta")
    _assert(theta.theta.grad.abs().sum() > 0, "Zero gradient in theta")


def test_backbone_is_frozen_after_client_init():
    client = _make_client()
    for p in client.model.parameters():
        _assert(not p.requires_grad, "Backbone parameter should be frozen")


def test_local_train_returns_correct_shape():
    client    = _make_client()
    theta_in  = ThetaVector(K_EDGE, K_NODE).theta.detach()
    theta_out = client.local_train(theta_in)
    _assert(theta_out.shape == (P,), f"Expected ({P},), got {theta_out.shape}")


def test_local_train_updates_theta():
    """Theta should change after training (loss is non-zero)."""
    client    = _make_client()
    theta_in  = ThetaVector(K_EDGE, K_NODE).theta.detach().clone()
    theta_out = client.local_train(theta_in.clone())
    _assert(not torch.allclose(theta_in, theta_out),
            "Theta unchanged after local_train -- no gradient update?")


def test_server_broadcast_before_first_aggregate():
    server = FederationServer(num_clients=3, fed_config=FED, mg_config=MG)
    broadcast = server.broadcast()
    _assert(len(broadcast) == 3)
    for cid, theta in broadcast.items():
        _assert(theta.shape == (P,), f"Client {cid} theta shape wrong: {theta.shape}")


def test_server_aggregate_intra_cluster():
    """At t_ic=2, round 2 should trigger intra-cluster aggregation."""
    server = FederationServer(num_clients=3, fed_config=FED, mg_config=MG)
    # Round 1 (initialise clustering)
    uploads_r1 = {i: torch.randn(P) for i in range(3)}
    server.aggregate(1, uploads_r1)

    # Round 2 (t_ic=2 -> intra-cluster)
    uploads_r2 = {i: torch.randn(P) for i in range(3)}
    out = server.aggregate(2, uploads_r2)
    _assert(len(out) == 3)
    for theta in out.values():
        _assert(theta.shape == (P,))
    # All clients in the same cluster should receive the same theta
    cluster_result = server.cluster_result
    if cluster_result.num_clusters < 3:
        # At least two clients are in the same cluster -> same theta
        seen: dict = {}
        for inst_idx, cid in enumerate(sorted(out.keys())):
            clu = cluster_result.cluster_of(inst_idx)
            if clu in seen:
                _assert(torch.allclose(out[cid], out[seen[clu]]),
                        f"Same-cluster clients should share theta at t_ic")
            else:
                seen[clu] = cid


def test_server_aggregate_inter_cluster():
    """Round >= T_init and round % t_bc == 0 -> all clients get grand mean."""
    server = FederationServer(num_clients=3, fed_config=FED, mg_config=MG)

    # Run enough rounds to reach inter-cluster aggregation
    for t in range(1, FED.t_bc + 1):  # t_bc=6
        uploads = {i: torch.randn(P) for i in range(3)}
        out = server.aggregate(t, uploads)

    # At round t_bc=6 >= T_init=4: inter-cluster should fire
    thetas = list(out.values())
    _assert(all(torch.allclose(thetas[0], t) for t in thetas),
            "Inter-cluster aggregation should give all clients the same theta")


def test_server_cluster_result_updated():
    """Cluster result should be set after first aggregate call."""
    server = FederationServer(num_clients=4, fed_config=FED, mg_config=MG)
    _assert(server.cluster_result is None, "cluster_result should start None")
    uploads = {i: torch.randn(P) for i in range(4)}
    server.aggregate(1, uploads)
    _assert(server.cluster_result is not None, "cluster_result not set after aggregate")


def test_fedavg_server_all_get_same_theta():
    """FedAvg: every round, all clients receive the same averaged theta."""
    server  = FedAvgServer(num_clients=3, fed_config=FED, mg_config=MG)
    uploads = {i: torch.ones(P) * float(i) for i in range(3)}
    out     = server.aggregate(1, uploads)

    # Expected: mean([0, 1, 2]) * ones = 1.0 * ones
    expected = torch.ones(P)
    for cid, theta in out.items():
        _assert(torch.allclose(theta, expected, atol=1e-5),
                f"Client {cid}: expected mean 1.0, got {theta.mean():.4f}")


def test_fedavg_all_clients_same_theta():
    """All clients should receive identical theta from FedAvg."""
    server  = FedAvgServer(num_clients=4, fed_config=FED, mg_config=MG)
    uploads = {i: torch.randn(P) for i in range(4)}
    out     = server.aggregate(1, uploads)
    thetas  = list(out.values())
    for t in thetas[1:]:
        _assert(torch.allclose(thetas[0], t), "FedAvg: clients should share theta")


def test_full_simulation_2_rounds():
    """End-to-end: 2 rounds of federation with 3 clients, no crash."""
    server  = FederationServer(num_clients=3, fed_config=FED, mg_config=MG)
    clients = [_make_client(i) for i in range(3)]

    for t in range(1, 3):
        thetas_in = server.broadcast()
        uploads   = {i: clients[i].local_train(thetas_in[i]) for i in range(3)}
        out       = server.aggregate(t, uploads)
        for cid, theta in out.items():
            _assert(theta.shape == (P,), f"Round {t}: client {cid} theta wrong shape")


def test_theta_size_transmitted_is_fixed():
    """Verify size of uploaded Theta is always p regardless of backbone."""
    # Two different backbone sizes
    model_a = _TinyLM(hidden=16)
    model_b = _TinyLM(hidden=32)

    clients = [
        FederationClient(0, model_a, _SyntheticLoader(2), K_EDGE, K_NODE, FED, TRAIN),
        FederationClient(1, model_b, _SyntheticLoader(2), K_EDGE, K_NODE, FED, TRAIN),
    ]
    theta_in = ThetaVector(K_EDGE, K_NODE).theta.detach()
    for client in clients:
        out = client.local_train(theta_in.clone())
        _assert(out.shape == (P,), f"Client theta shape is {out.shape}, expected ({P},)")


if __name__ == "__main__":
    tests = [
        test_assign_layer_groups_deterministic,
        test_assign_layer_groups_range,
        test_modulate_params_shapes,
        test_modulate_params_requires_grad_through_theta,
        test_backbone_is_frozen_after_client_init,
        test_local_train_returns_correct_shape,
        test_local_train_updates_theta,
        test_server_broadcast_before_first_aggregate,
        test_server_aggregate_intra_cluster,
        test_server_aggregate_inter_cluster,
        test_server_cluster_result_updated,
        test_fedavg_server_all_get_same_theta,
        test_fedavg_all_clients_same_theta,
        test_full_simulation_2_rounds,
        test_theta_size_transmitted_is_fixed,
    ]
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            import traceback
            print(f"FAIL  {t.__name__}: {e}")
            traceback.print_exc()
