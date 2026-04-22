"""
train.py  --  UnifiedEdu federation training entry point.

Usage
-----
# Quick smoke-run (CPU, tiny mock model, no download needed):
python train.py --method unifiededu --mock --num-rounds 4 --output-dir runs/smoke

# Real run with identical backbone for all clients:
python train.py --method unifiededu \\
    --model distilgpt2 \\
    --data-dir data/processed \\
    --num-rounds 100 \\
    --output-dir runs/exp1

# Per-client backbones (heterogeneous federation):
python train.py --method unifiededu \\
    --mit-model      distilgpt2 \\
    --stanford-model facebook/opt-125m \\
    --papers-model   EleutherAI/pythia-160m \\
    --data-dir data/processed \\
    --output-dir runs/hetero

Methods
-------
  individual   -- each client trains independently, no Theta sharing
  fedavg       -- standard FedAvg (plain average every round)
  static       -- topology-guided static clustering (cluster once at init)
  unifiededu   -- dynamic Theta-guided re-clustering (proposed method)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import transformers
transformers.logging.set_verbosity_error()

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy imports (avoid loading HuggingFace at module level)
# ---------------------------------------------------------------------------

def _load_hf_model(model_name: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log.info("Loading model %s ...", model_name)
    tok   = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return model, tok


def _load_mock_model():
    """Tiny in-memory model for smoke tests (no download needed)."""
    import torch.nn as nn
    # from unifiededu.tests_helpers import _TinyLM, _MockTokenizer  # type: ignore

    # Inline the minimal classes so train.py has no test dependency
    VOCAB = 64

    class TinyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(VOCAB, 32)
            self.fc1   = nn.Linear(32, 32)
            self.fc2   = nn.Linear(32, VOCAB)

        def forward(self, input_ids, attention_mask=None, labels=None, **_):
            h      = self.embed(input_ids.clamp(0, VOCAB - 1))
            h      = torch.relu(self.fc1(h))
            logits = self.fc2(h)
            loss   = None
            if labels is not None:
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, VOCAB), labels.clamp(min=-1).view(-1),
                    ignore_index=-100,
                )
            from types import SimpleNamespace
            return SimpleNamespace(loss=loss, logits=logits)

    class CharTokenizer:
        bos_token_id = 1
        eos_token_id = 2
        pad_token_id = 0
        model_max_length = 512

        def encode(self, text, add_special_tokens=True):
            ids = [ord(c) % (VOCAB - 3) + 3 for c in text]
            if add_special_tokens:
                ids = [self.bos_token_id] + ids
            return ids

        def decode(self, ids, skip_special_tokens=True):
            sp = {self.bos_token_id, self.eos_token_id, self.pad_token_id}
            return "".join(chr((i - 3) % (VOCAB - 3) + 32)
                           for i in ids if not (skip_special_tokens and i in sp))

        def __call__(self, text, max_length=512, truncation=True,
                     padding=False, add_special_tokens=True, **_):
            ids = self.encode(text, add_special_tokens=add_special_tokens)
            if truncation and len(ids) > max_length:
                ids = ids[:max_length]
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    return TinyLM(), CharTokenizer()


# ---------------------------------------------------------------------------
# Client + DataLoader factory
# ---------------------------------------------------------------------------

def _make_client(
    client_id:    int,
    model,
    tokenizer,
    train_samples: list,
    val_samples:   list,
    model_name:    str,
    cfg,
    device:        str,
):
    from unifiededu.data.dataset import make_dataloader
    from unifiededu.federation.client import FederationClient

    train_loader = make_dataloader(
        train_samples,
        tokenizer,
        batch_size=cfg.training.batch_size,
        max_length=cfg.training.max_seq_len,
        shuffle=True,
    )
    val_loader = make_dataloader(
        val_samples,
        tokenizer,
        batch_size=cfg.training.batch_size,
        max_length=cfg.training.max_seq_len,
        shuffle=False,
    ) if val_samples else None

    pool = val_samples or train_samples
    sample_context = pool[0].get("context", "") if pool else None

    log.info(
        "Client %d (%s) — %d train / %d val samples, model: %s",
        client_id, ["mit", "stanford", "papers"][client_id],
        len(train_samples), len(val_samples) if val_samples else 0,
        model_name,
    )

    return FederationClient(
        client_id=client_id,
        model=model,
        dataloader=train_loader,
        val_dataloader=val_loader,
        k_edge=cfg.model_graph.k_edge,
        k_node=cfg.model_graph.k_node,
        fed_config=cfg.federation,
        train_config=cfg.training,
        model_name=model_name,
        tokenizer=tokenizer,
        sample_context=sample_context,
        device=device,
    )


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(output_dir: str, round_num: int, state: dict) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(output_dir, f"round_{round_num:04d}.pt")
    torch.save(state, path)
    # Keep only the 3 most recent checkpoints to save disk space
    existing = sorted(Path(output_dir).glob("round_*.pt"))
    for old in existing[:-3]:
        old.unlink()


def _load_latest_checkpoint(output_dir: str) -> Optional[dict]:
    ckpts = sorted(Path(output_dir).glob("round_*.pt"))
    if not ckpts:
        return None
    state = torch.load(ckpts[-1], map_location="cpu")
    log.info("Resumed from %s", ckpts[-1])
    return state


# ---------------------------------------------------------------------------
# Per-round logging
# ---------------------------------------------------------------------------

def _log_round(
    round_num:  int,
    total_rounds: int,
    uploads:    Dict[int, torch.Tensor],
    cluster_result,
    elapsed_s:  float,
    log_path:   str,
    val_losses: Optional[Dict[int, float]] = None,
) -> None:
    theta_norms = {cid: float(t.norm()) for cid, t in uploads.items()}
    record: dict = {
        "round":        round_num,
        "num_clusters": cluster_result.num_clusters if cluster_result else None,
        "silhouette":   cluster_result.silhouette   if cluster_result else None,
        "theta_norms":  theta_norms,
        "elapsed_s":    elapsed_s,
    }
    if val_losses is not None:
        valid = [v for v in val_losses.values() if not math.isnan(v)]
        record["val_loss_mean"]    = sum(valid) / len(valid) if valid else None
        record["val_losses_per_client"] = {str(k): v for k, v in val_losses.items()}

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    if round_num % 5 == 0 or round_num == 1:
        k   = record["num_clusters"] or "?"
        sil = f"{record['silhouette']:.4f}" if record["silhouette"] else "?"
        val_str = ""
        if record.get("val_loss_mean") is not None:
            val_str = f"  val={record['val_loss_mean']:.4f}"
        log.info(
            "Round %3d/%d  K=%s  sil=%s  |theta|=%.4f%s  %.1fs",
            round_num, total_rounds, k, sil,
            sum(theta_norms.values()) / len(theta_norms),
            val_str,
            elapsed_s,
        )


# ---------------------------------------------------------------------------
# Training methods
# ---------------------------------------------------------------------------

def run_individual(
    clients: list,
    cfg,
    output_dir: str,
    num_rounds: int,
) -> Dict[int, torch.Tensor]:
    """
    Each client trains in isolation for num_rounds epochs.
    No Theta sharing.  Returns final per-client Theta vectors.
    """
    from unifiededu.models.gnn_params import ThetaVector

    log_path  = os.path.join(output_dir, "train_log.jsonl")
    thetas    = {i: ThetaVector(cfg.model_graph.k_edge, cfg.model_graph.k_node
                                ).theta.detach().clone()
                 for i in range(len(clients))}

    # with logging_redirect_tqdm(tqdm_class=tqdm):
    round_pbar = tqdm(range(1, num_rounds + 1), desc="individual", unit="round")
    for t in round_pbar:
        t0      = time.time()
        uploads = {i: clients[i].local_train(thetas[i]) for i in range(len(clients))}
        thetas  = uploads

        val_losses = None
        if t % 5 == 0:
            val_losses = {i: clients[i].compute_val_loss(thetas[i]) for i in range(len(clients))}
            valid = [v for v in val_losses.values() if not math.isnan(v)]
            if valid:
                round_pbar.set_postfix({"val": f"{sum(valid)/len(valid):.4f}"})
            for i, client in enumerate(clients):
                q = client.generate_qa(thetas[i])
                log.info("Round %d  C%d(%s) sample question: %s", t, i, client.model_name, q)

        _log_round(t, num_rounds, uploads, None, time.time() - t0, log_path, val_losses)
        if t % 10 == 0:
            _save_checkpoint(output_dir, t, {"thetas": thetas, "round": t})

    _save_checkpoint(output_dir, num_rounds, {"thetas": thetas, "round": num_rounds})
    return thetas


def _run_federated(
    clients:     list,
    server,
    cfg,
    output_dir:  str,
    num_rounds:  int,
) -> Dict[int, torch.Tensor]:
    """Shared loop for all federated methods (fedavg, static, unifiededu)."""
    log_path = os.path.join(output_dir, "train_log.jsonl")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint if available
    state = _load_latest_checkpoint(output_dir)
    start_round = 1
    if state:
        start_round = state["round"] + 1
        # Restore last per-client thetas as the broadcast for next round
        server._client_thetas = state["thetas"]

    method_name = type(server).__name__
    # with logging_redirect_tqdm(tqdm_class=tqdm):
    round_pbar  = tqdm(range(start_round, num_rounds + 1), desc=method_name, unit="round")
    for t in round_pbar:
        t0        = time.time()
        thetas_in = server.broadcast()
        uploads   = {i: clients[i].local_train(thetas_in[i]) for i in range(len(clients))}
        server.aggregate(t, uploads)

        val_losses = None
        if t % 5 == 0:
            val_losses = {i: clients[i].compute_val_loss(uploads[i]) for i in range(len(clients))}
            valid = [v for v in val_losses.values() if not math.isnan(v)]
            if valid:
                k = server.cluster_result.num_clusters if server.cluster_result else "?"
                round_pbar.set_postfix({"val": f"{sum(valid)/len(valid):.4f}", "K": k})
            for i, client in enumerate(clients):
                q = client.generate_qa(uploads[i])
                log.info("Round %d  C%d(%s) sample question: %s", t, i, client.model_name, q)

        _log_round(t, num_rounds, uploads, server.cluster_result,
                    time.time() - t0, log_path, val_losses)

        if t % 10 == 0:
            _save_checkpoint(output_dir, t,
                                {"thetas": server._client_thetas, "round": t})

    # Final checkpoint
    _save_checkpoint(output_dir, num_rounds,
                     {"thetas": server._client_thetas, "round": num_rounds})
    return server._client_thetas


def run_fedavg(clients, cfg, output_dir, num_rounds):
    from unifiededu.federation.fedavg import FedAvgServer
    server = FedAvgServer(
        num_clients=len(clients),
        fed_config=cfg.federation,
        mg_config=cfg.model_graph,
    )
    return _run_federated(clients, server, cfg, output_dir, num_rounds)


def run_unifiededu(clients, cfg, output_dir, num_rounds):
    from unifiededu.federation.server import FederationServer
    server = FederationServer(
        num_clients=len(clients),
        fed_config=cfg.federation,
        mg_config=cfg.model_graph,
    )
    return _run_federated(clients, server, cfg, output_dir, num_rounds)


def run_static(clients, models, cfg, output_dir, num_rounds):
    """Static clustering: build model-graphs, cluster once by topology."""
    from unifiededu.models.model_graph import build_model_graph
    from unifiededu.federation.server import StaticFederationServer

    log.info("Building model-graphs for topology clustering ...")
    graphs = [
        build_model_graph(m, k_node=cfg.model_graph.k_node,
                          k_edge=cfg.model_graph.k_edge)
        for m in models
    ]
    server = StaticFederationServer(
        graphs=graphs,
        num_clients=len(clients),
        fed_config=cfg.federation,
        mg_config=cfg.model_graph,
    )
    return _run_federated(clients, server, cfg, output_dir, num_rounds)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="UnifiedEdu federation training")
    p.add_argument("--method", choices=["individual", "fedavg", "static", "unifiededu"],
                   default="unifiededu")
    p.add_argument("--data-dir",  default="data/processed",
                   help="Directory containing {client}_{split}.jsonl files")
    p.add_argument("--output-dir", default="runs/default")
    # Backbone models
    p.add_argument("--model", default=None,
                   help="Single HuggingFace model name for all clients")
    p.add_argument("--mit-model",      default=None)
    p.add_argument("--stanford-model", default=None)
    p.add_argument("--papers-model",   default=None)
    p.add_argument("--mock", action="store_true",
                   help="Use a tiny in-memory model (no download, for testing)")
    # Training
    p.add_argument("--num-rounds",   type=int, default=None,
                   help="Override config num_rounds")
    p.add_argument("--local-epochs", type=int, default=None,
                   help="Local SGD epochs per round (E in FedAvg). Default: 1")
    p.add_argument("--device",       default="cpu")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--log-level",  default="INFO")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    torch.manual_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Config
    from unifiededu.config import UnifiedEduConfig, DEFAULT_CONFIG
    cfg = UnifiedEduConfig()
    if args.num_rounds is not None:
        cfg.federation.num_rounds = args.num_rounds
    if args.local_epochs is not None:
        cfg.training.local_epochs = args.local_epochs
    cfg.device = args.device

    # Load processed splits
    from unifiededu.data.preprocessing import load_splits, CLIENT_FILES
    client_names = list(CLIENT_FILES.keys())   # ["mit", "stanford", "papers"]

    if args.mock:
        log.info("Using mock model (--mock flag set)")
        models_and_toks = [_load_mock_model() for _ in client_names]
        per_client = {name: "mock" for name in client_names}
        # Synthetic samples for smoke test
        synthetic_sample = {
            "sample_id": "s0", "context": "Machine learning optimises a loss function.",
            "question":  "What does ML optimise?", "answer": "A loss function.",
            "difficulty": "easy", "bloom_level": "understand", "metadata": {},
        }
        splits = {name: {"train": [synthetic_sample] * 8,
                         "val":   [synthetic_sample] * 2,
                         "test":  [synthetic_sample] * 2}
                  for name in client_names}
    else:
        splits = load_splits(args.data_dir, client_names)

        # Resolve per-client model names
        per_client = {
            "mit":      args.mit_model      or args.model or "distilgpt2",
            "stanford": args.stanford_model or args.model or "facebook/opt-125m",
            "papers":   args.papers_model   or args.model or "EleutherAI/pythia-160m",
        }
        log.info("Per-client backbone models: %s", per_client)
        models_and_toks = [
            _load_hf_model(per_client[name], args.device)
            for name in client_names
        ]

    # Build clients
    clients = []
    models  = []
    for idx, name in enumerate(client_names):
        model, tokenizer = models_and_toks[idx]
        models.append(model)
        clients.append(
            _make_client(
                idx, model, tokenizer,
                splits[name]["train"],
                splits[name].get("val", []),
                per_client[name],
                cfg, args.device,
            )
        )
    log.info("Built %d clients: %s", len(clients), client_names)

    # Save run config
    with open(os.path.join(args.output_dir, "run_config.json"), "w") as f:
        json.dump({
            "method":       args.method,
            "num_rounds":   cfg.federation.num_rounds,
            "local_epochs": cfg.training.local_epochs,
            "clients":      client_names,
            "models":       per_client,
            "device":       args.device,
        }, f, indent=2)

    T = cfg.federation.num_rounds
    log.info("Starting %s training for %d rounds ...", args.method, T)

    if args.method == "individual":
        final_thetas = run_individual(clients, cfg, args.output_dir, T)
    elif args.method == "fedavg":
        final_thetas = run_fedavg(clients, cfg, args.output_dir, T)
    elif args.method == "static":
        final_thetas = run_static(clients, models, cfg, args.output_dir, T)
    else:  # unifiededu
        final_thetas = run_unifiededu(clients, cfg, args.output_dir, T)

    log.info("Training complete. Checkpoints in %s", args.output_dir)
    return final_thetas


if __name__ == "__main__":
    main()
