"""
sanity_check.py  --  Overfit one (context, question, answer) sample with Theta.

Verifies the full training pipeline works end-to-end before running federation:
  1. Builds a single batch from one hard-coded sample.
  2. Trains Theta on that batch for --steps iterations.
  3. Generates a QA pair from the same context and prints it alongside the target.

If Theta can learn to reproduce the target after enough steps, the pipeline is correct.

Usage:
    python sanity_check.py --model distilgpt2 --steps 500 --device cuda
    python sanity_check.py --model distilgpt2 --steps 500 --device cpu
    python sanity_check.py --model distilgpt2 --steps 500 --rank 16 --device cuda
    python sanity_check.py --model distilgpt2 --steps 500 --gnn --device cuda
"""

from __future__ import annotations

import argparse
import logging

import torch
import torch.nn as nn
from torch.func import functional_call
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModelForCausalLM, AutoTokenizer

from unifiededu.federation.client import (
    assign_layer_groups,
    get_layer_shapes,
    modulate_params,
    _functional_generate,
)
from unifiededu.models.gnn_params import ThetaVector, ThetaGNN
from unifiededu.data.dataset import _tokenize_sample

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

SAMPLE = {
    "sample_id": "sanity",
    "context":   "The mitochondria is the powerhouse of the cell. "
                 "It generates ATP through a process called oxidative phosphorylation, "
                 "which occurs in the inner mitochondrial membrane.",
    "question":  "What process does the mitochondria use to generate ATP?",
    "answer":    "Oxidative phosphorylation, which occurs in the inner mitochondrial membrane.",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",          default="distilgpt2")
    p.add_argument("--steps",          type=int,   default=500)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--rank",           type=int,   default=8,
                   help="LoRA rank (ignored when --gnn is set)")
    p.add_argument("--k",              type=int,   default=32,
                   help="k_edge = k_node for layer-group assignment")
    p.add_argument("--gnn",            action="store_true",
                   help="Use ThetaGNN instead of LoRA ThetaVector")
    p.add_argument("--gnn-hidden",     type=int,   default=64)
    p.add_argument("--max-new-tokens", type=int,   default=40)
    p.add_argument("--device",         default="cpu")
    args = p.parse_args()

    device = args.device

    # ------------------------------------------------------------------
    # Load model (frozen backbone)
    # ------------------------------------------------------------------
    log.info("Loading %s ...", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()

    # ------------------------------------------------------------------
    # Build single-sample batch
    # ------------------------------------------------------------------
    item  = _tokenize_sample(SAMPLE, tokenizer, max_length=512)
    batch = {key: val.unsqueeze(0).to(device) for key, val in item.items()}

    n_active = (batch["labels"] != -100).sum().item()
    log.info(
        "Batch: seq_len=%d  active_label_tokens=%d",
        batch["input_ids"].shape[1], n_active,
    )
    if n_active == 0:
        log.error("No active label tokens — check tokenisation.")
        return

    # ------------------------------------------------------------------
    # Layer groups and shapes
    # ------------------------------------------------------------------
    layer_groups = assign_layer_groups(model, args.k, args.k)
    layer_shapes = get_layer_shapes(model, layer_groups)

    # ------------------------------------------------------------------
    # Theta + optimiser
    # ------------------------------------------------------------------
    if args.gnn:
        theta = ThetaGNN(args.gnn_hidden).to(device)
        log.info("Using ThetaGNN  p=%d  hidden=%d", theta.p, args.gnn_hidden)
    else:
        theta = ThetaVector(layer_shapes, lora_rank=args.rank).to(device)
        log.info(
            "Using LoRA ThetaVector  rank=%d  n_layers=%d  p=%d",
            args.rank, len(layer_shapes), theta.p,
        )

    optimizer = AdamW(theta.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.05)

    log.info("Overfitting for %d steps (lr=%.0e, cosine) ...", args.steps, args.lr)

    # ------------------------------------------------------------------
    # Overfit loop
    # ------------------------------------------------------------------
    for step in range(1, args.steps + 1):
        theta.train()
        modulated = modulate_params(model, theta, layer_groups)
        outputs   = functional_call(model, modulated, args=(), kwargs=batch, strict=False)

        if hasattr(outputs, "loss") and outputs.loss is not None:
            loss = outputs.loss
        else:
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            loss   = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                batch["labels"].view(-1),
                ignore_index=-100,
            )

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(theta.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 50 == 0 or step == 1:
            log.info("Step %4d  loss=%.4f  lr=%.2e",
                     step, loss.item(), scheduler.get_last_lr()[0])

    # ------------------------------------------------------------------
    # Generate with the trained Theta
    # ------------------------------------------------------------------
    log.info("\n--- Generation after overfitting ---")
    theta.eval()
    modulated = modulate_params(model, theta, layer_groups)
    ctx       = SAMPLE["context"]
    question  = _functional_generate(
        model, tokenizer,
        f"Context: {ctx}\n\nQuestion:",
        modulated, args.max_new_tokens, device,
    )
    answer = _functional_generate(
        model, tokenizer,
        f"Context: {ctx}\n\nQuestion: {question}\nAnswer:",
        modulated, args.max_new_tokens, device,
    )

    log.info("Context      : %s", ctx)
    log.info("")
    log.info("Generated Q  : %s", question)
    log.info("Target    Q  : %s", SAMPLE["question"])
    log.info("")
    log.info("Generated A  : %s", answer)
    log.info("Target    A  : %s", SAMPLE["answer"])


if __name__ == "__main__":
    main()
