"""
evaluate.py -- Final evaluation entry point for UnifiedEdu.

Usage
-----
python evaluate.py \\
    --method unifiededu \\
    --checkpoint-dir runs/exp1 \\
    --data-dir data/processed \\
    --output-dir results/exp1 \\
    --device cpu
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List

import torch

log = logging.getLogger(__name__)


def _load_model_and_tokenizer(model_name: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok   = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return model, tok


def _generate_qa_pairs(
    model,
    tokenizer,
    samples:     List[dict],
    theta_flat: torch.Tensor,
    k_edge:     int,
    k_node:     int,
    device:     str,
    max_new_tokens: int = 128,
) -> List[Dict[str, str]]:
    from unifiededu.federation.client import assign_layer_groups, modulate_params
    from unifiededu.models.gnn_params import theta_from_flat

    # 1. Instantiate Theta and modulate params
    theta  = theta_from_flat(theta_flat.to(device), k_edge, k_node)
    groups = assign_layer_groups(model, k_edge, k_node)
    params = modulate_params(model, theta, groups)

    # 2. Swap the modulated parameters into the model temporarily
    orig_state = {k: v.clone() for k, v in model.state_dict().items() if k in params}
    model.load_state_dict(params, strict=False)

    results = []
    
    try:
        for s in samples:
            prompt = f"Context: {s['context']}\n\nQuestion: "
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                max_length=400,
                truncation=True,
            ).to(device)

            with torch.no_grad():
                # Now model.generate uses the Theta-modulated weights
                gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

            generated = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            
            if "Answer:" in generated:
                parts = generated.split("Answer:", 1)
                q_part = parts[0].replace("Question:", "").strip()
                a_part = parts[1].strip()
            else:
                q_part = generated.strip()
                a_part = ""

            results.append({
                "sample_id": s["sample_id"],
                "context":   s["context"],
                "question":  q_part,
                "answer":    a_part,
                "ref_question": s["question"],
                "ref_answer":   s["answer"],
            })
    finally:
        # 3. Restore the original frozen weights so we don't leak state
        model.load_state_dict(orig_state, strict=False)

    return results


def run_evaluation(
    generated:      List[Dict[str, str]],
    output_path:    str,
    device:         str = "cpu",
) -> Dict[str, float]:
    """Run all five evaluation dimensions and save results."""
    from unifiededu.evaluation import (
        compute_rquge,
        compute_faithfulness,
        compute_qafacteval,
        compute_rtc,
        compute_bloom_level,
        compute_evs,
    )

    questions = [g["question"] for g in generated]
    answers   = [g["answer"]   for g in generated]
    contexts  = [g["context"]  for g in generated]

    log.info("Computing RQUGE...")
    rquge_scores = compute_rquge(questions, answers, contexts, device=device)

    log.info("Computing Faithfulness...")
    faith_scores = compute_faithfulness(answers, contexts, device=device)

    log.info("Computing QAFactEval...")
    qafe_scores  = compute_qafacteval(answers, contexts, device=device)

    log.info("Computing RTC...")
    rtc_scores   = compute_rtc(questions, answers, contexts, device=device)

    log.info("Computing Bloom's levels...")
    bloom_levels = compute_bloom_level(questions, method="bert", device=device)
    evs_scores   = compute_evs(bloom_levels)

    # Aggregate
    agg = {
        "rquge_mean":       sum(rquge_scores) / len(rquge_scores),
        "faithfulness_mean": sum(faith_scores) / len(faith_scores),
        "qafacteval_mean":  sum(qafe_scores)  / len(qafe_scores),
        "rtc_mean":         sum(rtc_scores)   / len(rtc_scores),
        "evs_mean":         sum(evs_scores)   / len(evs_scores),
        "bloom_mean":       sum(bloom_levels) / len(bloom_levels),
    }

    # Save per-sample results
    per_sample = []
    for i, g in enumerate(generated):
        per_sample.append({
            **g,
            "rquge":       rquge_scores[i],
            "faithfulness": faith_scores[i],
            "qafacteval":  qafe_scores[i],
            "rtc":         rtc_scores[i],
            "bloom_level": bloom_levels[i],
            "evs":         evs_scores[i],
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"aggregate": agg, "per_sample": per_sample}, f, indent=2)

    log.info("Results saved to %s", output_path)
    return agg


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method",         default="unifiededu")
    p.add_argument("--checkpoint-dir", default="runs/default")
    p.add_argument("--data-dir",       default="data/processed")
    p.add_argument("--output-dir",     default="results/default")
    p.add_argument("--split",          default="test",
                   choices=["train", "val", "test"])
    p.add_argument("--model",          default="distilgpt2")
    p.add_argument("--device",         default="cpu")
    p.add_argument("--client",         default=None,
                   help="Evaluate a specific client only (e.g. 'mit')")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from unifiededu.config import UnifiedEduConfig
    from unifiededu.data.preprocessing import load_splits, CLIENT_FILES

    cfg = UnifiedEduConfig()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    client_names = [args.client] if args.client else list(CLIENT_FILES.keys())
    splits = load_splits(args.data_dir, client_names)

    # Load latest checkpoint
    ckpts = sorted(Path(args.checkpoint_dir).glob("round_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {args.checkpoint_dir}")
    state = torch.load(ckpts[-1], map_location="cpu")
    thetas: Dict = state["thetas"]

    model, tokenizer = _load_model_and_tokenizer(args.model, args.device)

    all_agg = {}
    for idx, client_name in enumerate(client_names):
        log.info("Evaluating client: %s", client_name)
        theta_flat = thetas.get(idx, list(thetas.values())[0])
        samples    = splits[client_name][args.split]

        generated = _generate_qa_pairs(
            model, tokenizer, samples, theta_flat,
            cfg.model_graph.k_edge, cfg.model_graph.k_node,
            args.device,
        )

        out_path = os.path.join(
            args.output_dir,
            f"{client_name}_{args.split}_{args.method}.json",
        )
        agg = run_evaluation(generated, out_path, device=args.device)
        all_agg[client_name] = agg
        log.info("%s: %s", client_name, agg)
        
    # --- Global test set evaluation ---
    if "global" in splits:
        log.info("Evaluating on global test set...")
        global_samples = splits["global"]["test"]

        # Use the grand-mean Theta (last inter-cluster average)
        # which is the most general parameter vector
        grand_theta = torch.stack(list(thetas.values())).mean(dim=0)

        generated_global = _generate_qa_pairs(
            model, tokenizer, global_samples, grand_theta,
            cfg.model_graph.k_edge, cfg.model_graph.k_node,
            args.device,
        )

        global_out_path = os.path.join(
            args.output_dir,
            f"global_test_{args.method}.json",
        )
        global_agg = run_evaluation(
            generated_global, global_out_path, device=args.device
        )
        all_agg["global"] = global_agg
        log.info("Global test: %s", global_agg)

    # Save summary
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_agg, f, indent=2)
    log.info("Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()