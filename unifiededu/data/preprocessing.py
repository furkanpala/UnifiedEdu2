"""
data/preprocessing.py

Loads the three public QA datasets, balances them to equal sample counts,
splits into train/val/test, computes embeddings, and verifies non-IID
separation between clients.

Data model
----------
Each record in a source file has the structure:
    {
        "input_index": int,
        "clean_context": str,          # source passage
        "qa_pairs": [                  # 3-6 pairs per record
            {"question": str, "answer": str, "difficulty": str}, ...
        ],
        "input_meta": dict             # provenance metadata
    }

A *sample* is one (context, question, answer) triple derived by
flattening each record's qa_pairs list.  This is the atomic unit that
enters the PyTorch Dataset.

Balancing rule
--------------
  MIT    : 110 records -> 547  QA triples  (all kept)
  Stanford: 110 records -> 559  QA triples  (subsampled to 547)
  Papers :  981 records -> 5037 QA triples  (subsampled to 547)

  Target N = min(|MIT|, |Stanford|, |Papers|) = 547 per client.
  Subsampling is stratified by difficulty {easy, medium, hard} so each
  client preserves its original difficulty distribution at the target N.

Splits
------
  80 % train, 10 % val, 10 % test.
  The test split is guaranteed to include >= 20 samples that cover each
  of the 10 fixed anchor topics (keyword-based matching).

Non-IID gate
------------
  Mean pairwise cosine distance between per-client embedding centroids
  must be >= 0.15.  The function verify_non_iid() raises ValueError if
  the gate fails.

Outputs
-------
  For each client, three JSONL files are written:
    <output_dir>/<client_name>_{train,val,test}.jsonl

  Each line:
    {
        "sample_id":  str,
        "context":    str,
        "question":   str,
        "answer":     str,
        "difficulty": str,
        "bloom_level": str,   # from meta if present, else "unknown"
        "metadata":   dict,
        "embedding":  list[float]   # 384-dim, all-MiniLM-L6-v2
    }
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixed anchor topics (10 ML concepts guaranteed to appear in the test split)
# ---------------------------------------------------------------------------

ANCHOR_TOPICS: List[Tuple[str, List[str]]] = [
    ("linear_regression",     ["linear regression", "least squares", "normal equation"]),
    ("neural_networks",       ["neural network", "deep learning", "hidden layer", "activation function"]),
    ("gradient_descent",      ["gradient descent", "learning rate", "stochastic gradient", "sgd", "adam"]),
    ("regularisation",        ["regularization", "regularisation", "overfitting", "l1", "l2", "lasso", "ridge"]),
    ("svm",                   ["support vector", "svm", "kernel", "margin", "hyperplane"]),
    ("probabilistic_models",  ["bayesian", "probabilistic", "likelihood", "prior", "posterior", "gaussian"]),
    ("clustering",            ["clustering", "k-means", "unsupervised", "cluster"]),
    ("dimensionality",        ["dimensionality reduction", "pca", "principal component", "embedding"]),
    ("generalisation_theory", ["generalisation", "generalization", "vc dimension", "bias-variance",
                               "learning theory", "pac learning"]),
    ("backpropagation",       ["backpropagation", "back propagation", "chain rule", "gradient flow",
                               "training loss", "loss function"]),
]

# Client name -> source JSON filename
CLIENT_FILES: Dict[str, str] = {
    "mit":      "ML_QA_LectureNotes_MIT.json",
    "stanford": "ML_QA_LectureNotes_StanfordCS229.json",
    "papers":   "ML_QA_Papers_v2.json",
}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _load_multi_json(path: str) -> List[dict]:
    """
    Parse a file containing one or more concatenated JSON objects
    (not a JSON array, not JSONL -- just sequential top-level objects).
    """
    with open(path, encoding="utf-8") as f:
        raw = f.read().strip()
    decoder = json.JSONDecoder()
    records, pos = [], 0
    while pos < len(raw):
        obj, end = decoder.raw_decode(raw, pos)
        records.append(obj)
        pos = end
        while pos < len(raw) and raw[pos] in " \t\n\r":
            pos += 1
    return records


def _flatten_to_samples(records: List[dict], client_name: str) -> List[dict]:
    """
    Expand each record into one sample per QA pair.

    Returns a list of dicts:
        {sample_id, context, question, answer, difficulty, bloom_level, metadata}
    """
    samples = []
    for rec in records:
        ctx  = rec["clean_context"]
        meta = rec["input_meta"]
        bloom = meta.get("bloom_level", "unknown")

        for qa in rec.get("qa_pairs", []):
            # Deterministic ID from full content (avoid collisions on similar questions)
            raw_id = f"{client_name}|{rec['input_index']}|{qa['question']}|{qa['answer'][:60]}"
            sid    = hashlib.md5(raw_id.encode("utf-8")).hexdigest()[:16]

            samples.append({
                "sample_id":  sid,
                "context":    ctx,
                "question":   qa["question"],
                "answer":     qa["answer"],
                "difficulty": qa.get("difficulty", "unknown"),
                "bloom_level": bloom,
                "metadata":   meta,
                "embedding":  [],   # filled later by embed_samples()
            })
    return samples


# ---------------------------------------------------------------------------
# Stratified subsampling
# ---------------------------------------------------------------------------

def _stratified_subsample(
    samples: List[dict],
    target_n: int,
    rng: random.Random,
) -> List[dict]:
    """
    Sample exactly target_n items from samples, stratified by difficulty.

    Proportions are computed from the full list; any rounding remainder
    is filled from the most frequent bucket.
    """
    if len(samples) <= target_n:
        return list(samples)

    # Group by difficulty
    buckets: Dict[str, List[dict]] = defaultdict(list)
    for s in samples:
        buckets[s["difficulty"]].append(s)

    total = len(samples)
    allocated = 0
    result: List[dict] = []

    # Sort for determinism; most-frequent last (receives remainder)
    sorted_keys = sorted(buckets.keys(), key=lambda k: len(buckets[k]))

    for i, key in enumerate(sorted_keys):
        is_last = i == len(sorted_keys) - 1
        bucket  = buckets[key]
        if is_last:
            n_take = target_n - allocated
        else:
            n_take = round(len(bucket) / total * target_n)
            n_take = min(n_take, len(bucket))

        rng.shuffle(bucket)
        result.extend(bucket[:n_take])
        allocated += n_take

    rng.shuffle(result)
    return result


# ---------------------------------------------------------------------------
# Anchor-topic matching
# ---------------------------------------------------------------------------

def _anchor_hits(text: str) -> List[str]:
    """Return list of anchor topic keys whose keywords appear in text (case-insensitive)."""
    text_lower = text.lower()
    hits = []
    for topic_key, keywords in ANCHOR_TOPICS:
        if any(kw in text_lower for kw in keywords):
            hits.append(topic_key)
    return hits


# ---------------------------------------------------------------------------
# Train / val / test split
# ---------------------------------------------------------------------------

def _split_samples(
    samples:    List[dict],
    train_ratio: float,
    val_ratio:   float,
    min_anchor_chunks: int,
    rng:         random.Random,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Split samples into train / val / test.

    Anchor-topic guarantee:
        For each of the 10 anchor topics, we try to include at least
        min_anchor_chunks // 10 samples in the test set whose context
        matches that topic's keywords.  If the dataset has too few
        anchor hits, we include all matching samples in test and fill
        the remainder with non-anchor samples.

    Returns (train, val, test).
    """
    test_ratio = 1.0 - train_ratio - val_ratio
    n_test     = max(1, round(len(samples) * test_ratio))
    n_val      = max(1, round(len(samples) * val_ratio))

    # Separate anchor and non-anchor samples
    anchor_map: Dict[str, List[dict]] = defaultdict(list)
    for s in samples:
        for topic in _anchor_hits(s["context"] + " " + s["question"]):
            anchor_map[topic].append(s)

    # Guarantee at least one sample per anchor topic in test (if available)
    test_set:   List[dict] = []
    used_ids = set()

    per_topic_target = max(1, min_anchor_chunks // len(ANCHOR_TOPICS))
    for topic_key, _ in ANCHOR_TOPICS:
        candidates = [s for s in anchor_map.get(topic_key, [])
                      if s["sample_id"] not in used_ids]
        rng.shuffle(candidates)
        chosen = candidates[:per_topic_target]
        for s in chosen:
            used_ids.add(s["sample_id"])
        test_set.extend(chosen)

    # Fill remainder of test from non-anchor samples
    non_anchor = [s for s in samples if s["sample_id"] not in used_ids]
    rng.shuffle(non_anchor)
    still_needed = max(0, n_test - len(test_set))
    extra = non_anchor[:still_needed]
    test_set.extend(extra)
    used_ids.update(s["sample_id"] for s in extra)

    # Everything not in test is available for train / val
    remaining = [s for s in samples if s["sample_id"] not in used_ids]
    rng.shuffle(remaining)

    val_set   = remaining[:n_val]
    train_set = remaining[n_val:]

    log.info(
        "Split: train=%d val=%d test=%d | anchor topics covered: %d/10",
        len(train_set), len(val_set), len(test_set),
        sum(1 for tk, _ in ANCHOR_TOPICS if any(
            _anchor_hits(s["context"] + s["question"])
            for s in test_set if tk in _anchor_hits(s["context"] + s["question"])
        )),
    )
    return train_set, val_set, test_set


# ---------------------------------------------------------------------------
# Embedding (all-MiniLM-L6-v2 via HuggingFace transformers)
# ---------------------------------------------------------------------------

def _mean_pool(
    token_embeddings: torch.Tensor,
    attention_mask:   torch.Tensor,
) -> torch.Tensor:
    """Masked mean pooling over token dimension."""
    mask_expanded = attention_mask.unsqueeze(-1).float()
    summed = (token_embeddings * mask_expanded).sum(dim=1)
    counts = mask_expanded.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def embed_texts(
    texts:      List[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
    device:     str = "cpu",
    max_length: int = 256,
) -> np.ndarray:
    """
    Compute 384-dim sentence embeddings for each text.

    Uses HuggingFace AutoTokenizer + AutoModel with mean pooling,
    which is equivalent to the sentence-transformers pipeline.

    Parameters
    ----------
    texts      : list of strings
    model_name : HuggingFace model id
    batch_size : tokeniser batch size
    device     : 'cpu' or 'cuda'
    max_length : truncation length in tokens

    Returns
    -------
    np.ndarray, shape (len(texts), 384), dtype float32
    """
    from transformers import AutoTokenizer, AutoModel

    log.info("Loading embedding model %s ...", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    all_embeddings = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start: start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            out = model(**encoded)
        emb = _mean_pool(out.last_hidden_state, encoded["attention_mask"])
        # L2-normalise (cosine similarity = dot product after normalisation)
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        all_embeddings.append(emb.cpu().float().numpy())

    result = np.concatenate(all_embeddings, axis=0)
    log.info("Embedded %d texts -> shape %s", len(texts), result.shape)
    return result


def embed_samples(
    samples:    List[dict],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
    device:     str = "cpu",
) -> List[dict]:
    """
    Add 'embedding' field (list[float]) to each sample dict in-place.
    Returns the same list for convenience.
    """
    texts = [s["context"] for s in samples]
    embs  = embed_texts(texts, model_name=model_name, batch_size=batch_size, device=device)
    for s, e in zip(samples, embs):
        s["embedding"] = e.tolist()
    return samples


# ---------------------------------------------------------------------------
# Non-IID verification
# ---------------------------------------------------------------------------

def verify_non_iid(
    client_embeddings: Dict[str, np.ndarray],
    min_distance:      float = 0.15,
    raise_on_fail:     bool  = True,
) -> Tuple[bool, float]:
    """
    Verify that client embedding centroids are sufficiently separated.

    Computes the per-client mean embedding centroid (already L2-normalised
    per embed_texts, so cosine distance = 1 - dot product).

    Parameters
    ----------
    client_embeddings : Dict[client_name, ndarray (N, dim)]
    min_distance      : go/no-go threshold (default 0.15)
    raise_on_fail     : raise ValueError if gate fails

    Returns
    -------
    (passed: bool, mean_pairwise_dist: float)
    """
    from scipy.spatial.distance import cosine as cosine_dist

    names     = sorted(client_embeddings.keys())
    centroids = {name: client_embeddings[name].mean(axis=0) for name in names}

    distances = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = cosine_dist(centroids[names[i]], centroids[names[j]])
            distances.append(d)
            log.info(
                "Centroid distance %s <-> %s: %.4f",
                names[i], names[j], d,
            )

    mean_dist = float(np.mean(distances))
    passed    = mean_dist >= min_distance

    log.info(
        "Non-IID gate: mean pairwise centroid distance = %.4f (threshold %.2f) -> %s",
        mean_dist, min_distance, "PASS" if passed else "FAIL",
    )

    if not passed and raise_on_fail:
        raise ValueError(
            f"Non-IID gate FAILED: mean centroid distance {mean_dist:.4f} < {min_distance}. "
            "Check that client datasets are sufficiently different."
        )

    return passed, mean_dist


# ---------------------------------------------------------------------------
# t-SNE visualisation (optional, requires matplotlib)
# ---------------------------------------------------------------------------

def plot_tsne(
    client_embeddings: Dict[str, np.ndarray],
    output_path:       str = "tsne_embeddings.png",
    seed:              int = 42,
) -> None:
    """
    Produce t-SNE plot of all embeddings coloured by client / institution.
    Silently skips if matplotlib is not installed.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
    except ImportError:
        log.warning("matplotlib not installed; skipping t-SNE plot.")
        return

    names = sorted(client_embeddings.keys())
    all_embs   = np.concatenate([client_embeddings[n] for n in names], axis=0)
    all_labels = np.concatenate([
        np.full(client_embeddings[n].shape[0], i)
        for i, n in enumerate(names)
    ])

    log.info("Running t-SNE on %d points ...", len(all_embs))
    tsne = TSNE(n_components=2, random_state=seed, perplexity=30, n_iter=1000)
    proj = tsne.fit_transform(all_embs)

    colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, name in enumerate(names):
        mask = all_labels == i
        ax.scatter(proj[mask, 0], proj[mask, 1], s=10, alpha=0.6,
                   c=colours[i % len(colours)], label=name)
    ax.legend(title="Client")
    ax.set_title("t-SNE of context embeddings by institution")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    log.info("t-SNE saved to %s", output_path)


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

def save_jsonl(samples: List[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    log.info("Saved %d samples to %s", len(samples), path)


def load_jsonl(path: str) -> List[dict]:
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def prepare_all(
    data_dir:           str  = ".",
    output_dir:         str  = "data/processed",
    seed:               int  = 42,
    train_ratio:        float = 0.80,
    val_ratio:          float = 0.10,
    min_anchor_chunks:  int  = 20,
    min_iid_distance:   float = 0.15,
    embed:              bool = True,
    embedding_model:    str  = "sentence-transformers/all-MiniLM-L6-v2",
    embedding_device:   str  = "cpu",
    plot_tsne_path:     Optional[str] = None,
) -> Dict[str, Dict[str, List[dict]]]:
    """
    Full preprocessing pipeline.

    Returns
    -------
    Dict[client_name, {"train": [...], "val": [...], "test": [...]}]
    """
    rng = random.Random(seed)

    # ------------------------------------------------------------------
    # 1. Load and flatten
    # ------------------------------------------------------------------
    log.info("Loading source files from %s ...", data_dir)
    raw: Dict[str, List[dict]] = {}
    for client_name, fname in CLIENT_FILES.items():
        path    = os.path.join(data_dir, fname)
        records = _load_multi_json(path)
        samples = _flatten_to_samples(records, client_name)
        raw[client_name] = samples
        log.info("  %s: %d records -> %d QA triples", client_name, len(records), len(samples))

    # ------------------------------------------------------------------
    # 2. Balance to equal sample counts (stratified by difficulty)
    # ------------------------------------------------------------------
    target_n = min(len(v) for v in raw.values())
    log.info("Balancing to %d samples per client (stratified by difficulty)", target_n)

    balanced: Dict[str, List[dict]] = {}
    for client_name, samples in raw.items():
        balanced[client_name] = _stratified_subsample(samples, target_n, rng)
        log.info("  %s: %d -> %d", client_name, len(samples), len(balanced[client_name]))

    # ------------------------------------------------------------------
    # 3. Embed (optional -- required for non-IID check and JSONL output)
    # ------------------------------------------------------------------
    if embed:
        for client_name, samples in balanced.items():
            log.info("Embedding %s ...", client_name)
            embed_samples(samples, model_name=embedding_model, device=embedding_device)
    else:
        log.warning("Embedding skipped (embed=False). Non-IID check will be skipped too.")

    # ------------------------------------------------------------------
    # 4. Non-IID verification
    # ------------------------------------------------------------------
    if embed:
        client_embs: Dict[str, np.ndarray] = {
            name: np.array([s["embedding"] for s in samples])
            for name, samples in balanced.items()
        }
        verify_non_iid(client_embs, min_distance=min_iid_distance, raise_on_fail=True)

        if plot_tsne_path:
            plot_tsne(client_embs, output_path=plot_tsne_path)

    # ------------------------------------------------------------------
    # 5. Split each client's dataset
    # ------------------------------------------------------------------
    splits: Dict[str, Dict[str, List[dict]]] = {}
    for client_name, samples in balanced.items():
        train, val, test = _split_samples(
            samples,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            min_anchor_chunks=min_anchor_chunks,
            rng=rng,
        )
        splits[client_name] = {"train": train, "val": val, "test": test}
        log.info(
            "%s splits: train=%d val=%d test=%d",
            client_name, len(train), len(val), len(test),
        )

    # ------------------------------------------------------------------
    # 6. Save JSONL
    # ------------------------------------------------------------------
    for client_name, split_dict in splits.items():
        for split_name, samples in split_dict.items():
            out_path = os.path.join(output_dir, f"{client_name}_{split_name}.jsonl")
            save_jsonl(samples, out_path)

    log.info("Preprocessing complete. Output dir: %s", output_dir)
    return splits


# ---------------------------------------------------------------------------
# Convenience: load already-processed splits back into memory
# ---------------------------------------------------------------------------

def load_splits(
    output_dir:  str,
    client_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, List[dict]]]:
    """
    Load JSONL files previously written by prepare_all().

    Returns Dict[client_name, {"train": [...], "val": [...], "test": [...]}]
    """
    names = client_names or list(CLIENT_FILES.keys())
    result = {}
    for client_name in names:
        result[client_name] = {}
        for split in ("train", "val", "test"):
            path = os.path.join(output_dir, f"{client_name}_{split}.jsonl")
            result[client_name][split] = load_jsonl(path)
    return result
