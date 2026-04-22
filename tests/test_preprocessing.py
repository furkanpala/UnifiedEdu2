"""
Tests for data/preprocessing.py

Run: /c/Users/fp223/AppData/Local/anaconda3/envs/ML/python.exe tests/test_preprocessing.py

Embedding is skipped for speed (embed=False); the non-IID and JSONL
tests use random vectors instead of real embeddings.
"""

import sys, os, json, tempfile, random, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from unifiededu.data.preprocessing import (
    _load_multi_json,
    _flatten_to_samples,
    _stratified_subsample,
    _anchor_hits,
    _split_samples,
    verify_non_iid,
    save_jsonl,
    load_jsonl,
    prepare_all,
    CLIENT_FILES,
    ANCHOR_TOPICS,
)

DATA_DIR = "."   # JSON files live in the project root


def _assert(cond, msg=""):
    if not cond:
        raise AssertionError(msg)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _load_client(name):
    fname = CLIENT_FILES[name]
    path  = os.path.join(DATA_DIR, fname)
    records = _load_multi_json(path)
    return _flatten_to_samples(records, name)


# -----------------------------------------------------------------------
# 1. Parsing
# -----------------------------------------------------------------------

def test_load_all_files():
    for name, fname in CLIENT_FILES.items():
        path    = os.path.join(DATA_DIR, fname)
        records = _load_multi_json(path)
        _assert(len(records) > 0, f"{fname}: no records parsed")
        for r in records:
            _assert("clean_context" in r, f"{fname}: missing clean_context")
            _assert("qa_pairs"      in r, f"{fname}: missing qa_pairs")
            _assert("input_meta"    in r, f"{fname}: missing input_meta")
        print(f"  {fname}: {len(records)} records OK")


def test_flatten_sample_structure():
    samples = _load_client("mit")
    _assert(len(samples) > 0)
    s = samples[0]
    for key in ("sample_id", "context", "question", "answer", "difficulty", "bloom_level", "metadata"):
        _assert(key in s, f"Missing key '{key}' in sample")
    _assert(isinstance(s["sample_id"], str) and len(s["sample_id"]) == 16)
    _assert(isinstance(s["context"],   str) and len(s["context"]) > 0)
    _assert(isinstance(s["question"],  str) and len(s["question"]) > 0)
    _assert(isinstance(s["answer"],    str) and len(s["answer"]) > 0)
    _assert(s["difficulty"] in {"easy", "medium", "hard", "unknown"})


def test_sample_counts():
    """Verify exact counts match inspection findings."""
    mit      = _load_client("mit")
    stanford = _load_client("stanford")
    papers   = _load_client("papers")
    _assert(len(mit)      == 547, f"MIT count: {len(mit)}")
    _assert(len(stanford) == 559, f"Stanford count: {len(stanford)}")
    _assert(len(papers)   == 5037, f"Papers count: {len(papers)}")


def test_sample_ids_unique():
    for name in CLIENT_FILES:
        samples = _load_client(name)
        ids = [s["sample_id"] for s in samples]
        _assert(len(ids) == len(set(ids)), f"{name}: duplicate sample_ids")


# -----------------------------------------------------------------------
# 2. Balancing
# -----------------------------------------------------------------------

def test_stratified_subsample_exact_count():
    samples = _load_client("papers")   # 5037
    rng     = random.Random(42)
    result  = _stratified_subsample(samples, 547, rng)
    _assert(len(result) == 547, f"Expected 547, got {len(result)}")


def test_stratified_subsample_no_op_when_small():
    samples = _load_client("mit")     # 547 -- already at target
    rng     = random.Random(42)
    result  = _stratified_subsample(samples, 547, rng)
    _assert(len(result) == 547)


def test_stratified_subsample_preserves_difficulty_approx():
    """Subsampled distribution should be close to original."""
    samples = _load_client("papers")
    rng     = random.Random(42)
    result  = _stratified_subsample(samples, 547, rng)

    orig_dist  = {}
    for s in samples:
        orig_dist[s["difficulty"]] = orig_dist.get(s["difficulty"], 0) + 1
    samp_dist  = {}
    for s in result:
        samp_dist[s["difficulty"]] = samp_dist.get(s["difficulty"], 0) + 1

    for diff in orig_dist:
        orig_frac = orig_dist[diff] / len(samples)
        samp_frac = samp_dist.get(diff, 0) / len(result)
        _assert(
            abs(orig_frac - samp_frac) < 0.05,
            f"Difficulty '{diff}' fraction drifted: orig={orig_frac:.3f} sampled={samp_frac:.3f}"
        )


def test_balance_gives_equal_counts():
    """All three clients should end up with target_n samples."""
    datasets = {name: _load_client(name) for name in CLIENT_FILES}
    target_n = min(len(v) for v in datasets.values())
    rng      = random.Random(42)
    for name, samples in datasets.items():
        balanced = _stratified_subsample(samples, target_n, rng)
        _assert(len(balanced) == target_n, f"{name}: {len(balanced)} != {target_n}")


# -----------------------------------------------------------------------
# 3. Anchor topics
# -----------------------------------------------------------------------

def test_anchor_hits_linear_regression():
    hits = _anchor_hits("This chapter introduces linear regression and least squares.")
    _assert("linear_regression" in hits)


def test_anchor_hits_neural_networks():
    hits = _anchor_hits("Deep learning uses multiple hidden layers in a neural network.")
    _assert("neural_networks" in hits)


def test_anchor_hits_empty_text():
    hits = _anchor_hits("nothing relevant here about xyz abc")
    _assert(len(hits) == 0)


def test_anchor_topics_count():
    _assert(len(ANCHOR_TOPICS) == 10, f"Expected 10 anchor topics, got {len(ANCHOR_TOPICS)}")


# -----------------------------------------------------------------------
# 4. Splitting
# -----------------------------------------------------------------------

def test_split_sizes_sum_to_total():
    samples = _load_client("mit")
    rng     = random.Random(42)
    train, val, test = _split_samples(samples, 0.80, 0.10, 20, rng)
    _assert(len(train) + len(val) + len(test) == len(samples),
            f"Split sizes don't sum: {len(train)+len(val)+len(test)} != {len(samples)}")


def test_split_no_overlap():
    samples = _load_client("mit")
    rng     = random.Random(42)
    train, val, test = _split_samples(samples, 0.80, 0.10, 20, rng)
    all_ids = [s["sample_id"] for s in train + val + test]
    _assert(len(all_ids) == len(set(all_ids)), "Samples appear in multiple splits")


def test_split_train_ratio_approx():
    samples = _load_client("mit")
    rng     = random.Random(42)
    train, val, test = _split_samples(samples, 0.80, 0.10, 20, rng)
    train_frac = len(train) / len(samples)
    _assert(0.70 <= train_frac <= 0.88,
            f"Train fraction {train_frac:.3f} outside expected range [0.70, 0.88]")


def test_split_test_has_anchor_coverage():
    """Test split should contain samples covering multiple anchor topics."""
    samples = _load_client("stanford")
    rng     = random.Random(42)
    _, _, test = _split_samples(samples, 0.80, 0.10, 20, rng)
    covered = set()
    for s in test:
        covered.update(_anchor_hits(s["context"] + " " + s["question"]))
    _assert(len(covered) >= 3,
            f"Test split covers only {len(covered)} anchor topics (want >= 3)")


# -----------------------------------------------------------------------
# 5. Non-IID verification
# -----------------------------------------------------------------------

def test_non_iid_passes_with_separated_centroids():
    rng = np.random.default_rng(42)
    embs = {
        "mit":      rng.normal([1, 0], 0.1, (100, 2)).astype(np.float32),
        "stanford": rng.normal([0, 1], 0.1, (100, 2)).astype(np.float32),
        "papers":   rng.normal([-1, 0], 0.1, (100, 2)).astype(np.float32),
    }
    # L2-normalise so cosine distance is meaningful
    for k in embs:
        norms = np.linalg.norm(embs[k], axis=1, keepdims=True) + 1e-9
        embs[k] = embs[k] / norms

    passed, dist = verify_non_iid(embs, min_distance=0.05, raise_on_fail=True)
    _assert(passed, f"Expected pass, got dist={dist:.4f}")


def test_non_iid_fails_with_identical_embeddings():
    e = np.ones((50, 10), dtype=np.float32)
    embs = {"a": e, "b": e, "c": e}
    passed, dist = verify_non_iid(embs, min_distance=0.15, raise_on_fail=False)
    _assert(not passed, f"Expected fail with identical embeddings, dist={dist:.4f}")


# -----------------------------------------------------------------------
# 6. JSONL I/O
# -----------------------------------------------------------------------

def test_save_and_load_jsonl_roundtrip():
    samples = _load_client("mit")[:5]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test.jsonl")
        save_jsonl(samples, path)
        loaded = load_jsonl(path)
    _assert(len(loaded) == 5)
    for orig, back in zip(samples, loaded):
        _assert(orig["sample_id"] == back["sample_id"])
        _assert(orig["question"]  == back["question"])


def test_save_jsonl_unicode():
    """Non-ASCII characters must survive the round-trip."""
    samples = [{"sample_id": "x", "text": "α-regularisation: λ=0.01 θ∈ℝ^p"}]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "u.jsonl")
        save_jsonl(samples, path)
        loaded = load_jsonl(path)
    _assert(loaded[0]["text"] == samples[0]["text"])


# -----------------------------------------------------------------------
# 7. Full pipeline (embed=False for speed)
# -----------------------------------------------------------------------

def test_prepare_all_no_embed():
    """End-to-end pipeline without embedding; checks file creation and sizes."""
    with tempfile.TemporaryDirectory() as tmp:
        splits = prepare_all(
            data_dir=DATA_DIR,
            output_dir=tmp,
            seed=42,
            embed=False,        # skip slow model download in CI
        )

    # Three clients
    _assert(set(splits.keys()) == {"mit", "stanford", "papers"})

    for client_name, split_dict in splits.items():
        _assert(set(split_dict.keys()) == {"train", "val", "test"})
        total = sum(len(v) for v in split_dict.values())
        # All three should have the same total (balanced)
        _assert(total == 547, f"{client_name}: total {total} != 547")
        # No overlap between splits
        all_ids = [s["sample_id"] for split in split_dict.values() for s in split]
        _assert(len(all_ids) == len(set(all_ids)), f"{client_name}: split overlap")


def test_prepare_all_creates_jsonl_files():
    with tempfile.TemporaryDirectory() as tmp:
        prepare_all(data_dir=DATA_DIR, output_dir=tmp, seed=42, embed=False)
        for client_name in CLIENT_FILES:
            for split in ("train", "val", "test"):
                path = os.path.join(tmp, f"{client_name}_{split}.jsonl")
                _assert(os.path.exists(path), f"Missing file: {path}")
                lines = open(path, encoding="utf-8").readlines()
                _assert(len(lines) > 0, f"Empty file: {path}")
                # Each line must be valid JSON
                for line in lines:
                    obj = json.loads(line)
                    _assert("sample_id" in obj)
                    _assert("question"  in obj)
                    _assert("answer"    in obj)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    tests = [
        test_load_all_files,
        test_flatten_sample_structure,
        test_sample_counts,
        test_sample_ids_unique,
        test_stratified_subsample_exact_count,
        test_stratified_subsample_no_op_when_small,
        test_stratified_subsample_preserves_difficulty_approx,
        test_balance_gives_equal_counts,
        test_anchor_hits_linear_regression,
        test_anchor_hits_neural_networks,
        test_anchor_hits_empty_text,
        test_anchor_topics_count,
        test_split_sizes_sum_to_total,
        test_split_no_overlap,
        test_split_train_ratio_approx,
        test_split_test_has_anchor_coverage,
        test_non_iid_passes_with_separated_centroids,
        test_non_iid_fails_with_identical_embeddings,
        test_save_and_load_jsonl_roundtrip,
        test_save_jsonl_unicode,
        test_prepare_all_no_embed,
        test_prepare_all_creates_jsonl_files,
    ]
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            import traceback
            print(f"FAIL  {t.__name__}: {e}")
            traceback.print_exc()
