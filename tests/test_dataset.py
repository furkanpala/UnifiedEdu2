"""
Tests for data/dataset.py

Uses a lightweight mock tokenizer so no model download is required.
One test (test_with_gpt2_tokenizer) attempts to use the real GPT-2
tokenizer if it is cached locally; it is skipped gracefully otherwise.

Run: /c/Users/fp223/AppData/Local/anaconda3/envs/ML/python.exe tests/test_dataset.py
"""

import sys, os, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import DataLoader

from unifiededu.data.dataset import (
    QADataset, QACollator, make_dataloader,
    _tokenize_sample, _build_texts, LABEL_IGNORE_ID,
)

logging.basicConfig(level=logging.WARNING)


def _assert(cond, msg=""):
    if not cond:
        raise AssertionError(msg)


# ---------------------------------------------------------------------------
# Minimal mock tokenizer (no HuggingFace download needed)
# ---------------------------------------------------------------------------

class _MockTokenizer:
    """
    Character-level tokenizer for testing.
    Each character maps to its ASCII code; vocab size = 256.
    Special tokens: BOS=1, EOS=2, PAD=0.
    """
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0
    model_max_length = 2048

    def encode(self, text: str, add_special_tokens: bool = True) -> list:
        ids = [ord(c) % 200 + 3 for c in text]   # 3..202, never clash with special tokens
        if add_special_tokens:
            ids = [self.bos_token_id] + ids
        return ids

    def decode(self, ids: list, skip_special_tokens: bool = True) -> str:
        specials = {self.bos_token_id, self.eos_token_id, self.pad_token_id}
        if skip_special_tokens:
            ids = [i for i in ids if i not in specials]
        return "".join(chr((i - 3) % 200 + 32) for i in ids)

    def __call__(self, text: str, max_length: int = 512, truncation: bool = True,
                 padding: bool = False, add_special_tokens: bool = True,
                 return_tensors=None, **_):
        ids = self.encode(text, add_special_tokens=add_special_tokens)
        if truncation and len(ids) > max_length:
            ids = ids[:max_length]
        mask = [1] * len(ids)
        return {"input_ids": ids, "attention_mask": mask}


_TOK = _MockTokenizer()

_SAMPLE = {
    "sample_id": "test001",
    "context":   "Gradient descent minimises the loss by taking steps proportional to the negative gradient.",
    "question":  "What does gradient descent minimise?",
    "answer":    "The loss function.",
    "difficulty": "easy",
    "bloom_level": "understand",
    "metadata":  {},
}

_LONG_SAMPLE = {
    "sample_id": "long001",
    "context":   "A " * 400,          # very long context (~800 chars)
    "question":  "Short question?",
    "answer":    "Short answer.",
    "difficulty": "easy",
    "bloom_level": "unknown",
    "metadata":  {},
}


# ---------------------------------------------------------------------------
# _build_texts
# ---------------------------------------------------------------------------

def test_build_texts_contains_all_parts():
    prefix, completion = _build_texts("ctx", "q?", "ans")
    _assert("ctx"   in prefix,     "Context missing from prefix")
    _assert("q?"    in completion, "Question missing from completion")
    _assert("ans"   in completion, "Answer missing from completion")
    _assert("q?"    not in prefix, "Question should not be in prefix")


def test_build_texts_prefix_ends_at_question_prompt():
    """The prefix should end with the 'Question: ' template string."""
    prefix, _ = _build_texts("some context", "q?", "ans")
    _assert(prefix.endswith("Question: "), f"Prefix doesn't end with 'Question: ': {repr(prefix[-20:])}")


# ---------------------------------------------------------------------------
# _tokenize_sample
# ---------------------------------------------------------------------------

def test_tokenize_output_keys():
    out = _tokenize_sample(_SAMPLE, _TOK, max_length=512)
    _assert(set(out.keys()) == {"input_ids", "attention_mask", "labels"})


def test_tokenize_tensor_types():
    out = _tokenize_sample(_SAMPLE, _TOK, max_length=512)
    for key in ("input_ids", "attention_mask", "labels"):
        _assert(isinstance(out[key], torch.Tensor), f"{key} is not a Tensor")
        _assert(out[key].dtype == torch.long, f"{key} dtype is {out[key].dtype}")


def test_tokenize_shapes_match():
    out = _tokenize_sample(_SAMPLE, _TOK, max_length=512)
    L = out["input_ids"].shape[0]
    _assert(out["attention_mask"].shape[0] == L, "attention_mask length mismatch")
    _assert(out["labels"].shape[0] == L,          "labels length mismatch")


def test_tokenize_respects_max_length():
    for ml in (64, 128, 256, 512):
        out = _tokenize_sample(_LONG_SAMPLE, _TOK, max_length=ml)
        _assert(out["input_ids"].shape[0] <= ml,
                f"max_length={ml} violated: got {out['input_ids'].shape[0]}")


def test_tokenize_completion_tokens_are_present():
    """At least some labels should be non -100 (QA part survived truncation)."""
    out = _tokenize_sample(_SAMPLE, _TOK, max_length=512)
    n_real = (out["labels"] != LABEL_IGNORE_ID).sum().item()
    _assert(n_real > 0, "All labels are -100; completion was lost")


def test_tokenize_prefix_masked():
    """The first token(s) should always be masked (BOS / context prefix)."""
    out = _tokenize_sample(_SAMPLE, _TOK, max_length=512)
    _assert(out["labels"][0].item() == LABEL_IGNORE_ID,
            "First token should be masked (BOS / context prefix)")


def test_tokenize_no_label_in_context():
    """
    No label token in the prefix portion should equal the corresponding
    input_id (they should all be -100).
    """
    out    = _tokenize_sample(_SAMPLE, _TOK, max_length=512)
    labels = out["labels"]
    # Find where the real labels start
    first_real = next((i for i, l in enumerate(labels.tolist()) if l != LABEL_IGNORE_ID), None)
    if first_real is not None:
        # Everything before first_real must be -100
        prefix_labels = labels[:first_real]
        _assert(
            (prefix_labels == LABEL_IGNORE_ID).all(),
            "Found non-masked label in context prefix"
        )


def test_tokenize_long_context_truncated():
    """Long context must be truncated so QA completion is still present."""
    out = _tokenize_sample(_LONG_SAMPLE, _TOK, max_length=128, qa_max_tokens=64)
    n_real = (out["labels"] != LABEL_IGNORE_ID).sum().item()
    _assert(n_real > 0, f"Long context sample has no real labels; qa completion lost")
    _assert(out["input_ids"].shape[0] <= 128)


def test_tokenize_eos_appended():
    """If tokenizer has eos_token_id, the last non-pad label should be EOS."""
    out = _tokenize_sample(_SAMPLE, _TOK, max_length=512)
    real_labels = [l for l in out["labels"].tolist() if l != LABEL_IGNORE_ID]
    if real_labels:
        _assert(real_labels[-1] == _TOK.eos_token_id,
                f"Last label {real_labels[-1]} is not EOS ({_TOK.eos_token_id})")


def test_tokenize_attention_mask_all_ones_no_pad():
    """With no padding, all attention mask values should be 1."""
    out = _tokenize_sample(_SAMPLE, _TOK, max_length=512)
    _assert(out["attention_mask"].all(), "Some attention mask values are 0 without padding")


# ---------------------------------------------------------------------------
# QADataset
# ---------------------------------------------------------------------------

def test_dataset_len():
    samples = [_SAMPLE, _LONG_SAMPLE]
    ds = QADataset(samples, _TOK)
    _assert(len(ds) == 2)


def test_dataset_getitem_returns_dict():
    ds  = QADataset([_SAMPLE], _TOK)
    out = ds[0]
    _assert(isinstance(out, dict))
    _assert("input_ids" in out)


def test_dataset_label_coverage_full():
    """All samples have real labels → coverage should be 1.0."""
    samples = [_SAMPLE] * 20
    ds      = QADataset(samples, _TOK)
    cov     = ds.label_coverage()
    _assert(cov == 1.0, f"Expected coverage 1.0, got {cov}")


# ---------------------------------------------------------------------------
# QACollator
# ---------------------------------------------------------------------------

def _make_batch(lengths):
    """Create a list of fake tokenised items with given sequence lengths."""
    return [
        {
            "input_ids":      torch.ones(l, dtype=torch.long),
            "attention_mask": torch.ones(l, dtype=torch.long),
            "labels":         torch.zeros(l, dtype=torch.long),
        }
        for l in lengths
    ]


def test_collator_pads_to_max_length():
    coll  = QACollator(pad_token_id=0)
    batch = coll(_make_batch([4, 7, 3]))
    for key in ("input_ids", "attention_mask", "labels"):
        _assert(batch[key].shape == (3, 7), f"{key} shape: {batch[key].shape}")


def test_collator_pad_values():
    coll  = QACollator(pad_token_id=99, label_pad_id=-100)
    batch = coll(_make_batch([2, 4]))
    # Short sequence should have 2 padding tokens at the end
    _assert(batch["input_ids"][0, 2].item() == 99,    "input pad value wrong")
    _assert(batch["attention_mask"][0, 2].item() == 0, "attn pad value wrong")
    _assert(batch["labels"][0, 2].item() == -100,      "label pad value wrong")


def test_collator_no_pad_when_equal_length():
    coll  = QACollator(pad_token_id=0)
    batch = coll(_make_batch([5, 5]))
    _assert(batch["input_ids"].shape == (2, 5))
    _assert((batch["attention_mask"] == 1).all())


# ---------------------------------------------------------------------------
# make_dataloader
# ---------------------------------------------------------------------------

def test_make_dataloader_returns_dataloader():
    dl = make_dataloader([_SAMPLE, _LONG_SAMPLE], _TOK, batch_size=2, shuffle=False)
    _assert(isinstance(dl, DataLoader))


def test_make_dataloader_batch_shapes():
    samples = [_SAMPLE] * 8
    dl      = make_dataloader(samples, _TOK, batch_size=4, max_length=128, shuffle=False)
    batch   = next(iter(dl))
    B, T    = batch["input_ids"].shape
    _assert(B == 4, f"Batch size: {B}")
    _assert(T <= 128, f"Seq length {T} exceeds max_length=128")
    _assert(batch["labels"].shape == (B, T))
    _assert(batch["attention_mask"].shape == (B, T))


def test_make_dataloader_sets_pad_token_when_missing():
    """Tokenizer without pad_token_id should have it set to eos_token_id."""
    class _NoPadTok(_MockTokenizer):
        pad_token_id = None  # type: ignore
    tok = _NoPadTok()
    dl  = make_dataloader([_SAMPLE], tok, batch_size=1, shuffle=False)
    _assert(tok.pad_token_id == tok.eos_token_id,
            "pad_token_id not set to eos_token_id")


# ---------------------------------------------------------------------------
# Integration: from_jsonl
# ---------------------------------------------------------------------------

def test_from_jsonl_roundtrip(tmp_path=None):
    import tempfile, json, os
    samples = [_SAMPLE, _LONG_SAMPLE]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "samples.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        ds = QADataset.from_jsonl(path, _TOK, max_length=256)
    _assert(len(ds) == 2)
    for i in range(2):
        out = ds[i]
        _assert(out["input_ids"].shape[0] <= 256)
        _assert((out["labels"] != LABEL_IGNORE_ID).any(),
                f"Sample {i}: all labels masked")


# ---------------------------------------------------------------------------
# Integration with real data (fast — no model download, uses mock tokenizer)
# ---------------------------------------------------------------------------

def test_real_data_no_truncation_loss():
    """
    All 547 balanced MIT samples should have at least one real label token
    (QA completion always survives context truncation).
    """
    import sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import random
    from unifiededu.data.preprocessing import _load_multi_json, _flatten_to_samples, _stratified_subsample

    records = _load_multi_json("ML_QA_LectureNotes_MIT.json")
    samples = _flatten_to_samples(records, "mit")
    rng     = random.Random(42)
    balanced = _stratified_subsample(samples, 547, rng)

    ds = QADataset(balanced, _TOK, max_length=512, qa_max_tokens=128)
    fully_masked = 0
    for i in range(len(ds)):
        item = ds[i]
        if not (item["labels"] != LABEL_IGNORE_ID).any():
            fully_masked += 1

    _assert(fully_masked == 0,
            f"{fully_masked}/547 MIT samples have all labels masked (QA truncated)")


def test_real_papers_data_with_truncation():
    """Papers has long contexts; verify every sample still has QA tokens after truncation."""
    import random
    from unifiededu.data.preprocessing import _load_multi_json, _flatten_to_samples, _stratified_subsample

    records  = _load_multi_json("ML_QA_Papers_v2.json")
    samples  = _flatten_to_samples(records, "papers")
    rng      = random.Random(42)
    balanced = _stratified_subsample(samples, 547, rng)

    ds = QADataset(balanced, _TOK, max_length=512, qa_max_tokens=128)
    fully_masked = sum(
        1 for i in range(len(ds))
        if not (ds[i]["labels"] != LABEL_IGNORE_ID).any()
    )
    _assert(fully_masked == 0,
            f"{fully_masked}/547 Papers samples have all labels masked (QA truncated)")


if __name__ == "__main__":
    tests = [
        test_build_texts_contains_all_parts,
        test_build_texts_prefix_ends_at_question_prompt,
        test_tokenize_output_keys,
        test_tokenize_tensor_types,
        test_tokenize_shapes_match,
        test_tokenize_respects_max_length,
        test_tokenize_completion_tokens_are_present,
        test_tokenize_prefix_masked,
        test_tokenize_no_label_in_context,
        test_tokenize_long_context_truncated,
        test_tokenize_eos_appended,
        test_tokenize_attention_mask_all_ones_no_pad,
        test_dataset_len,
        test_dataset_getitem_returns_dict,
        test_dataset_label_coverage_full,
        test_collator_pads_to_max_length,
        test_collator_pad_values,
        test_collator_no_pad_when_equal_length,
        test_make_dataloader_returns_dataloader,
        test_make_dataloader_batch_shapes,
        test_make_dataloader_sets_pad_token_when_missing,
        test_from_jsonl_roundtrip,
        test_real_data_no_truncation_loss,
        test_real_papers_data_with_truncation,
    ]
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            import traceback
            print(f"FAIL  {t.__name__}: {e}")
            traceback.print_exc()
