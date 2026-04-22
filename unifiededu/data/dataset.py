"""
data/dataset.py

PyTorch Dataset for QA generation fine-tuning.

Each sample is a (context, question, answer) triple.  The tokenised
sequence is formatted as:

    Context: {context}

    Question: {question}
    Answer: {answer}{EOS}

Labels follow the causal LM convention:
  - -100 (ignore) for the "Context: ... Question: " prefix  ← model sees but doesn't predict
  - actual token IDs for "{question}\\nAnswer: {answer}{EOS}" ← model must predict

Truncation strategy (Papers contexts can be ~1 800 estimated tokens):
  - Tokenise the Q+A completion first; count the tokens needed (≤ qa_max_tokens).
  - Remaining budget = max_length − qa_tokens − template_overhead_tokens goes to context.
  - Context is truncated from the END (start of context is usually most informative).
  - If after truncation the QA still doesn't fit, the sample is silently replaced with
    an empty-label sentinel (all labels = -100); this is logged once per dataset.

Works with any HuggingFace tokenizer (BERT, LLaMA, GPT-2, Qwen, …).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from .preprocessing import load_jsonl

log = logging.getLogger(__name__)

# -100 is the standard PyTorch ignore index for cross-entropy
LABEL_IGNORE_ID = -100

# Prompt components (kept short to minimise overhead tokens)
_CTX_PREFIX = "Context: "
_QA_SEP     = "\n\nQuestion: "
_ANS_SEP    = "\nAnswer: "


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------

def _build_texts(context: str, question: str, answer: str):
    """
    Return (prefix_text, completion_text).

    prefix     = "Context: {context}\\n\\nQuestion: "
    completion = "{question}\\nAnswer: {answer}"

    Labels are -100 for prefix tokens; completion tokens are the targets.
    """
    prefix     = f"{_CTX_PREFIX}{context}{_QA_SEP}"
    completion = f"{question}{_ANS_SEP}{answer}"
    return prefix, completion


def _tokenize_sample(
    sample:         dict,
    tokenizer,
    max_length:     int  = 512,
    qa_max_tokens:  int  = 128,
) -> Dict[str, torch.Tensor]:
    """
    Tokenise one sample into {input_ids, attention_mask, labels}.

    Context is truncated from the right if the full sequence exceeds max_length.
    The Q+A completion is NEVER truncated; if it alone exceeds qa_max_tokens,
    the completion is preserved and the context is shrunk to zero.

    Parameters
    ----------
    sample        : dict with keys 'context', 'question', 'answer'
    tokenizer     : HuggingFace tokenizer
    max_length    : maximum sequence length in tokens (default 512)
    qa_max_tokens : token budget reserved for Q+A (default 128)

    Returns
    -------
    {input_ids: LongTensor, attention_mask: LongTensor, labels: LongTensor}
    """
    ctx  = sample["context"]
    q    = sample["question"]
    a    = sample["answer"]

    # ------------------------------------------------------------------
    # Step 1: tokenise the completion to know its exact length
    # ------------------------------------------------------------------
    prefix_text, completion_text = _build_texts(ctx, q, a)

    completion_ids = tokenizer.encode(
        completion_text, add_special_tokens=False
    )

    # Add EOS if the tokenizer has one (helps decoder-only models learn to stop)
    eos = tokenizer.eos_token_id
    if eos is not None and (not completion_ids or completion_ids[-1] != eos):
        completion_ids = completion_ids + [eos]

    # ------------------------------------------------------------------
    # Step 2: figure out token budget for the context
    # ------------------------------------------------------------------
    # BOS token (if any) takes 1 slot
    has_bos   = (tokenizer.bos_token_id is not None)
    bos_slots = 1 if has_bos else 0

    # Tokens used by the template scaffolding (everything except the raw context):
    # _CTX_PREFIX + _QA_SEP  →  tokenise without the context to get overhead
    template_overhead_ids = tokenizer.encode(
        f"{_CTX_PREFIX}{_QA_SEP}", add_special_tokens=False
    )
    overhead = len(template_overhead_ids) + bos_slots

    # Max tokens the context may use
    ctx_budget = max(0, max_length - len(completion_ids) - overhead)

    # ------------------------------------------------------------------
    # Step 3: tokenise and (if needed) truncate the context
    # ------------------------------------------------------------------
    ctx_ids = tokenizer.encode(ctx, add_special_tokens=False)
    if len(ctx_ids) > ctx_budget:
        ctx_ids = ctx_ids[:ctx_budget]

    # Rebuild prefix text from (possibly truncated) context tokens and re-decode
    # to get a clean string — avoids half-word tokenisation artefacts.
    truncated_ctx  = tokenizer.decode(ctx_ids, skip_special_tokens=True)
    prefix_text, _ = _build_texts(truncated_ctx, q, a)

    # ------------------------------------------------------------------
    # Step 4: assemble and tokenise the full sequence
    # ------------------------------------------------------------------
    full_text = prefix_text + completion_text

    full_enc = tokenizer(
        full_text,
        max_length=max_length,
        truncation=True,        # safety net; should not fire after step 3
        padding=False,
        add_special_tokens=True,
        return_tensors=None,
    )
    input_ids      = full_enc["input_ids"]
    attention_mask = full_enc["attention_mask"]

    # Explicitly append EOS if the tokenizer uses one and there is budget left.
    # (The tokenizer.__call__ may or may not add EOS depending on its config;
    # we normalise by always appending it so the model learns to stop.)
    if (
        eos is not None
        and len(input_ids) < max_length
        and (not input_ids or input_ids[-1] != eos)
    ):
        input_ids      = input_ids + [eos]
        attention_mask = attention_mask + [1]

    # ------------------------------------------------------------------
    # Step 5: build labels — find where the completion begins
    # ------------------------------------------------------------------
    # Encode prefix to count its tokens (with same special-token settings)
    prefix_enc_ids = tokenizer.encode(prefix_text, add_special_tokens=True)
    n_prefix       = len(prefix_enc_ids)

    # Sanity: if truncation removed part of the completion, keep what remains.
    n_prefix = min(n_prefix, len(input_ids))

    # Warn once if the completion was truncated away entirely
    if n_prefix == len(input_ids):
        log.warning(
            "Sample '%s': completion fully truncated; all labels = -100.",
            sample.get("sample_id", "?"),
        )

    labels = (
        [LABEL_IGNORE_ID] * n_prefix
        + list(input_ids[n_prefix:])
    )
    # Trim to input length (should already match, but defensive)
    labels = labels[:len(input_ids)]

    return {
        "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels":         torch.tensor(labels,         dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class QADataset(Dataset):
    """
    PyTorch Dataset for QA generation.

    Parameters
    ----------
    samples       : list of dicts (from preprocessing.load_jsonl / prepare_all)
    tokenizer     : HuggingFace tokenizer
    max_length    : token sequence length cap (default 512)
    qa_max_tokens : token budget reserved for Q+A completion (default 128)
    """

    def __init__(
        self,
        samples:       List[dict],
        tokenizer,
        max_length:    int = 512,
        qa_max_tokens: int = 128,
    ) -> None:
        self.samples       = samples
        self.tokenizer     = tokenizer
        self.max_length    = max_length
        self.qa_max_tokens = qa_max_tokens

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return _tokenize_sample(
            self.samples[idx],
            self.tokenizer,
            max_length=self.max_length,
            qa_max_tokens=self.qa_max_tokens,
        )

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_jsonl(
        cls,
        path:          str,
        tokenizer,
        max_length:    int = 512,
        qa_max_tokens: int = 128,
    ) -> "QADataset":
        """Load a .jsonl file produced by preprocessing.save_jsonl()."""
        return cls(
            samples=load_jsonl(path),
            tokenizer=tokenizer,
            max_length=max_length,
            qa_max_tokens=qa_max_tokens,
        )

    @classmethod
    def from_splits(
        cls,
        splits:        Dict[str, List[dict]],
        split_name:    str,
        tokenizer,
        max_length:    int = 512,
        qa_max_tokens: int = 128,
    ) -> "QADataset":
        """Build from an in-memory splits dict (output of prepare_all)."""
        return cls(
            samples=splits[split_name],
            tokenizer=tokenizer,
            max_length=max_length,
            qa_max_tokens=qa_max_tokens,
        )

    def label_coverage(self) -> float:
        """
        Fraction of samples that have at least one non-masked label token.
        A value < 1.0 indicates truncation is swallowing some completions.
        (Computes on a 10 % random sample to avoid full-dataset tokenisation.)
        """
        import random
        k     = max(1, len(self.samples) // 10)
        idxs  = random.sample(range(len(self.samples)), k)
        valid = sum(
            1 for i in idxs
            if (self[i]["labels"] != LABEL_IGNORE_ID).any()
        )
        return valid / k


# ---------------------------------------------------------------------------
# Collator (handles variable-length padding within a batch)
# ---------------------------------------------------------------------------

class QACollator:
    """
    Pads variable-length sequences in a batch to the longest sequence.

    Parameters
    ----------
    pad_token_id : int   (from tokenizer.pad_token_id)
    label_pad_id : int   (default -100)
    """

    def __init__(self, pad_token_id: int, label_pad_id: int = LABEL_IGNORE_ID) -> None:
        self.pad_token_id = pad_token_id
        self.label_pad_id = label_pad_id

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(item["input_ids"].shape[0] for item in batch)

        input_ids_list      = []
        attention_mask_list = []
        labels_list         = []

        for item in batch:
            seq_len  = item["input_ids"].shape[0]
            pad_len  = max_len - seq_len

            input_ids_list.append(
                torch.cat([item["input_ids"],
                           torch.full((pad_len,), self.pad_token_id, dtype=torch.long)])
            )
            attention_mask_list.append(
                torch.cat([item["attention_mask"],
                           torch.zeros(pad_len, dtype=torch.long)])
            )
            labels_list.append(
                torch.cat([item["labels"],
                           torch.full((pad_len,), self.label_pad_id, dtype=torch.long)])
            )

        return {
            "input_ids":      torch.stack(input_ids_list),
            "attention_mask": torch.stack(attention_mask_list),
            "labels":         torch.stack(labels_list),
        }


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_dataloader(
    samples:       List[dict],
    tokenizer,
    batch_size:    int  = 8,
    max_length:    int  = 512,
    qa_max_tokens: int  = 128,
    shuffle:       bool = True,
    num_workers:   int  = 0,
) -> DataLoader:
    """
    Build a DataLoader ready for the FederationClient training loop.

    Parameters
    ----------
    samples       : output of preprocessing.prepare_all() or load_jsonl()
    tokenizer     : HuggingFace tokenizer (must have pad_token_id set)
    batch_size    : samples per batch
    max_length    : token sequence length cap
    qa_max_tokens : token budget reserved for Q+A
    shuffle       : shuffle at each epoch (True for train, False for val/test)
    num_workers   : DataLoader worker processes
    """
    # Ensure the tokenizer has a pad token (required for collation)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            log.info("Tokenizer has no pad_token; using eos_token as pad.")
        else:
            raise ValueError(
                "Tokenizer has neither pad_token_id nor eos_token_id. "
                "Set tokenizer.pad_token_id manually before calling make_dataloader."
            )

    dataset  = QADataset(samples, tokenizer, max_length=max_length, qa_max_tokens=qa_max_tokens)
    collator = QACollator(pad_token_id=tokenizer.pad_token_id)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=False,
    )
