<<<<<<< HEAD
"""
evaluation/faithfulness.py

NLI-based RAGAS Faithfulness (Section 7.2 of the paper).

Replaces the LLM verifier with DeBERTa-large-mnli so that the
faithfulness check is deterministic and hallucination-free.

Score in [0, 1]: fraction of atomic claims entailed by the context.
"""

from __future__ import annotations

import logging
import re
from typing import List

log = logging.getLogger(__name__)

_claim_splitter = None
_nli_model      = None
_nli_tokenizer  = None


def _load_nli(device: str = "cpu"):
    global _nli_model, _nli_tokenizer
    if _nli_model is None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        model_name = "cross-encoder/nli-deberta-v3-large"
        log.info("Loading NLI model %s...", model_name)
        _nli_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _nli_model     = AutoModelForSequenceClassification.from_pretrained(
            model_name
        ).to(device)
        _nli_model.eval()
    return _nli_model, _nli_tokenizer


def _split_into_claims(text: str) -> List[str]:
    """
    Split answer text into atomic claims.

    Uses simple sentence splitting as a lightweight alternative to
    an LLM-based semantic parser.  Each sentence is treated as one claim.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _is_entailed(context: str, claim: str, device: str = "cpu") -> bool:
    """Return True if context entails claim according to the NLI model."""
    import torch
    model, tokenizer = _load_nli(device)
    inputs = tokenizer(
        context, claim,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True,
    ).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    # Label order for cross-encoder/nli: 0=contradiction, 1=neutral, 2=entailment
    pred = logits.argmax(dim=-1).item()
    return pred == 2


def compute_faithfulness(
    answers:  List[str],
    contexts: List[str],
    device:   str = "cpu",
) -> List[float]:
    """
    Compute NLI-based faithfulness for each (answer, context) pair.

    Returns
    -------
    List of float in [0, 1].
    """
    scores = []
    for a, c in zip(answers, contexts):
        claims = _split_into_claims(a)
        if not claims:
            scores.append(0.0)
            continue
        entailed = sum(1 for s in claims if _is_entailed(c, s, device=device))
        scores.append(entailed / len(claims))
=======
"""
evaluation/faithfulness.py

NLI-based RAGAS Faithfulness (Section 7.2 of the paper).

Replaces the LLM verifier with DeBERTa-large-mnli so that the
faithfulness check is deterministic and hallucination-free.

Score in [0, 1]: fraction of atomic claims entailed by the context.
"""

from __future__ import annotations

import logging
import re
from typing import List

log = logging.getLogger(__name__)

_claim_splitter = None
_nli_model      = None
_nli_tokenizer  = None


def _load_nli(device: str = "cpu"):
    global _nli_model, _nli_tokenizer
    if _nli_model is None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        model_name = "cross-encoder/nli-deberta-v3-large"
        log.info("Loading NLI model %s...", model_name)
        _nli_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _nli_model     = AutoModelForSequenceClassification.from_pretrained(
            model_name
        ).to(device)
        _nli_model.eval()
    return _nli_model, _nli_tokenizer


def _split_into_claims(text: str) -> List[str]:
    """
    Split answer text into atomic claims.

    Uses simple sentence splitting as a lightweight alternative to
    an LLM-based semantic parser.  Each sentence is treated as one claim.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _is_entailed(context: str, claim: str, device: str = "cpu") -> bool:
    """Return True if context entails claim according to the NLI model."""
    import torch
    model, tokenizer = _load_nli(device)
    inputs = tokenizer(
        context, claim,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True,
    ).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    # Label order for cross-encoder/nli: 0=contradiction, 1=neutral, 2=entailment
    pred = logits.argmax(dim=-1).item()
    return pred == 2


def compute_faithfulness(
    answers:  List[str],
    contexts: List[str],
    device:   str = "cpu",
) -> List[float]:
    """
    Compute NLI-based faithfulness for each (answer, context) pair.

    Returns
    -------
    List of float in [0, 1].
    """
    scores = []
    for a, c in zip(answers, contexts):
        claims = _split_into_claims(a)
        if not claims:
            scores.append(0.0)
            continue
        entailed = sum(1 for s in claims if _is_entailed(c, s, device=device))
        scores.append(entailed / len(claims))
>>>>>>> 9e0cc5677294928b6f41dd80931d26dd2eac2e06
    return scores