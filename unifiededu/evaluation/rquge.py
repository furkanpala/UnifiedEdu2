<<<<<<< HEAD
"""
evaluation/rquge.py

RQUGE: Reference-free QUestion Generation Evaluation.

Score range: [1, 5].
a_c = QA(q, c)          -- UnifiedQAv2 answers the question given context
kappa = S(q, a_c, a, c) -- RoBERTa-based MOCHA scorer

In our setting the generated answer a serves as the gold span a_r.
"""

from __future__ import annotations

import logging
from typing import List, Optional

log = logging.getLogger(__name__)

_qa_model    = None
_qa_tokenizer = None
_scorer      = None


def _load_qa_model(device: str = "cpu"):
    global _qa_model, _qa_tokenizer
    if _qa_model is None:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        model_name = "allenai/unifiedqa-v2-t5-large-1251000"
        log.info("Loading UnifiedQAv2 from %s...", model_name)
        _qa_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _qa_model     = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        _qa_model.eval()
    return _qa_model, _qa_tokenizer


def _load_scorer(device: str = "cpu"):
    """
    RQUGE scorer: RoBERTa fine-tuned on MOCHA.
    We use the rquge package if available, else fall back to a
    token-F1 approximation between a_c and a_r.
    """
    global _scorer
    if _scorer is None:
        try:
            from rquge_score import RQUGE
            _scorer = RQUGE(device=device)
            log.info("Loaded RQUGE scorer.")
        except ImportError:
            log.warning(
                "rquge_score package not found. "
                "Falling back to token-F1 as RQUGE approximation."
            )
            _scorer = "token_f1"
    return _scorer


def _token_f1(pred: str, gold: str) -> float:
    """Token-level F1 between two strings (fallback scorer)."""
    pred_toks = set(pred.lower().split())
    gold_toks = set(gold.lower().split())
    if not pred_toks or not gold_toks:
        return 0.0
    common = pred_toks & gold_toks
    if not common:
        return 0.0
    p = len(common) / len(pred_toks)
    r = len(common) / len(gold_toks)
    return 2 * p * r / (p + r)


def _answer_question(question: str, context: str, device: str = "cpu") -> str:
    """Run UnifiedQAv2 to answer `question` given `context`."""
    model, tokenizer = _load_qa_model(device)
    import torch
    input_text = f"{question} \\n {context}"
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
    ).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64)
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()


def compute_rquge(
    questions: List[str],
    answers:   List[str],
    contexts:  List[str],
    device:    str = "cpu",
) -> List[float]:
    """
    Compute RQUGE score for each (question, answer, context) triple.

    Parameters
    ----------
    questions : list of generated questions
    answers   : list of generated answers (used as gold span a_r)
    contexts  : list of source contexts

    Returns
    -------
    List of float scores in [1, 5] (or [0, 1] if using token-F1 fallback).
    """
    scorer = _load_scorer(device)
    scores = []

    for q, a, c in zip(questions, answers, contexts):
        a_c = _answer_question(q, c, device=device)

        if scorer == "token_f1":
            # Scale token-F1 from [0,1] to [1,5]
            f1 = _token_f1(a_c, a)
            scores.append(1.0 + 4.0 * f1)
        else:
            try:
                kappa = scorer.score(
                    question=q,
                    candidate_answer=a_c,
                    reference_answer=a,
                    context=c,
                )
                scores.append(float(kappa))
            except Exception as e:
                log.warning("RQUGE scorer failed for sample: %s", e)
                f1 = _token_f1(a_c, a)
                scores.append(1.0 + 4.0 * f1)

=======
"""
evaluation/rquge.py

RQUGE: Reference-free QUestion Generation Evaluation.

Score range: [1, 5].
a_c = QA(q, c)          -- UnifiedQAv2 answers the question given context
kappa = S(q, a_c, a, c) -- RoBERTa-based MOCHA scorer

In our setting the generated answer a serves as the gold span a_r.
"""

from __future__ import annotations

import logging
from typing import List, Optional

log = logging.getLogger(__name__)

_qa_model    = None
_qa_tokenizer = None
_scorer      = None


def _load_qa_model(device: str = "cpu"):
    global _qa_model, _qa_tokenizer
    if _qa_model is None:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        model_name = "allenai/unifiedqa-v2-t5-large-1251000"
        log.info("Loading UnifiedQAv2 from %s...", model_name)
        _qa_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _qa_model     = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        _qa_model.eval()
    return _qa_model, _qa_tokenizer


def _load_scorer(device: str = "cpu"):
    """
    RQUGE scorer: RoBERTa fine-tuned on MOCHA.
    We use the rquge package if available, else fall back to a
    token-F1 approximation between a_c and a_r.
    """
    global _scorer
    if _scorer is None:
        try:
            from rquge_score import RQUGE
            _scorer = RQUGE(device=device)
            log.info("Loaded RQUGE scorer.")
        except ImportError:
            log.warning(
                "rquge_score package not found. "
                "Falling back to token-F1 as RQUGE approximation."
            )
            _scorer = "token_f1"
    return _scorer


def _token_f1(pred: str, gold: str) -> float:
    """Token-level F1 between two strings (fallback scorer)."""
    pred_toks = set(pred.lower().split())
    gold_toks = set(gold.lower().split())
    if not pred_toks or not gold_toks:
        return 0.0
    common = pred_toks & gold_toks
    if not common:
        return 0.0
    p = len(common) / len(pred_toks)
    r = len(common) / len(gold_toks)
    return 2 * p * r / (p + r)


def _answer_question(question: str, context: str, device: str = "cpu") -> str:
    """Run UnifiedQAv2 to answer `question` given `context`."""
    model, tokenizer = _load_qa_model(device)
    import torch
    input_text = f"{question} \\n {context}"
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
    ).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64)
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()


def compute_rquge(
    questions: List[str],
    answers:   List[str],
    contexts:  List[str],
    device:    str = "cpu",
) -> List[float]:
    """
    Compute RQUGE score for each (question, answer, context) triple.

    Parameters
    ----------
    questions : list of generated questions
    answers   : list of generated answers (used as gold span a_r)
    contexts  : list of source contexts

    Returns
    -------
    List of float scores in [1, 5] (or [0, 1] if using token-F1 fallback).
    """
    scorer = _load_scorer(device)
    scores = []

    for q, a, c in zip(questions, answers, contexts):
        a_c = _answer_question(q, c, device=device)

        if scorer == "token_f1":
            # Scale token-F1 from [0,1] to [1,5]
            f1 = _token_f1(a_c, a)
            scores.append(1.0 + 4.0 * f1)
        else:
            try:
                kappa = scorer.score(
                    question=q,
                    candidate_answer=a_c,
                    reference_answer=a,
                    context=c,
                )
                scores.append(float(kappa))
            except Exception as e:
                log.warning("RQUGE scorer failed for sample: %s", e)
                f1 = _token_f1(a_c, a)
                scores.append(1.0 + 4.0 * f1)

>>>>>>> 9e0cc5677294928b6f41dd80931d26dd2eac2e06
    return scores