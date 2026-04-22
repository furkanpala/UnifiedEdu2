"""
evaluation/qafacteval.py

QAFactEval: QA-based factual consistency score.
Uses a QA model to verify whether answer spans can be recovered
from the context, scored by token-F1 (LERC approximation).
"""

from __future__ import annotations

import logging
from typing import List

log = logging.getLogger(__name__)


def _generate_internal_qa_pairs(
    answer: str,
    n: int = 3,
) -> List[str]:
    """
    Extract n key spans from the answer as internal questions.
    Simple heuristic: split into sentences and use each as a question.
    """
    import re
    sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    # Convert to cloze-style questions
    questions = []
    for sent in sentences[:n]:
        words = sent.split()
        if len(words) > 3:
            # Blank out the last noun-like word as the answer span
            questions.append(" ".join(words[:-1]) + " ___?")
        else:
            questions.append(sent + "?")
    return questions[:n] if questions else [answer[:50] + "?"]


def _token_f1(pred: str, gold: str) -> float:
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


def compute_qafacteval(
    answers:  List[str],
    contexts: List[str],
    device:   str = "cpu",
    n:        int = 3,
) -> List[float]:
    """
    Compute QAFactEval score for each (answer, context) pair.

    Returns
    -------
    List of float in [0, 1].
    """
    from.rquge import _answer_question

    scores = []
    for a, c in zip(answers, contexts):
        internal_qs = _generate_internal_qa_pairs(a, n=n)
        pair_scores = []
        for q_int in internal_qs:
            # Gold span: extract from answer
            gold_span = a.split()[-3:] if a.split() else [""]
            gold_span = " ".join(gold_span)
            # Predicted span: QA model reads context
            pred_span = _answer_question(q_int, c, device=device)
            pair_scores.append(_token_f1(pred_span, gold_span))
        scores.append(sum(pair_scores) / len(pair_scores) if pair_scores else 0.0)
    return scores