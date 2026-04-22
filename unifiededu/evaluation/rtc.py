"""
evaluation/rtc.py

Round-Trip Consistency (RTC): token-level F1 between
QA(q, c) and the generated answer a.
"""

from __future__ import annotations

from typing import List


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


def compute_rtc(
    questions: List[str],
    answers:   List[str],
    contexts:  List[str],
    device:    str = "cpu",
) -> List[float]:
    """
    Compute Round-Trip Consistency for each (question, answer, context).

    RTC(q, a, c) = token_F1(QA(q, c), a)

    Returns
    -------
    List of float in [0, 1].
    """
    from.rquge import _answer_question

    scores = []
    for q, a, c in zip(questions, answers, contexts):
        a_c = _answer_question(q, c, device=device)
        scores.append(_token_f1(a_c, a))
    return scores


def compute_answer_relevancy(
    questions: List[str],
    answers:   List[str],
    n_reverse: int = 3,
    device:    str = "cpu",
) -> List[float]:
    """
    RAGAS Answer Relevancy: cosine similarity between
    embed(q) and mean embed(reverse_questions from a).

    Returns
    -------
    List of float in [-1, 1] (typically [0, 1]).
    """
    import torch
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device=device,
    )

    scores = []
    for q, a in zip(questions, answers):
        # Generate reverse questions from the answer
        words = a.split()
        reverse_qs = []
        chunk = max(1, len(words) // n_reverse)
        for i in range(n_reverse):
            span = " ".join(words[i * chunk: (i + 1) * chunk])
            reverse_qs.append(f"What is {span}?")

        q_emb   = model.encode([q],          convert_to_tensor=True)
        rq_embs = model.encode(reverse_qs,   convert_to_tensor=True)

        q_norm  = torch.nn.functional.normalize(q_emb,   dim=-1)
        rq_norm = torch.nn.functional.normalize(rq_embs, dim=-1)

        sim = (q_norm * rq_norm).sum(dim=-1).mean().item()
        scores.append(float(sim))

    return scores