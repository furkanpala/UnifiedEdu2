"""
evaluation/blooms.py

Bloom's Taxonomy classification and Educational Value Score (EVS).

Two methods:
  1. BERT-based classifier (fine-tuned on annotated questions)
  2. LLM judge with structured JSON output

EVS = (bloom_level - 1) / 5  in [0, 1]
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional, Tuple

log = logging.getLogger(__name__)

BLOOM_LEVELS = {
    1: "Remember",
    2: "Understand",
    3: "Apply",
    4: "Analyse",
    5: "Evaluate",
    6: "Create",
}

_bert_classifier = None
_bert_tokenizer  = None


def _load_bert_classifier(
    model_name: str = "facebook/bart-large-mnli",
    device: str = "cpu",
):
    """
    Load a zero-shot classifier as a proxy for Bloom's level classification.
    Replace model_name with a fine-tuned Bloom's BERT checkpoint when available.
    """
    global _bert_classifier, _bert_tokenizer
    if _bert_classifier is None:
        from transformers import pipeline
        log.info("Loading Bloom's classifier (%s)...", model_name)
        _bert_classifier = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=0 if device == "cuda" else -1,
        )
    return _bert_classifier


def compute_bloom_level_bert(
    questions: List[str],
    device:    str = "cpu",
) -> List[int]:
    """
    Classify each question into a Bloom's level [1, 6] using a
    zero-shot BART-based classifier.

    Returns
    -------
    List of int in [1, 6].
    """
    classifier = _load_bert_classifier(device=device)
    candidate_labels = [f"Level {k}: {v}" for k, v in BLOOM_LEVELS.items()]

    levels = []
    for q in questions:
        result = classifier(q, candidate_labels, multi_label=False)
        top_label = result["labels"][0]
        # Extract level number from label string "Level N:..."
        try:
            level = int(top_label.split(":")[0].split()[-1])
        except (ValueError, IndexError):
            level = 1
        levels.append(max(1, min(6, level)))
    return levels


def compute_bloom_level_llm(
    questions:  List[str],
    course:     str = "machine learning",
    model_name: str = "gpt-3.5-turbo",
    api_key:    Optional[str] = None,
) -> List[Tuple[int, str]]:
    """
    Classify each question using an LLM judge with structured JSON output.

    Returns
    -------
    List of (level: int, reason: str) tuples.
    """
    try:
        from openai import OpenAI
    except ImportError:
        log.warning("openai package not found; returning level=1 for all questions.")
        return [(1, "openai not available")] * len(questions)

    client = OpenAI(api_key=api_key)

    bloom_desc = "\n".join(
        f"  {k}: {v}" for k, v in BLOOM_LEVELS.items()
    )
    results = []
    for q in questions:
        prompt = (
            f"You are an expert in educational assessment for a {course} course.\n"
            f"Classify the following question according to Bloom's Revised Taxonomy:\n\n"
            f"Question: {q}\n\n"
            f"Bloom's levels:\n{bloom_desc}\n\n"
            f"Respond ONLY with valid JSON: "
            f'{{ "level": <int 1-6>, "reason": "<brief justification>" }}'
        )
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100,
            )
            raw = response.choices[0].message.content.strip()
            parsed = json.loads(raw)
            level  = max(1, min(6, int(parsed.get("level", 1))))
            reason = str(parsed.get("reason", ""))
            results.append((level, reason))
        except Exception as e:
            log.warning("LLM Bloom's judge failed: %s", e)
            results.append((1, "error"))
    return results


def compute_bloom_level(
    questions: List[str],
    method:    str = "bert",
    device:    str = "cpu",
    **kwargs,
) -> List[int]:
    """
    Unified interface. method in {"bert", "llm"}.
    Returns list of int in [1, 6].
    """
    if method == "llm":
        pairs = compute_bloom_level_llm(questions, **kwargs)
        return [p[0] for p in pairs]
    return compute_bloom_level_bert(questions, device=device)


def compute_evs(bloom_levels: List[int]) -> List[float]:
    """
    Educational Value Score: EVS(q) = (b - 1) / 5 in [0, 1].
    """
    return [(b - 1) / 5.0 for b in bloom_levels]