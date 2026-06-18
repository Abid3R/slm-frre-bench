"""Core scoring primitives shared by all tasks.

Two primitives cover the whole benchmark:
  1. sentence_log_likelihood  -> used by fairness (CrowS-Pairs) and any LL-based metric
  2. zero_shot_classify       -> used by robustness + capability (SST-2 style) tasks

Keeping scoring centralized is what makes the protocol "controlled": every model is
scored the exact same way.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def sentence_log_likelihood(model, tokenizer, text: str, device: str = "cuda") -> float:
    """Sum of per-token log-probabilities of `text` under a causal LM.

    This is the pseudo-log-likelihood used to compare two sentences in CrowS-Pairs.
    Higher = the model finds the sentence more probable.
    """
    enc = tokenizer(text, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    if input_ids.shape[1] < 2:
        return 0.0

    logits = model(**enc).logits  # [1, T, V]
    # Shift: predict token t from tokens < t.
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_ll = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # [1, T-1]
    return float(token_ll.sum().item())


@torch.no_grad()
def zero_shot_classify(
    model,
    tokenizer,
    text: str,
    labels: list[str],
    template: str = "Review: {text}\nSentiment ({options}):",
    device: str = "cuda",
) -> int:
    """Zero-shot single-label classification by comparing label log-likelihoods.

    We score the prompt + each candidate label and pick the most likely label.
    This avoids free-form generation parsing and works on base (non-chat) models.

    Returns the index into `labels`.
    """
    options = "/".join(labels)
    prompt = template.format(text=text, options=options)

    scores = []
    for label in labels:
        full = prompt + " " + label
        scores.append(sentence_log_likelihood(model, tokenizer, full, device=device))
    return int(max(range(len(labels)), key=lambda i: scores[i]))


def spearman(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation without scipy (small-n trade-off analysis)."""
    n = len(x)
    if n < 2:
        return float("nan")

    def rank(vals):
        order = sorted(range(n), key=lambda i: vals[i])
        r = [0.0] * n
        for pos, idx in enumerate(order):
            r[idx] = pos + 1
        return r

    rx, ry = rank(x), rank(y)
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1 - (6 * d2) / (n * (n * n - 1))
