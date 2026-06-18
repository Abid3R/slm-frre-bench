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


DEFAULT_TEMPLATE = "Review: {text}\nSentiment ({options}):"


@torch.no_grad()
def _continuation_ll(model, tokenizer, prompt: str, continuation: str,
                     device: str = "cuda") -> float:
    """Summed log-prob of `continuation` tokens *conditioned on* `prompt`.

    Unlike scoring the whole `prompt + continuation` string, this isolates the
    label tokens, which is what calibration needs (the prompt-likelihood term
    must not leak in when we subtract a null-prompt baseline).
    """
    p_ids = tokenizer(prompt, return_tensors="pt").to(device)["input_ids"]
    full = tokenizer(prompt + continuation, return_tensors="pt").to(device)
    ids = full["input_ids"]
    p_len = p_ids.shape[1]
    if ids.shape[1] <= p_len:
        return 0.0
    logits = model(**full).logits  # [1, T, V]
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    tok_ll = log_probs.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)[0]  # [T-1]
    # Token at position p_len in `ids` is predicted from logits[p_len-1]; that
    # index in the shifted `tok_ll` is p_len-1. So continuation tokens start there.
    return float(tok_ll[p_len - 1:].sum().item())


@torch.no_grad()
def null_label_scores(model, tokenizer, labels: list[str],
                      template: str = DEFAULT_TEMPLATE, device: str = "cuda",
                      null_text: str = "N/A") -> list[float]:
    """Per-label log-likelihood under a content-free prompt (contextual calibration).

    Computed ONCE per (model, labels, template); subtracted from every example's
    label scores to remove each model's intrinsic label prior. See Zhao et al.
    (2021), "Calibrate Before Use".
    """
    options = "/".join(labels)
    null_prompt = template.format(text=null_text, options=options)
    return [_continuation_ll(model, tokenizer, null_prompt, " " + lb, device)
            for lb in labels]


@torch.no_grad()
def zero_shot_classify(
    model,
    tokenizer,
    text: str,
    labels: list[str],
    template: str = DEFAULT_TEMPLATE,
    device: str = "cuda",
    null_scores: list[float] | None = None,
) -> int:
    """Zero-shot single-label classification by comparing label log-likelihoods.

    We score the conditional log-likelihood of each candidate label given the
    prompt and pick the most likely label. This avoids free-form generation
    parsing and works on base (non-chat) models.

    If `null_scores` is provided (from `null_label_scores`), we apply contextual
    calibration: label_score = logP(label | prompt) - logP(label | null_prompt).
    This cancels each model's intrinsic preference for one label word, which is
    what otherwise pins weak models to the class-balance (chance) accuracy.

    Returns the index into `labels`.
    """
    options = "/".join(labels)
    prompt = template.format(text=text, options=options)

    scores = [_continuation_ll(model, tokenizer, prompt, " " + lb, device=device)
              for lb in labels]
    if null_scores is not None:
        scores = [s - n for s, n in zip(scores, null_scores)]
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
