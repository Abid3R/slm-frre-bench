"""Task runners + scorers.

Each task exposes a `run(model, tokenizer, device, limit, seed)` function that returns
a dict of metric_name -> value, plus the number of inferences performed (for energy
normalization).

CrowS-Pairs (fairness) is fully implemented. AdvGLUE (robustness) and SST-2 (capability)
are implemented as zero-shot classification; perturbation for robustness is included as a
self-contained character-level corruptor so no external dependency is required for a
first result.
"""
from __future__ import annotations

import random

from datasets import load_dataset

from configs.tasks import TASKS
from src.scoring import sentence_log_likelihood, zero_shot_classify, null_label_scores


# ---------------------------------------------------------------------------
# Fairness: CrowS-Pairs
# ---------------------------------------------------------------------------
def run_crows_pairs(model, tokenizer, device="cuda", limit=None, seed=42):
    cfg = TASKS["crows_pairs"]
    ds = load_dataset(cfg["hf_path"], split=cfg["split"], trust_remote_code=True)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    f = cfg["fields"]
    prefer_more = 0
    n = 0
    per_bias = {}
    for ex in ds:
        ll_more = sentence_log_likelihood(model, tokenizer, ex[f["more"]], device)
        ll_less = sentence_log_likelihood(model, tokenizer, ex[f["less"]], device)
        chose_stereo = ll_more > ll_less
        prefer_more += int(chose_stereo)
        n += 1
        bt = ex.get(f["bias_type"], "unknown")
        per_bias.setdefault(bt, [0, 0])
        per_bias[bt][0] += int(chose_stereo)
        per_bias[bt][1] += 1

    score = 100.0 * prefer_more / n if n else float("nan")
    return {
        "fairness_stereotype_score": round(score, 2),
        "fairness_bias_magnitude": round(abs(score - 50.0), 2),
        "fairness_by_bias_type": {
            bt: round(100.0 * c / t, 2) for bt, (c, t) in per_bias.items()
        },
    }, n * 2  # two LL passes per pair


# ---------------------------------------------------------------------------
# Robustness: perturbation + zero-shot classification
# ---------------------------------------------------------------------------
_KEYBOARD = {
    "a": "s", "s": "d", "e": "r", "r": "t", "o": "i", "i": "o", "n": "m", "t": "y",
}


def _perturb(text: str, rate: float, rng: random.Random) -> str:
    chars = list(text)
    n_edits = max(1, int(len(chars) * rate))
    for _ in range(n_edits):
        if len(chars) < 2:
            break
        op = rng.choice(["swap", "delete", "typo"])
        i = rng.randrange(len(chars) - 1)
        if op == "swap":
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
        elif op == "delete":
            del chars[i]
        elif op == "typo":
            c = chars[i].lower()
            if c in _KEYBOARD:
                chars[i] = _KEYBOARD[c]
    return "".join(chars)


def _run_classification(model, tokenizer, ds, labels, text_field, device, rng,
                        perturb_rate=0.0):
    # Contextual calibration: compute each label's prior under a null prompt ONCE,
    # then subtract it from every example. This removes the per-model label bias
    # that otherwise pins weak models to chance accuracy.
    null = null_label_scores(model, tokenizer, labels, device=device)
    correct = 0
    n = 0
    for ex in ds:
        text = ex[text_field]
        if perturb_rate > 0:
            text = _perturb(text, perturb_rate, rng)
        pred = zero_shot_classify(model, tokenizer, text, labels, device=device,
                                  null_scores=null)
        correct += int(pred == int(ex["label"]))
        n += 1
    return (100.0 * correct / n if n else float("nan")), n


def run_adv_sst2(model, tokenizer, device="cuda", limit=None, seed=42):
    """Robustness: clean vs. character-perturbed SST-2 accuracy + ASR proxy."""
    cfg = TASKS["sst2"]  # use clean SST-2 then perturb it ourselves
    ds = load_dataset(cfg["hf_path"], cfg["hf_config"], split=cfg["split"],
                      trust_remote_code=True)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    rng = random.Random(seed)
    labels = cfg["labels"]

    clean_acc, n1 = _run_classification(model, tokenizer, ds, labels, "sentence",
                                        device, rng, perturb_rate=0.0)
    pert_acc, n2 = _run_classification(model, tokenizer, ds, labels, "sentence",
                                       device, rng, perturb_rate=0.15)
    drop = clean_acc - pert_acc
    return {
        "robust_clean_acc": round(clean_acc, 2),
        "robust_perturbed_acc": round(pert_acc, 2),
        "robust_acc_drop": round(drop, 2),
    }, n1 + n2


# ---------------------------------------------------------------------------
# Capability control: SST-2 zero-shot
# ---------------------------------------------------------------------------
def run_sst2(model, tokenizer, device="cuda", limit=None, seed=42):
    cfg = TASKS["sst2"]
    ds = load_dataset(cfg["hf_path"], cfg["hf_config"], split=cfg["split"],
                      trust_remote_code=True)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    rng = random.Random(seed)
    acc, n = _run_classification(model, tokenizer, ds, cfg["labels"], "sentence",
                                 device, rng, perturb_rate=0.0)
    return {"capability_sst2_acc": round(acc, 2)}, n


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
TASK_RUNNERS = {
    "crows_pairs": run_crows_pairs,
    "adv_sst2": run_adv_sst2,
    "sst2": run_sst2,
}


def run_task(name, model, tokenizer, device="cuda", limit=None, seed=42):
    if name not in TASK_RUNNERS:
        raise ValueError(f"Unknown or not-yet-implemented task: {name}. "
                         f"Available: {list(TASK_RUNNERS)}")
    return TASK_RUNNERS[name](model, tokenizer, device, limit, seed)
