"""Bootstrap confidence intervals for SLM-FRRE-Bench (reviewer concern C6).

This is a STANDALONE post-hoc analysis utility. It does not import, modify, or
depend on the evaluation harness in any way. It only reads the per-example
``.jsonl`` outcome logs emitted by a logging-enabled run and writes a single
``bootstrap_ci.csv`` with a point estimate and a 95% percentile bootstrap
confidence interval for each per-model metric.

Expected input files (one JSON object per line) in the --logs directory:

    per_example_crows_<model>_<seed>.jsonl
        fields: index, bias_type, chose_stereo   (chose_stereo in {0,1})

    per_example_sst2clean_<model>_<seed>.jsonl
        fields: index, pred, gold, correct        (correct in {0,1})

    per_example_sst2pert_<model>_<seed>.jsonl
        fields: index, pred, gold, correct

    per_example_capability_<model>_<seed>.jsonl
        fields: index, pred, gold, correct

Filename rule: the trailing token before ``.jsonl`` is the integer seed; the
token(s) between the task name and the seed are the model id (model ids may
contain ``_``, ``-`` or ``.``).

Metrics produced (matching the paper):
    fairness_stereotype_score   100 * mean(chose_stereo)
    fairness_bias_magnitude     |stereotype_score - 50|
    capability_sst2_acc         100 * mean(correct)            [capability log]
    robust_clean_acc            100 * mean(correct)            [sst2clean log]
    robust_perturbed_acc        100 * mean(correct)            [sst2pert log]
    robust_acc_drop             clean_acc - perturbed_acc      [PAIRED by index]

The acc-drop interval uses paired resampling: a single set of example positions
is drawn per bootstrap replicate and applied to BOTH the clean and perturbed
arrays (which are aligned on the shared ``index`` field), so the difference CI
correctly reflects the within-example correlation between the two passes.

Usage:
    python bootstrap_ci.py --logs results/ --out results/bootstrap_ci.csv
    python bootstrap_ci.py --logs results/ --B 10000 --seed 12345
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------
_TASKS = ("crows", "sst2clean", "sst2pert", "capability")
_FNAME_RE = re.compile(
    r"^per_example_(crows|sst2clean|sst2pert|capability)_(.+)_(\d+)\.jsonl$"
)


def parse_filename(path):
    """Return (task, model, seed) or None if the name does not match."""
    base = os.path.basename(path)
    m = _FNAME_RE.match(base)
    if not m:
        return None
    task, model, seed = m.group(1), m.group(2), int(m.group(3))
    return task, model, seed


# ---------------------------------------------------------------------------
# JSONL loading
# ---------------------------------------------------------------------------
def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  ! skipping bad line {ln} in {path}: {e}", file=sys.stderr)
    return rows


def _outcome_array(rows, key, alt_keys=()):
    """Pull a 0/1 outcome column out of loaded rows, tolerant to key naming."""
    keys = (key,) + tuple(alt_keys)
    out = []
    for r in rows:
        v = None
        for k in keys:
            if k in r:
                v = r[k]
                break
        if v is None:
            # last resort: derive correctness from pred/gold if present
            if "pred" in r and "gold" in r:
                v = int(int(r["pred"]) == int(r["gold"]))
            else:
                raise KeyError(
                    f"row missing any of {keys} (and no pred/gold): {r}"
                )
        out.append(int(bool(v)) if isinstance(v, bool) else int(v))
    return np.asarray(out, dtype=np.float64)


def _indexed_outcomes(rows, key, alt_keys=()):
    """Return dict index -> 0/1 outcome, for paired alignment."""
    vals = _outcome_array(rows, key, alt_keys)
    idx = [int(r["index"]) for r in rows]
    return dict(zip(idx, vals.tolist()))


# ---------------------------------------------------------------------------
# Bootstrap core
# ---------------------------------------------------------------------------
def _resample_means(arr, B, rng):
    """Return a length-B array of bootstrap means of `arr` (with replacement)."""
    n = arr.shape[0]
    idx = rng.integers(0, n, size=(B, n))
    return arr[idx].mean(axis=1)


def ci_proportion(arr, B, rng, scale=100.0, transform=None):
    """Point estimate + 95% percentile CI for `scale * mean(arr)`.

    `transform` optionally maps the scaled mean to another statistic
    (used for bias magnitude = |score - 50|).
    """
    point = scale * float(arr.mean())
    boot = scale * _resample_means(arr, B, rng)
    if transform is not None:
        point = transform(point)
        boot = transform(boot)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return point, float(lo), float(hi)


def ci_paired_diff(clean, pert, B, rng, scale=100.0):
    """Point estimate + 95% CI for scale*(mean(clean) - mean(pert)), PAIRED.

    `clean` and `pert` must be aligned element-wise (same example order).
    """
    n = clean.shape[0]
    point = scale * (float(clean.mean()) - float(pert.mean()))
    idx = rng.integers(0, n, size=(B, n))
    boot = scale * (clean[idx].mean(axis=1) - pert[idx].mean(axis=1))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return point, float(lo), float(hi)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Bootstrap CIs for SLM-FRRE-Bench (C6).")
    ap.add_argument("--logs", default="results",
                    help="directory containing per_example_*.jsonl files")
    ap.add_argument("--out", default="results/bootstrap_ci.csv",
                    help="output CSV path")
    ap.add_argument("--B", type=int, default=10000,
                    help="number of bootstrap resamples (default 10000)")
    ap.add_argument("--seed", type=int, default=12345,
                    help="RNG seed for reproducible resampling")
    args = ap.parse_args()

    pattern = os.path.join(args.logs, "per_example_*.jsonl")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No per_example_*.jsonl files found under {args.logs!r}.\n"
              f"Run the logging-enabled evaluation first.", file=sys.stderr)
        sys.exit(1)

    # group[(model, seed)][task] = filepath
    group = defaultdict(dict)
    for path in files:
        parsed = parse_filename(path)
        if parsed is None:
            print(f"  ! unrecognized filename, skipping: {path}", file=sys.stderr)
            continue
        task, model, seed = parsed
        group[(model, seed)][task] = path

    rng = np.random.default_rng(args.seed)
    records = []  # dict rows for CSV

    for (model, seed) in sorted(group):
        tasks = group[(model, seed)]
        print(f"[{model} | seed {seed}] tasks: {sorted(tasks)}")

        # --- Fairness (CrowS-Pairs) ---
        if "crows" in tasks:
            rows = load_jsonl(tasks["crows"])
            stereo = _outcome_array(rows, "chose_stereo", alt_keys=("stereo",))
            pt, lo, hi = ci_proportion(stereo, args.B, rng)
            records.append(dict(model=model, seed=seed,
                                metric="fairness_stereotype_score",
                                n=stereo.size, point=pt, ci_low=lo, ci_high=hi))
            ptm, lom, him = ci_proportion(
                stereo, args.B, rng, transform=lambda x: np.abs(x - 50.0))
            records.append(dict(model=model, seed=seed,
                                metric="fairness_bias_magnitude",
                                n=stereo.size, point=ptm, ci_low=lom, ci_high=him))
        else:
            print(f"  ! no crows log for {model} seed {seed}", file=sys.stderr)

        # --- Capability (clean SST-2, capability axis) ---
        if "capability" in tasks:
            rows = load_jsonl(tasks["capability"])
            corr = _outcome_array(rows, "correct")
            pt, lo, hi = ci_proportion(corr, args.B, rng)
            records.append(dict(model=model, seed=seed,
                                metric="capability_sst2_acc",
                                n=corr.size, point=pt, ci_low=lo, ci_high=hi))
        else:
            print(f"  ! no capability log for {model} seed {seed}", file=sys.stderr)

        # --- Robustness (clean vs perturbed, paired) ---
        if "sst2clean" in tasks and "sst2pert" in tasks:
            cmap = _indexed_outcomes(load_jsonl(tasks["sst2clean"]), "correct")
            pmap = _indexed_outcomes(load_jsonl(tasks["sst2pert"]), "correct")
            shared = sorted(set(cmap) & set(pmap))
            if not shared:
                print(f"  ! clean/pert logs share no indices for {model}",
                      file=sys.stderr)
            else:
                clean = np.asarray([cmap[i] for i in shared], dtype=np.float64)
                pert = np.asarray([pmap[i] for i in shared], dtype=np.float64)
                pc, lc, hc = ci_proportion(clean, args.B, rng)
                records.append(dict(model=model, seed=seed,
                                    metric="robust_clean_acc",
                                    n=clean.size, point=pc, ci_low=lc, ci_high=hc))
                pp, lp, hp = ci_proportion(pert, args.B, rng)
                records.append(dict(model=model, seed=seed,
                                    metric="robust_perturbed_acc",
                                    n=pert.size, point=pp, ci_low=lp, ci_high=hp))
                pd_, ld, hd = ci_paired_diff(clean, pert, args.B, rng)
                records.append(dict(model=model, seed=seed,
                                    metric="robust_acc_drop",
                                    n=clean.size, point=pd_, ci_low=ld, ci_high=hd))
        else:
            print(f"  ! need BOTH sst2clean and sst2pert logs for robustness "
                  f"({model} seed {seed})", file=sys.stderr)

    if not records:
        print("No metrics computed -- check that the logs contain the expected "
              "fields.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fields = ["model", "seed", "metric", "n", "point", "ci_low", "ci_high",
              "ci_halfwidth"]
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in records:
            r["ci_halfwidth"] = round((r["ci_high"] - r["ci_low"]) / 2.0, 4)
            for k in ("point", "ci_low", "ci_high"):
                r[k] = round(r[k], 4)
            w.writerow(r)

    print(f"\nWrote {len(records)} rows -> {args.out}")
    print("Columns: model, seed, metric, n, point, ci_low, ci_high, ci_halfwidth")


if __name__ == "__main__":
    main()
