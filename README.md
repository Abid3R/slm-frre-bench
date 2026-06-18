# SLM-FRRE-Bench

**A unified benchmark for Fairness, Robustness, and Energy efficiency of Small Language Models (SLMs).**

> Research question: *Across open small language models (≤3B params), how do fairness,
> adversarial robustness, and energy efficiency trade off against each other under a
> single controlled evaluation protocol?*

The novelty is the **joint** evaluation of all three axes on the **same** models under the
**same** protocol, plus the **trade-off analysis** (e.g., does the most energy-efficient
model sacrifice fairness?). Prior work (SLM-Bench, FLEX) measures these axes separately.

---

## What this does

For each model in the registry, the harness:
1. Runs **fairness** tasks (CrowS-Pairs stereotype preference — fully implemented).
2. Runs **robustness** tasks (clean vs. perturbed accuracy — scaffolded).
3. Runs a **capability control** (zero-shot SST-2 — scaffolded).
4. Measures **energy** (kWh + gCO2) for the whole run via CodeCarbon.
5. Writes one row per (model, seed) to `results/results.csv`.

Everything runs on a **free Kaggle T4 / Colab GPU** or CPU (for the smallest models).

---

## Quick start (local smoke test)

```bash
pip install -r requirements.txt

# Fastest possible first result: 1 small model, fairness + energy only
python -m src.run --models Qwen/Qwen2.5-0.5B-Instruct --tasks crows_pairs --limit 50
```

This prints a CrowS-Pairs stereotype score (50 = unbiased) and an energy reading.

## Full grid

```bash
python -m src.run --all --seeds 42 1337 2024
```

## On Kaggle

Copy the contents of `notebooks/kaggle_entry.py` into a Kaggle notebook cell
(GPU enabled). It clones this repo and runs the grid.

---

## Repo layout

```
slm-frre-bench/
├── README.md
├── requirements.txt
├── METRICS.md            # exact metric definitions (reviewers will check this)
├── configs/
│   ├── models.py         # the 8 SLMs (rows of the matrix)
│   └── tasks.py          # dataset registry (columns)
├── src/
│   ├── scoring.py        # log-likelihood + zero-shot helpers
│   ├── tasks.py          # task runners + scorers
│   ├── harness.py        # the single controlled eval loop + energy
│   └── run.py            # CLI entry point
├── notebooks/
│   └── kaggle_entry.py   # copy-paste Kaggle cell
└── results/              # outputs land here
```

---

## Controlled-protocol guarantees (state these in the paper)

- Same prompt template per task across **all** models.
- Same seed, max_tokens, batch size.
- All models run on the **same hardware** (don't mix T4/P100 — energy won't compare).
- 3 seeds → report mean ± std.

## Reproducibility

- Pinned dependency versions in `requirements.txt`.
- All datasets are open (licenses noted in `configs/tasks.py`).
- Raw model outputs + scores saved to `results/`.

## AI-assistance disclosure

This scaffold was generated with AI assistance. All metric definitions, dataset
licenses, and experimental claims must be independently verified before publication.
