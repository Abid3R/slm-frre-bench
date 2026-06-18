"""Post-run analysis: trade-off table + plots + Spearman correlations.

Run after you have results/results.csv:
    python -m src.analysis --csv results/results.csv --out results/

Produces:
  * results/summary.csv          -- mean +/- std per model across seeds
  * results/tradeoff_fairness_energy.png
  * results/tradeoff_robust_energy.png
  * correlation lines printed to stdout (the paper's headline trade-off claim)
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from src.scoring import spearman


# numeric metric columns we care about for the trade-off analysis
METRIC_COLS = [
    "fairness_stereotype_score",
    "fairness_bias_magnitude",
    "robust_clean_acc",
    "robust_perturbed_acc",
    "robust_acc_drop",
    "capability_sst2_acc",
    "co2_g_per_1k_inf",
    "inferences_per_sec",
]


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- std per model across seeds for every numeric metric present."""
    present = [c for c in METRIC_COLS if c in df.columns]
    g = df.groupby("model")[present]
    summary = g.agg(["mean", "std"])
    summary.columns = [f"{m}_{stat}" for m, stat in summary.columns]
    return summary.reset_index()


def _scatter(df, x, y, out_path, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[analysis] matplotlib unavailable, skipping plot: {e}")
        return
    if x not in df.columns or y not in df.columns:
        print(f"[analysis] missing {x} or {y}, skipping {out_path}")
        return

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(df[x], df[y])
    for _, r in df.iterrows():
        ax.annotate(str(r.get("model", "")), (r[x], r[y]),
                    fontsize=8, xytext=(4, 2), textcoords="offset points")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[analysis] wrote {out_path}")


def _corr(df, x, y):
    if x not in df.columns or y not in df.columns:
        return
    sub = df[[x, y]].dropna()
    if len(sub) < 2:
        return
    rho = spearman(list(sub[x]), list(sub[y]))
    print(f"[analysis] Spearman({x}, {y}) = {rho:.3f}  (n={len(sub)})")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="results/results.csv")
    p.add_argument("--out", default="results/")
    args = p.parse_args(argv)

    df = pd.read_csv(args.csv)
    df = df[df.get("error").isna()] if "error" in df.columns else df
    os.makedirs(args.out, exist_ok=True)

    summary = summarize(df)
    summary_path = os.path.join(args.out, "summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"[analysis] wrote {summary_path}")
    print(summary.to_string(index=False))

    # headline trade-off questions for the paper
    _corr(df, "co2_g_per_1k_inf", "fairness_bias_magnitude")
    _corr(df, "co2_g_per_1k_inf", "robust_acc_drop")
    _corr(df, "capability_sst2_acc", "fairness_bias_magnitude")

    _scatter(df, "co2_g_per_1k_inf", "fairness_bias_magnitude",
             os.path.join(args.out, "tradeoff_fairness_energy.png"),
             "Fairness bias vs. energy")
    _scatter(df, "co2_g_per_1k_inf", "robust_acc_drop",
             os.path.join(args.out, "tradeoff_robust_energy.png"),
             "Robustness drop vs. energy")


if __name__ == "__main__":
    main()
