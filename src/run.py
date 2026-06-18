"""CLI entry point for SLM-FRRE-Bench.

Examples
--------
# Smoke test on CPU (fast, one model, one task, 50 examples):
python -m src.run --models Qwen/Qwen2.5-0.5B-Instruct --tasks crows_pairs --limit 50 --device cpu

# Full fairness+robustness+capability grid on the default model set (Kaggle T4):
python -m src.run --all --tasks crows_pairs adv_sst2 sst2 --limit 300

# Multiple seeds for variance reporting:
python -m src.run --all --tasks crows_pairs --seeds 13 42 123 --limit 300
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import torch

from configs.models import all_model_ids, CPU_FRIENDLY
from configs.tasks import implemented_tasks
from src.harness import evaluate_model


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="SLM Fairness/Robustness/Energy benchmark")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--models", nargs="+", help="HF model ids to evaluate")
    g.add_argument("--all", action="store_true", help="use the full model registry")
    g.add_argument("--cpu-set", action="store_true",
                   help="use the small CPU-friendly model set")

    p.add_argument("--tasks", nargs="+", default=implemented_tasks(),
                   help=f"tasks to run (default: {implemented_tasks()})")
    p.add_argument("--seeds", nargs="+", type=int, default=[42],
                   help="random seeds; one row per (model, seed)")
    p.add_argument("--limit", type=int, default=None,
                   help="cap examples per task (None = full split)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   choices=["cuda", "cpu"])
    p.add_argument("--no-4bit", action="store_true",
                   help="disable bitsandbytes 4-bit loading")
    p.add_argument("--out", default="results/results.csv",
                   help="CSV path to append results to")
    return p.parse_args(argv)


def _resolve_models(args) -> list[str]:
    if args.all:
        return all_model_ids()
    if args.cpu_set:
        return CPU_FRIENDLY
    return args.models


def _append_rows(rows: list[dict], out_path: str):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    # union of all keys keeps the CSV stable even as tasks add columns
    fieldnames: list[str] = []
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)

    existing = os.path.exists(out_path)
    # if appending to an existing file with a different header, fall back to a
    # fresh file so the schema stays consistent for downstream analysis.
    if existing:
        with open(out_path, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), [])
        if header != fieldnames:
            existing = False
            out_path = out_path.replace(".csv", "_new.csv")

    mode = "a" if existing else "w"
    with open(out_path, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not existing:
            w.writeheader()
        for r in rows:
            w.writerow(r)
    return out_path


def main(argv=None):
    args = _parse_args(argv)
    models = _resolve_models(args)
    load_in_4bit = not args.no_4bit and args.device == "cuda"

    print(f"[run] device={args.device}  4bit={load_in_4bit}  "
          f"models={len(models)}  tasks={args.tasks}  seeds={args.seeds}")

    rows = []
    for model_id in models:
        for seed in args.seeds:
            print(f"[run] -> {model_id}  (seed={seed})")
            try:
                row = evaluate_model(
                    model_id, args.tasks, device=args.device,
                    limit=args.limit, seed=seed, load_in_4bit=load_in_4bit,
                )
                rows.append(row)
                print(f"[run]    {row}")
            except Exception as e:  # keep the grid alive if one model fails
                print(f"[run]    FAILED {model_id} (seed={seed}): {e}",
                      file=sys.stderr)
                rows.append({"model_id": model_id, "seed": seed, "error": str(e)})

    final_path = _append_rows(rows, args.out)
    print(f"[run] wrote {len(rows)} rows -> {final_path}")


if __name__ == "__main__":
    main()
