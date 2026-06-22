"""Controlled evaluation harness.

One model at a time:
  1. load (optionally 4-bit) -> 2. wrap energy tracker -> 3. run every selected task
  -> 4. record metrics + energy -> 5. free GPU memory.

The "controlled protocol" guarantee is that every model goes through the exact same
load path, the exact same scoring primitives, and the exact same energy accounting.
"""
from __future__ import annotations

import gc
import time

import torch

from configs.models import short_name
from src.tasks import run_task

try:
    from codecarbon import EmissionsTracker
    _HAS_CODECARBON = True
except Exception:  # codecarbon optional at import time (e.g. offline lint)
    _HAS_CODECARBON = False


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(model_id: str, device: str = "cuda", load_in_4bit: bool = True):
    """Load a causal LM + tokenizer with the same recipe for every model.

    4-bit (bitsandbytes) is used on CUDA to fit 2-3B models on a single T4.
    On CPU we fall back to fp32 and ignore the quant flag.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {}
    if device == "cuda" and load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": 0}
    elif device == "cuda":
        kwargs["torch_dtype"] = torch.float16
        kwargs["device_map"] = {"": 0}
    else:
        kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if device == "cpu":
        model = model.to("cpu")
    model.eval()
    return model, tokenizer


def free_model(model):
    """Release GPU memory between models so the grid runs in one Kaggle session."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Energy accounting
# ---------------------------------------------------------------------------
class _NullTracker:
    """Fallback when codecarbon is unavailable: still times the run."""

    def start(self):
        self._t0 = time.time()

    def stop(self):
        return 0.0


def _make_tracker(project_name: str, output_dir: str):
    if not _HAS_CODECARBON:
        return _NullTracker()
    return EmissionsTracker(
        project_name=project_name,
        output_dir=output_dir,
        log_level="error",
        save_to_file=True,
        measure_power_secs=5,
    )


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------
def evaluate_model(
    model_id: str,
    tasks: list[str],
    device: str = "cuda",
    limit: int | None = None,
    seed: int = 42,
    load_in_4bit: bool = True,
    energy_dir: str = "results",
):
    """Run all `tasks` for one model and return a flat result row.

    Energy is tracked across the *whole* set of tasks for the model, then divided
    by total inferences to get a per-1k-inference figure that is comparable across
    models regardless of how many examples each task contributed.
    """
    gpu_name = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() and device == "cuda"
        else "cpu"
    )

    t_load0 = time.time()
    model, tokenizer = load_model(model_id, device=device, load_in_4bit=load_in_4bit)
    load_secs = time.time() - t_load0

    tracker = _make_tracker(project_name=short_name(model_id), output_dir=energy_dir)
    tracker.start()
    t0 = time.time()

    row = {
        "model_id": model_id,
        "model": short_name(model_id),
        "device": device,
        "gpu": gpu_name,
        "seed": seed,
        "limit": limit if limit is not None else "full",
        "load_secs": round(load_secs, 1),
    }
    total_inferences = 0
    for task in tasks:
        metrics, n_inf = run_task(task, model, tokenizer, device=device,
                                  limit=limit, seed=seed, model_name=row["model"])
        # flatten nested dicts (e.g. fairness_by_bias_type) into JSON-ish strings
        for k, v in metrics.items():
            row[k] = v if not isinstance(v, dict) else str(v)
        total_inferences += n_inf

    wall_secs = time.time() - t0
    emissions_kg = tracker.stop() or 0.0

    # CodeCarbon also writes energy_consumed (kWh) to its csv; we approximate here
    # from emissions if the kWh field isn't surfaced by the return value.
    row["wall_secs"] = round(wall_secs, 1)
    row["total_inferences"] = total_inferences
    row["inferences_per_sec"] = round(total_inferences / wall_secs, 2) if wall_secs else 0.0
    row["co2_kg"] = round(emissions_kg, 6)
    row["co2_g_per_1k_inf"] = (
        round(1000.0 * emissions_kg * 1000.0 / total_inferences, 4)
        if total_inferences else float("nan")
    )

    free_model(model)
    return row
