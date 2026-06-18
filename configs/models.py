"""Model registry: the rows of the benchmark matrix.

All models are <=3B params so they fit a free Kaggle/Colab T4 (16 GB), using 4-bit
quantization where needed. Keep this list stable across the whole study so every
result is comparable.
"""

# Each entry: HF id -> metadata used for reporting and loading.
MODELS = {
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {
        "short": "TinyLlama-1.1B",
        "params_b": 1.1,
        "family": "Llama",
        "instruct": True,
    },
    "Qwen/Qwen2.5-0.5B-Instruct": {
        "short": "Qwen2.5-0.5B",
        "params_b": 0.5,
        "family": "Qwen",
        "instruct": True,
    },
    "Qwen/Qwen2.5-1.5B-Instruct": {
        "short": "Qwen2.5-1.5B",
        "params_b": 1.5,
        "family": "Qwen",
        "instruct": True,
    },
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": {
        "short": "SmolLM2-1.7B",
        "params_b": 1.7,
        "family": "SmolLM",
        "instruct": True,
    },
    "microsoft/phi-2": {
        "short": "Phi-2",
        "params_b": 2.7,
        "family": "Phi",
        "instruct": False,
    },
    "google/gemma-2-2b-it": {
        "short": "Gemma-2-2B",
        "params_b": 2.0,
        "family": "Gemma",
        "instruct": True,
        "gated": True,  # requires HF token + license acceptance
    },
    "EleutherAI/pythia-1.4b": {
        "short": "Pythia-1.4B",
        "params_b": 1.4,
        "family": "Pythia",
        "instruct": False,
    },
    "facebook/opt-1.3b": {
        "short": "OPT-1.3B",
        "params_b": 1.3,
        "family": "OPT",
        "instruct": False,
    },
}

# Models small enough to run on CPU for a quick local smoke test.
CPU_FRIENDLY = ["Qwen/Qwen2.5-0.5B-Instruct", "facebook/opt-1.3b"]


def all_model_ids():
    return list(MODELS.keys())


def short_name(model_id: str) -> str:
    return MODELS.get(model_id, {}).get("short", model_id.split("/")[-1])
