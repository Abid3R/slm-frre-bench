"""Task / dataset registry: the columns of the benchmark matrix.

All datasets are open. Licenses are noted for the paper's data-availability section.
Verify each license/schema against the current Hugging Face dataset card before
publishing -- dataset schemas occasionally change.
"""

TASKS = {
    # ---- Fairness ----------------------------------------------------------
    "crows_pairs": {
        "axis": "fairness",
        "hf_path": "crows_pairs",          # nyu-mll/crows_pairs
        "hf_config": None,
        "split": "test",
        "license": "CC-BY-SA-4.0 (verify on dataset card)",
        "fields": {"more": "sent_more", "less": "sent_less", "bias_type": "bias_type"},
        "status": "implemented",
    },
    "bbq": {
        "axis": "fairness",
        "hf_path": "heegyu/bbq",           # verify; original: nyu-mll/BBQ repo
        "hf_config": None,
        "split": "test",
        "license": "CC-BY-4.0 (verify)",
        "status": "scaffold",
    },

    # ---- Robustness --------------------------------------------------------
    "adv_sst2": {
        "axis": "robustness",
        "hf_path": "adv_glue",
        "hf_config": "adv_sst2",
        "split": "validation",
        "license": "CC-BY-SA-4.0 (verify)",
        "labels": ["negative", "positive"],
        "status": "scaffold",
    },

    # ---- Capability control -----------------------------------------------
    "sst2": {
        "axis": "capability",
        "hf_path": "glue",
        "hf_config": "sst2",
        "split": "validation",
        "license": "CC-BY-4.0 / GLUE terms (verify)",
        "labels": ["negative", "positive"],
        "status": "scaffold",
    },
}

AXES = ["fairness", "robustness", "capability", "energy"]


def tasks_for_axis(axis: str):
    return [k for k, v in TASKS.items() if v["axis"] == axis]


def implemented_tasks():
    return [k for k, v in TASKS.items() if v["status"] == "implemented"]
