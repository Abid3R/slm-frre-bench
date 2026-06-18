# ============================================================================
# SLM-FRRE-Bench  --  Kaggle copy-paste cell
# ----------------------------------------------------------------------------
# How to use on Kaggle (Notebook with GPU T4 x2 enabled, Internet ON):
#   1. Create a new Notebook, Settings -> Accelerator -> GPU T4 x2, Internet ON.
#   2. Paste THIS whole file into the first cell and run it.
#   3. It clones the repo (or uploads), installs deps, and runs a first result.
# The first run does CrowS-Pairs (fairness) + energy on two small models so you
# get a complete row of the results table within a single free session.
# ============================================================================

import os, subprocess, sys

REPO_DIR = "/kaggle/working/slm-frre-bench"

# --- Option A: clone the repo from GitHub (default) --------------------------
GIT_URL = "https://github.com/Abid3R/slm-frre-bench.git"
if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", GIT_URL, REPO_DIR], check=True)

# --- Option B: if you uploaded the folder as a Kaggle Dataset ----------------
# Add the dataset via "Add Input", then copy it into working:
# import shutil; shutil.copytree("/kaggle/input/slm-frre-bench", REPO_DIR)

assert os.path.exists(REPO_DIR), (
    "Put the repo at %s first (clone from GitHub or add as a Dataset)." % REPO_DIR
)
os.chdir(REPO_DIR)

# --- Install pinned dependencies --------------------------------------------
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                "requirements.txt"], check=True)

# --- Full study grid: 7 SLMs x (Fairness, Robustness, Capability) + energy ---
# Gemma-2-2B is gated (needs an HF token + license accept); excluded here so the
# first full run is clean. To add it: accept the license on its HF page, set
#   os.environ["HF_TOKEN"] = "hf_..."  and append the model id below.
GRID_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "facebook/opt-1.3b",
    "EleutherAI/pythia-1.4b",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "microsoft/phi-2",
]
subprocess.run([
    sys.executable, "-m", "src.run",
    "--models", *GRID_MODELS,
    "--tasks", "crows_pairs", "adv_sst2", "sst2",
    "--limit", "300",
    "--seeds", "42",
], check=True)

# --- Inspect results (handles the schema-fork _new.csv fallback) -------------
import glob
import pandas as pd
csvs = sorted(glob.glob("results/results*.csv"))
df = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)
print(df.to_string(index=False))
print("\n[kaggle] result files:", csvs)
