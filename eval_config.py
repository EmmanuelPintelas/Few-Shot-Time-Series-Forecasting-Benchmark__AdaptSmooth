"""Global evaluation configuration.

This file controls only the benchmark/evaluation protocol and output behavior.
Model-specific hyperparameters live inside each model package under models/.
"""

from pathlib import Path

# -----------------------------
# Data and output paths
# -----------------------------
TASKS_DIR = Path("FST_BENCHMARK_TASKS")
PRED_DIR = Path("FST_BENCHMARK_PREDICTIONS")
INDEX_FILE = TASKS_DIR / "fst_tasks_index.csv"

# -----------------------------
# Task selection
# -----------------------------
MAX_TASKS = None
MAX_TASKS_PER_DOMAIN = 100
RANDOM_SAMPLE_TASKS_PER_DOMAIN = True
# None means fresh random sample every run. Use an integer for reproducibility.
TASK_SELECTION_RANDOM_SEED = 40
TASK_SELECTION_RUN_FILE = PRED_DIR / "task_selection_this_run.json"

# -----------------------------
# Evaluation protocol
# -----------------------------
CLEAR_OUTPUT = False
EVAL_HORIZONS = [5]
RANDOM_SEED = 42

# -----------------------------
# Target transformation
# -----------------------------
APPLY_LOG = False
# Options:
#   "log"          -> strictly positive values
#   "log1p"        -> values >= 0
#   "signed_log1p" -> positive, zero, and negative values
#   "shifted_log"  -> shifts each origin so min(history) becomes positive
LOG_MODE = "signed_log1p"
LOG_EPS = 1e-8

# -----------------------------
# Optional pretrained/foundation baselines
# -----------------------------
ENABLE_FOUNDATION_MODELS = False
ENABLE_CHRONOS = True
# ENABLE_TIMESFM = False
# ENABLE_LAG_LLAMA = False

CHRONOS_MODEL_ID = "amazon/chronos-bolt-mini"   #"amazon/chronos-bolt-tiny"
# TIMESFM_REPO_ID = "google/timesfm-1.0-200m-pytorch"
# LAG_LLAMA_CHECKPOINT_PATH = "lag-llama.ckpt"
