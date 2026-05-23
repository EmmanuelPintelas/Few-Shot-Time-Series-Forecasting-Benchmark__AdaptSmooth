# AdaptSmooth

**Few-shot time-series forecasting through online-adapted smoothing signals**

AdaptSmooth is a research codebase for evaluating few-shot time-series forecasting methods on a heterogeneous benchmark of univariate forecasting tasks. The repository contains the curated datasets, the generated few-shot task benchmark, an online rolling evaluator, statistical ranking utilities, classical and neural forecasting baselines, Chronos-family foundation-model wrappers, and AdaptSmooth variants.

The central idea of AdaptSmooth is simple: in few-shot forecasting, the available history is often too short and noisy for a single fixed preprocessing rule to be reliable. AdaptSmooth therefore selects a smoothing signal online. At every prediction origin, it scores candidate smoothing operators using only the currently observed prefix, chooses the operator whose endpoint signal has recently been most predictive, and uses that adapted signal to stabilize a base forecaster.

The current main instantiation is:

```text
AdaptSmooth-Chronos2 = Chronos-2 base forecaster + online-adapted smoothing signal
```

The same mechanism is also implemented for other base forecasters, such as SES and Theta, to study model-agnosticity.

---

## Repository contents

```text
.
├── CURATED_TIME_SERIES_DATASETS/
│   ├── BTC.csv
│   ├── Electricity consumption.csv
│   ├── ETD-OT.csv
│   ├── Exchange rate.csv
│   ├── Gold prices.csv
│   ├── Natural gas.csv
│   ├── Solar power.csv
│   ├── Temperature.csv
│   ├── Traffic.csv
│   └── Wind speed.csv
│
├── FST_BENCHMARK_TASKS/
│   ├── fst_tasks_index.csv
│   ├── fst_tasks_long.csv
│   └── <dataset_slug>/<task_id>/
│       ├── support.csv
│       ├── query.csv
│       └── full_task.csv
│
├── LOADER_EVALUATOR.py
├── TASK_GENERATOR.py
├── STATISTICAL_RANKING.py
├── eval_config.py
├── eval_utils.py
│
├── models/
│   ├── base.py
│   ├── registry.py
│   ├── registry_types.py
│   ├── model_utils.py
│   │
│   ├── classical/
│   │   ├── naive.py
│   │   ├── ses.py
│   │   ├── theta.py
│   │   ├── ets.py
│   │   ├── autoarima.py
│   │   └── ridge.py
│   │
│   ├── smoothers/
│   │   ├── config.py
│   │   └── smoothers.py
│   │
│   ├── adaptsmooth_chronos2/
│   ├── adaptsmooth_ses/
│   ├── adaptsmooth_theta/
│   ├── dlinear/
│   ├── fits/
│   ├── foundation/
│   └── template_model/
│
├── vizualize_AadaptSmoth_Chronos2.py
├── vizualizer - datasets_overview.py
└── vizualizer - task_illustrator.py
```

---

## Benchmark overview

The benchmark is designed for **domain-agnostic few-shot time-series forecasting**. It contains ten heterogeneous univariate datasets spanning weather, renewable energy, electricity systems, traffic, commodities, foreign exchange, and cryptocurrency markets.

| Dataset | Domain | Curated file |
|---|---|---|
| Temperature | Weather / environment | `Temperature.csv` |
| Wind speed | Weather / environment | `Wind speed.csv` |
| Solar power | Renewable energy | `Solar power.csv` |
| Natural gas | Commodity market | `Natural gas.csv` |
| Gold prices | Commodity / financial market | `Gold prices.csv` |
| ETD-OT | Transformer oil temperature | `ETD-OT.csv` |
| BTC | Cryptocurrency market | `BTC.csv` |
| Electricity consumption | Load forecasting | `Electricity consumption.csv` |
| Exchange rate | Foreign exchange | `Exchange rate.csv` |
| Traffic | Road occupancy | `Traffic.csv` |

All curated series use a common schema:

```text
unique_id, ds, y
```

where `unique_id` identifies the series, `ds` is the timestamp or time index, and `y` is the observed univariate value.

The generated few-shot benchmark contains:

```text
10 datasets × 100 tasks per dataset = 1000 few-shot forecasting tasks
```

Each task is a contiguous temporal segment split into:

```text
support length: n_support ∈ [10, 100]
query length:   n_query   ∈ [40, 400]
```

The support segment is the only information initially available to the model. The query segment is evaluated sequentially in rolling online blocks.

---

## Few-shot task format

Each task folder contains:

```text
support.csv    # observed support segment
query.csv      # future query segment
full_task.csv  # support + query in one file
```

A typical `full_task.csv` contains:

```text
task_id, dataset_name, split, relative_t, unique_id, ds, y
```

The benchmark index is stored in:

```text
FST_BENCHMARK_TASKS/fst_tasks_index.csv
```

It records task metadata such as dataset name, source file, support length, query length, time-span boundaries, and available horizons.

---

## AdaptSmooth method

AdaptSmooth is based on **online smoothing-signal adaptation**.

At prediction origin `t`, the method has access only to the observed prefix:

```text
U_1, ..., U_t
```

where `U` denotes the working-scale target used by the evaluator. This may be the original target or a transformed version depending on `eval_config.py`.

For each candidate smoother `g`, AdaptSmooth computes:

```text
z^{(g)} = g(U_1:t)
```

and extracts a local endpoint signal from the selected smoother. Candidate smoothers are scored by a causal rolling-origin criterion: each smoother is tested on recent prefixes inside the currently observed history, and the smoother whose endpoint signal best predicts the following validation blocks is selected.

The selected signal is then used to stabilize the base forecaster. In the Chronos-2 variant, the final forecast is produced by coupling:

```text
1. the Chronos-2 forecast
2. the online-selected smoothing signal
```

The smoothing term is not a separately trained model. It is an online-selected representation-level anchor that helps reduce sensitivity to short histories, noise, spikes, local regime changes, and unstable forecast extrapolations.

---

## Smoother catalogue

The smoother pool is implemented in:

```text
models/smoothers/
├── config.py
└── smoothers.py
```

Every smoother implements:

```python
z = smoother.transform(y)
```

and returns a finite same-length vector. This design allows the same smoother pool to be reused by AdaptSmooth variants and by other smoother-based forecasters without special-case inverse transformations.

The full smoother pool includes:

```text
identity / no smoothing
finite-window exponential smoothing
recursive EWMA ablation
windowed convolution / FIR smoothers
spectral FFT low-pass smoothers
polynomial trend smoothers
linear, cubic, and natural spline smoothers
Gaussian RBF regression smoothers
binner / piecewise-constant smoothers
LOWESS
additive seasonal decomposition smoothers
Kalman level and local-linear trend smoothers
difference-domain smoothers reconstructed back to level scale
```

The active pool is controlled by:

```python
# models/smoothers/config.py
SMOOTHER_CANDIDATES = SMOOTHER_CANDIDATES_FULL
```

For faster experiments, use:

```python
SMOOTHER_CANDIDATES = SMOOTHER_CANDIDATES_FAST
```

The default fast pool is:

```python
SMOOTHER_CANDIDATES_FAST = [
    "identity",
    "exp_alpha_0.30",
    "conv_7_hanning",
    "fft_cutoff_0.25_pad_10",
    "poly_degree_2",
    "lowess_frac_0.30",
    "kalman_local_linear",
]
```

---

## Implemented forecasting competitors

### Classical baselines

Implemented under `models/classical/`:

| Model | Description |
|---|---|
| `Naive` | Repeats the last observed value. |
| `Drift` | Extrapolates a linear drift from the first to the last observed value. |
| `SeasonalNaive` | Repeats the most recent value from the corresponding seasonal phase. |
| `SES` | Simple exponential smoothing with optional grid-based alpha selection. |
| `Theta` | Low-parameter Theta forecaster via `statsmodels`. |
| `ETS` | Error-trend-seasonal exponential-smoothing model. |
| `AutoARIMA` | Automatic ARIMA order selection via `pmdarima`, when enabled and installed. |
| `DirectRidge` | Simple direct autoregressive ridge regression baseline. |

Classical model settings are in:

```text
models/classical/config.py
```

### Lightweight train-from-scratch neural baselines

Implemented as independent model folders:

| Model | Folder | Description |
|---|---|---|
| `DLinear` | `models/dlinear/` | Local decomposition-linear model trained from scratch on the current prefix. |
| `FITS` | `models/fits/` | Lightweight frequency-domain forecasting model trained from scratch on the current prefix. |

These models are intentionally small and non-pretrained. They are included to compare AdaptSmooth against modern efficient forecasting architectures rather than only classical baselines.

### Foundation / pretrained baselines

Implemented under:

```text
models/foundation/
```

The repository includes wrappers for Chronos-family models, including Chronos-2. These wrappers follow the same `BaseForecaster` interface as all other models.

Chronos-2 settings are in:

```text
models/foundation/config.py
```

### AdaptSmooth variants

Implemented variants include:

```text
models/adaptsmooth_chronos2/
models/adaptsmooth_ses/
models/adaptsmooth_theta/
```

The main variant is:

```text
AdaptSmooth-Chronos2
```

Its settings are in:

```text
models/adaptsmooth_chronos2/config.py
```

Important parameters include:

```python
USE_FAST_POOL = True
MAX_INTERNAL_ORIGINS = 2
MIN_INTERNAL_ORIGIN = 6
RESCORE_EVERY_NEW_POINTS = 1
SIGNAL_ALPHA = 0.90
BASE_FORECASTER_WEIGHT = 0.50
```

`RESCORE_EVERY_NEW_POINTS = 1` means the smoother is re-selected at every online prediction origin.

---

## Installation

A minimal environment for classical and local neural models:

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows PowerShell

python -m pip install --upgrade pip
python -m pip install numpy pandas scipy scikit-learn statsmodels matplotlib torch
```

Optional packages:

```bash
# AutoARIMA baseline
python -m pip install pmdarima

# Chronos / Chronos-2 baselines and AdaptSmooth-Chronos2
python -m pip install "chronos-forecasting[extras]>=2.2"
```

If you only want to run the classical and lightweight local baselines, Chronos is not required. If Chronos-2 is enabled, the model weights are loaded through the Chronos/Hugging Face stack and may require internet access on the first run.

---

## Reproducing the benchmark workflow

### 1. Generate or regenerate few-shot tasks

The repository already includes generated tasks in `FST_BENCHMARK_TASKS/`. To regenerate them from the curated datasets, run:

```bash
python TASK_GENERATOR.py
```

This reads:

```text
CURATED_TIME_SERIES_DATASETS/
```

and writes:

```text
FST_BENCHMARK_TASKS/
├── fst_tasks_index.csv
├── fst_tasks_long.csv
└── <dataset_slug>/<task_id>/
    ├── support.csv
    ├── query.csv
    └── full_task.csv
```

Task-generation settings are at the top of `TASK_GENERATOR.py`:

```python
TASKS_PER_DATASET = 100
MIN_SUPPORT = 10
MAX_SUPPORT = 100
MIN_QUERY = 40
MAX_QUERY = 400
```

### 2. Select models

Models are registered in:

```text
models/registry.py
```

The evaluator obtains all competitors through:

```python
from models.registry import get_model_specs
```

Each model provider returns a list of `ModelSpec` objects. A `ModelSpec` contains a model name and a factory that creates a fresh model instance for every task.

To activate or deactivate model families, edit `MODEL_PROVIDER_FUNCTIONS` in `models/registry.py` and the relevant `config.py` files. For example:

```python
MODEL_PROVIDER_FUNCTIONS = [
    _classical_model_specs,
    dlinear_specs,
    fits_specs,
    adaptsmooth_chronos2_specs,
    chronos_specs,
    template_model_specs,
]
```

The final project version may keep only the target model family enabled by default to avoid long accidental runs.

### 3. Configure the evaluator

Global evaluation settings are in:

```text
eval_config.py
```

Important fields:

```python
TASKS_DIR = Path("FST_BENCHMARK_TASKS")
PRED_DIR = Path("FST_BENCHMARK_PREDICTIONS")
MAX_TASKS_PER_DOMAIN = 100
RANDOM_SAMPLE_TASKS_PER_DOMAIN = True
TASK_SELECTION_RANDOM_SEED = 40
EVAL_HORIZONS = [5]
```



### 4. Run online evaluation

```bash
python LOADER_EVALUATOR.py
```

The evaluator follows a strict rolling protocol:

```text
1. fit the model on the currently observed prefix
2. forecast the next h query values
3. reveal the true query block
4. append it to the history
5. repeat until the query segment is exhausted
```

Outputs are written to:

```text
FST_BENCHMARK_PREDICTIONS/
├── evaluation_config.json
├── metrics_by_task.csv
├── summary_by_model_horizon.csv
├── task_selection_this_run.json
└── <model_slug>/h_<horizon>/<dataset_slug>_predictions.csv
```

`metrics_by_task.csv` is the main file used for statistical ranking. It contains one row per model, task, and horizon, with task-level RMSE, MAE, and SMAPE.

### 5. Run statistical ranking

After evaluation, run:

```bash
python STATISTICAL_RANKING.py
```

This applies a Friedman aligned-ranks comparison with Finner post-hoc tests and writes:

```text
FST_BENCHMARK_PREDICTIONS/STATISTICAL_RANKING/
├── ranking_h_<h>_rmse.csv
├── ranking_h_<h>_rmse.tex
├── ranking_all_horizons_rmse.csv
├── global_far_tests_rmse.csv
└── latex_tables_all_horizons_rmse.tex
```

The ranking procedure compares methods on complete task blocks and avoids directly averaging raw errors across heterogeneous datasets with different scales.

---

## Visualization scripts

### Dataset overview

```bash
python "vizualizer - datasets_overview.py"
```

This creates a visual overview of the curated datasets.

### Few-shot task examples

```bash
python "vizualizer - task_illustrator.py"
```

This plots sampled support/query tasks from the benchmark.

### AdaptSmooth-Chronos2 visualization study

```bash
python vizualize_AadaptSmoth_Chronos2.py
```

This script creates paper-style diagnostic figures showing:

```text
actual query series
selected adapted smoothing signal
Chronos-2 baseline prediction
AdaptSmooth-Chronos2 prediction
selected smoother name at each visible prediction origin
```

Outputs are written to:

```text
FST_BENCHMARK_PLOTS/adaptsmooth_chronos2/
```

The visualization is useful for inspecting smoother switching behavior and the way the adapted signal stabilizes the Chronos-2 forecast over a small number of prediction origins.

---

## Adding a new forecasting model

Every forecasting model must implement the common interface:

```python
class BaseForecaster:
    name = "base"

    def fit(self, y: np.ndarray, horizon: int = 1, context: Optional[dict] = None):
        raise NotImplementedError

    def predict(self, n_steps: int) -> np.ndarray:
        raise NotImplementedError
```

The recommended workflow is:

1. Copy `models/template_model/` to `models/my_model/`.
2. Implement your model in `models/my_model/my_model.py`.
3. Add a local `config.py` for hyperparameters.
4. Expose:

```python
def get_model_specs() -> list[ModelSpec]:
    ...
```

5. Import the provider in `models/registry.py` and add it to `MODEL_PROVIDER_FUNCTIONS`.

The evaluator will automatically instantiate a fresh model for every task through `spec.factory()`.

---

## Output files and interpretation

### Per-step predictions

Each model writes prediction traces to:

```text
FST_BENCHMARK_PREDICTIONS/<model_slug>/h_<horizon>/<dataset_slug>_predictions.csv
```

Typical columns include:

```text
task_id
dataset_name
model_name
horizon
origin_relative_t
target_relative_t
target_query_index
horizon_step
ds
y_true
y_pred
```

### Task-level metrics

```text
FST_BENCHMARK_PREDICTIONS/metrics_by_task.csv
```

This is the main evaluation output. It contains one metric row per task, model, and horizon.

### Summary metrics

```text
FST_BENCHMARK_PREDICTIONS/summary_by_model_horizon.csv
```

This file reports mean and median RMSE, MAE, and SMAPE by model and horizon. These summaries are useful for quick inspection, but the paper ranking uses the nonparametric aligned-ranks procedure.

### Statistical ranking

```text
FST_BENCHMARK_PREDICTIONS/STATISTICAL_RANKING/
```

This folder contains CSV and LaTeX outputs for the Friedman aligned-ranks test and Finner post-hoc comparisons.

---

## Reproducibility notes

The benchmark and evaluator expose several explicit seeds:

```python
# TASK_GENERATOR.py
RANDOM_SEED = 42

# eval_config.py
RANDOM_SEED = 42
TASK_SELECTION_RANDOM_SEED = 40

# models/dlinear/config.py
DLINEAR_SEED = 42

# models/fits/config.py
FITS_SEED = 42
```

For deterministic task subsets, keep `TASK_SELECTION_RANDOM_SEED` fixed. If `TASK_SELECTION_RANDOM_SEED = None`, a fresh task sample is selected on each run.

Chronos-family results can depend on installed package versions, hardware, numerical dtype, and model-loading behavior. For final experiments, record the contents of:

```text
FST_BENCHMARK_PREDICTIONS/evaluation_config.json
```

and the package versions used in your environment.

---

## Practical recommendations

For a quick smoke test:

```python
# eval_config.py
MAX_TASKS_PER_DOMAIN = 1
EVAL_HORIZONS = [5]
```

Then run:

```bash
python LOADER_EVALUATOR.py
```

For a full benchmark run:

```python
# eval_config.py
MAX_TASKS_PER_DOMAIN = 100
EVAL_HORIZONS = [5]
```



## Project philosophy

This repository separates the benchmark protocol from the forecasting models.

```text
LOADER_EVALUATOR.py       -> online evaluation protocol
models/                   -> forecasting competitors
models/smoothers/         -> representation/smoothing operators
STATISTICAL_RANKING.py    -> task-wise nonparametric comparison
```

This separation is intentional. It makes it easier to add new models, swap base forecasters, test new smoother pools, and reproduce the same online evaluation protocol across different forecasting families.

AdaptSmooth is intended to be a lightweight, modular mechanism rather than a monolithic architecture. Its main contribution is not a new large forecasting model, but an online adaptation layer that selects a stabilizing smoothing signal at task time and can be coupled with different base forecasters.

---

## Citation

If you use this repository, please cite the corresponding paper once available.

```bibtex
@article{pintelas2026adaptsmooth,
  title   = {AdaptSmooth: Few-shot Time-Series Forecasting through Online-Adapted Smoothing Signals},
  author  = {Pintelas, Emmanuel and Livieris, Ioannis E. and Karagrigoriou, Alex},
  journal = {Manuscript under review},
  year    = {2026}
}
```

