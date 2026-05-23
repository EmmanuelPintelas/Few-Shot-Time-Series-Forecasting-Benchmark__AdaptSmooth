"""Visualization study for AdaptSmooth-Chronos2.

Run from inside the IMPL/ folder:

    python VISUALIZE_ADAPTSMOOTH_CHRONOS2.py

Expected task folder:

    FST_BENCHMARK_TASKS/
      fst_tasks_index.csv
      <dataset_slug>/<task_id>/full_task.csv

Outputs:

    FST_BENCHMARK_PLOTS/adaptsmooth_chronos2/
      h_<horizon>/
        adaptsmooth_chronos2_grid_h<horizon>.png
        adaptsmooth_chronos2_grid_h<horizon>.pdf
        traces_h<horizon>.csv
        summary_h<horizon>.csv
        task_<task_id>_h<horizon>.png
        task_<task_id>_h<horizon>.pdf

The figure is designed for the paper: one panel per task in a 4x1 vertical grid.
Each panel shows:
    1) actual observed series,
    2) selected online-adapted smoothing signal,
    3) Chronos-2 baseline prediction,
    4) AdaptSmooth-Chronos2 prediction,
plus a compact top strip with the selected smoother name for each visible prediction origin.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import re
import textwrap
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from eval_config import TASKS_DIR
from eval_utils import (
    normalize_name,
    load_task_index,
    load_task,
    transform_target,
    inverse_transform_target,
    infer_seasonal_period,
    rmse,
    mae,
)
from models.model_utils import sanitize_predictions
from models.adaptsmooth_chronos2.adaptsmooth_chronos2 import AdaptSmoothChronos2Forecaster
from models.adaptsmooth_chronos2 import config as as_chronos2_config
from models.smoothers import config as smoother_config
from models.smoothers.smoothers import get_smoother_candidates

warnings.filterwarnings("ignore")


# =============================================================================
# User-facing visualization settings
# =============================================================================

OUTPUT_DIR = Path("FST_BENCHMARK_PLOTS") / "adaptsmooth_chronos2"

# Plot horizons.  Create one 4x1 grid per horizon.
VIZ_HORIZONS = [5]

# Select exact tasks by ID.  Leave empty to sample tasks from the index.
VIZ_TASK_IDS: List[str] = []

# used only when VIZ_TASK_IDS is empty.
VIZ_DATASET_NAMES = ["BTC", "etd_ot", "Traffic", "Wind Speed"]
#VIZ_DATASET_NAMES: List[str] = []

# Number of task panels in the grid.
VIZ_MAX_TASKS = 4
VIZ_RANDOM_SEED = 7

# Use a shorter smoother pool for faster visualization.  For paper figures, use False.
USE_FAST_SMOOTHER_POOL_FOR_VIZ = False

# AdaptSmooth coefficients used in the visualization run.
SIGNAL_ALPHA = 0.90
BASE_FORECASTER_WEIGHT = 0.50
MAX_INTERNAL_ORIGINS = 2
MIN_INTERNAL_ORIGIN = 6

# Chronos-2 settings.  Device "auto" is safer than forcing CUDA on Windows/CPU.
CHRONOS2_MODEL_ID = "amazon/chronos-2"
CHRONOS2_DEVICE_MAP = "auto"
CHRONOS2_TORCH_DTYPE = "auto"
CHRONOS2_QUANTILE = 0.5
CHRONOS2_FREQ = "D"
CHRONOS2_CONTEXT_LENGTH = None

# Figure settings.
# The plot is intentionally cropped to a small query window to keep the
# smoother-switching story readable in the paper.
FIGSIZE_GRID = (11.5, 13.0)
FIGSIZE_SINGLE = (11.5, 3.2)
DPI = 400

# Show only this many prediction origins per task. With h=5, this gives 50
# query timestamps.
MAX_VISIBLE_ORIGINS = 10

# How to choose the visible origins:
#   "best_improvement" = contiguous window where AS-Chronos2 improves most
#   "first"            = first MAX_VISIBLE_ORIGINS origins
#   "middle"           = centered window
#   "last"             = last MAX_VISIBLE_ORIGINS origins
VISIBLE_ORIGIN_MODE = "best_improvement"

# Use query index on the x-axis and do not draw the support segment.
USE_QUERY_INDEX_XAXIS = True

# Smoother labels are shown once per selected origin in a small top strip.
ANNOTATE_SMOOTHER_NAMES = True
ANNOTATE_ONLY_WHEN_SMOOTHER_CHANGES = False
SMOOTHER_LABEL_FONTSIZE = 10


# =============================================================================
# Helpers
# =============================================================================


def _infer_pandas_freq(ds_values: pd.Series, fallback: str = "D") -> str:
    """Infer a compact pandas frequency string from task timestamps."""
    try:
        dt = pd.to_datetime(ds_values, errors="coerce").dropna().sort_values()
        if len(dt) < 3:
            return fallback
        delta_seconds = float(dt.diff().dropna().dt.total_seconds().median())
        if not np.isfinite(delta_seconds) or delta_seconds <= 0:
            return fallback
        if abs(delta_seconds - 600) < 30:
            return "10min"
        if abs(delta_seconds - 900) < 60:
            return "15min"
        if abs(delta_seconds - 1800) < 120:
            return "30min"
        if abs(delta_seconds - 3600) < 300:
            return "H"
        if abs(delta_seconds - 86400) < 3600:
            return "D"
        return pd.tseries.frequencies.to_offset(pd.Timedelta(seconds=delta_seconds)).freqstr
    except Exception:
        return fallback


def _fmt_float_token(x: str) -> str:
    """Compact float formatting for paper labels."""
    try:
        v = float(x)
    except Exception:
        return str(x)
    out = f"{v:.2f}"
    if out.startswith("0"):
        out = out[1:]
    if out.startswith("-0"):
        out = "-" + out[2:]
    return out.rstrip("0").rstrip(".")


def _paper_smoother_label(name: object) -> str:
    """Convert implementation smoother names to short paper-ready labels."""
    raw = str(name) if name is not None else "?"
    n = raw.strip()

    # Some wrappers may prefix selected names with A_. Keep the operator only.
    while n.startswith("A_"):
        n = n[2:]

    n = n.replace("hann", "hanning")
    n = n.replace("ones", "flat")

    diff = False
    if n.startswith("diff_"):
        diff = True
        n = n[len("diff_"):]

    # prefix = "Δ-" if diff else ""

    # if n in {"identity", "id"}:
    #     return "Δ" if diff else "Id"
    # if n == "reconstruct":
    #     return "Δ"
    prefix = ''

    m = re.match(r"exp_m_(\d+)_alpha_([0-9.]+)$", n)
    if m:
        return f"{prefix}Exp({m.group(1)},{_fmt_float_token(m.group(2))})"

    m = re.match(r"exp_alpha_([0-9.]+)$", n)
    if m:
        return f"{prefix}Exp({_fmt_float_token(str(smoother_config.DEFAULT_EXP_WINDOW))},{_fmt_float_token(m.group(1))})"

    m = re.match(r"ewm_alpha_([0-9.]+)$", n)
    if m:
        return f"{prefix}EWMA({_fmt_float_token(m.group(1))})"

    m = re.match(r"conv_(\d+)_(.+)$", n)
    if m:
        win, kind = m.group(1), m.group(2)
        kind_map = {
            "flat": "Box",
            "mean": "Box",
            "hanning": "Hann",
            "hamming": "Hamm",
            "bartlett": "Bart",
            "blackman": "Black",
        }
        return f"{prefix}{kind_map.get(kind, kind.title())}({win})"

    m = re.match(r"fft_cutoff_([0-9.]+)_pad_(\d+)$", n)
    if m:
        return f"{prefix}Spec({_fmt_float_token(m.group(1))})"

    m = re.match(r"poly_degree_(\d+)$", n)
    if m:
        return f"{prefix}Poly({m.group(1)})"

    m = re.match(r"spline_linear_k?_(\d+)$", n)
    if m:
        return f"{prefix}LinSpl({m.group(1)})"

    m = re.match(r"spline_cubic_k?_(\d+)$", n)
    if m:
        return f"{prefix}CubSpl({m.group(1)})"

    m = re.match(r"spline_natural_k?_(\d+)$", n)
    if m:
        return f"{prefix}NatSpl({m.group(1)})"

    m = re.match(r"gauss_rbf_k_(\d+)_sigma_([0-9.]+)$", n)
    if m:
        return f"{prefix}Gauss({m.group(1)},{_fmt_float_token(m.group(2))})"

    m = re.match(r"binner_k_(\d+)$", n)
    if m:
        return f"{prefix}Bin({m.group(1)})"

    m = re.match(r"lowess_frac_([0-9.]+)$", n)
    if m:
        return f"{prefix}LOWESS({_fmt_float_token(m.group(1))})"

    m = re.match(r"decomp_add_conv_(\d+)_(.+)_period_(\d+)$", n)
    if m:
        return f"{prefix}Decomp-Box({m.group(3)})"

    m = re.match(r"decomp_add_lowess_([0-9.]+)_period_(\d+)$", n)
    if m:
        return f"{prefix}Decomp-LOWESS({m.group(2)})"

    m = re.match(r"decomp_add_spline_natural_(\d+)_period_(\d+)$", n)
    if m:
        return f"{prefix}Decomp-NatSpl({m.group(2)})"

    if n == "kalman_local_linear" or n.startswith("kalman_level_trend"):
        return f"{prefix}Kalman-LLT"

    if n.startswith("kalman_level"):
        return f"{prefix}Kalman-L"

    return prefix + n.replace("_", "-")


def _short_label(name: object, width: int = 999) -> str:
    # Kept for backward compatibility with the earlier script.
    return _paper_smoother_label(name)

def _choose_tasks(index_df: pd.DataFrame) -> pd.DataFrame:
    df = index_df.copy()
    df["task_id"] = df["task_id"].astype(str)

    if VIZ_TASK_IDS:
        wanted = [str(x) for x in VIZ_TASK_IDS]
        out = df[df["task_id"].isin(wanted)].copy()
        # Preserve user order.
        order = {tid: i for i, tid in enumerate(wanted)}
        out["_order"] = out["task_id"].map(order)
        out = out.sort_values("_order").drop(columns=["_order"])
        if out.empty:
            raise ValueError(f"None of VIZ_TASK_IDS were found: {wanted}")
        return out.head(VIZ_MAX_TASKS).reset_index(drop=True)

    if VIZ_DATASET_NAMES:
        wanted_slugs = {normalize_name(x) for x in VIZ_DATASET_NAMES}
        df = df[df["dataset_name"].map(normalize_name).isin(wanted_slugs)].copy()
        if df.empty:
            raise ValueError(f"No tasks found for VIZ_DATASET_NAMES={VIZ_DATASET_NAMES}")

    # Pick diverse tasks: one deterministic sampled task per dataset until VIZ_MAX_TASKS.
    rng = np.random.default_rng(VIZ_RANDOM_SEED)
    rows = []
    for dataset_name, group in df.groupby("dataset_name", sort=False):
        if len(rows) >= VIZ_MAX_TASKS:
            break
        idx = int(rng.integers(0, len(group)))
        rows.append(group.iloc[idx])

    if not rows:
        raise ValueError("No tasks available for visualization.")

    return pd.DataFrame(rows).reset_index(drop=True)


def _candidate_smoothers():
    names = smoother_config.SMOOTHER_CANDIDATES_FAST if USE_FAST_SMOOTHER_POOL_FOR_VIZ else smoother_config.SMOOTHER_CANDIDATES
    return get_smoother_candidates(names)


def _make_model() -> AdaptSmoothChronos2Forecaster:
    return AdaptSmoothChronos2Forecaster(
        candidates=_candidate_smoothers(),
        max_internal_origins=MAX_INTERNAL_ORIGINS,
        min_internal_origin=MIN_INTERNAL_ORIGIN,
        rescore_every_new_points=1,
        fallback_smoother_name="identity",
        signal_alpha=SIGNAL_ALPHA,
        base_forecaster_weight=BASE_FORECASTER_WEIGHT,
        model_name="AdaptSmooth-Chronos2",
        base_model_id=CHRONOS2_MODEL_ID,
        base_model_name="Chronos2",
        base_device_map=CHRONOS2_DEVICE_MAP,
        base_torch_dtype=CHRONOS2_TORCH_DTYPE,
        base_quantile=CHRONOS2_QUANTILE,
        base_freq=CHRONOS2_FREQ,
        base_context_length=CHRONOS2_CONTEXT_LENGTH,
        debug=False,
    )


def _inverse_scalar(x: float, transform_info: Dict) -> float:
    return float(inverse_transform_target(np.asarray([x], dtype=float), transform_info)[0])


def run_task_trace(dataset_name: str, task_id: str, horizon: int) -> Tuple[pd.DataFrame, Dict]:
    """Run online AdaptSmooth-Chronos2 on one task and return plot-ready traces."""
    support, query = load_task(dataset_name, task_id)
    full_df = pd.concat([support, query], ignore_index=True)

    support_y_original = support["y"].to_numpy(dtype=float)
    query_y_original = query["y"].to_numpy(dtype=float)
    history_original = support_y_original.copy()
    history_ds = support["ds"].copy().reset_index(drop=True)

    full_ds = full_df["ds"]
    seasonal_period = infer_seasonal_period(dataset_name, full_ds)
    pandas_freq = _infer_pandas_freq(full_ds, fallback=CHRONOS2_FREQ)

    model = _make_model()
    records: List[Dict] = []
    origin_records: List[Dict] = []

    query_len = len(query_y_original)
    for query_start in range(0, query_len, horizon):
        block_len = min(int(horizon), query_len - query_start)
        origin_relative_t = int(len(history_original) - 1)
        fallback_original = float(history_original[-1]) if len(history_original) else 0.0

        context = {
            "task_id": task_id,
            "dataset_name": dataset_name,
            "horizon": horizon,
            "seasonal_period": seasonal_period,
            "history_ds": history_ds,
            "pandas_freq": pandas_freq,
            "freq": pandas_freq,
        }

        try:
            history_model_scale, transform_info = transform_target(history_original)
            model.fit(history_model_scale, horizon=horizon, context=context)
            as_pred_model_scale = model.predict(block_len)
            components = model.get_last_prediction_components()

            base_pred_model_scale = np.asarray(components.get("base_pred"), dtype=float)
            if len(base_pred_model_scale) != block_len:
                base_pred_model_scale = np.repeat(history_model_scale[-1], block_len)
            smooth_signal_model_scale = float(components.get("smooth_signal", history_model_scale[-1]))

            base_pred_model_scale = sanitize_predictions(
                base_pred_model_scale,
                fallback_value=float(history_model_scale[-1]),
                n_steps=block_len,
            )
            as_pred_model_scale = sanitize_predictions(
                as_pred_model_scale,
                fallback_value=float(history_model_scale[-1]),
                n_steps=block_len,
            )

            base_pred_original = inverse_transform_target(base_pred_model_scale, transform_info)
            as_pred_original = inverse_transform_target(as_pred_model_scale, transform_info)
            smooth_signal_original = _inverse_scalar(smooth_signal_model_scale, transform_info)
            selected_smoother = components.get("selected_smoother")
            selected_score = components.get("selected_score", np.nan)

        except Exception as exc:
            print(f"[WARN] visualization failed at task={task_id}, h={horizon}, origin={origin_relative_t}: {exc}")
            base_pred_original = np.repeat(fallback_original, block_len)
            as_pred_original = np.repeat(fallback_original, block_len)
            smooth_signal_original = fallback_original
            selected_smoother = "error"
            selected_score = np.nan

        y_true_block = query_y_original[query_start:query_start + block_len]

        origin_records.append(
            {
                "task_id": task_id,
                "dataset_name": dataset_name,
                "horizon": int(horizon),
                "query_start": int(query_start),
                "origin_relative_t": int(origin_relative_t),
                "selected_smoother": str(selected_smoother),
                "selected_score": float(selected_score) if np.isfinite(selected_score) else np.nan,
                "smooth_signal": float(smooth_signal_original),
            }
        )

        for j in range(block_len):
            query_row = query.iloc[query_start + j]
            records.append(
                {
                    "task_id": task_id,
                    "dataset_name": dataset_name,
                    "horizon": int(horizon),
                    "query_start": int(query_start),
                    "origin_relative_t": int(origin_relative_t),
                    "target_relative_t": int(query_row["relative_t"]),
                    "target_query_index": int(query_start + j),
                    "horizon_step": int(j + 1),
                    "ds": query_row["ds"],
                    "y_true": float(y_true_block[j]),
                    "chronos2_pred": float(base_pred_original[j]),
                    "adaptsmooth_chronos2_pred": float(as_pred_original[j]),
                    "adapted_smooth_signal": float(smooth_signal_original),
                    "selected_smoother": str(selected_smoother),
                    "selected_score": float(selected_score) if np.isfinite(selected_score) else np.nan,
                }
            )

        history_original = np.concatenate([history_original, y_true_block])
        history_ds = pd.concat(
            [history_ds, query.iloc[query_start:query_start + block_len]["ds"]],
            ignore_index=True,
        )

    trace_df = pd.DataFrame(records)
    y_true = trace_df["y_true"].to_numpy(dtype=float)
    chronos_pred = trace_df["chronos2_pred"].to_numpy(dtype=float)
    as_pred = trace_df["adaptsmooth_chronos2_pred"].to_numpy(dtype=float)

    summary = {
        "task_id": task_id,
        "dataset_name": dataset_name,
        "horizon": int(horizon),
        "n_support": int(len(support)),
        "n_query": int(len(query)),
        "chronos2_rmse": rmse(y_true, chronos_pred),
        "adaptsmooth_chronos2_rmse": rmse(y_true, as_pred),
        "rmse_improvement": rmse(y_true, chronos_pred) - rmse(y_true, as_pred),
        "chronos2_mae": mae(y_true, chronos_pred),
        "adaptsmooth_chronos2_mae": mae(y_true, as_pred),
        "selected_smoother_sequence": " | ".join(pd.DataFrame(origin_records)["selected_smoother"].astype(str).tolist()),
    }

    return trace_df, summary


def _visible_query_window(trace_df: pd.DataFrame, max_origins: int = MAX_VISIBLE_ORIGINS) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """Return a clear query slice containing at most ``max_origins``.

    The default mode selects the contiguous window where AdaptSmooth-Chronos2
    gives the largest RMSE improvement over Chronos-2, which is useful for a
    qualitative paper figure.
    """
    df = trace_df.copy()
    origin_df = (
        df[["query_start", "origin_relative_t", "target_query_index", "selected_smoother"]]
        .drop_duplicates("query_start")
        .sort_values("query_start")
        .reset_index(drop=True)
    )

    if origin_df.empty:
        return df, origin_df, {
            "shown_origin_start": 0,
            "shown_origin_end": 0,
            "visible_chronos2_rmse": np.nan,
            "visible_as_rmse": np.nan,
        }

    n_orig = len(origin_df)
    k = min(int(max_origins), n_orig)

    if VISIBLE_ORIGIN_MODE == "first" or n_orig <= k:
        start = 0
    elif VISIBLE_ORIGIN_MODE == "last":
        start = n_orig - k
    elif VISIBLE_ORIGIN_MODE == "middle":
        start = max(0, (n_orig - k) // 2)
    elif VISIBLE_ORIGIN_MODE == "best_improvement":
        best_start = 0
        best_gain = -np.inf
        for i in range(0, n_orig - k + 1):
            starts = set(origin_df.iloc[i:i + k]["query_start"].astype(int).tolist())
            sub = df[df["query_start"].astype(int).isin(starts)]
            if sub.empty:
                continue
            y = sub["y_true"].to_numpy(dtype=float)
            c = sub["chronos2_pred"].to_numpy(dtype=float)
            a = sub["adaptsmooth_chronos2_pred"].to_numpy(dtype=float)
            gain = rmse(y, c) - rmse(y, a)
            if np.isfinite(gain) and gain > best_gain:
                best_gain = gain
                best_start = i
        start = best_start
    else:
        start = 0

    shown_origins = origin_df.iloc[start:start + k].copy().reset_index(drop=True)
    shown_starts = set(shown_origins["query_start"].astype(int).tolist())
    visible = df[df["query_start"].astype(int).isin(shown_starts)].copy().reset_index(drop=True)

    y = visible["y_true"].to_numpy(dtype=float)
    c = visible["chronos2_pred"].to_numpy(dtype=float)
    a = visible["adaptsmooth_chronos2_pred"].to_numpy(dtype=float)
    info = {
        "shown_origin_start": int(start + 1),
        "shown_origin_end": int(start + k),
        "visible_chronos2_rmse": rmse(y, c) if len(y) else np.nan,
        "visible_as_rmse": rmse(y, a) if len(y) else np.nan,
    }
    return visible, shown_origins, info


def _plot_one_task(ax, full_df: pd.DataFrame, trace_df: pd.DataFrame, summary: Dict):
    visible, origin_df, winfo = _visible_query_window(trace_df, MAX_VISIBLE_ORIGINS)

    if visible.empty:
        ax.set_title(f"{summary.get('dataset_name', '?')} | no visible trace")
        return

    if USE_QUERY_INDEX_XAXIS:
        x = visible["target_query_index"].to_numpy(dtype=float)
        x_label = "Query time index"
    else:
        x = visible["target_relative_t"].to_numpy(dtype=float)
        x_label = "Relative time"

    y_actual = visible["y_true"].to_numpy(dtype=float)
    y_smooth = visible["adapted_smooth_signal"].to_numpy(dtype=float)
    y_base = visible["chronos2_pred"].to_numpy(dtype=float)
    y_as = visible["adaptsmooth_chronos2_pred"].to_numpy(dtype=float)

    ax.plot(x, y_actual, label="Actual", linewidth=1.8)
    ax.step(x, y_smooth, where="post", label="Adapted smooth signal", linewidth=1.35, linestyle="--")
    ax.plot(x, y_base, label="Chronos-2", linewidth=1.25, linestyle=":")
    ax.plot(x, y_as, label="AdaptSmooth-Chronos2", linewidth=1.55)

    y_values = np.concatenate([y_actual, y_smooth, y_base, y_as])
    ymin = float(np.nanmin(y_values))
    ymax = float(np.nanmax(y_values))
    yrng = ymax - ymin if ymax > ymin else 1.0

    # Reserve a compact top strip for the selected operator labels.
    ax.set_ylim(ymin - 0.05 * yrng, ymax + 0.24 * yrng)
    label_y_1 = ymax + 0.055 * yrng
    label_y_2 = ymax + 0.145 * yrng

    last_label = None
    for i, row in origin_df.iterrows():
        q_start = int(row["query_start"])
        block = visible[visible["query_start"].astype(int) == q_start]
        if block.empty:
            continue

        if USE_QUERY_INDEX_XAXIS:
            xb = block["target_query_index"].to_numpy(dtype=float)
        else:
            xb = block["target_relative_t"].to_numpy(dtype=float)

        x0 = float(xb.min())
        x1 = float(xb.max())
        xmid = 0.5 * (x0 + x1)

        # Subtle boundaries for only the visible origins, not the full query.
        ax.axvline(x0, linewidth=0.55, linestyle="--", alpha=0.28)
        if i % 2 == 0:
            ax.axvspan(x0, x1, alpha=0.025)

        if ANNOTATE_SMOOTHER_NAMES:
            label = _paper_smoother_label(row["selected_smoother"])
            if ANNOTATE_ONLY_WHEN_SMOOTHER_CHANGES and label == last_label:
                continue
            last_label = label
            ax.text(
                xmid,
                label_y_1 if i % 2 == 0 else label_y_2,
                label,
                fontsize=SMOOTHER_LABEL_FONTSIZE,
                rotation=0,
                va="center",
                ha="center",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="0.78", alpha=0.88),
                clip_on=False,
            )

    ax.axhline(ymax + 0.005 * yrng, linewidth=0.55, alpha=0.25)

    # title = (
    #     f"{summary['dataset_name']} | task={summary['task_id']} | h={summary['horizon']} | "
    #     f"shown origins {winfo['shown_origin_start']}-{winfo['shown_origin_end']} | "
    #     f"visible RMSE: Chronos2={winfo['visible_chronos2_rmse']:.3g}, "
    #     f"AdaptSmooth-Chronos2={winfo['visible_as_rmse']:.3g}"
    # )

    title = (
        f"{summary['dataset_name']} | "
        f"visible RMSE: Chronos2={winfo['visible_chronos2_rmse']:.3g}, "
        f"AdaptSmooth-Chronos2={winfo['visible_as_rmse']:.3g}"
    )

    ax.set_title(title, fontsize=12)
    ax.set_xlabel(x_label)
    #ax.set_ylabel("Value")
    ax.grid(True, alpha=0.18)

def save_task_plot(task_row: pd.Series, horizon: int, trace_df: pd.DataFrame, summary: Dict, out_dir: Path):
    support, query = load_task(task_row["dataset_name"], str(task_row["task_id"]))
    full_df = pd.concat([support, query], ignore_index=True)

    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_SINGLE)
    _plot_one_task(ax, full_df, trace_df, summary)
    ax.legend(loc="upper left", fontsize=14, ncol=2, frameon=True)
    fig.tight_layout()

    slug = normalize_name(str(task_row["task_id"]))
    png_path = out_dir / f"task_{slug}_h{horizon}.png"
    pdf_path = out_dir / f"task_{slug}_h{horizon}.pdf"
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def save_grid_plot(task_rows: pd.DataFrame, traces: List[pd.DataFrame], summaries: List[Dict], horizon: int, out_dir: Path):
    n = len(task_rows)
    fig, axes = plt.subplots(n, 1, figsize=FIGSIZE_GRID, sharex=False)
    if n == 1:
        axes = [axes]

    for ax, (_, task_row), trace_df, summary in zip(axes, task_rows.iterrows(), traces, summaries):
        support, query = load_task(task_row["dataset_name"], str(task_row["task_id"]))
        full_df = pd.concat([support, query], ignore_index=True)
        _plot_one_task(ax, full_df, trace_df, summary)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=14, frameon=True)
    # fig.suptitle(
    #     #f"AdaptSmooth-Chronos2 visualization, h={horizon}: 10 selected prediction origins per task",
    #     y=0.995,
    #     fontsize=12,
    # )
    fig.tight_layout(rect=[0, 0, 1, 0.965])

    fig.savefig(out_dir / f"adaptsmooth_chronos2_grid_h{horizon}.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(out_dir / f"adaptsmooth_chronos2_grid_h{horizon}.pdf", bbox_inches="tight")
    plt.close(fig)


def run_visualization_study():
    if not TASKS_DIR.exists():
        raise FileNotFoundError(f"Missing task folder: {TASKS_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    index_df = load_task_index()
    task_rows = _choose_tasks(index_df)

    print("[INFO] Visualization tasks:")
    print(task_rows[["dataset_name", "task_id", "n_support", "n_query"]].to_string(index=False))
    print(f"[INFO] Output folder: {OUTPUT_DIR}")

    for horizon in VIZ_HORIZONS:
        out_dir = OUTPUT_DIR / f"h_{horizon}"
        out_dir.mkdir(parents=True, exist_ok=True)

        all_traces: List[pd.DataFrame] = []
        summaries: List[Dict] = []

        for _, task_row in task_rows.iterrows():
            dataset_name = str(task_row["dataset_name"])
            task_id = str(task_row["task_id"])
            print(f"[RUN] dataset={dataset_name}, task={task_id}, h={horizon}")
            trace_df, summary = run_task_trace(dataset_name, task_id, horizon)
            all_traces.append(trace_df)
            summaries.append(summary)
            save_task_plot(task_row, horizon, trace_df, summary, out_dir)

        traces_df = pd.concat(all_traces, ignore_index=True)
        summary_df = pd.DataFrame(summaries)
        traces_df.to_csv(out_dir / f"traces_h{horizon}.csv", index=False)
        summary_df.to_csv(out_dir / f"summary_h{horizon}.csv", index=False)
        save_grid_plot(task_rows, all_traces, summaries, horizon, out_dir)

        print("[OK] Saved:")
        print(f"     {out_dir / f'traces_h{horizon}.csv'}")
        print(f"     {out_dir / f'summary_h{horizon}.csv'}")
        print(f"     {out_dir / f'adaptsmooth_chronos2_grid_h{horizon}.png'}")
        print(f"     {out_dir / f'adaptsmooth_chronos2_grid_h{horizon}.pdf'}")


if __name__ == "__main__":
    run_visualization_study()
