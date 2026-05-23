# Run with:
#   python LOADER_EVALUATOR.py
#
# Expected input:
#   FST_BENCHMARK_TASKS/
#     fst_tasks_index.csv
#     <dataset_slug>/<task_id>/full_task.csv
#
# Creates:
#   FST_BENCHMARK_PREDICTIONS/
#     evaluation_config.json
#     metrics_by_task.csv
#     summary_by_model_horizon.csv
#     <model_slug>/h_<horizon>/<dataset_slug>_predictions.csv

from __future__ import annotations

from typing import Dict, Tuple
import json
import shutil
import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from eval_config import (
    TASKS_DIR,
    PRED_DIR,
    INDEX_FILE,
    CLEAR_OUTPUT,
    EVAL_HORIZONS,
    ENABLE_FOUNDATION_MODELS,
    ENABLE_CHRONOS,
    # ENABLE_TIMESFM,
    # ENABLE_LAG_LLAMA,
    CHRONOS_MODEL_ID,
    # TIMESFM_REPO_ID,
    # LAG_LLAMA_CHECKPOINT_PATH,
)
from eval_utils import (
    normalize_name,
    safe_model_name,
    load_task_index,
    load_task,
    transform_target,
    inverse_transform_target,
    infer_seasonal_period,
    rmse,
    mae,
    smape,
)
from models.base import BaseForecaster
from models.model_utils import sanitize_predictions
from models.registry import get_model_specs
from models.smoothers.config import SMOOTHER_CANDIDATES

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def evaluate_model_on_task(
    model: BaseForecaster,
    support: pd.DataFrame,
    query: pd.DataFrame,
    task_id: str,
    dataset_name: str,
    horizon: int,
) -> Tuple[pd.DataFrame, Dict]:

    support_y_original = support["y"].to_numpy(dtype=float)
    query_y_original = query["y"].to_numpy(dtype=float)
    full_ds = pd.concat([support["ds"], query["ds"]], ignore_index=True)

    history_original = support_y_original.copy()
    history_ds = support["ds"].copy().reset_index(drop=True)

    records = []
    query_len = len(query_y_original)
    seasonal_period = infer_seasonal_period(dataset_name, full_ds)

    for query_start in range(0, query_len, horizon):
        block_len = min(horizon, query_len - query_start)
        origin_relative_t = int(len(history_original) - 1)
        fallback_original = float(history_original[-1]) if len(history_original) else 0.0

        context = {
            "task_id": task_id,
            "dataset_name": dataset_name,
            "horizon": horizon,
            "seasonal_period": seasonal_period,
            "history_ds": history_ds,
        }

        try:
            history_model_scale, transform_info = transform_target(history_original)

            model.fit(history_model_scale, horizon=horizon, context=context)
            pred_model_scale = model.predict(block_len)

            fallback_model_scale = transform_target(np.asarray([fallback_original]))[0][0]

            pred_model_scale = sanitize_predictions(
                pred=pred_model_scale,
                fallback_value=fallback_model_scale,
                n_steps=block_len,
            )

            pred_original = inverse_transform_target(pred_model_scale, transform_info)

        except NotImplementedError as exc:
            raise exc
        except Exception as exc:
            print(f"[WARN] {model.name} failed on task={task_id}, h={horizon}: {exc}")
            pred_original = np.repeat(fallback_original, block_len)

        pred_original = sanitize_predictions(
            pred=pred_original,
            fallback_value=fallback_original,
            n_steps=block_len,
        )

        y_true_block_original = query_y_original[query_start:query_start + block_len]

        for j in range(block_len):
            query_row = query.iloc[query_start + j]
            records.append({
                "task_id": task_id,
                "dataset_name": dataset_name,
                "model_name": model.name,
                "horizon": horizon,
                "origin_relative_t": origin_relative_t,
                "target_relative_t": int(query_row["relative_t"]),
                "target_query_index": int(query_start + j),
                "horizon_step": int(j + 1),
                "ds": query_row["ds"],
                "y_true": float(y_true_block_original[j]),
                "y_pred": float(pred_original[j]),
            })

        # Online evaluation: reveal true query values on the ORIGINAL scale.
        history_original = np.concatenate([history_original, y_true_block_original])
        history_ds = pd.concat(
            [history_ds, query.iloc[query_start:query_start + block_len]["ds"]],
            ignore_index=True,
        )

    pred_df = pd.DataFrame(records)
    y_true_all = pred_df["y_true"].to_numpy(dtype=float)
    y_pred_all = pred_df["y_pred"].to_numpy(dtype=float)

    metric_row = {
        "task_id": task_id,
        "dataset_name": dataset_name,
        "model_name": model.name,
        "horizon": horizon,
        "n_support": int(len(support)),
        "n_query": int(len(query)),
        "seasonal_period": seasonal_period if seasonal_period is not None else np.nan,
        "rmse": rmse(y_true_all, y_pred_all),
        "mae": mae(y_true_all, y_pred_all),
        "smape": smape(y_true_all, y_pred_all),
    }

    return pred_df, metric_row


def append_csv(df: pd.DataFrame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    df.to_csv(path, mode="a", header=write_header, index=False)


def prepare_output_folder():
    if CLEAR_OUTPUT and PRED_DIR.exists():
        shutil.rmtree(PRED_DIR)
    PRED_DIR.mkdir(parents=True, exist_ok=True)


def write_config(model_specs):
    config = {
        "tasks_dir": str(TASKS_DIR),
        "pred_dir": str(PRED_DIR),
        "index_file": str(INDEX_FILE),
        "eval_horizons": EVAL_HORIZONS,
        "models": [spec.name for spec in model_specs],
        "rolling_evaluation": True,
        "foundation_models_enabled": ENABLE_FOUNDATION_MODELS,
        "enable_chronos": ENABLE_CHRONOS,
        ###"enable_timesfm": ENABLE_TIMESFM,
        ###"enable_lag_llama": ENABLE_LAG_LLAMA,
        "chronos_model_id": CHRONOS_MODEL_ID if ENABLE_FOUNDATION_MODELS else None,
        ###"timesfm_repo_id": TIMESFM_REPO_ID if ENABLE_FOUNDATION_MODELS else None,
        ###"lag_llama_checkpoint_path": LAG_LLAMA_CHECKPOINT_PATH if ENABLE_FOUNDATION_MODELS else None,
        "smoother_candidates": SMOOTHER_CANDIDATES,
        "description": (
            "For each task and horizon h, each model is fitted on the current observed history, "
            "forecasts the next h query values, then the true query block is revealed before "
            "the next origin. Models are registered through models/registry.py."
        ),
    }
    with open(PRED_DIR / "evaluation_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)


def run_benchmark_evaluation():
    prepare_output_folder()

    index_df = load_task_index()
    model_specs = get_model_specs()
    write_config(model_specs)

    all_metrics = []

    print(f"[INFO] Loaded task index: {INDEX_FILE}")
    print(f"[INFO] Number of tasks: {len(index_df):,}")
    print(f"[INFO] Models: {[spec.name for spec in model_specs]}")
    print(f"[INFO] Horizons: {EVAL_HORIZONS}")
    print(f"[INFO] Output directory: {PRED_DIR}")

    total_runs = len(index_df) * len(model_specs) * len(EVAL_HORIZONS)
    run_counter = 0

    for _, row in index_df.iterrows():
        task_id = row["task_id"]
        dataset_name = row["dataset_name"]
        support, query = load_task(dataset_name, task_id)

        for horizon in EVAL_HORIZONS:
            for spec in model_specs:
                model = spec.factory()

                try:
                    pred_df, metric_row = evaluate_model_on_task(
                        model=model,
                        support=support,
                        query=query,
                        task_id=task_id,
                        dataset_name=dataset_name,
                        horizon=horizon,
                    )
                except NotImplementedError as exc:
                    print(f"[WARN] Skipping {model.name}: {exc}")
                    run_counter += 1
                    continue

                model_slug = safe_model_name(model.name)
                dataset_slug = normalize_name(dataset_name)

                model_horizon_path = (
                    PRED_DIR
                    / model_slug
                    / f"h_{horizon}"
                    / f"{dataset_slug}_predictions.csv"
                )

                append_csv(pred_df, model_horizon_path)
                all_metrics.append(metric_row)

                run_counter += 1
                if run_counter % 100 == 0 or run_counter == total_runs:
                    print(f"[PROGRESS] {run_counter:,}/{total_runs:,} model-task-horizon runs completed")

    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = PRED_DIR / "metrics_by_task.csv"
    metrics_df.to_csv(metrics_path, index=False)

    summary = (
        metrics_df
        .groupby(["model_name", "horizon"], as_index=False)
        .agg(
            mean_rmse=("rmse", "mean"),
            median_rmse=("rmse", "median"),
            mean_mae=("mae", "mean"),
            median_mae=("mae", "median"),
            mean_smape=("smape", "mean"),
            median_smape=("smape", "median"),
            n_tasks=("task_id", "count"),
        )
        .sort_values(["horizon", "mean_rmse"])
    )

    summary_path = PRED_DIR / "summary_by_model_horizon.csv"
    summary.to_csv(summary_path, index=False)

    print("\n[OK] Benchmark evaluation completed.")
    print(f"[OK] Task-level metrics:    {metrics_path}")
    print(f"[OK] Summary metrics:       {summary_path}")
    print("\n[SUMMARY]")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    run_benchmark_evaluation()
