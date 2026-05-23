"""Evaluation-side utility functions.

This module intentionally contains only benchmark/data/metric/target-transform logic.
Forecasting model code lives under models/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple
import json

import numpy as np
import pandas as pd

from eval_config import (
    TASKS_DIR,
    INDEX_FILE,
    MAX_TASKS,
    MAX_TASKS_PER_DOMAIN,
    RANDOM_SAMPLE_TASKS_PER_DOMAIN,
    TASK_SELECTION_RANDOM_SEED,
    TASK_SELECTION_RUN_FILE,
    APPLY_LOG,
    LOG_MODE,
    LOG_EPS,
)


def normalize_name(name: str) -> str:
    return str(name).lower().replace(" ", "_").replace("-", "_")


def safe_model_name(name: str) -> str:
    return normalize_name(name)


def load_task_index() -> pd.DataFrame:
    if not INDEX_FILE.exists():
        raise FileNotFoundError(f"Missing task index file: {INDEX_FILE}")

    df = pd.read_csv(INDEX_FILE)

    required = {"task_id", "dataset_name", "source_file", "n_support", "n_query"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Task index is missing columns: {missing}")

    df = df.reset_index(drop=True)
    df["_original_order"] = np.arange(len(df))

    if MAX_TASKS_PER_DOMAIN is not None:
        if MAX_TASKS_PER_DOMAIN <= 0:
            raise ValueError("MAX_TASKS_PER_DOMAIN must be positive or None.")

        rng = np.random.default_rng(TASK_SELECTION_RANDOM_SEED)

        selected_groups = []
        selected_payload = {}

        for dataset_name, group in df.groupby("dataset_name", sort=False):
            group = group.copy()
            n_available = len(group)
            n_select = min(MAX_TASKS_PER_DOMAIN, n_available)

            if RANDOM_SAMPLE_TASKS_PER_DOMAIN and n_select < n_available:
                sample_seed = int(rng.integers(0, 2**32 - 1))
                selected = group.sample(
                    n=n_select,
                    replace=False,
                    random_state=sample_seed,
                )
            else:
                selected = group.head(n_select)

            selected = selected.sort_values("_original_order")
            selected_groups.append(selected)

            selected_payload[str(dataset_name)] = {
                "n_available": int(n_available),
                "n_selected": int(len(selected)),
                "selected_task_ids": selected["task_id"].astype(str).tolist(),
            }

        df = (
            pd.concat(selected_groups, axis=0)
            .sort_values("_original_order")
            .reset_index(drop=True)
        )

        TASK_SELECTION_RUN_FILE.parent.mkdir(parents=True, exist_ok=True)
        TASK_SELECTION_RUN_FILE.write_text(
            json.dumps(
                {
                    "max_tasks_per_domain": MAX_TASKS_PER_DOMAIN,
                    "random_sample_tasks_per_domain": RANDOM_SAMPLE_TASKS_PER_DOMAIN,
                    "task_selection_random_seed": TASK_SELECTION_RANDOM_SEED,
                    "selected_by_dataset": selected_payload,
                },
                indent=4,
            ),
            encoding="utf-8",
        )

        print(f"[INFO] Saved task selection for this run: {TASK_SELECTION_RUN_FILE}")

    if MAX_TASKS is not None:
        df = df.iloc[:MAX_TASKS].copy()

    df = df.drop(columns=["_original_order"])

    print("[INFO] Tasks selected per dataset:")
    print(df.groupby("dataset_name")["task_id"].count().to_string())

    print(f"[INFO] Total selected tasks: {len(df):,}")
    print(f"[INFO] Unique selected task IDs: {df['task_id'].nunique():,}")

    return df.reset_index(drop=True)


def infer_task_folder(dataset_name: str, task_id: str) -> Path:
    dataset_folder = TASKS_DIR / normalize_name(dataset_name)
    task_folder = dataset_folder / str(task_id)
    if task_folder.exists():
        return task_folder

    matches = list(TASKS_DIR.glob(f"*/{task_id}"))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Could not find folder for task_id={task_id}")


def load_task(dataset_name: str, task_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    task_folder = infer_task_folder(dataset_name, task_id)
    full_path = task_folder / "full_task.csv"
    if not full_path.exists():
        raise FileNotFoundError(f"Missing full_task.csv: {full_path}")

    df = pd.read_csv(full_path)
    required = {"split", "relative_t", "ds", "y"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{full_path} is missing columns: {missing}")

    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["y"]).reset_index(drop=True)

    support = df[df["split"] == "support"].copy().reset_index(drop=True)
    query = df[df["split"] == "query"].copy().reset_index(drop=True)

    if support.empty:
        raise ValueError(f"{task_id}: empty support set")
    if query.empty:
        raise ValueError(f"{task_id}: empty query set")

    return support, query


def transform_target(y: np.ndarray) -> Tuple[np.ndarray, Dict]:
    y = np.asarray(y, dtype=float)

    if not APPLY_LOG:
        return y, {"mode": "none"}

    if LOG_MODE == "log":
        if np.any(y <= 0):
            raise ValueError("LOG_MODE='log' requires strictly positive y values.")
        return np.log(y + LOG_EPS), {"mode": "log"}

    if LOG_MODE == "log1p":
        if np.any(y < 0):
            raise ValueError("LOG_MODE='log1p' requires y >= 0.")
        return np.log1p(y), {"mode": "log1p"}

    if LOG_MODE == "signed_log1p":
        z = np.sign(y) * np.log1p(np.abs(y))
        return z, {"mode": "signed_log1p"}

    if LOG_MODE == "shifted_log":
        min_y = float(np.min(y)) if len(y) else 0.0
        shift = 0.0 if min_y > 0 else abs(min_y) + 1.0
        z = np.log(y + shift + LOG_EPS)
        return z, {"mode": "shifted_log", "shift": shift}

    raise ValueError(f"Unknown LOG_MODE: {LOG_MODE}")


def inverse_transform_target(y_transformed: np.ndarray, transform_info: Dict) -> np.ndarray:
    y_transformed = np.asarray(y_transformed, dtype=float)

    mode = transform_info.get("mode", "none")

    if mode == "none":
        return y_transformed

    if mode == "log":
        return np.exp(y_transformed) - LOG_EPS

    if mode == "log1p":
        return np.expm1(y_transformed)

    if mode == "signed_log1p":
        return np.sign(y_transformed) * np.expm1(np.abs(y_transformed))

    if mode == "shifted_log":
        shift = float(transform_info.get("shift", 0.0))
        return np.exp(y_transformed) - shift - LOG_EPS

    raise ValueError(f"Unknown inverse LOG_MODE: {mode}")


def infer_seasonal_period(dataset_name: str, ds_values: Optional[pd.Series] = None) -> Optional[int]:
    slug = normalize_name(dataset_name)

    explicit = {
        "btc": 24,
        "bitcoin": 24,
        "electricity_consumption": 24,
        "electricity": 24,
        "traffic": 24,
        "etd_ot": 24,
        "ett_ot": 24,
        "temperature": 144,
        "wind_speed": 144,
        "solar_power": 144,
        "exchange_rate": 5,
        "gold_prices": 5,
        "gold": 5,
        "natural_gas": 5,
    }
    for key, value in explicit.items():
        if key in slug:
            return value

    if ds_values is not None:
        try:
            dt = pd.to_datetime(ds_values, errors="coerce").dropna()
            if len(dt) >= 3:
                delta_seconds = dt.sort_values().diff().dropna().dt.total_seconds().median()
                if np.isfinite(delta_seconds) and delta_seconds > 0:
                    if abs(delta_seconds - 3600) < 60:
                        return 24
                    if abs(delta_seconds - 600) < 30:
                        return 144
                    if abs(delta_seconds - 86400) < 3600:
                        return 7
        except Exception:
            pass

    return None


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    valid = denom > 0
    if valid.sum() == 0:
        return np.nan
    return float(np.mean(2.0 * np.abs(y_true[valid] - y_pred[valid]) / denom[valid]))
