from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

CURATED_DIR = Path("CURATED_TIME_SERIES_DATASETS")
TASKS_DIR = Path("FST_BENCHMARK_TASKS")

TASKS_PER_DATASET = 100
RANDOM_SEED = 42

MIN_SUPPORT = 10
MAX_SUPPORT = 100

MIN_QUERY = 40
MAX_QUERY = 400

HORIZONS = [1, 2, 3, 4, 5, 10]

DATASETS = [
    ("Temperature.csv", "Temperature"),
    ("Wind speed.csv", "Wind speed"),
    ("Solar power.csv", "Solar power"),
    ("Natural gas.csv", "Natural gas"),
    ("Gold prices.csv", "Gold prices"),
    ("ETD-OT.csv", "ETD-OT"),
    ("BTC.csv", "BTC"),
    ("Electricity consumption.csv", "Electricity consumption"),
    ("Exchange rate.csv", "Exchange rate"),
    ("Traffic.csv", "Traffic"),
]


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class FSTTaskMeta:
    task_id: str
    dataset_name: str
    source_file: str

    n_support: int
    n_query: int
    n_total: int

    start_idx: int
    support_start_idx: int
    support_end_idx: int
    query_start_idx: int
    query_end_idx: int

    support_start_ds: str
    support_end_ds: str
    query_start_ds: str
    query_end_ds: str

    horizons: str


# ============================================================
# HELPERS
# ============================================================

def find_file(data_dir: Path, filename: str) -> Path:
    path = data_dir / filename
    if path.exists():
        return path

    target_stem = Path(filename).stem.lower()

    for p in data_dir.glob("*.csv"):
        if p.stem.lower() == target_stem:
            return p

    raise FileNotFoundError(f"Could not find dataset file: {filename}")


def load_curated_series(path: Path) -> pd.DataFrame:
    """
    Load a curated univariate series with columns:
    unique_id, ds, y
    """
    df = pd.read_csv(path)

    required_cols = {"unique_id", "ds", "y"}
    missing = required_cols.difference(df.columns)

    if missing:
        raise ValueError(
            f"{path.name}: missing columns {missing}. "
            f"Expected columns: {required_cols}"
        )

    df = df[["unique_id", "ds", "y"]].copy()

    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["y"]).reset_index(drop=True)

    if len(df) == 0:
        raise ValueError(f"{path.name}: empty after cleaning.")

    return df


def sample_task_indices(
    series_length: int,
    rng: np.random.Generator,
    min_support: int,
    max_support: int,
    min_query: int,
    max_query: int,
) -> Tuple[int, int, int]:
    """
    Sample a contiguous few-shot forecasting task.

    Returns:
        start_idx, n_support, n_query

    The task is:
        support = [start_idx, ..., start_idx + n_support - 1]
        query   = [start_idx + n_support, ..., start_idx + n_support + n_query - 1]
    """
    if series_length < min_support + min_query:
        raise ValueError(
            f"Series too short: length={series_length}, "
            f"minimum required={min_support + min_query}"
        )

    # Avoid sampling support/query sizes that exceed the series
    max_possible_support = min(max_support, series_length - min_query)
    n_support = int(rng.integers(min_support, max_possible_support + 1))

    max_possible_query = min(max_query, series_length - n_support)
    n_query = int(rng.integers(min_query, max_possible_query + 1))

    max_start = series_length - n_support - n_query
    start_idx = int(rng.integers(0, max_start + 1))

    return start_idx, n_support, n_query


def create_single_task(
    df: pd.DataFrame,
    dataset_name: str,
    source_file: str,
    task_number: int,
    rng: np.random.Generator,
) -> Tuple[FSTTaskMeta, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    start_idx, n_support, n_query = sample_task_indices(
        series_length=len(df),
        rng=rng,
        min_support=MIN_SUPPORT,
        max_support=MAX_SUPPORT,
        min_query=MIN_QUERY,
        max_query=MAX_QUERY,
    )

    support_start_idx = start_idx
    support_end_idx = start_idx + n_support - 1

    query_start_idx = start_idx + n_support
    query_end_idx = start_idx + n_support + n_query - 1

    support = df.iloc[support_start_idx:support_end_idx + 1].copy()
    query = df.iloc[query_start_idx:query_end_idx + 1].copy()

    task_id = f"{dataset_name.lower().replace(' ', '_').replace('-', '_')}_task_{task_number:03d}"

    support["task_id"] = task_id
    support["dataset_name"] = dataset_name
    support["split"] = "support"
    support["relative_t"] = np.arange(len(support))

    query["task_id"] = task_id
    query["dataset_name"] = dataset_name
    query["split"] = "query"
    query["relative_t"] = np.arange(len(support), len(support) + len(query))

    full_task = pd.concat([support, query], axis=0, ignore_index=True)

    full_task = full_task[
        [
            "task_id",
            "dataset_name",
            "split",
            "relative_t",
            "unique_id",
            "ds",
            "y",
        ]
    ]

    meta = FSTTaskMeta(
        task_id=task_id,
        dataset_name=dataset_name,
        source_file=source_file,

        n_support=n_support,
        n_query=n_query,
        n_total=n_support + n_query,

        start_idx=start_idx,
        support_start_idx=support_start_idx,
        support_end_idx=support_end_idx,
        query_start_idx=query_start_idx,
        query_end_idx=query_end_idx,

        support_start_ds=str(support["ds"].iloc[0]),
        support_end_ds=str(support["ds"].iloc[-1]),
        query_start_ds=str(query["ds"].iloc[0]),
        query_end_ds=str(query["ds"].iloc[-1]),

        horizons=",".join(map(str, HORIZONS)),
    )

    return meta, support, query, full_task


# ============================================================
# MAIN GENERATOR
# ============================================================

def generate_fst_benchmark_tasks():
    TASKS_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(RANDOM_SEED)

    all_metadata: List[Dict] = []
    all_tasks_long: List[pd.DataFrame] = []

    print(f"[INFO] Curated directory: {CURATED_DIR}")
    print(f"[INFO] Output directory:  {TASKS_DIR}")
    print(f"[INFO] Tasks per dataset: {TASKS_PER_DATASET}")
    print(f"[INFO] Random seed:       {RANDOM_SEED}")

    for filename, dataset_name in DATASETS:
        path = find_file(CURATED_DIR, filename)
        df = load_curated_series(path)

        print(f"\n[DATASET] {dataset_name}")
        print(f"  file: {path.name}")
        print(f"  rows: {len(df):,}")

        dataset_folder = TASKS_DIR / dataset_name.lower().replace(" ", "_").replace("-", "_")
        dataset_folder.mkdir(parents=True, exist_ok=True)

        for task_number in range(1, TASKS_PER_DATASET + 1):
            meta, support, query, full_task = create_single_task(
                df=df,
                dataset_name=dataset_name,
                source_file=path.name,
                task_number=task_number,
                rng=rng,
            )

            task_folder = dataset_folder / meta.task_id
            task_folder.mkdir(parents=True, exist_ok=True)

            support.to_csv(task_folder / "support.csv", index=False)
            query.to_csv(task_folder / "query.csv", index=False)
            full_task.to_csv(task_folder / "full_task.csv", index=False)

            all_metadata.append(asdict(meta))
            all_tasks_long.append(full_task)

        print(f"  generated tasks: {TASKS_PER_DATASET}")

    metadata_df = pd.DataFrame(all_metadata)
    metadata_path = TASKS_DIR / "fst_tasks_index.csv"
    metadata_df.to_csv(metadata_path, index=False)

    long_df = pd.concat(all_tasks_long, axis=0, ignore_index=True)
    long_path = TASKS_DIR / "fst_tasks_long.csv"
    long_df.to_csv(long_path, index=False)

    print("\n[OK] Few-shot benchmark generated.")
    print(f"[OK] Metadata index: {metadata_path}")
    print(f"[OK] Long-format tasks file: {long_path}")
    print(f"[OK] Total tasks: {len(metadata_df):,}")
    print(f"[OK] Total rows in long file: {len(long_df):,}")


if __name__ == "__main__":
    generate_fst_benchmark_tasks()