from pathlib import Path
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

# ============================================================
# CONFIG
# ============================================================

TASKS_DIR = Path("FST_BENCHMARK_TASKS")
FIG_DIR = TASKS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

INDEX_FILE = TASKS_DIR / "fst_tasks_index.csv"

RANDOM_SEED = 42

# ============================================================
# CONFIG
# ============================================================

TASKS_PER_DOMAIN_TO_PLOT = 3   # set to 2 or 3



# nice colors
SUPPORT_COLOR = "#1f77b4"   # blue
QUERY_COLOR = "#d62728"     # red
SPLIT_COLOR = "black"


# ============================================================
# HELPERS
# ============================================================

def normalize_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


def load_index(index_file: Path) -> pd.DataFrame:
    df = pd.read_csv(index_file)
    required = {
        "task_id", "dataset_name", "source_file",
        "n_support", "n_query"
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in fst_tasks_index.csv: {missing}")
    return df


def infer_task_folder(dataset_name: str, task_id: str) -> Path:
    dataset_folder = TASKS_DIR / normalize_name(dataset_name)
    task_folder = dataset_folder / task_id
    if not task_folder.exists():
        raise FileNotFoundError(f"Task folder not found: {task_folder}")
    return task_folder


def load_full_task(dataset_name: str, task_id: str) -> pd.DataFrame:
    task_folder = infer_task_folder(dataset_name, task_id)
    full_path = task_folder / "full_task.csv"

    if not full_path.exists():
        raise FileNotFoundError(f"Missing full_task.csv: {full_path}")

    df = pd.read_csv(full_path)

    required = {"split", "y"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"{full_path}: expected at least columns {required}, got {df.columns.tolist()}"
        )

    if "relative_t" not in df.columns:
        df["relative_t"] = np.arange(len(df))

    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["y"]).reset_index(drop=True)

    return df


def sample_one_task_per_domain(index_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)

    sampled_rows = []
    for dataset_name, group in index_df.groupby("dataset_name", sort=True):
        row = group.sample(n=1, random_state=rng.randint(0, 10_000)).iloc[0]
        sampled_rows.append(row)

    out = pd.DataFrame(sampled_rows).reset_index(drop=True)
    return out


def sample_random_stream(index_df: pd.DataFrame, k: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    k = min(k, len(index_df))
    idx = rng.choice(len(index_df), size=k, replace=False)
    return index_df.iloc[idx].reset_index(drop=True)


# ============================================================
# FIGURE 1: RANDOM TASKS FROM EACH DOMAIN
# ============================================================





# ============================================================
# SAMPLING: K TASKS FROM EACH DOMAIN
# ============================================================

def sample_k_tasks_per_domain(
    index_df: pd.DataFrame,
    k: int = 3,
    seed: int = 42
) -> pd.DataFrame:
    rng = random.Random(seed)

    sampled_rows = []

    for dataset_name, group in index_df.groupby("dataset_name", sort=True):
        n = min(k, len(group))
        rows = group.sample(
            n=n,
            random_state=rng.randint(0, 10_000)
        )
        sampled_rows.append(rows)

    out = pd.concat(sampled_rows, axis=0).reset_index(drop=True)
    return out





# def plot_sample_tasks_from_each_domain(index_df: pd.DataFrame):
#     sampled = sample_one_task_per_domain(index_df, seed=RANDOM_SEED)

#     fig, axes = plt.subplots(5, 2, figsize=(15, 12))
#     axes = axes.flatten()

#     legend_handles = [
#         Line2D([0], [0], color=SUPPORT_COLOR, lw=2, label="Support"),
#         Line2D([0], [0], color=QUERY_COLOR, lw=2, label="Query"),
#         Line2D([0], [0], color=SPLIT_COLOR, lw=1.2, linestyle="--", label="Support/Query split"),
#     ]

#     for ax, (_, row) in zip(axes, sampled.iterrows()):
#         dataset_name = row["dataset_name"]
#         task_id = row["task_id"]
#         n_support = int(row["n_support"])
#         n_query = int(row["n_query"])

#         df = load_full_task(dataset_name, task_id)

#         support_df = df[df["split"] == "support"].copy()
#         query_df = df[df["split"] == "query"].copy()

#         # x-axis = relative position inside task
#         x_support = support_df["relative_t"].to_numpy()
#         y_support = support_df["y"].to_numpy()

#         x_query = query_df["relative_t"].to_numpy()
#         y_query = query_df["y"].to_numpy()

#         if len(x_support) > 0:
#             ax.plot(x_support, y_support, color=SUPPORT_COLOR, linewidth=1.5)

#         if len(x_query) > 0:
#             ax.plot(x_query, y_query, color=QUERY_COLOR, linewidth=1.5)

#         # vertical split line
#         split_x = n_support - 0.5
#         ax.axvline(split_x, color=SPLIT_COLOR, linestyle="--", linewidth=1.0)

#         # light background shading
#         ax.axvspan(-0.5, n_support - 0.5, alpha=0.08, color=SUPPORT_COLOR)
#         ax.axvspan(n_support - 0.5, n_support + n_query - 0.5, alpha=0.08, color=QUERY_COLOR)

#         ax.set_title(dataset_name, fontsize=11, pad=6)
#         ax.set_xlabel("Relative time within task", fontsize=9)
#         ax.set_ylabel("y", fontsize=9)
#         ax.grid(True, alpha=0.25)
#         ax.tick_params(axis="both", labelsize=8)

#         ax.text(
#             0.01, 0.95,
#             f"{task_id}\n$n_S$={n_support}, $n_Q$={n_query}",
#             transform=ax.transAxes,
#             fontsize=8,
#             va="top"
#         )

#     # hide unused axes if any
#     for i in range(len(sampled), len(axes)):
#         axes[i].axis("off")


#     fig.legend(
#         handles=legend_handles,
#         loc="upper center",
#         ncol=3,
#         bbox_to_anchor=(0.5, 0.985),
#         fontsize=10,
#         frameon=False
#     )

#     plt.subplots_adjust(
#         left=0.06,
#         right=0.99,
#         top=0.90,
#         bottom=0.06,
#         hspace=0.65,
#         wspace=0.22
#     )

#     out_png = FIG_DIR / "sample_tasks_each_domain.png"
#     out_pdf = FIG_DIR / "sample_tasks_each_domain.pdf"

#     fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.03)
#     fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.03)
#     plt.show()

#     print(f"[OK] Saved:\n{out_png}\n{out_pdf}")


# ============================================================
# FIGURE: MULTIPLE RANDOM TASKS FROM EACH DOMAIN
# ============================================================

def plot_sample_tasks_from_each_domain2(index_df: pd.DataFrame):
    sampled = sample_k_tasks_per_domain(
        index_df,
        k=TASKS_PER_DOMAIN_TO_PLOT,
        seed=RANDOM_SEED
    )

    domains = sorted(sampled["dataset_name"].unique())
    n_domains = len(domains)
    n_cols = TASKS_PER_DOMAIN_TO_PLOT

    fig, axes = plt.subplots(
        n_domains,
        n_cols,
        figsize=(5.2 * n_cols, 1.65 * n_domains),
        sharey=False
    )

    if n_cols == 1:
        axes = np.array(axes).reshape(n_domains, 1)

    legend_handles = [
        Line2D([0], [0], color=SUPPORT_COLOR, lw=2, label="Support"),
        Line2D([0], [0], color=QUERY_COLOR, lw=2, label="Query"),
        Line2D([0], [0], color=SPLIT_COLOR, lw=1.2, linestyle="--", label="Support/Query split"),
    ]

    def format_thousands_as_k(x, pos):
        if abs(x) >= 1000:
            return f"{x / 1000:g}k"
        return f"{x:g}"

    for row_idx, dataset_name in enumerate(domains):
        domain_tasks = sampled[sampled["dataset_name"] == dataset_name].reset_index(drop=True)

        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]

            if col_idx >= len(domain_tasks):
                ax.axis("off")
                continue

            row = domain_tasks.iloc[col_idx]

            task_id = row["task_id"]
            n_support = int(row["n_support"])
            n_query = int(row["n_query"])

            df = load_full_task(dataset_name, task_id)

            support_df = df[df["split"] == "support"].copy()
            query_df = df[df["split"] == "query"].copy()

            x_support = support_df["relative_t"].to_numpy()
            y_support = support_df["y"].to_numpy()

            x_query = query_df["relative_t"].to_numpy()
            y_query = query_df["y"].to_numpy()

            if len(x_support) > 0:
                ax.plot(
                    x_support,
                    y_support,
                    color=SUPPORT_COLOR,
                    linewidth=1.25
                )

            if len(x_query) > 0:
                ax.plot(
                    x_query,
                    y_query,
                    color=QUERY_COLOR,
                    linewidth=1.25
                )

            split_x = n_support - 0.5

            ax.axvline(
                split_x,
                color=SPLIT_COLOR,
                linestyle="--",
                linewidth=0.9
            )

            ax.axvspan(
                -0.5,
                n_support - 0.5,
                alpha=0.08,
                color=SUPPORT_COLOR
            )

            ax.axvspan(
                n_support - 0.5,
                n_support + n_query - 0.5,
                alpha=0.08,
                color=QUERY_COLOR
            )

            if row_idx == 0:
                ax.set_title(
                    f"Sample task {col_idx + 1}",
                    fontsize=13,
                    pad=6
                )

            if col_idx == 0:
                ax.set_ylabel(dataset_name, fontsize=10)
            else:
                ax.set_ylabel("")

            if row_idx == n_domains - 1:
                ax.set_xlabel("Relative time", fontsize=11)
            else:
                ax.set_xlabel("")

            # BTC y-axis values as 21k instead of 21000
            if "btc" in dataset_name.lower():
                ax.yaxis.set_major_formatter(FuncFormatter(format_thousands_as_k))

            ax.grid(True, alpha=0.25)

            ax.tick_params(
                axis="both",
                labelsize=10
            )

            ax.text(
                0.01,
                0.93,
                f"$n_S$={n_support}, $n_Q$={n_query}",
                transform=ax.transAxes,
                fontsize=10,
                va="top"
            )

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.995),
        fontsize=12,
        frameon=False
    )

    plt.subplots_adjust(
        left=0.075,
        right=0.995,
        top=0.955,
        bottom=0.055,
        hspace=0.55,
        wspace=0.16
    )

    out_png = FIG_DIR / f"sample_{TASKS_PER_DOMAIN_TO_PLOT}_tasks_per_domain.png"
    out_pdf = FIG_DIR / f"sample_{TASKS_PER_DOMAIN_TO_PLOT}_tasks_per_domain.pdf"

    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.03)

    plt.show()

    print(f"[OK] Saved:\n{out_png}\n{out_pdf}")


# ============================================================
# MAIN
# ============================================================

def main():
    index_df = load_index(INDEX_FILE)

    print(f"[INFO] Loaded task index: {INDEX_FILE}")
    print(f"[INFO] Total tasks: {len(index_df):,}")
    print(f"[INFO] Domains: {index_df['dataset_name'].nunique()}")

    ### plot_sample_tasks_from_each_domain(index_df)
    plot_sample_tasks_from_each_domain2(index_df)



if __name__ == "__main__":
    main()