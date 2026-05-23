# Run:
# python STATISTICAL_RANKING.py
# It will create:
# FST_BENCHMARK_PREDICTIONS/
# └── STATISTICAL_RANKING/
#     ├── ranking_h_5_rmse.csv
#     ├── ranking_h_5_rmse.tex
#     ├── ranking_h_10_rmse.csv
#     ├── ranking_h_10_rmse.tex
#     ├── ranking_all_horizons_rmse.csv
#     ├── global_far_tests_rmse.csv
#     └── latex_tables_all_horizons_rmse.tex


from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata, chi2, norm


# ============================================================
# CONFIG
# ============================================================

PRED_DIR = Path("FST_BENCHMARK_PREDICTIONS")
METRICS_FILE = PRED_DIR / "metrics_by_task.csv"

OUT_DIR = PRED_DIR / "STATISTICAL_RANKING"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRIC = "rmse"
ALPHA = 0.05

# If True, uses table* instead of table.
# For only 5 baselines, False usually fits in one IEEE column.
USE_TABLE_STAR = False


# ============================================================
# HELPERS
# ============================================================

def latex_escape(text: str) -> str:
    """
    Escape special LaTeX characters in model names.
    """
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "$": r"\$",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }

    out = str(text)
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


def format_rank(x: float) -> str:
    return f"{x:.1f}"


def format_pvalue_cell(p: float) -> str:
    if pd.isna(p):
        return "--"

    if p < 1e-12:
        return r"$<10^{-12}$"

    if p < 1e-6:
        return rf"${p:.2e}$"

    return rf"${p:.6f}$"


def format_pvalue_note(p: float) -> str:
    if pd.isna(p):
        return r"$p$ unavailable"

    if p < 1e-12:
        return r"$p<10^{-12}$"

    if p < 1e-6:
        return rf"$p={p:.2e}$"

    return rf"$p={p:.6f}$"


def normalize_horizon_label(h) -> str:
    return str(h).replace(".", "_")


# ============================================================
# FINNER POST-HOC CORRECTION
# ============================================================

def finner_adjust_pvalues(raw_pvalues: np.ndarray) -> np.ndarray:
    """
    Finner step-up adjusted p-values.

    For ordered p-values p_(i), i=1,...,m:
        q_i = 1 - (1 - p_(i))^(m / i)

    Adjusted values are made monotone using:
        p_adj_(i) = min_{j >= i} q_j
    """
    raw_pvalues = np.asarray(raw_pvalues, dtype=float)

    if len(raw_pvalues) == 0:
        return np.array([])

    m = len(raw_pvalues)

    order = np.argsort(raw_pvalues)
    p_sorted = raw_pvalues[order]

    q_sorted = np.zeros(m, dtype=float)

    for i in range(m):
        rank_i = i + 1
        p_i = min(max(p_sorted[i], 0.0), 1.0)
        q_sorted[i] = 1.0 - (1.0 - p_i) ** (m / rank_i)

    # Step-up monotonicity
    adj_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    adj_sorted = np.clip(adj_sorted, 0.0, 1.0)

    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adj_sorted

    return adjusted


# ============================================================
# FRIEDMAN ALIGNED RANKS TEST
# ============================================================

def friedman_aligned_ranks(
    data: pd.DataFrame,
    metric: str,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Computes Friedman aligned-ranks statistic.

    Input:
        data columns:
            task_id, model_name, metric

    Output:
        ranking_df with method-wise aligned ranks
        global_info with FAR statistic and p-value
    """
    pivot = data.pivot_table(
        index="task_id",
        columns="model_name",
        values=metric,
        aggfunc="mean"
    )

    # Keep only complete task blocks
    before_tasks = len(pivot)
    pivot = pivot.dropna(axis=0, how="any")
    after_tasks = len(pivot)

    if after_tasks == 0:
        raise ValueError("No complete task blocks available for ranking.")

    models = list(pivot.columns)
    values = pivot.to_numpy(dtype=float)

    N, A = values.shape

    # Align by removing the task-wise mean error
    aligned = values - values.mean(axis=1, keepdims=True)

    # Rank all aligned errors jointly.
    # Lower aligned error is better, hence ascending rank.
    ranks_flat = rankdata(aligned.ravel(), method="average")
    ranks = ranks_flat.reshape(N, A)

    method_rank_sums = ranks.sum(axis=0)
    method_mean_ranks = method_rank_sums / N

    task_rank_sums = ranks.sum(axis=1)

    numerator = (A - 1) * (
        np.sum(method_rank_sums ** 2)
        - (A * (N ** 2) / 4.0) * ((A * N + 1) ** 2)
    )

    denominator = (
        (A * N * (A * N + 1) * (2 * A * N + 1)) / 6.0
        - (1.0 / A) * np.sum(task_rank_sums ** 2)
    )

    if denominator <= 0:
        raise ValueError("Invalid denominator in Friedman aligned-ranks statistic.")

    far_stat = numerator / denominator
    far_pvalue = chi2.sf(far_stat, df=A - 1)

    ranking_df = pd.DataFrame({
        "model_name": models,
        "rank_sum": method_rank_sums,
        "friedman_ranking": method_mean_ranks,
    })

    ranking_df = ranking_df.sort_values(
        "friedman_ranking",
        ascending=True
    ).reset_index(drop=True)

    global_info = {
        "n_tasks_original": before_tasks,
        "n_tasks_complete": after_tasks,
        "n_models": A,
        "far_statistic": float(far_stat),
        "far_df": int(A - 1),
        "far_pvalue": float(far_pvalue),
    }

    return ranking_df, global_info


# ============================================================
# POST-HOC COMPARISONS AGAINST CONTROL
# ============================================================

def add_finner_posthoc(
    ranking_df: pd.DataFrame,
    n_tasks: int,
    n_models: int,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Compares every model against the best-ranked control method.

    Uses aligned-rank post-hoc z statistic:
        z = (R_i - R_control) / sqrt(A*N*(A*N+1)/6) / N

    where R_i are mean aligned ranks.
    """
    out = ranking_df.copy()

    control_model = out.iloc[0]["model_name"]
    control_rank = float(out.iloc[0]["friedman_ranking"])

    denom_sum = np.sqrt(n_models * n_tasks * (n_models * n_tasks + 1) / 6.0)
    denom_mean = denom_sum / n_tasks

    raw_pvalues = []
    compared_indices = []

    out["control_model"] = control_model
    out["posthoc_z"] = np.nan
    out["finner_raw_pvalue"] = np.nan
    out["finner_adjusted_pvalue"] = np.nan
    out["h0_decision"] = "--"

    for idx in range(len(out)):
        model = out.loc[idx, "model_name"]

        if model == control_model:
            continue

        rank_i = float(out.loc[idx, "friedman_ranking"])
        z = (rank_i - control_rank) / denom_mean

        raw_p = 2.0 * norm.sf(abs(z))

        out.loc[idx, "posthoc_z"] = z
        out.loc[idx, "finner_raw_pvalue"] = raw_p

        raw_pvalues.append(raw_p)
        compared_indices.append(idx)

    adjusted = finner_adjust_pvalues(np.array(raw_pvalues, dtype=float))

    for idx, adj_p in zip(compared_indices, adjusted):
        out.loc[idx, "finner_adjusted_pvalue"] = adj_p
        out.loc[idx, "h0_decision"] = "Rejected" if adj_p < alpha else "Not rejected"

    return out


# ============================================================
# LATEX TABLE WRITER
# ============================================================

def make_latex_table(
    ranking_df: pd.DataFrame,
    global_info: Dict,
    horizon,
    metric: str,
    use_table_star: bool = False,
) -> str:
    env = "table*" if use_table_star else "table"

    h_label = normalize_horizon_label(horizon)

    caption = (
        f"Friedman aligned-ranks statistical comparison for horizon $h={horizon}$ "
        f"using {metric.upper()} as the task-level error metric. "
        f"Lower ranking values indicate better performance. "
        f"The best-ranked method is used as the control in the Finner post-hoc test."
    )

    label = f"tab:far_ranking_h_{h_label}_{metric.lower()}"

    lines = []

    lines.append(f"\\begin{{{env}}}[!t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\toprule")
    lines.append(
        "Model & "
        "\\begin{tabular}{c}Friedman\\\\Ranking\\end{tabular} & "
        "\\multicolumn{2}{c}{Finner post-hoc test} \\\\"
    )
    lines.append("\\cmidrule(lr){3-4}")
    lines.append(" & & $p$-value & $H_0$ \\\\")
    lines.append("\\midrule")

    for _, row in ranking_df.iterrows():
        model = latex_escape(row["model_name"])
        rank = format_rank(row["friedman_ranking"])

        if pd.isna(row["finner_adjusted_pvalue"]):
            pval = "--"
            decision = "--"
        else:
            pval = format_pvalue_cell(float(row["finner_adjusted_pvalue"]))
            decision = latex_escape(row["h0_decision"])

        lines.append(f"{model} & {rank} & {pval} & {decision} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    lines.append(
        "\\par\\vspace{1mm}\n"
        "\\begin{minipage}{0.95\\linewidth}\n"
        "\\footnotesize\n"
        "\\emph{Note.} Global Friedman aligned-ranks test: "
        f"$T={global_info['far_statistic']:.4f}$, "
        f"$df={global_info['far_df']}$, "
        f"{format_pvalue_note(global_info['far_pvalue'])}, "
        f"$N={global_info['n_tasks_complete']}$ complete task blocks.\n"
        "\\end{minipage}"
    )

    lines.append(f"\\end{{{env}}}")

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def run_statistical_ranking():
    if not METRICS_FILE.exists():
        raise FileNotFoundError(f"Missing metrics file: {METRICS_FILE}")

    metrics_df = pd.read_csv(METRICS_FILE)

    required = {"task_id", "dataset_name", "model_name", "horizon", METRIC}
    missing = required.difference(metrics_df.columns)

    if missing:
        raise ValueError(f"metrics_by_task.csv is missing columns: {missing}")

    metrics_df[METRIC] = pd.to_numeric(metrics_df[METRIC], errors="coerce")
    metrics_df = metrics_df.replace([np.inf, -np.inf], np.nan)
    metrics_df = metrics_df.dropna(subset=[METRIC])

    horizons = sorted(metrics_df["horizon"].unique())

    all_rankings = []
    all_global_tests = []
    all_latex_tables = []

    print(f"[INFO] Loaded metrics: {METRICS_FILE}")
    print(f"[INFO] Metric: {METRIC}")
    print(f"[INFO] Horizons found: {horizons}")
    print(f"[INFO] Output directory: {OUT_DIR}")

    for horizon in horizons:
        h_df = metrics_df[metrics_df["horizon"] == horizon].copy()

        ranking_df, global_info = friedman_aligned_ranks(
            data=h_df,
            metric=METRIC,
        )

        ranking_df = add_finner_posthoc(
            ranking_df=ranking_df,
            n_tasks=global_info["n_tasks_complete"],
            n_models=global_info["n_models"],
            alpha=ALPHA,
        )

        ranking_df.insert(0, "horizon", horizon)

        h_label = normalize_horizon_label(horizon)

        out_csv = OUT_DIR / f"ranking_h_{h_label}_{METRIC}.csv"
        ranking_df.to_csv(out_csv, index=False)

        latex_table = make_latex_table(
            ranking_df=ranking_df,
            global_info=global_info,
            horizon=horizon,
            metric=METRIC,
            use_table_star=USE_TABLE_STAR,
        )

        out_tex = OUT_DIR / f"ranking_h_{h_label}_{METRIC}.tex"
        out_tex.write_text(latex_table, encoding="utf-8")

        global_info_with_h = dict(global_info)
        global_info_with_h["horizon"] = horizon
        global_info_with_h["metric"] = METRIC

        all_rankings.append(ranking_df)
        all_global_tests.append(global_info_with_h)
        all_latex_tables.append(latex_table)

        print(f"\n[HORIZON h={horizon}]")
        print(f"  complete tasks: {global_info['n_tasks_complete']}")
        print(f"  models:         {global_info['n_models']}")
        print(f"  FAR statistic:  {global_info['far_statistic']:.6f}")
        print(f"  FAR p-value:    {global_info['far_pvalue']:.6g}")
        print(f"  saved:          {out_csv}")
        print(f"  saved:          {out_tex}")

        display_cols = [
            "model_name",
            "friedman_ranking",
            "finner_adjusted_pvalue",
            "h0_decision",
        ]
        print(ranking_df[display_cols].to_string(index=False))

    combined_rankings = pd.concat(all_rankings, axis=0, ignore_index=True)
    combined_rankings.to_csv(
        OUT_DIR / f"ranking_all_horizons_{METRIC}.csv",
        index=False
    )

    global_tests_df = pd.DataFrame(all_global_tests)
    global_tests_df.to_csv(
        OUT_DIR / f"global_far_tests_{METRIC}.csv",
        index=False
    )

    all_tables_path = OUT_DIR / f"latex_tables_all_horizons_{METRIC}.tex"
    all_tables_path.write_text("\n\n".join(all_latex_tables), encoding="utf-8")

    print("\n[OK] Statistical ranking completed.")
    print(f"[OK] Combined rankings: {OUT_DIR / f'ranking_all_horizons_{METRIC}.csv'}")
    print(f"[OK] Global FAR tests:  {OUT_DIR / f'global_far_tests_{METRIC}.csv'}")
    print(f"[OK] LaTeX tables:      {all_tables_path}")


if __name__ == "__main__":
    run_statistical_ranking()