from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path("CURATED_TIME_SERIES_DATASETS")
FIG_DIR = DATA_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

STANDARDIZE = False      # True for z-score comparison, False for raw values
MAX_POINTS = 5000        # downsampling for clean figures

# Flat paper-like layout
FIGSIZE = (16, 8.2)

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


YLABELS = {
    "Temperature": "Temperature (°C)",
    "Wind speed": "Wind speed (m/s)",
    "Solar power": "Power (MW)",
    "Natural gas": "Value",
    "Gold prices": "Open price",
    "ETD-OT": "Oil temperature",
    "BTC": "Open price",
    "Electricity consumption": "Consumption",
    "Exchange rate": "Exchange rate",
    "Traffic": "Occupancy rate",
}

# ============================================================
# HELPERS
# ============================================================

def find_file(data_dir: Path, filename: str) -> Path:
    p = data_dir / filename
    if p.exists():
        return p

    target_stem = Path(filename).stem.lower()
    for q in data_dir.glob("*.csv"):
        if q.stem.lower() == target_stem:
            return q

    raise FileNotFoundError(f"Could not find {filename}")


def load_plot_df(path: Path, title: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "ds" not in df.columns or "y" not in df.columns:
        raise ValueError(f"{path.name}: expected columns ['ds', 'y'].")

    df = df[["ds", "y"]].copy()
    df["y"] = pd.to_numeric(df["y"], errors="coerce")

    # Keep NaNs in y for visualization:
    # they are intentional gap-breakers in temperature/wind viz CSVs

    # Special x-axis handling for benchmark datasets with integer ds
    if title == "Exchange rate":
        df["ds_plot"] = pd.date_range(
            start="1990-01-01",
            end="2016-12-31",
            periods=len(df)
        )
        df["x_is_datetime"] = True
        return df

    if title == "Traffic":
        df["ds_plot"] = pd.date_range(
            start="2015-01-01 00:00:00",
            periods=len(df),
            freq="1h"
        )
        df["x_is_datetime"] = True
        return df

    # General case
    ds_dt = pd.to_datetime(df["ds"], errors="coerce")
    if ds_dt.notna().mean() > 0.8:
        df["ds_plot"] = ds_dt
        df["x_is_datetime"] = True
        return df

    ds_num = pd.to_numeric(df["ds"], errors="coerce")
    if ds_num.notna().mean() > 0.8:
        df["ds_plot"] = ds_num
        df["x_is_datetime"] = False
        return df

    df["ds_plot"] = np.arange(len(df))
    df["x_is_datetime"] = False
    return df


def format_datetime_axis(ax):
    locator = mdates.AutoDateLocator(minticks=4, maxticks=6)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=25, labelsize=8)


def style_axis(ax, title: str, df: pd.DataFrame):
    ax.plot(df["ds_plot"], df["y"], linewidth=0.8)

    ax.set_title(title, fontsize=11, pad=6)
    ax.set_ylabel(YLABELS.get(title, "y"), fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="y", labelsize=8)

    n_valid = df["y"].notna().sum()
    ax.text(
        0.01,
        0.93,
        f"n={n_valid:,}",
        transform=ax.transAxes,
        fontsize=8,
        va="top"
    )

    if bool(df["x_is_datetime"].iloc[0]):
        format_datetime_axis(ax)
    else:
        ax.ticklabel_format(style="plain", axis="x", useOffset=False)
        ax.tick_params(axis="x", labelsize=8, rotation=0)


# ============================================================
# LOAD DATA
# ============================================================

loaded = []
for filename, title in DATASETS:
    path = find_file(DATA_DIR, filename)
    df = load_plot_df(path, title)
    loaded.append((title, df))


# ============================================================
# COMBINED FLAT FIGURE
# ============================================================

fig, axes = plt.subplots(5, 2, figsize=FIGSIZE)
axes = axes.flatten()

for ax, (title, df) in zip(axes, loaded):
    style_axis(ax, title, df)

for i in range(len(loaded), len(axes)):
    axes[i].axis("off")



plt.subplots_adjust(
    left=0.055,
    right=0.995,
    top=0.92,
    bottom=0.07,
    hspace=0.95,
    wspace=0.18
)

out_png = FIG_DIR / "datasets_overview.png"

fig.savefig(out_png, dpi=600, bbox_inches="tight", pad_inches=0.03)

plt.show()

