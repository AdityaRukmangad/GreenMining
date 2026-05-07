"""
Phase 2 — Dataset Validation
============================
Performs physical and numerical validation on the merged CFD dataset.

Input:
    data/raw/greenmining_master_dataset.csv

Outputs:
    reports/summaries/dataset_validation.txt
    reports/plots/val_*.png

Usage:
    python 02_validate_dataset.py
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================
# Matplotlib setup
# ============================================================================

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ============================================================================
# Repository paths
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parent

PLOTS_DIR = REPO_ROOT / "reports" / "plots"
SUMMARY_DIR = REPO_ROOT / "reports" / "summaries"

# ============================================================================
# Physical bounds
# ============================================================================

BOUNDS = {
    "CH4":         (0.0,   1.0,    "vol fraction"),
    "CO":          (0.0,   1.0,    "vol fraction"),
    "H2":          (0.0,   1.0,    "vol fraction"),
    "Temperature": (200.0, 400.0,  "K"),
    "Velocity":    (0.0,   30.0,   "m/s"),
    "Pressure":    (8e4,   1.5e5,  "Pa"),
    "Time":        (0.0,   300.0,  "s"),
}

THRESHOLDS = {
    "CH4": {
        "WARNING": 0.010,
        "DANGER": 0.025,
        "LEL": 0.050,
    },
    "CO": {
        "WARNING": 0.0001,
        "DANGER": 0.0005,
        "TLV": 0.00005,
    },
    "H2": {
        "WARNING": 0.010,
        "DANGER": 0.020,
        "LEL": 0.040,
    },
}

RISK_ORDER = ["SAFE", "WARNING", "DANGER"]

RISK_COLOURS = {
    "SAFE": "#2ecc71",
    "WARNING": "#f39c12",
    "DANGER": "#e74c3c",
}

# ============================================================================
# Report helper
# ============================================================================

class Report:

    def __init__(self):
        self.lines = []
        self.issues = []

    def h1(self, text):
        self.lines += ["", "=" * 72, f"  {text}", "=" * 72]

    def h2(self, text):
        self.lines += ["", f"  --- {text} ---"]

    def line(self, text=""):
        self.lines.append(f"  {text}" if text else "")

    def ok(self, text):
        self.lines.append(f"  [OK]       {text}")

    def issue(self, severity, text):
        tag = severity.upper()
        self.lines.append(f"  [{tag:8s}] {text}")
        self.issues.append((severity, text))

    def save(self, path: Path):

        path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text("\n".join(self.lines), encoding="utf-8")

        print(f"  Report → {path.relative_to(REPO_ROOT)}")

    def print_summary(self):

        if not self.issues:
            print("\n  All validation checks passed.")
            return

        print(f"\n  Issues found: {len(self.issues)}")

        for sev, msg in self.issues:
            print(f"    [{sev.upper():6s}] {msg}")


# ============================================================================
# Validation checks
# ============================================================================

def check_missing(df, report):

    report.h2("Missing Values")

    missing = df.isna().sum()

    total = missing.sum()

    if total == 0:
        report.ok("No missing values")
        return

    for col, n in missing[missing > 0].items():
        pct = 100 * n / len(df)

        report.issue(
            "ERROR",
            f"{col}: {n:,} missing values ({pct:.2f}%)"
        )


def check_duplicates(df, report):

    report.h2("Duplicate Rows")

    n_dup = df.duplicated().sum()

    if n_dup == 0:
        report.ok("No duplicate rows")
    else:
        report.issue(
            "WARN",
            f"{n_dup:,} duplicate rows remain"
        )


def check_inf_nan(df, report):

    report.h2("Inf / NaN Values")

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    found = False

    for col in numeric_cols:

        n_inf = np.isinf(df[col]).sum()
        n_nan = df[col].isna().sum()

        if n_inf > 0:
            found = True
            report.issue(
                "ERROR",
                f"{col}: {n_inf:,} Inf values"
            )

        if n_nan > 0:
            found = True
            report.issue(
                "ERROR",
                f"{col}: {n_nan:,} NaN values"
            )

    if not found:
        report.ok("No Inf or NaN values")


def check_negative_concentrations(df, report):

    report.h2("Negative Concentrations")

    for col in ["CH4", "CO", "H2"]:

        if col not in df.columns:
            continue

        n_neg = (df[col] < 0).sum()

        if n_neg == 0:
            report.ok(f"{col}: all values >= 0")
        else:
            report.issue(
                "WARN",
                f"{col}: {n_neg:,} negative values remain"
            )


def check_physical_bounds(df, report):

    report.h2("Physical Bounds")

    for col, (lo, hi, desc) in BOUNDS.items():

        if col not in df.columns:
            continue

        below = (df[col] < lo).sum()
        above = (df[col] > hi).sum()

        actual_min = df[col].min()
        actual_max = df[col].max()

        if below == 0 and above == 0:

            report.ok(
                f"{col:<12} [{actual_min:.3e}, {actual_max:.3e}]"
            )

        else:

            if below > 0:
                report.issue(
                    "WARN",
                    f"{col}: {below:,} values below {lo} ({desc})"
                )

            if above > 0:
                report.issue(
                    "WARN",
                    f"{col}: {above:,} values above {hi} ({desc})"
                )


def check_class_distribution(df, report):

    report.h2("Risk Class Distribution")

    counts = (
        df["Risk"]
        .value_counts()
        .reindex(RISK_ORDER, fill_value=0)
    )

    total = len(df)

    for cls, n in counts.items():

        pct = 100 * n / total

        report.line(
            f"{cls:<8}: {n:>10,} ({pct:5.1f}%)"
        )

    safe_pct = 100 * counts["SAFE"] / total
    danger_pct = 100 * counts["DANGER"] / total

    if safe_pct > 90:
        report.issue(
            "WARN",
            f"Severe SAFE dominance ({safe_pct:.1f}%)"
        )

    if danger_pct < 5:
        report.issue(
            "WARN",
            f"DANGER underrepresented ({danger_pct:.1f}%)"
        )


def check_scenario_distribution(df, report):

    report.h2("Scenario Distribution")

    counts = df.groupby("Scenario").size()

    for sid, n in counts.items():
        report.line(f"Scenario {sid}: {n:,} rows")

    if counts.std() / counts.mean() > 0.1:
        report.issue(
            "WARN",
            "Scenario row counts vary >10%"
        )
    else:
        report.ok("Scenario row counts consistent")


def check_temporal_coverage(df, report):

    report.h2("Temporal Coverage")

    if "Time" not in df.columns:
        return

    times = np.sort(df["Time"].unique())

    report.line(f"Unique timesteps: {len(times)}")
    report.line(f"Time range      : {times.min()} → {times.max()}")

    dt = np.diff(times)

    if len(dt) > 0:

        if np.std(dt) > 1e-6:
            report.issue(
                "WARN",
                "Non-uniform timestep spacing detected"
            )
        else:
            report.ok(
                f"Uniform timestep spacing ({dt[0]:.2f} s)"
            )


def descriptive_statistics(df, report):

    report.h2("Descriptive Statistics")

    cols = [
        "CH4",
        "CO",
        "H2",
        "Velocity",
        "Temperature",
        "Pressure",
    ]

    cols = [c for c in cols if c in df.columns]

    stats = df[cols].describe(
        percentiles=[0.01, 0.5, 0.99]
    )

    for col in cols:

        s = stats[col]

        report.line(
            f"{col:<12} "
            f"mean={s['mean']:.3e} "
            f"std={s['std']:.3e} "
            f"p99={s['99%']:.3e} "
            f"max={s['max']:.3e}"
        )


def check_correlation(df, report):

    report.h2("Correlation Analysis")

    cols = [
        "CH4",
        "CO",
        "H2",
        "Velocity",
        "Temperature",
        "Pressure",
    ]

    cols = [c for c in cols if c in df.columns]

    corr = df[cols].corr(numeric_only=True)

    strongest = []

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):

            c = corr.iloc[i, j]

            strongest.append((abs(c), cols[i], cols[j], c))

    strongest.sort(reverse=True)

    report.line("Strongest correlations:")

    for _, a, b, c in strongest[:5]:
        report.line(f"{a} ↔ {b}: {c:.3f}")

# ============================================================================
# Plotting helpers
# ============================================================================

def _save_fig(fig, name):

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    path = PLOTS_DIR / f"val_{name}.png"

    fig.savefig(path, dpi=120, bbox_inches="tight")

    plt.close(fig)

    print(f"    Saved → reports/plots/val_{name}.png")


# ============================================================================
# Plots
# ============================================================================

def plot_concentration_histograms(df):

    for gas in ["CH4", "CO", "H2"]:

        if gas not in df.columns:
            continue

        fig, ax = plt.subplots(figsize=(8, 4))

        data = df[gas]

        ax.hist(
            data,
            bins=100,
            color="#3498db",
            alpha=0.85,
            edgecolor="none",
        )

        meta = THRESHOLDS.get(gas, {})

        for level in ["WARNING", "DANGER"]:

            thresh = meta.get(level)

            if thresh:
                ax.axvline(
                    thresh,
                    linestyle="--",
                    linewidth=1.5,
                    color=RISK_COLOURS[level],
                    label=f"{level} ({thresh:.4f})"
                )

        ax.set_title(f"{gas} Distribution")
        ax.set_xlabel("Volume fraction")
        ax.set_ylabel("Cell count")

        ax.legend()

        fig.tight_layout()

        _save_fig(fig, f"hist_{gas.lower()}")


def plot_velocity_histogram(df):

    if "Velocity" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 4))

    data = df["Velocity"]

    ax.hist(
        data,
        bins=100,
        color="#2980b9",
        alpha=0.85,
        edgecolor="none",
    )

    ax.axvline(
        data.mean(),
        color="red",
        linestyle="--",
        label=f"Mean {data.mean():.2f}"
    )

    ax.set_title("Velocity Distribution")
    ax.set_xlabel("Velocity [m/s]")
    ax.set_ylabel("Count")

    ax.legend()

    fig.tight_layout()

    _save_fig(fig, "hist_velocity")


def plot_risk_distribution(df):

    counts = (
        df["Risk"]
        .value_counts()
        .reindex(RISK_ORDER, fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(7, 4))

    bars = ax.bar(
        RISK_ORDER,
        counts.values,
        color=[RISK_COLOURS[r] for r in RISK_ORDER]
    )

    total = len(df)

    for bar, n in zip(bars, counts.values):

        pct = 100 * n / total

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{pct:.1f}%",
            ha="center",
            va="bottom"
        )

    ax.set_title("Risk Distribution")

    fig.tight_layout()

    _save_fig(fig, "risk_distribution")


def plot_scenario_distribution(df):

    counts = df.groupby("Scenario").size()

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(
        counts.index.astype(str),
        counts.values,
        color="#8e44ad"
    )

    ax.set_title("Rows per Scenario")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Rows")

    fig.tight_layout()

    _save_fig(fig, "scenario_distribution")


def plot_temporal_evolution(df):

    if "Time" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(9, 4))

    grouped = df.groupby("Time")["CH4"].mean()

    ax.plot(
        grouped.index,
        grouped.values,
        linewidth=2
    )

    ax.set_title("Mean CH4 Evolution")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Mean CH4")

    fig.tight_layout()

    _save_fig(fig, "temporal_ch4")


def plot_correlation_matrix(df):

    cols = [
        "CH4",
        "CO",
        "H2",
        "Velocity",
        "Temperature",
        "Pressure",
    ]

    cols = [c for c in cols if c in df.columns]

    corr = df[cols].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(7, 6))

    im = ax.imshow(corr, aspect="auto")

    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))

    ax.set_xticklabels(cols, rotation=45)
    ax.set_yticklabels(cols)

    fig.colorbar(im)

    ax.set_title("Correlation Matrix")

    fig.tight_layout()

    _save_fig(fig, "correlation_matrix")

# ============================================================================
# Main
# ============================================================================

def main(input_path: Path, report_path: Path):

    print("=" * 72)
    print("  GreenMining — Phase 2: Dataset Validation")
    print("=" * 72)

    if not input_path.exists():

        print(f"\nERROR: {input_path} not found")
        print("Run 01_merge_datasets.py first")

        sys.exit(1)

    print(f"\nLoading dataset:")
    print(f"  {input_path.relative_to(REPO_ROOT)}")

    df = pd.read_csv(input_path)

    print(f"\nRows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")

    if "Risk" in df.columns:
        df["Risk"] = pd.Categorical(
            df["Risk"],
            categories=RISK_ORDER,
            ordered=True
        )

    report = Report()

    report.h1("GreenMining Dataset Validation")

    report.line(f"Rows    : {len(df):,}")
    report.line(f"Columns : {len(df.columns)}")

    print("\nRunning validation checks ...")

    check_missing(df, report)
    check_duplicates(df, report)
    check_inf_nan(df, report)
    check_negative_concentrations(df, report)
    check_physical_bounds(df, report)
    check_class_distribution(df, report)
    check_scenario_distribution(df, report)
    check_temporal_coverage(df, report)
    descriptive_statistics(df, report)
    check_correlation(df, report)

    print("\nGenerating validation plots ...")

    plot_concentration_histograms(df)
    plot_velocity_histogram(df)
    plot_risk_distribution(df)
    plot_scenario_distribution(df)
    plot_temporal_evolution(df)
    plot_correlation_matrix(df)

    report.save(report_path)

    report.print_summary()

    print("\nValidation complete.")
    print("Next step: python 03_feature_engineering.py\n")


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Phase 2 — Dataset Validation"
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "greenmining_master_dataset.csv",
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "reports" / "summaries" / "dataset_validation.txt",
    )

    args = parser.parse_args()

    main(args.input, args.report)