"""
Phase 2 — Dataset Validation
==============================
Performs physical and numerical quality checks on the master dataset.
Identifies impossible values, numerical instability artefacts, and
class imbalance before any feature engineering is applied.

Input:   data/raw/greenmining_master_dataset.csv
Outputs: reports/summaries/dataset_validation.txt
         reports/plots/val_*.png  (6 figures)

Usage:
  python 02_validate_dataset.py
  python 02_validate_dataset.py --input data/raw/greenmining_master_dataset.csv
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# Matplotlib backend — use non-interactive Agg for headless environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ---------------------------------------------------------------------------
# Repository paths
# ---------------------------------------------------------------------------

REPO_ROOT   = Path(__file__).resolve().parent
PLOTS_DIR   = REPO_ROOT / "reports" / "plots"
SUMMARY_DIR = REPO_ROOT / "reports" / "summaries"

# ---------------------------------------------------------------------------
# Physical bounds — values outside these ranges are physically impossible
# in a mine environment and indicate numerical error or unit mismatch.
# ---------------------------------------------------------------------------

BOUNDS = {
    # (column, hard_min, hard_max, description)
    "CH4":         (0.0,    1.0,    "vol fraction [0,1]"),
    "CO":          (0.0,    1.0,    "vol fraction [0,1]"),
    "H2":          (0.0,    1.0,    "vol fraction [0,1]"),
    "Temperature": (200.0,  400.0,  "K — mine ambient 293K ± 100K"),
    "Velocity":    (0.0,    30.0,   "m/s — max 2x inlet, plus recirculation"),
    "Pressure":    (8e4,    1.5e5,  "Pa — atmospheric ± 20%"),
    "Time":        (0.0,    300.0,  "s — simulation window"),
}

# Gas LEL and TLV thresholds for annotation
THRESHOLDS = {
    "CH4": {
        "WARNING": 0.010,   "DANGER": 0.025,
        "LEL":     0.050,   "label": "CH4 vol fraction",
    },
    "CO":  {
        "WARNING": 0.0001,  "DANGER": 0.0005,
        "TLV":     0.000050, "label": "CO vol fraction",
    },
    "H2":  {
        "WARNING": 0.010,   "DANGER": 0.020,
        "LEL":     0.040,   "label": "H2 vol fraction",
    },
}

RISK_ORDER   = ["SAFE", "WARNING", "DANGER"]
RISK_COLOURS = {"SAFE": "#2ecc71", "WARNING": "#f39c12", "DANGER": "#e74c3c"}

SCENARIO_LABELS = {
    1: "S1 Normal\n(2.0 m/s)",
    2: "S2 Moderate\nLeak (1.5 m/s)",
    3: "S3 Vent\nFailure (0.5 m/s)",
    4: "S4 Dead-Zone\n(1.0 m/s)",
    5: "S5 Emergency\n(2.0 m/s)",
}


# ---------------------------------------------------------------------------
# Text report helper
# ---------------------------------------------------------------------------

class Report:
    """Collects validation findings into a structured text report."""

    def __init__(self):
        self._lines = []
        self._issues = []

    def h1(self, text: str) -> None:
        self._lines += ["", "=" * 64, f"  {text}", "=" * 64]

    def h2(self, text: str) -> None:
        self._lines += ["", f"  --- {text} ---"]

    def line(self, text: str = "") -> None:
        self._lines.append(f"  {text}" if text else "")

    def issue(self, severity: str, text: str) -> None:
        tag = f"[{severity.upper()}]"
        self._lines.append(f"  {tag:10s} {text}")
        self._issues.append((severity, text))

    def ok(self, text: str) -> None:
        self._lines.append(f"  [OK]       {text}")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        content = "\n".join(self._lines)
        path.write_text(content)
        print(f"  Report → {path.relative_to(REPO_ROOT)}")

    def print_issues_summary(self) -> None:
        if not self._issues:
            print("\n  All checks passed — no issues found.")
            return
        print(f"\n  Issues found: {len(self._issues)}")
        for sev, msg in self._issues:
            print(f"    [{sev.upper():6s}] {msg}")

    def to_string(self) -> str:
        return "\n".join(self._lines)


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------

def check_missing(df: pd.DataFrame, report: Report) -> None:
    report.h2("Missing Values")
    missing = df.isnull().sum()
    total_missing = missing.sum()
    if total_missing == 0:
        report.ok("No missing values in any column")
    else:
        for col, n in missing[missing > 0].items():
            report.issue("ERROR", f"{col}: {n:,} missing values ({100*n/len(df):.2f}%)")


def check_duplicates(df: pd.DataFrame, report: Report) -> None:
    report.h2("Duplicate Rows")
    n_dup = df.duplicated().sum()
    if n_dup == 0:
        report.ok("No duplicate rows")
    else:
        report.issue("WARN", f"{n_dup:,} exact duplicate rows remain (run 01_merge_datasets.py again)")


def check_inf_nan(df: pd.DataFrame, report: Report) -> None:
    report.h2("NaN / Inf Values")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    found_any = False
    for col in numeric_cols:
        n_inf = np.isinf(df[col]).sum()
        n_nan = df[col].isna().sum()
        if n_inf > 0:
            report.issue("ERROR", f"{col}: {n_inf:,} Inf values — numerical divergence in CFD")
            found_any = True
        if n_nan > 0:
            report.issue("ERROR", f"{col}: {n_nan:,} NaN values")
            found_any = True
    if not found_any:
        report.ok("No NaN or Inf values in numeric columns")


def check_negative_concentrations(df: pd.DataFrame, report: Report) -> None:
    report.h2("Negative Concentrations (post-clamp check)")
    for col in ["CH4", "CO", "H2"]:
        if col not in df.columns:
            continue
        n_neg = (df[col] < 0).sum()
        if n_neg == 0:
            report.ok(f"{col}: all values ≥ 0.0")
        else:
            report.issue("WARN", f"{col}: {n_neg:,} negative values still present — re-run Phase 1")


def check_physical_bounds(df: pd.DataFrame, report: Report) -> None:
    report.h2("Physical Bounds Check")
    for col, (lo, hi, desc) in BOUNDS.items():
        if col not in df.columns:
            continue
        actual_min = df[col].min()
        actual_max = df[col].max()
        out_lo = (df[col] < lo).sum()
        out_hi = (df[col] > hi).sum()
        if out_lo == 0 and out_hi == 0:
            report.ok(f"{col:12s} [{actual_min:.3e}, {actual_max:.3e}]  within [{lo:.3e}, {hi:.3e}] ({desc})")
        else:
            if out_lo > 0:
                report.issue("WARN", f"{col}: {out_lo:,} values below physical minimum {lo} ({desc})")
            if out_hi > 0:
                report.issue("WARN", f"{col}: {out_hi:,} values above physical maximum {hi} ({desc})")


def check_class_distribution(df: pd.DataFrame, report: Report) -> None:
    report.h2("Risk Class Distribution")
    counts = df["Risk"].value_counts()
    total = len(df)
    for cls in RISK_ORDER:
        n = counts.get(cls, 0)
        pct = 100 * n / total
        report.line(f"  {cls:8s}: {n:>8,}  ({pct:5.1f}%)")

    # Imbalance diagnosis
    safe_pct    = 100 * counts.get("SAFE",    0) / total
    danger_pct  = 100 * counts.get("DANGER",  0) / total
    warning_pct = 100 * counts.get("WARNING", 0) / total

    if safe_pct > 90:
        report.issue(
            "WARN",
            f"Severe class imbalance: SAFE={safe_pct:.1f}% — "
            "use class_weight='balanced' or SMOTE in ML models"
        )
    if danger_pct < 5:
        report.issue(
            "WARN",
            f"DANGER class is rare ({danger_pct:.1f}%) — "
            "prioritise DANGER recall in evaluation (see Phase 7)"
        )
    if warning_pct < 5:
        report.issue(
            "WARN",
            f"WARNING class is rare ({warning_pct:.1f}%) — "
            "WARNING recall is critical for early hazard detection"
        )
    report.line()
    report.line("  Note: class imbalance is PHYSICALLY EXPECTED (most of the mine")
    report.line("  is safe at any given time). Do NOT upsample to 50/50 — it would")
    report.line("  introduce synthetic physics that do not exist in the simulation.")


def check_scenario_distribution(df: pd.DataFrame, report: Report) -> None:
    report.h2("Rows per Scenario")
    counts = df.groupby("Scenario").size()
    for sid, n in counts.items():
        label = SCENARIO_LABELS.get(sid, f"Scenario {sid}")
        report.line(f"  S{sid} ({label.split(chr(10))[0].strip():30s}): {n:>8,} rows")

    # Verify uniform temporal coverage (all scenarios should have same cell × timestep count)
    if counts.std() / counts.mean() > 0.1:
        report.issue(
            "WARN",
            "Scenario row counts differ >10% — some scenarios may have fewer time steps"
        )
    else:
        report.ok("Row counts are consistent across scenarios")


def full_statistics(df: pd.DataFrame, report: Report) -> None:
    report.h2("Descriptive Statistics — Concentration Fields")
    numeric_cols = ["CH4", "CO", "H2", "Velocity", "Temperature", "Pressure"]
    available = [c for c in numeric_cols if c in df.columns]
    stats = df[available].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99])
    for col in available:
        col_stats = stats[col]
        report.line(
            f"  {col:12s} | mean={col_stats['mean']:.3e}  std={col_stats['std']:.3e}"
            f"  p1={col_stats['1%']:.3e}  p99={col_stats['99%']:.3e}"
            f"  max={col_stats['max']:.3e}"
        )


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _save_fig(fig: plt.Figure, name: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / f"val_{name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved → reports/plots/val_{name}.png")


def plot_concentration_histograms(df: pd.DataFrame) -> None:
    """One figure per gas — log-scale y axis, threshold vlines."""
    for gas, meta in THRESHOLDS.items():
        if gas not in df.columns:
            continue

        data = df[gas]
        # Drop exact zeros for log scale (they dominate for SAFE cells)
        nonzero = data[data > 0]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f"{gas} Concentration Distribution", fontsize=13, fontweight="bold")

        # Left: full range including zeros
        ax = axes[0]
        ax.hist(data, bins=80, color="#3498db", edgecolor="none", alpha=0.8)
        ax.set_xlabel(meta["label"])
        ax.set_ylabel("Cell count")
        ax.set_title("All values (linear scale)")
        for level_name in ["WARNING", "DANGER"]:
            thresh = meta.get(level_name)
            if thresh:
                ax.axvline(thresh, color=RISK_COLOURS[level_name], linewidth=1.5,
                           linestyle="--", label=f"{level_name} ({thresh:.4f})")
        if "LEL" in meta:
            ax.axvline(meta["LEL"], color="#8e44ad", linewidth=1.5,
                       linestyle=":", label=f"LEL ({meta['LEL']:.3f})")
        if "TLV" in meta:
            ax.axvline(meta["TLV"], color="#8e44ad", linewidth=1.5,
                       linestyle=":", label=f"TLV ({meta['TLV']:.5f})")
        ax.legend(fontsize=8)

        # Right: non-zero only on log scale
        ax = axes[1]
        if len(nonzero) > 0:
            ax.hist(nonzero, bins=80, color="#e67e22", edgecolor="none", alpha=0.8)
            ax.set_yscale("log")
        ax.set_xlabel(meta["label"])
        ax.set_title("Non-zero values (log y-scale)")
        ax.yaxis.set_major_formatter(ticker.LogFormatter())

        fig.tight_layout()
        _save_fig(fig, f"hist_{gas.lower()}")


def plot_velocity_histogram(df: pd.DataFrame) -> None:
    if "Velocity" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    data = df["Velocity"]
    ax.hist(data, bins=100, color="#2980b9", edgecolor="none", alpha=0.85)
    ax.axvline(data.mean(), color="red",    linewidth=1.5, linestyle="--", label=f"Mean {data.mean():.2f} m/s")
    ax.axvline(data.median(), color="orange", linewidth=1.5, linestyle="-.", label=f"Median {data.median():.2f} m/s")
    ax.set_xlabel("Velocity magnitude [m/s]")
    ax.set_ylabel("Cell count")
    ax.set_title("Velocity Magnitude Distribution")
    ax.legend()
    fig.tight_layout()
    _save_fig(fig, "hist_velocity")


def plot_risk_class_barchart(df: pd.DataFrame) -> None:
    counts = df["Risk"].value_counts().reindex(RISK_ORDER, fill_value=0)
    total  = len(df)

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(
        RISK_ORDER,
        counts.values,
        color=[RISK_COLOURS[r] for r in RISK_ORDER],
        edgecolor="white",
        width=0.55,
    )
    for bar, n in zip(bars, counts.values):
        pct = 100 * n / total
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.005,
            f"{n:,}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=10,
        )
    ax.set_ylabel("Number of cells × timesteps")
    ax.set_title("Risk Class Distribution — Master Dataset")
    ax.set_ylim(0, counts.max() * 1.18)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.tight_layout()
    _save_fig(fig, "risk_class_bar")


def plot_scenario_distribution(df: pd.DataFrame) -> None:
    counts = df.groupby("Scenario").size()
    labels = [SCENARIO_LABELS.get(int(s), f"S{s}") for s in counts.index]
    colors = ["#3498db", "#27ae60", "#e67e22", "#8e44ad", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(range(len(counts)), counts.values,
                  color=colors[:len(counts)], edgecolor="white", width=0.6)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(labels, fontsize=9)
    for bar, n in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + len(df) * 0.003,
                f"{n:,}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Rows (cells × timesteps)")
    ax.set_title("Data Volume per Scenario")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.tight_layout()
    _save_fig(fig, "scenario_distribution")


def plot_concentration_by_risk(df: pd.DataFrame) -> None:
    """Box plots of CH4 and CO per risk class — shows threshold alignment."""
    gases = [g for g in ["CH4", "CO", "H2"] if g in df.columns]
    if not gases:
        return

    fig, axes = plt.subplots(1, len(gases), figsize=(5 * len(gases), 5))
    if len(gases) == 1:
        axes = [axes]

    for ax, gas in zip(axes, gases):
        risk_data = [
            df.loc[df["Risk"].astype(str) == cls, gas].values
            for cls in RISK_ORDER
        ]
        bp = ax.boxplot(
            risk_data, labels=RISK_ORDER, patch_artist=True,
            medianprops={"color": "black", "linewidth": 2},
            flierprops={"marker": ".", "markersize": 2, "alpha": 0.3},
        )
        for patch, cls in zip(bp["boxes"], RISK_ORDER):
            patch.set_facecolor(RISK_COLOURS[cls])
            patch.set_alpha(0.7)

        meta = THRESHOLDS.get(gas, {})
        for level in ["WARNING", "DANGER"]:
            thresh = meta.get(level)
            if thresh:
                ax.axhline(thresh, linestyle="--", linewidth=1.2,
                           color=RISK_COLOURS[level], label=f"{level} ({thresh:.4f})")
        ax.set_ylabel(f"{gas} vol fraction")
        ax.set_title(f"{gas} Distribution by Risk Class")
        ax.legend(fontsize=8)

    fig.tight_layout()
    _save_fig(fig, "concentration_by_risk")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(input_path: Path, report_path: Path) -> None:
    print("=" * 64)
    print("  GreenMining — Phase 2: Dataset Validation")
    print("=" * 64)

    # Load
    if not input_path.exists():
        print(f"\n  ERROR: {input_path} not found.")
        print("  Run 01_merge_datasets.py first.")
        sys.exit(1)

    print(f"\n  Loading {input_path.relative_to(REPO_ROOT)} ...")
    df = pd.read_csv(input_path)
    print(f"  Rows: {len(df):,}   Columns: {len(df.columns)}")

    # Restore Risk as Categorical
    if "Risk" in df.columns:
        df["Risk"] = pd.Categorical(df["Risk"], categories=RISK_ORDER, ordered=True)

    report = Report()
    report.h1("GreenMining — Dataset Validation Report")
    report.line(f"Input: {input_path.relative_to(REPO_ROOT)}")
    report.line(f"Rows: {len(df):,}   Columns: {len(df.columns)}")
    report.line(f"Columns: {list(df.columns)}")

    # Run all checks
    print("\n  Running physical and numerical checks ...")

    check_missing(df, report)
    check_duplicates(df, report)
    check_inf_nan(df, report)
    check_negative_concentrations(df, report)
    check_physical_bounds(df, report)
    check_class_distribution(df, report)
    check_scenario_distribution(df, report)
    full_statistics(df, report)

    # Plots
    print("\n  Generating validation plots ...")
    plot_concentration_histograms(df)
    plot_velocity_histogram(df)
    plot_risk_class_barchart(df)
    plot_scenario_distribution(df)
    plot_concentration_by_risk(df)

    # Save report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.save(report_path)

    # Print issues to stdout
    report.print_issues_summary()

    print(f"\n  Validation complete.")
    print(f"  Summary : {report_path.relative_to(REPO_ROOT)}")
    print(f"  Plots   : reports/plots/val_*.png")
    print("\n  Run 03_feature_engineering.py next.\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: Validate master dataset.")
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
