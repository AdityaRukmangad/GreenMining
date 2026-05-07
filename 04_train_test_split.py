"""
Phase 4 — Train / Test Split
=============================
Creates physically meaningful ML splits.

Optimized for:
- large datasets (~3M+ rows)
- low memory overhead
- reproducible ML workflows
"""

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ============================================================================
# Configuration
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parent

RISK_ORDER = ["SAFE", "WARNING", "DANGER"]

RISK_COLOURS = {
    "SAFE": "#2ecc71",
    "WARNING": "#f39c12",
    "DANGER": "#e74c3c",
}

TRAIN_SCENARIOS = [1, 2, 3, 4]
TEST_SCENARIOS  = [5]

TEMPORAL_CUTOFF = 225.0

SPATIAL_DEAD_END_COL = "in_chamber"

VALIDATION_FRAC = 0.15

FLOAT_DTYPE = "float32"
INT_DTYPE   = "int32"


# ============================================================================
# Memory optimization
# ============================================================================

def optimize_memory(df):

    float_cols = df.select_dtypes(include=["float64"]).columns
    int_cols = df.select_dtypes(include=["int64"]).columns

    if len(float_cols):
        df[float_cols] = df[float_cols].astype(FLOAT_DTYPE)

    if len(int_cols):
        df[int_cols] = df[int_cols].astype(INT_DTYPE)

    return df


# ============================================================================
# Leakage verification
# ============================================================================

def hash_rows(df, cols):

    vals = (
        df[cols]
        .astype(str)
        .agg("|".join, axis=1)
    )

    return pd.util.hash_pandas_object(vals, index=False)


def verify_no_leakage(train, test, split_name):

    key_cols = [
        c for c in
        ["Scenario", "Time", "x", "y", "z"]
        if c in train.columns
    ]

    train_hash = set(hash_rows(train, key_cols))
    test_hash  = set(hash_rows(test, key_cols))

    overlap = train_hash & test_hash

    if overlap:

        print(
            f"  [ERROR] {split_name}: "
            f"{len(overlap):,} overlapping rows"
        )

    else:

        print(
            f"  [{split_name}] "
            f"No leakage detected ✓"
        )


# ============================================================================
# Distribution summary
# ============================================================================

def distribution_summary(df, label):

    lines = [
        f"  {label}: {len(df):,} rows"
    ]

    if "Risk" in df.columns:

        counts = (
            df["Risk"]
            .astype(str)
            .value_counts()
        )

        total = len(df)

        for cls in RISK_ORDER:

            n = counts.get(cls, 0)

            pct = 100 * n / total

            lines.append(
                f"    {cls:<8}: "
                f"{n:>10,} ({pct:5.1f}%)"
            )

    return lines


# ============================================================================
# Split functions
# ============================================================================

def scenario_split(df):

    train = df[
        df["Scenario"].isin(TRAIN_SCENARIOS)
    ]

    test = df[
        df["Scenario"].isin(TEST_SCENARIOS)
    ]

    return train, test


def temporal_split(df):

    train = df[df["Time"] < TEMPORAL_CUTOFF]

    test = df[df["Time"] >= TEMPORAL_CUTOFF]

    return train, test


def spatial_split(df):

    if SPATIAL_DEAD_END_COL not in df.columns:

        raise ValueError(
            f"{SPATIAL_DEAD_END_COL} missing"
        )

    train = df[
        df[SPATIAL_DEAD_END_COL] == 0
    ]

    test = df[
        df[SPATIAL_DEAD_END_COL] == 1
    ]

    return train, test


# ============================================================================
# Validation split
# ============================================================================

def validation_split(train_df):

    train_df = train_df.sample(
        frac=1.0,
        random_state=42
    )

    n_val = int(len(train_df) * VALIDATION_FRAC)

    val = train_df.iloc[:n_val]

    train = train_df.iloc[n_val:]

    return train, val


# ============================================================================
# Saving
# ============================================================================

def save_dataset(df, path):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df = optimize_memory(df)

    if path.suffix == ".parquet":

        df.to_parquet(path, index=False)

    else:

        df.to_csv(path, index=False)

    size_mb = path.stat().st_size / (1024**2)

    print(
        f"    Saved {path.name:<30}"
        f"{size_mb:8.1f} MB"
    )


# ============================================================================
# Plot helpers
# ============================================================================

def _save_fig(fig, name):

    path = (
        REPO_ROOT /
        "reports" /
        "plots" /
        f"split_{name}.png"
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fig.savefig(
        path,
        dpi=120,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"    Saved → {path.name}")


# ============================================================================
# Split plots
# ============================================================================

def plot_split_distribution(splits):

    fig, axes = plt.subplots(
        1,
        len(splits),
        figsize=(6 * len(splits), 5)
    )

    if len(splits) == 1:
        axes = [axes]

    for ax, (name, (train, test)) in zip(
        axes,
        splits.items()
    ):

        width = 0.35

        x = np.arange(len(RISK_ORDER))

        train_counts = (
            train["Risk"]
            .astype(str)
            .value_counts(normalize=True)
        )

        test_counts = (
            test["Risk"]
            .astype(str)
            .value_counts(normalize=True)
        )

        train_pct = [
            100 * train_counts.get(r, 0)
            for r in RISK_ORDER
        ]

        test_pct = [
            100 * test_counts.get(r, 0)
            for r in RISK_ORDER
        ]

        ax.bar(
            x - width/2,
            train_pct,
            width,
            label="Train"
        )

        ax.bar(
            x + width/2,
            test_pct,
            width,
            label="Test",
            hatch="//",
            alpha=0.7
        )

        ax.set_xticks(x)
        ax.set_xticklabels(RISK_ORDER)

        ax.set_ylim(0, 100)

        ax.set_title(name)

        ax.yaxis.set_major_formatter(
            ticker.PercentFormatter()
        )

        ax.legend()

    fig.tight_layout()

    _save_fig(fig, "distribution")


# ============================================================================
# Main
# ============================================================================

def main(input_path, output_dir, report_path):

    print("=" * 72)
    print("  GreenMining — Phase 4: Train/Test Split")
    print("=" * 72)

    if not input_path.exists():

        print(f"\nERROR: {input_path} not found")

        sys.exit(1)

    print(f"\nLoading:")
    print(f"  {input_path.relative_to(REPO_ROOT)}")

    df = pd.read_csv(input_path)

    print(f"\nRows loaded: {len(df):,}")

    print("\nOptimizing memory ...")

    df = optimize_memory(df)

    # ----------------------------------------------------------------------

    print("\n[1/3] Scenario split ...")

    train_sc, test_sc = scenario_split(df)

    verify_no_leakage(
        train_sc,
        test_sc,
        "SCENARIO"
    )

    # ----------------------------------------------------------------------

    print("\nCreating validation split ...")

    train_sc, val_sc = validation_split(train_sc)

    # ----------------------------------------------------------------------

    print("\nSaving primary datasets ...")

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    save_dataset(
        train_sc,
        output_dir / "train.csv"
    )

    save_dataset(
        val_sc,
        output_dir / "val.csv"
    )

    save_dataset(
        test_sc,
        output_dir / "test.csv"
    )

    # ----------------------------------------------------------------------

    print("\n[2/3] Temporal split ...")

    train_t, test_t = temporal_split(df)

    verify_no_leakage(
        train_t,
        test_t,
        "TEMPORAL"
    )

    save_dataset(
        train_t,
        output_dir / "train_temporal.csv"
    )

    save_dataset(
        test_t,
        output_dir / "test_temporal.csv"
    )

    # ----------------------------------------------------------------------

    print("\n[3/3] Spatial split ...")

    try:

        train_sp, test_sp = spatial_split(df)

        verify_no_leakage(
            train_sp,
            test_sp,
            "SPATIAL"
        )

        save_dataset(
            train_sp,
            output_dir / "train_spatial.csv"
        )

        save_dataset(
            test_sp,
            output_dir / "test_spatial.csv"
        )

    except ValueError as exc:

        print(f"  SKIP: {exc}")

        train_sp = test_sp = None

    # ----------------------------------------------------------------------

    print("\nGenerating plots ...")

    splits = {
        "Scenario": (train_sc, test_sc),
        "Temporal": (train_t, test_t),
    }

    if train_sp is not None:

        splits["Spatial"] = (
            train_sp,
            test_sp
        )

    plot_split_distribution(splits)

    # ----------------------------------------------------------------------

    print("\nWriting report ...")

    lines = [
        "=" * 72,
        "GreenMining Split Report",
        "=" * 72,
        "",
    ]

    lines += distribution_summary(
        train_sc,
        "Primary Train"
    )

    lines += [""]

    lines += distribution_summary(
        val_sc,
        "Validation"
    )

    lines += [""]

    lines += distribution_summary(
        test_sc,
        "Primary Test"
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print(
        f"  Report → "
        f"{report_path.relative_to(REPO_ROOT)}"
    )

    # ----------------------------------------------------------------------

    print("\nDONE.")
    print("\nPrimary ML files:")
    print("  data/final/train.csv")
    print("  data/final/val.csv")
    print("  data/final/test.csv")

    print("\nNext step: baseline ML models.\n")


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Phase 4 — Train/Test Split"
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=(
            REPO_ROOT /
            "data" /
            "processed" /
            "greenmining_features.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT /
            "data" /
            "final"
        ),
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=(
            REPO_ROOT /
            "reports" /
            "summaries" /
            "split_report.txt"
        ),
    )

    args = parser.parse_args()

    main(
        args.input,
        args.output_dir,
        args.report
    )