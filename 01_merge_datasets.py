"""
Phase 1 — Master Dataset Creation
==================================
Merges all scenario CSVs from the OpenFOAM post-processing pipeline into
a single master dataset for the ML pipeline.

Input:
    results/scenario*/mine_cfd_output_scenario*.csv

Output:
    data/raw/greenmining_master_dataset.csv

Operations:
  1. Discover and load all scenario CSVs
  2. Verify required columns and schema consistency
  3. Concatenate into a single DataFrame
  4. Remove duplicate rows
  5. Clamp tiny negative concentrations → 0.0
  6. Enforce efficient dtypes
  7. Sort rows reproducibly
  8. Save master dataset

Usage:
    python 01_merge_datasets.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# Configuration
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parent

REQUIRED_COLUMNS = [
    "Time",
    "x",
    "y",
    "z",
    "CH4",
    "CO",
    "H2",
    "Temperature",
    "Velocity",
    "Pressure",
    "Risk",
    "Scenario",
]

CONCENTRATION_COLS = ["CH4", "CO", "H2"]

RISK_ORDER = ["SAFE", "WARNING", "DANGER"]

# Physical sanity limits
CONCENTRATION_MAX = 1.0
VELOCITY_MAX = 50.0
TEMPERATURE_MAX = 1000.0
PRESSURE_MAX = 2e6

# Memory optimization dtypes
FLOAT_DTYPE = "float32"
INT_DTYPE = "int32"


# ============================================================================
# Helpers
# ============================================================================

def discover_csvs(results_dir: Path) -> list[Path]:
    """Discover all scenario CSVs."""
    paths = sorted(results_dir.glob("scenario*/mine_cfd_output_scenario*.csv"))

    if not paths:
        paths = sorted(results_dir.glob("mine_cfd_output_scenario*.csv"))

    return paths


def extract_scenario_id(path: Path) -> int:
    """Extract scenario ID from filename."""
    stem = path.stem

    try:
        return int(stem.replace("mine_cfd_output_scenario", ""))
    except Exception:
        return -1


def load_csv(path: Path, scenario_id: int) -> pd.DataFrame:
    """Load a single CSV safely."""
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to read {path}: {exc}") from exc

    # Normalize columns
    df.columns = df.columns.str.strip()

    # Inject Scenario if missing
    if "Scenario" not in df.columns:
        df["Scenario"] = scenario_id
        print(f"  [WARN] Injected missing Scenario={scenario_id} into {path.name}")

    return df


def verify_required_columns(df: pd.DataFrame, path: Path) -> None:
    """Ensure all required columns exist."""
    missing = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(
            f"{path.name} is missing required columns:\n"
            f"  {sorted(missing)}"
        )


def check_column_consistency(dfs: list[pd.DataFrame], paths: list[Path]) -> None:
    """Check schema consistency."""
    reference = set(dfs[0].columns)

    all_match = True

    for df, path in zip(dfs[1:], paths[1:]):
        cols = set(df.columns)

        missing = reference - cols
        extra = cols - reference

        if missing or extra:
            all_match = False
            print(f"\n  [WARN] {path.name}")
            print(f"    Missing: {sorted(missing)}")
            print(f"    Extra  : {sorted(extra)}")

    if all_match:
        print("  Column consistency: OK")


def clamp_negative_concentrations(df: pd.DataFrame) -> dict:
    """Clamp tiny negative concentrations to zero."""
    clamp_counts = {}

    for col in CONCENTRATION_COLS:
        if col not in df.columns:
            continue

        mask = df[col] < 0.0
        count = int(mask.sum())

        if count:
            df.loc[mask, col] = 0.0

        clamp_counts[col] = count

    return clamp_counts


def apply_memory_optimization(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce dataframe memory footprint."""
    float_cols = df.select_dtypes(include=["float64"]).columns
    int_cols = df.select_dtypes(include=["int64"]).columns

    if len(float_cols):
        df[float_cols] = df[float_cols].astype(FLOAT_DTYPE)

    if len(int_cols):
        df[int_cols] = df[int_cols].astype(INT_DTYPE)

    return df


def physical_sanity_checks(df: pd.DataFrame) -> None:
    """Print warnings for suspicious physical values."""

    print("\n  Physical sanity checks:")

    for col in CONCENTRATION_COLS:
        if col not in df.columns:
            continue

        mx = df[col].max()

        if mx > CONCENTRATION_MAX:
            print(f"    [WARN] {col} exceeds physical limit ({mx:.3f})")

    if "Velocity" in df.columns:
        mx = df["Velocity"].max()
        if mx > VELOCITY_MAX:
            print(f"    [WARN] Velocity unusually high ({mx:.2f} m/s)")

    if "Temperature" in df.columns:
        mx = df["Temperature"].max()
        if mx > TEMPERATURE_MAX:
            print(f"    [WARN] Temperature unusually high ({mx:.2f} K)")

    if "Pressure" in df.columns:
        mx = df["Pressure"].max()
        if mx > PRESSURE_MAX:
            print(f"    [WARN] Pressure unusually high ({mx:.2f} Pa)")


# ============================================================================
# Main
# ============================================================================

def main(results_dir: Path, output_path: Path) -> None:

    print("=" * 72)
    print("  GreenMining — Phase 1: Master Dataset Creation")
    print("=" * 72)

    # ----------------------------------------------------------------------
    # Discover CSVs
    # ----------------------------------------------------------------------

    print(f"\n[1/8] Discovering scenario CSVs in:")
    print(f"        {results_dir}")

    csv_paths = discover_csvs(results_dir)

    if not csv_paths:
        print("\nERROR: No scenario CSVs found.")
        sys.exit(1)

    print(f"\n  Found {len(csv_paths)} CSV files:")

    for path in csv_paths:
        size_mb = path.stat().st_size / (1024 ** 2)
        print(f"    {path.name:<40} {size_mb:8.1f} MB")

    # ----------------------------------------------------------------------
    # Load CSVs
    # ----------------------------------------------------------------------

    print(f"\n[2/8] Loading CSV files ...")

    dfs = []

    for path in csv_paths:

        sid = extract_scenario_id(path)

        df = load_csv(path, sid)

        verify_required_columns(df, path)

        print(f"  Loaded {path.name:<40} {len(df):>12,} rows")

        dfs.append(df)

    # ----------------------------------------------------------------------
    # Schema consistency
    # ----------------------------------------------------------------------

    print(f"\n[3/8] Verifying schema consistency ...")

    check_column_consistency(dfs, csv_paths)

    # ----------------------------------------------------------------------
    # Concatenate
    # ----------------------------------------------------------------------

    print(f"\n[4/8] Concatenating DataFrames ...")

    master = pd.concat(dfs, ignore_index=True, sort=False)

    rows_before = len(master)

    print(f"  Rows after concat: {rows_before:,}")

    # ----------------------------------------------------------------------
    # Remove duplicates
    # ----------------------------------------------------------------------

    print(f"\n[5/8] Removing duplicate rows ...")

    master = master.drop_duplicates()

    rows_after = len(master)

    removed = rows_before - rows_after

    print(f"  Duplicates removed: {removed:,}")
    print(f"  Remaining rows    : {rows_after:,}")

    # ----------------------------------------------------------------------
    # Clamp negative concentrations
    # ----------------------------------------------------------------------

    print(f"\n[6/8] Clamping negative concentrations ...")

    clamp_counts = clamp_negative_concentrations(master)

    for col, n in clamp_counts.items():

        if n == 0:
            print(f"  {col:<4} : OK")
        else:
            print(f"  {col:<4} : clamped {n:,} values")

    # ----------------------------------------------------------------------
    # Dtypes + sorting
    # ----------------------------------------------------------------------

    print(f"\n[7/8] Optimizing dtypes and sorting ...")

    master["Scenario"] = master["Scenario"].astype(INT_DTYPE)

    master["Risk"] = pd.Categorical(
        master["Risk"],
        categories=RISK_ORDER,
        ordered=True
    )

    master = apply_memory_optimization(master)

    sort_cols = [c for c in ["Scenario", "Time", "x", "y", "z"] if c in master.columns]

    master = master.sort_values(sort_cols).reset_index(drop=True)

    # ----------------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------------

    print("\n" + "=" * 72)
    print("  MASTER DATASET SUMMARY")
    print("=" * 72)

    print(f"\n  Rows            : {len(master):,}")
    print(f"  Columns         : {len(master.columns)}")

    mem_mb = master.memory_usage(deep=True).sum() / (1024 ** 2)

    print(f"  Memory usage    : {mem_mb:.1f} MB")

    print("\n  Rows per scenario:")

    per_scenario = master.groupby("Scenario").size()

    for sid, count in per_scenario.items():
        print(f"    Scenario {sid}: {count:,}")

    print("\n  Risk distribution:")

    risk_counts = (
        master["Risk"]
        .value_counts()
        .reindex(RISK_ORDER, fill_value=0)
    )

    total = len(master)

    for cls, count in risk_counts.items():
        pct = 100 * count / total
        print(f"    {cls:<8}: {count:>10,} ({pct:5.1f}%)")

    print("\n  Concentration ranges:")

    for col in CONCENTRATION_COLS:

        mn = master[col].min()
        mx = master[col].max()

        print(f"    {col:<4}: [{mn:.4e}, {mx:.4e}]")

    print("\n  Other field ranges:")

    for col, unit in [
        ("Velocity", "m/s"),
        ("Temperature", "K"),
        ("Pressure", "Pa"),
    ]:

        if col not in master.columns:
            continue

        mn = master[col].min()
        mx = master[col].max()

        print(f"    {col:<12}: [{mn:.2f}, {mx:.2f}] {unit}")

    physical_sanity_checks(master)

    print("\n  Final columns:")
    print(f"    {list(master.columns)}")

    # ----------------------------------------------------------------------
    # Save
    # ----------------------------------------------------------------------

    print(f"\n[8/8] Saving master dataset ...")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    master.to_csv(output_path, index=False)

    out_mb = output_path.stat().st_size / (1024 ** 2)

    print(f"\n  Saved:")
    print(f"    {output_path.relative_to(REPO_ROOT)}")
    print(f"    Size: {out_mb:.1f} MB")

    print("\nDONE.")
    print("Next step: python 02_validate_dataset.py\n")


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Phase 1 — Merge scenario CSVs into master dataset."
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "results",
        help="Directory containing scenario CSVs."
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "greenmining_master_dataset.csv",
        help="Output path for merged dataset."
    )

    args = parser.parse_args()

    main(args.results_dir, args.output)