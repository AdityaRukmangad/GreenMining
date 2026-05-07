"""
Phase 1 — Master Dataset Creation
==================================
Merges all 5 scenario CSVs from the OpenFOAM post-processing pipeline into
a single master dataset for the ML pipeline.

Input:   results/scenario*/mine_cfd_output_scenario*.csv
Output:  data/raw/greenmining_master_dataset.csv

Operations (in order):
  1. Discover and load all scenario CSVs
  2. Verify column consistency across all files
  3. Concatenate into a single DataFrame
  4. Remove exact duplicate rows
  5. Clamp tiny negative concentrations → 0.0  (CH4, CO, H2)
  6. Enforce correct dtypes (Scenario: int, Risk: Categorical)
  7. Sort by Scenario → Time → x → y → z  (reproducible row order)
  8. Print statistics and save

Usage:
  python 01_merge_datasets.py
  python 01_merge_datasets.py --results-dir /custom/path/results
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent

# Column specification — must match postProcess_extract.py output exactly.
REQUIRED_COLUMNS = [
    "Time", "x", "y", "z",
    "CH4", "CO", "H2",
    "Temperature", "Velocity", "Pressure",
    "Risk", "Scenario",
]

CONCENTRATION_COLS = ["CH4", "CO", "H2"]

# Physical upper bounds for sanity check during load (not hard clamping).
CONCENTRATION_MAX = 1.0    # volume fraction
VELOCITY_MAX      = 50.0   # m/s  — well above any mine ventilation speed
TEMPERATURE_MAX   = 1000.0 # K
PRESSURE_MAX      = 2e6    # Pa


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def discover_csvs(results_dir: Path) -> list[Path]:
    """Return sorted list of scenario CSV paths that exist on disk."""
    paths = sorted(results_dir.glob("scenario*/mine_cfd_output_scenario*.csv"))
    if not paths:
        # Also check if CSVs are at results root (alternate layout)
        paths = sorted(results_dir.glob("mine_cfd_output_scenario*.csv"))
    return paths


def load_csv(path: Path, scenario_id: int) -> pd.DataFrame:
    """Load a single scenario CSV with minimal validation."""
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read {path}: {exc}") from exc

    # Normalise column names (strip whitespace)
    df.columns = df.columns.str.strip()

    # Inject Scenario if missing (legacy CSVs from old postprocessor)
    if "Scenario" not in df.columns:
        df["Scenario"] = scenario_id
        print(f"  [warn] Scenario column missing in {path.name} — injected {scenario_id}")

    return df


def check_column_consistency(dfs: list[pd.DataFrame], paths: list[Path]) -> set:
    """Verify all CSVs have the same columns.  Returns the union column set."""
    col_sets = [set(df.columns) for df in dfs]
    reference = col_sets[0]
    all_match = True
    for i, cols in enumerate(col_sets[1:], start=1):
        missing  = reference - cols
        extra    = cols - reference
        if missing or extra:
            print(f"  [warn] {paths[i].name}: missing={missing}, extra={extra}")
            all_match = False

    if all_match:
        print("  Column consistency: OK — all files share identical columns")
    else:
        print("  Column consistency: WARN — differences detected (see above)")

    return reference.union(*col_sets[1:])


def clamp_concentrations(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Set sub-zero concentration values to 0.0 (numerical diffusion artefacts)."""
    clamp_counts = {}
    for col in CONCENTRATION_COLS:
        if col not in df.columns:
            continue
        mask = df[col] < 0.0
        count = int(mask.sum())
        if count:
            df.loc[mask, col] = 0.0
        clamp_counts[col] = count
    return df, clamp_counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(results_dir: Path, output_path: Path) -> None:
    print("=" * 64)
    print("  GreenMining — Phase 1: Master Dataset Creation")
    print("=" * 64)

    # ------------------------------------------------------------------ #
    # 1. Discover CSVs
    # ------------------------------------------------------------------ #
    print(f"\n[1/7] Scanning {results_dir} ...")
    csv_paths = discover_csvs(results_dir)

    if not csv_paths:
        print(f"\n  ERROR: No scenario CSVs found in {results_dir}")
        print("  Expected pattern: results/scenarioN/mine_cfd_output_scenarioN.csv")
        print("  Run the CFD pipeline first: cd openfoam && ./Allrun [1-5]")
        print("  Then copy outputs:  cp openfoam/mine_cfd_output_scenario*.csv results/scenarioN/")
        sys.exit(1)

    print(f"  Found {len(csv_paths)} CSV file(s):")
    for p in csv_paths:
        size_mb = p.stat().st_size / (1024 ** 2)
        print(f"    {p.relative_to(REPO_ROOT)}  ({size_mb:.1f} MB)")

    # ------------------------------------------------------------------ #
    # 2. Load all CSVs
    # ------------------------------------------------------------------ #
    print(f"\n[2/7] Loading CSV files ...")
    dfs = []
    for path in csv_paths:
        # Extract scenario ID from filename (mine_cfd_output_scenarioN.csv)
        stem = path.stem  # e.g. mine_cfd_output_scenario3
        try:
            sid = int(stem.replace("mine_cfd_output_scenario", ""))
        except ValueError:
            sid = -1
        df = load_csv(path, sid)
        print(f"  {path.name}: {len(df):,} rows  {list(df.columns)}")
        dfs.append(df)

    # ------------------------------------------------------------------ #
    # 3. Verify column consistency
    # ------------------------------------------------------------------ #
    print(f"\n[3/7] Checking column consistency ...")
    check_column_consistency(dfs, csv_paths)

    # ------------------------------------------------------------------ #
    # 4. Concatenate
    # ------------------------------------------------------------------ #
    print(f"\n[4/7] Concatenating {len(dfs)} DataFrames ...")
    master = pd.concat(dfs, ignore_index=True, sort=False)
    rows_before_dedup = len(master)
    print(f"  Rows after concat: {rows_before_dedup:,}")

    # ------------------------------------------------------------------ #
    # 5. Remove duplicate rows
    # ------------------------------------------------------------------ #
    print(f"\n[5/7] Removing duplicate rows ...")
    master = master.drop_duplicates()
    rows_after_dedup = len(master)
    removed = rows_before_dedup - rows_after_dedup
    print(f"  Duplicates removed: {removed:,}")
    print(f"  Rows remaining:     {rows_after_dedup:,}")

    # ------------------------------------------------------------------ #
    # 6. Clamp negative concentrations → 0.0
    # ------------------------------------------------------------------ #
    print(f"\n[6/7] Clamping sub-zero concentrations ...")
    master, clamp_counts = clamp_concentrations(master)
    for col, n in clamp_counts.items():
        status = "OK" if n == 0 else f"clamped {n:,} cells"
        print(f"  {col:4s}: {status}")

    # ------------------------------------------------------------------ #
    # 7. Dtype enforcement and sort
    # ------------------------------------------------------------------ #
    print(f"\n[7/7] Enforcing dtypes and sorting ...")

    # Scenario → int
    master["Scenario"] = master["Scenario"].astype(int)

    # Risk → Categorical with defined order (helps downstream label encoding)
    risk_order = ["SAFE", "WARNING", "DANGER"]
    master["Risk"] = pd.Categorical(master["Risk"], categories=risk_order, ordered=True)

    # Canonical sort for reproducibility
    sort_cols = [c for c in ["Scenario", "Time", "x", "y", "z"] if c in master.columns]
    master = master.sort_values(sort_cols).reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Summary statistics
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 64)
    print("  MASTER DATASET SUMMARY")
    print("=" * 64)

    print(f"\n  Total rows        : {len(master):,}")
    print(f"  Total columns     : {len(master.columns)}")
    mem_mb = master.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"  Memory usage      : {mem_mb:.1f} MB")

    print("\n  Rows per scenario:")
    per_scenario = master.groupby("Scenario").size()
    for sid, count in per_scenario.items():
        print(f"    Scenario {sid}: {count:,} rows")

    print("\n  Risk class distribution:")
    risk_counts = master["Risk"].value_counts().reindex(risk_order, fill_value=0)
    total = len(master)
    for cls, count in risk_counts.items():
        pct = 100 * count / total
        print(f"    {cls:8s}: {count:>8,}  ({pct:5.1f}%)")

    print("\n  Concentration ranges (volume fraction):")
    for col in CONCENTRATION_COLS:
        if col in master.columns:
            mn, mx = master[col].min(), master[col].max()
            print(f"    {col:4s}: [{mn:.4e}, {mx:.4e}]")

    print("\n  Other field ranges:")
    for col, unit in [("Velocity", "m/s"), ("Temperature", "K"), ("Pressure", "Pa")]:
        if col in master.columns:
            mn, mx = master[col].min(), master[col].max()
            print(f"    {col:12s}: [{mn:.2f}, {mx:.2f}] {unit}")

    print(f"\n  Final column list:")
    print(f"    {list(master.columns)}")

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #
    output_path.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(output_path, index=False)
    out_mb = output_path.stat().st_size / (1024 ** 2)
    print(f"\n  Saved → {output_path.relative_to(REPO_ROOT)}  ({out_mb:.1f} MB)")
    print("\n  DONE.  Run 02_validate_dataset.py next.\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: Merge scenario CSVs into master dataset.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "results",
        help="Directory containing scenario sub-directories with CSVs. Default: ./results",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "greenmining_master_dataset.csv",
        help="Output path for master CSV. Default: data/raw/greenmining_master_dataset.csv",
    )
    args = parser.parse_args()
    main(args.results_dir, args.output)
