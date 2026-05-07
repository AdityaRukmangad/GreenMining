"""
Phase 3 — Feature Engineering
================================
Adds physics-informed features to the master dataset.
All features are derived from quantities that have a direct physical meaning
in the mine ventilation context — no abstract latent representations.

Input:   data/raw/greenmining_master_dataset.csv
Output:  data/processed/greenmining_features.csv

Features added:
  --- Geometry ---
  dist_inlet          Euclidean distance from cell centre to inlet face (x=0)
  dist_outlet         Euclidean distance to outlet face (x=80)
  dist_source         Min. Euclidean distance to the active source point(s)
                      for this scenario (from scalarSourceTerms calibration)
  in_chamber          1 if cell is inside a dead-end chamber or stub, else 0
  zone                Categorical: INLET_SECTION / JUNCTION_1 / MID_TUNNEL /
                                   JUNCTION_2 / JUNCTION_3 / OUTLET_SECTION /
                                   CHAMBER_1 / CHAMBER_2 / SOUTH_STUB

  --- Concentration derived ---
  total_gas           CH4 + CO + H2  (total syngas volume fraction)
  CH4_frac            CH4 / (total_gas + eps)  — CH4 share of gas mixture
  CO_frac             CO  / (total_gas + eps)
  H2_frac             H2  / (total_gas + eps)
  gas_LEL_equiv       Normalised explosion risk:
                        CH4/LEL_CH4 + H2/LEL_H2  (additive LEL rule)
  co_toxicity_ratio   CO / CO_DANGER_THRESH       (1.0 = DANGER threshold)
  CH4_log             log10(CH4 + 1e-6)           — log-scale for SAFE cells
  CO_log              log10(CO  + 1e-6)
  H2_log              log10(H2  + 1e-6)

  --- Velocity ---
  vel_x               x-component of velocity (from Velocity in single column)
                      NOTE: Velocity in CSV is the magnitude |U|; if U_x is
                      not available, this is set to NaN and noted.
  low_velocity        1 if |U| < 0.3 m/s  (stagnation indicator)
  recirculation_proxy 1 if cell is in dead-end AND |U| < 0.5 × inlet_velocity

  --- Labels ---
  hazard_binary       0 = SAFE,  1 = WARNING or DANGER  (binary classification)
  hazard_3class       0 = SAFE,  1 = WARNING,  2 = DANGER (multi-class)

  --- Temporal gradients ---
  dCH4_dt             ΔCH4 / Δt  per spatial cell  [vol_frac / s]
  dCO_dt              ΔCO  / Δt
  dH2_dt              ΔH2  / Δt
  dCH4_dt_abs         |dCH4_dt|  (unsigned accumulation rate)
  accumulating        1 if dCH4_dt > 1e-5 (CH4 actively increasing)

  --- Time ---
  time_norm           Time / 300   in [0, 1]

Usage:
  python 03_feature_engineering.py
  python 03_feature_engineering.py --input data/raw/greenmining_master_dataset.csv
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

# Physical constants
EPS          = 1e-9    # avoid division by zero
LEL_CH4      = 0.050   # vol fraction — lower explosive limit
LEL_H2       = 0.040   # vol fraction
CO_DANGER    = 0.0005  # vol fraction (500 ppm OSHA danger threshold)

# Geometry — must match blockMeshDict
INLET_X      = 0.0
OUTLET_X     = 80.0
TUNNEL_Y_MIN = 0.0
TUNNEL_Y_MAX = 4.0

# Inlet velocity per scenario [m/s] — from scenario U files
INLET_VELOCITY = {1: 2.0, 2: 1.5, 3: 0.5, 4: 1.0, 5: 2.0}

# Stagnation threshold: below this |U| we call the cell a stagnation zone
STAGNATION_VELOCITY = 0.3   # m/s

# Source locations per scenario (from scalarSourceTerms)
# Each scenario maps to a list of (x, y, z) injection points.
# For scenarios with no source, the distance is set to the tunnel length.
SOURCE_POINTS = {
    1: [],                               # no sources
    2: [(18.0, 2.0, 1.5)],              # seam seepage, Chamber 1 approach
    3: [(25.0, 13.0, 1.5),             # Chamber 1 dead end
        (56.0, 17.0, 1.5),             # Chamber 2 dead end
        (39.0, -4.0, 1.5)],            # South Stub
    4: [(25.0, 14.0, 1.5)],            # Chamber 1 dead end
    5: [(25.0, 13.0, 1.5),             # Chamber 1 dead end (blowout)
        (56.0, 17.0, 1.5),             # Chamber 2 dead end (blowout)
        (39.0, -4.0, 1.5)],            # South Stub (blowout)
}

# Zone x-boundaries in the main tunnel
ZONE_X = {
    "INLET_SECTION":  (0.0,  20.0),
    "JUNCTION_1":     (20.0, 30.0),
    "MID_TUNNEL":     (30.0, 50.0),
    "JUNCTION_2_3":   (50.0, 62.0),
    "OUTLET_SECTION": (62.0, 80.0),
}

# Dead-end chamber/stub bounds [(x_min, x_max, y_min, y_max, z_min, z_max, name), ...]
DEAD_END_ZONES = [
    (20.0, 30.0,  4.0, 16.0, 0.0, 3.0, "CHAMBER_1"),
    (50.0, 62.0,  4.0, 20.0, 0.0, 3.0, "CHAMBER_2"),
    (35.0, 43.0, -6.0,  0.0, 0.0, 3.0, "SOUTH_STUB"),
]


# ---------------------------------------------------------------------------
# Feature functions
# ---------------------------------------------------------------------------

def add_geometry_features(df: pd.DataFrame) -> pd.DataFrame:
    """Distance metrics and spatial zone labels."""

    # Distance to inlet (x=0, y=2, z=1.5 — centreline of inlet face)
    # Simplified: Euclidean distance using x, y, z from each cell centre.
    df["dist_inlet"]  = np.sqrt(df["x"]**2 + (df["y"] - 2.0)**2 + (df["z"] - 1.5)**2)
    df["dist_outlet"] = np.sqrt((df["x"] - OUTLET_X)**2 + (df["y"] - 2.0)**2 + (df["z"] - 1.5)**2)

    # Distance to nearest active source for this scenario
    def _dist_to_nearest_source(row):
        pts = SOURCE_POINTS.get(int(row["Scenario"]), [])
        if not pts:
            return float(OUTLET_X)  # no source — use tunnel length as sentinel
        return min(
            np.sqrt((row["x"] - px)**2 + (row["y"] - py)**2 + (row["z"] - pz)**2)
            for px, py, pz in pts
        )

    df["dist_source"] = df.apply(_dist_to_nearest_source, axis=1)

    # in_chamber: 1 if inside any dead-end, 0 otherwise
    in_chamber = pd.Series(False, index=df.index)
    zone_label = pd.Series("MAIN_TUNNEL", index=df.index)

    for x0, x1, y0, y1, z0, z1, name in DEAD_END_ZONES:
        mask = (
            (df["x"] >= x0) & (df["x"] <= x1) &
            (df["y"] >= y0) & (df["y"] <= y1) &
            (df["z"] >= z0) & (df["z"] <= z1)
        )
        in_chamber |= mask
        zone_label[mask] = name

    # Main tunnel zones (only cells NOT already in a dead-end)
    main = ~in_chamber
    for name, (x0, x1) in ZONE_X.items():
        mask = main & (df["x"] >= x0) & (df["x"] < x1)
        zone_label[mask] = name

    df["in_chamber"] = in_chamber.astype(np.int8)
    df["zone"]       = zone_label.astype("category")

    return df


def add_concentration_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derived concentration quantities with physical meaning."""

    total_gas = df["CH4"] + df["CO"] + df["H2"]
    df["total_gas"]   = total_gas

    df["CH4_frac"] = df["CH4"] / (total_gas + EPS)
    df["CO_frac"]  = df["CO"]  / (total_gas + EPS)
    df["H2_frac"]  = df["H2"]  / (total_gas + EPS)

    # Additive LEL rule (Le Chatelier):
    # sum(phi_i / LEL_i) >= 1.0 means the mixture is at or above its LEL.
    # Only CH4 and H2 are flammable; CO does not have a standard LEL in
    # the additive rule at these concentrations.
    df["gas_LEL_equiv"] = df["CH4"] / LEL_CH4 + df["H2"] / LEL_H2

    # CO toxicity relative to danger threshold (1.0 = DANGER)
    df["co_toxicity_ratio"] = df["CO"] / CO_DANGER

    # Log-scale concentrations (compress the wide dynamic range for SAFE cells)
    # Using log10(phi + 1e-6) so that phi=0 maps to -6 (not -inf).
    df["CH4_log"] = np.log10(df["CH4"] + 1e-6)
    df["CO_log"]  = np.log10(df["CO"]  + 1e-6)
    df["H2_log"]  = np.log10(df["H2"]  + 1e-6)

    return df


def add_velocity_features(df: pd.DataFrame, inlet_vel_col: bool = True) -> pd.DataFrame:
    """Stagnation and recirculation indicators."""

    # Low-velocity / stagnation indicator
    df["low_velocity"] = (df["Velocity"] < STAGNATION_VELOCITY).astype(np.int8)

    # Recirculation proxy: inside a dead-end AND significantly slower than
    # the inlet velocity for this scenario.
    def _inlet_vel(sid):
        return INLET_VELOCITY.get(int(sid), 2.0)

    inlet_v = df["Scenario"].map(_inlet_vel)
    df["recirculation_proxy"] = (
        (df["in_chamber"] == 1) & (df["Velocity"] < 0.5 * inlet_v)
    ).astype(np.int8)

    return df


def add_label_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode Risk into numeric labels for ML models."""

    # Binary: SAFE=0, WARNING or DANGER=1
    df["hazard_binary"] = (df["Risk"].astype(str) != "SAFE").astype(np.int8)

    # 3-class: SAFE=0, WARNING=1, DANGER=2
    label_map = {"SAFE": 0, "WARNING": 1, "DANGER": 2}
    df["hazard_3class"] = df["Risk"].astype(str).map(label_map).astype(np.int8)

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalised time in [0, 1]."""
    t_max = 300.0
    df["time_norm"] = df["Time"] / t_max
    return df


def add_temporal_gradients(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute dCH4/dt, dCO/dt, dH2/dt per spatial cell.

    The dataset has one row per (Scenario, Time, x, y, z) combination.
    For each unique spatial cell in a scenario, we compute the forward
    finite difference of concentration across consecutive time steps.

    t=0 row: gradient set to 0 (no previous time step).
    """
    print("  Computing temporal concentration gradients ...")

    species = ["CH4", "CO", "H2"]
    grad_cols = {s: f"d{s}_dt" for s in species}

    # Sort to ensure forward-difference is computed correctly
    df = df.sort_values(["Scenario", "x", "y", "z", "Time"]).reset_index(drop=True)

    # Groupby (Scenario, x, y, z) → compute diff() / dt_diff()
    group_keys = ["Scenario", "x", "y", "z"]

    grp = df.groupby(group_keys, sort=False)

    # Initialise gradient columns to 0
    for s in species:
        df[grad_cols[s]] = 0.0

    time_diff = grp["Time"].diff()  # Δt between consecutive rows in each group

    for s in species:
        conc_diff = grp[s].diff()   # Δφ between consecutive rows
        # Forward difference: dφ/dt = Δφ / Δt  (NaN at first row of each group → fill 0)
        df[grad_cols[s]] = (conc_diff / time_diff).fillna(0.0)

    # Unsigned accumulation rate for CH4 (magnitude of change)
    df["dCH4_dt_abs"] = df["dCH4_dt"].abs()

    # Binary: is CH4 actively increasing above noise floor?
    NOISE_FLOOR = 1e-5   # vol_frac / s — below this is numerical diffusion
    df["accumulating"] = (df["dCH4_dt"] > NOISE_FLOOR).astype(np.int8)

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(input_path: Path, output_path: Path) -> None:
    print("=" * 64)
    print("  GreenMining — Phase 3: Feature Engineering")
    print("=" * 64)

    if not input_path.exists():
        print(f"\n  ERROR: {input_path} not found.")
        print("  Run 01_merge_datasets.py first.")
        sys.exit(1)

    print(f"\n  Loading {input_path.relative_to(REPO_ROOT)} ...")
    df = pd.read_csv(input_path)
    n_rows_in = len(df)
    print(f"  Rows loaded: {n_rows_in:,}")
    print(f"  Columns in:  {list(df.columns)}")

    # ------------------------------------------------------------------ #
    # Apply feature groups
    # ------------------------------------------------------------------ #
    print("\n  [1/6] Geometry features (dist_inlet, dist_outlet, dist_source, zone) ...")
    df = add_geometry_features(df)

    print("  [2/6] Concentration derived features (total_gas, fractions, LEL, logs) ...")
    df = add_concentration_features(df)

    print("  [3/6] Velocity features (low_velocity, recirculation_proxy) ...")
    df = add_velocity_features(df)

    print("  [4/6] Label encoding (hazard_binary, hazard_3class) ...")
    df = add_label_features(df)

    print("  [5/6] Time features (time_norm) ...")
    df = add_time_features(df)

    print("  [6/6] Temporal gradients (dCH4/dt, dCO/dt, dH2/dt) ...")
    df = add_temporal_gradients(df)

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 64)
    print("  FEATURE ENGINEERING SUMMARY")
    print("=" * 64)

    print(f"\n  Rows: {len(df):,}  (unchanged from input: {n_rows_in:,})")
    print(f"  Columns: {len(df.columns)}")

    new_cols = [c for c in df.columns if c not in [
        "Time", "x", "y", "z", "CH4", "CO", "H2",
        "Temperature", "Velocity", "Pressure", "Risk", "Scenario"
    ]]
    print(f"\n  New features added ({len(new_cols)}):")
    for col in new_cols:
        dtype = df[col].dtype
        if hasattr(dtype, "categories"):
            dtype_str = f"categorical({len(dtype.categories)} classes)"
        else:
            dtype_str = str(dtype)
        print(f"    {col:30s}  {dtype_str}")

    print("\n  Feature statistics:")
    check_cols = ["dist_inlet", "dist_outlet", "dist_source", "total_gas",
                  "gas_LEL_equiv", "co_toxicity_ratio", "dCH4_dt"]
    for col in check_cols:
        if col not in df.columns:
            continue
        mn, mx, med = df[col].min(), df[col].max(), df[col].median()
        print(f"    {col:30s}  min={mn:.3e}  median={med:.3e}  max={mx:.3e}")

    print("\n  Zone distribution:")
    zone_counts = df["zone"].value_counts().sort_index()
    for zone, count in zone_counts.items():
        pct = 100 * count / len(df)
        print(f"    {zone:20s}: {count:>8,}  ({pct:5.1f}%)")

    print("\n  in_chamber distribution:")
    ch_counts = df["in_chamber"].value_counts()
    for val, count in ch_counts.sort_index().items():
        label = "main tunnel" if val == 0 else "dead-end/stub"
        print(f"    {val} ({label:12s}): {count:>8,}  ({100*count/len(df):.1f}%)")

    print("\n  Accumulating cells (dCH4/dt > 1e-5):")
    n_acc = df["accumulating"].sum()
    print(f"    {n_acc:,}  ({100*n_acc/len(df):.1f}%)")

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    out_mb = output_path.stat().st_size / (1024 ** 2)
    print(f"\n  Saved → {output_path.relative_to(REPO_ROOT)}  ({out_mb:.1f} MB)")
    print("\n  Run 04_train_test_split.py next.\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3: Feature engineering.")
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "greenmining_master_dataset.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "greenmining_features.csv",
    )
    args = parser.parse_args()
    main(args.input, args.output)
