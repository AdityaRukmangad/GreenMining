"""
Phase 3 — Feature Engineering
==============================
Physics-informed feature engineering for GreenMining CFD datasets.

Optimized for:
- large datasets (~3M+ rows)
- memory efficiency
- vectorized operations
- reproducible ML workflows
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

EPS = 1e-9

LEL_CH4   = 0.050
LEL_H2    = 0.040
CO_DANGER = 0.0005

INLET_X  = 0.0
OUTLET_X = 80.0

FLOAT_DTYPE = np.float32
INT_DTYPE   = np.int32

STAGNATION_VELOCITY = 0.3

INLET_VELOCITY = {
    1: 2.0,
    2: 1.5,
    3: 0.5,
    4: 1.0,
    5: 2.0,
}

SOURCE_POINTS = {
    1: [],
    2: [(18.0, 2.0, 1.5)],
    3: [
        (25.0, 13.0, 1.5),
        (56.0, 17.0, 1.5),
        (39.0, -4.0, 1.5),
    ],
    4: [(25.0, 14.0, 1.5)],
    5: [
        (25.0, 13.0, 1.5),
        (56.0, 17.0, 1.5),
        (39.0, -4.0, 1.5),
    ],
}

ZONE_X = {
    "INLET_SECTION":  (0.0, 20.0),
    "JUNCTION_1":     (20.0, 30.0),
    "MID_TUNNEL":     (30.0, 50.0),
    "JUNCTION_2_3":   (50.0, 62.0),
    "OUTLET_SECTION": (62.0, 80.0),
}

DEAD_END_ZONES = [
    (20.0, 30.0,  4.0, 16.0, 0.0, 3.0, "CHAMBER_1"),
    (50.0, 62.0,  4.0, 20.0, 0.0, 3.0, "CHAMBER_2"),
    (35.0, 43.0, -6.0,  0.0, 0.0, 3.0, "SOUTH_STUB"),
]

ORIGINAL_COLUMNS = [
    "Time", "x", "y", "z",
    "CH4", "CO", "H2",
    "Temperature", "Velocity",
    "Pressure", "Risk", "Scenario"
]


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
# Geometry features
# ============================================================================

def add_geometry_features(df):

    print("  Geometry features ...")

    x = df["x"].values
    y = df["y"].values
    z = df["z"].values

    df["dist_inlet"] = np.sqrt(
        x**2 +
        (y - 2.0)**2 +
        (z - 1.5)**2
    ).astype(FLOAT_DTYPE)

    df["dist_outlet"] = np.sqrt(
        (x - OUTLET_X)**2 +
        (y - 2.0)**2 +
        (z - 1.5)**2
    ).astype(FLOAT_DTYPE)

    # ----------------------------------------------------------------------
    # Vectorized source distance
    # ----------------------------------------------------------------------

    dist_source = np.full(len(df), OUTLET_X, dtype=FLOAT_DTYPE)

    for sid, points in SOURCE_POINTS.items():

        mask = (df["Scenario"].values == sid)

        if not np.any(mask):
            continue

        if not points:
            continue

        xs = x[mask]
        ys = y[mask]
        zs = z[mask]

        min_dist = np.full(mask.sum(), np.inf)

        for px, py, pz in points:

            d = np.sqrt(
                (xs - px)**2 +
                (ys - py)**2 +
                (zs - pz)**2
            )

            min_dist = np.minimum(min_dist, d)

        dist_source[mask] = min_dist

    df["dist_source"] = dist_source

    # ----------------------------------------------------------------------
    # Chamber masks
    # ----------------------------------------------------------------------

    in_chamber = np.zeros(len(df), dtype=np.int8)

    zone = np.full(len(df), "MAIN_TUNNEL", dtype=object)

    for x0, x1, y0, y1, z0, z1, name in DEAD_END_ZONES:

        mask = (
            (x >= x0) & (x <= x1) &
            (y >= y0) & (y <= y1) &
            (z >= z0) & (z <= z1)
        )

        in_chamber[mask] = 1
        zone[mask] = name

    main = (in_chamber == 0)

    for name, (x0, x1) in ZONE_X.items():

        mask = main & (x >= x0) & (x < x1)

        zone[mask] = name

    df["in_chamber"] = in_chamber
    df["zone"] = pd.Categorical(zone)

    return df


# ============================================================================
# Concentration features
# ============================================================================

def add_concentration_features(df):

    print("  Concentration features ...")

    total = df["CH4"] + df["CO"] + df["H2"]

    df["total_gas"] = total.astype(FLOAT_DTYPE)

    df["CH4_frac"] = (df["CH4"] / (total + EPS)).astype(FLOAT_DTYPE)
    df["CO_frac"]  = (df["CO"]  / (total + EPS)).astype(FLOAT_DTYPE)
    df["H2_frac"]  = (df["H2"]  / (total + EPS)).astype(FLOAT_DTYPE)

    df["gas_LEL_equiv"] = (
        df["CH4"] / LEL_CH4 +
        df["H2"] / LEL_H2
    ).astype(FLOAT_DTYPE)

    df["co_toxicity_ratio"] = (
        df["CO"] / CO_DANGER
    ).astype(FLOAT_DTYPE)

    for gas in ["CH4", "CO", "H2"]:

        df[f"{gas}_log"] = np.log10(
            df[gas] + 1e-6
        ).astype(FLOAT_DTYPE)

    return df


# ============================================================================
# Velocity features
# ============================================================================

def add_velocity_features(df):

    print("  Velocity features ...")

    vel = df["Velocity"]

    df["low_velocity"] = (
        vel < STAGNATION_VELOCITY
    ).astype(np.int8)

    inlet_vel = df["Scenario"].map(INLET_VELOCITY)

    df["recirculation_proxy"] = (
        (df["in_chamber"] == 1) &
        (vel < 0.5 * inlet_vel)
    ).astype(np.int8)

    return df


# ============================================================================
# Labels
# ============================================================================

def add_label_features(df):

    print("  Label features ...")

    risk = df["Risk"].astype(str)

    df["hazard_binary"] = (
        risk != "SAFE"
    ).astype(np.int8)

    label_map = {
        "SAFE": 0,
        "WARNING": 1,
        "DANGER": 2,
    }

    df["hazard_3class"] = (
        risk.map(label_map)
    ).astype(np.int8)

    return df


# ============================================================================
# Time features
# ============================================================================

def add_time_features(df):

    print("  Time features ...")

    df["time_norm"] = (
        df["Time"] / 300.0
    ).astype(FLOAT_DTYPE)

    return df


# ============================================================================
# Temporal gradients
# ============================================================================

def add_temporal_gradients(df):

    print("  Temporal gradients ...")

    species = ["CH4", "CO", "H2"]

    group_cols = ["Scenario", "x", "y", "z"]

    df = df.sort_values(
        group_cols + ["Time"]
    ).reset_index(drop=True)

    grp = df.groupby(group_cols, sort=False)

    dt = grp["Time"].diff()

    for gas in species:

        diff = grp[gas].diff()

        grad = (diff / dt).fillna(0.0)

        df[f"d{gas}_dt"] = grad.astype(FLOAT_DTYPE)

    df["dCH4_dt_abs"] = (
        np.abs(df["dCH4_dt"])
    ).astype(FLOAT_DTYPE)

    df["accumulating"] = (
        df["dCH4_dt"] > 1e-5
    ).astype(np.int8)

    return df


# ============================================================================
# Validation
# ============================================================================

def validate_features(df):

    print("\n  Feature sanity checks ...")

    checks = [
        ("dist_source", 0),
        ("gas_LEL_equiv", 0),
        ("time_norm", 0),
    ]

    for col, lo in checks:

        if col not in df.columns:
            continue

        mn = df[col].min()

        if mn < lo:
            print(f"    [WARN] {col} below expected range")


# ============================================================================
# Main
# ============================================================================

def main(input_path, output_path):

    print("=" * 72)
    print("  GreenMining — Phase 3: Feature Engineering")
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

    mem_before = (
        df.memory_usage(deep=True).sum() / (1024**2)
    )

    print(f"  Memory: {mem_before:.1f} MB")

    # ----------------------------------------------------------------------

    df = add_geometry_features(df)

    df = add_concentration_features(df)

    df = add_velocity_features(df)

    df = add_label_features(df)

    df = add_time_features(df)

    df = add_temporal_gradients(df)

    # ----------------------------------------------------------------------

    validate_features(df)

    # ----------------------------------------------------------------------

    print("\nFinal memory optimization ...")

    df = optimize_memory(df)

    mem_after = (
        df.memory_usage(deep=True).sum() / (1024**2)
    )

    print(f"  Final memory: {mem_after:.1f} MB")

    # ----------------------------------------------------------------------

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("\nSaving dataset ...")

    df.to_csv(output_path, index=False)

    out_mb = output_path.stat().st_size / (1024**2)

    print(f"\nSaved:")
    print(f"  {output_path.relative_to(REPO_ROOT)}")
    print(f"  Size: {out_mb:.1f} MB")

    print("\nDONE.")
    print("Next step: python 04_train_test_split.py\n")


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Phase 3 — Feature Engineering"
    )

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