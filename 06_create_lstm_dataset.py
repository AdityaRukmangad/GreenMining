"""
Phase 6 — Create LSTM Forecasting Dataset
=========================================
Build temporal sequence datasets for hazard forecasting.

Goal
----
Learn:
    [t-45, t-30, t-15, t]
            ↓
    predict hazard at t+30

This script converts the engineered CFD dataset into:
- temporal sequence tensors
- future hazard labels
- train / validation / test forecasting datasets

Outputs
-------
data/lstm/
├── train_X.npy
├── train_y.npy
├── val_X.npy
├── val_y.npy
├── test_X.npy
├── test_y.npy
├── train_y_multiclass.npy
├── val_y_multiclass.npy
├── test_y_multiclass.npy
├── feature_columns.json
└── dataset_metadata.json

Important
---------
This script preserves:
- spatial consistency
- scenario consistency
- temporal ordering

No future leakage is introduced.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


# ============================================================================
# Repository paths
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parent

OUTPUT_DIR = REPO_ROOT / "data" / "lstm"

# ============================================================================
# Configuration
# ============================================================================

SEQUENCE_LENGTH = 4

FORECAST_HORIZON_STEPS = 2

TIME_DELTA = 15.0

VALIDATION_FRACTION = 0.15

RANDOM_STATE = 42

# ============================================================================
# Features to exclude
# ============================================================================

DROP_COLUMNS = [

    # Labels
    "Risk",
    "hazard_binary",
    "hazard_3class",

    # Future labels
    "future_hazard_binary",
    "future_hazard_3class",

    # Avoid scenario memorization
    "Scenario",
]

# ============================================================================
# Helpers
# ============================================================================

def optimize_memory(df):

    float_cols = df.select_dtypes(include=["float64"]).columns
    int_cols = df.select_dtypes(include=["int64"]).columns

    if len(float_cols):
        df[float_cols] = df[float_cols].astype("float32")

    if len(int_cols):
        df[int_cols] = df[int_cols].astype("int32")

    return df


def encode_categorical(df):

    if "zone" in df.columns:

        df = pd.get_dummies(
            df,
            columns=["zone"],
            drop_first=True
        )

    return df


def prepare_feature_columns(df):

    drop_cols = [
        c for c in DROP_COLUMNS
        if c in df.columns
    ]

    feature_df = df.drop(columns=drop_cols)

    return feature_df.columns.tolist()


# ============================================================================
# Future target generation
# ============================================================================

def create_future_targets(df):

    print("\nCreating future forecasting targets ...")

    df = df.sort_values(
        ["Scenario", "x", "y", "z", "Time"]
    ).reset_index(drop=True)

    group_cols = ["Scenario", "x", "y", "z"]

    grouped = df.groupby(group_cols, sort=False)

    df["future_hazard_binary"] = (
        grouped["hazard_binary"]
        .shift(-FORECAST_HORIZON_STEPS)
    )

    df["future_hazard_3class"] = (
        grouped["hazard_3class"]
        .shift(-FORECAST_HORIZON_STEPS)
    )

    before = len(df)

    df = df.dropna(
        subset=[
            "future_hazard_binary",
            "future_hazard_3class",
        ]
    ).reset_index(drop=True)

    after = len(df)

    print(
        f"  Removed incomplete future rows: "
        f"{before - after:,}"
    )

    df["future_hazard_binary"] = (
        df["future_hazard_binary"]
        .astype("int8")
    )

    df["future_hazard_3class"] = (
        df["future_hazard_3class"]
        .astype("int8")
    )

    return df


# ============================================================================
# Sequence generation
# ============================================================================

def build_sequences(df, feature_columns):

    print("\nBuilding temporal sequences ...")

    X_sequences = []

    y_binary = []

    y_multiclass = []

    group_cols = ["Scenario", "x", "y", "z"]

    grouped = df.groupby(group_cols, sort=False)

    total_groups = len(grouped)

    print(f"  Spatial cells: {total_groups:,}")

    for idx, (_, group) in enumerate(grouped):

        group = group.sort_values("Time")

        required = (
            SEQUENCE_LENGTH +
            FORECAST_HORIZON_STEPS
        )

        if len(group) < required:
            continue

        feature_matrix = (
            group[feature_columns]
            .values
            .astype(np.float32)
        )

        binary_targets = (
            group["future_hazard_binary"]
            .values
        )

        multiclass_targets = (
            group["future_hazard_3class"]
            .values
        )

        n = len(group)

        max_start = (
            n - SEQUENCE_LENGTH
        )

        for start in range(max_start):

            end = start + SEQUENCE_LENGTH

            seq_x = feature_matrix[start:end]

            target_idx = end - 1

            X_sequences.append(seq_x)

            y_binary.append(
                binary_targets[target_idx]
            )

            y_multiclass.append(
                multiclass_targets[target_idx]
            )

        if (idx + 1) % 5000 == 0:

            print(
                f"    Processed groups: "
                f"{idx + 1:,}/{total_groups:,}"
            )

    X_sequences = np.array(
        X_sequences,
        dtype=np.float32
    )

    y_binary = np.array(
        y_binary,
        dtype=np.int8
    )

    y_multiclass = np.array(
        y_multiclass,
        dtype=np.int8
    )

    print("\nSequence dataset created:")
    print(f"  X shape: {X_sequences.shape}")
    print(f"  Binary labels: {y_binary.shape}")
    print(f"  Multiclass labels: {y_multiclass.shape}")

    return (
        X_sequences,
        y_binary,
        y_multiclass,
    )


# ============================================================================
# Save arrays
# ============================================================================

def save_array(path, arr):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(path, arr)

    size_mb = (
        path.stat().st_size /
        (1024**2)
    )

    print(
        f"    Saved {path.name:<35}"
        f"{size_mb:8.1f} MB"
    )


# ============================================================================
# Main
# ============================================================================

def main(input_path):

    print("=" * 72)
    print("  GreenMining — Phase 6: LSTM Forecast Dataset")
    print("=" * 72)

    if not input_path.exists():

        print(f"\nERROR: Missing dataset:")
        print(f"  {input_path}")

        sys.exit(1)

    # ----------------------------------------------------------------------

    print("\nLoading engineered dataset ...")

    df = pd.read_csv(input_path)

    df = optimize_memory(df)

    print(f"  Rows loaded: {len(df):,}")

    # ----------------------------------------------------------------------
    # Future forecasting targets
    # ----------------------------------------------------------------------

    df = create_future_targets(df)

    # ----------------------------------------------------------------------
    # Encode categoricals
    # ----------------------------------------------------------------------

    df = encode_categorical(df)

    # ----------------------------------------------------------------------
    # Feature columns
    # ----------------------------------------------------------------------

    feature_columns = prepare_feature_columns(df)

    print("\nFeatures used:")

    for col in feature_columns:
        print(f"  {col}")

    # ----------------------------------------------------------------------
    # Split scenarios
    # ----------------------------------------------------------------------

    print("\nApplying scenario forecasting split ...")

    train_df = df[
        df["Scenario"].isin([1, 2, 3, 4])
    ]

    test_df = df[
        df["Scenario"] == 5
    ]

    print(f"  Train rows: {len(train_df):,}")
    print(f"  Test rows : {len(test_df):,}")

    # ----------------------------------------------------------------------
    # Build training sequences
    # ----------------------------------------------------------------------

    train_X_all, train_y_all, train_y_multi_all = (
        build_sequences(
            train_df,
            feature_columns
        )
    )

    # ----------------------------------------------------------------------
    # Build test sequences
    # ----------------------------------------------------------------------

    test_X, test_y, test_y_multi = (
        build_sequences(
            test_df,
            feature_columns
        )
    )

    # ----------------------------------------------------------------------
    # Validation split AFTER sequence generation
    # ----------------------------------------------------------------------

    print("\nCreating validation split ...")

    (
        train_X,
        val_X,
        train_y,
        val_y,
        train_y_multi,
        val_y_multi,
    ) = train_test_split(

        train_X_all,
        train_y_all,
        train_y_multi_all,

        test_size=VALIDATION_FRACTION,

        random_state=RANDOM_STATE,

        stratify=train_y_all,
    )

    print("\nFinal dataset shapes:")

    print(f"  Train X: {train_X.shape}")
    print(f"  Train y: {train_y.shape}")

    print(f"  Val X  : {val_X.shape}")
    print(f"  Val y  : {val_y.shape}")

    print(f"  Test X : {test_X.shape}")
    print(f"  Test y : {test_y.shape}")

    # ----------------------------------------------------------------------
    # Save binary forecasting dataset
    # ----------------------------------------------------------------------

    print("\nSaving binary forecasting dataset ...")

    save_array(
        OUTPUT_DIR / "train_X.npy",
        train_X
    )

    save_array(
        OUTPUT_DIR / "train_y.npy",
        train_y
    )

    save_array(
        OUTPUT_DIR / "val_X.npy",
        val_X
    )

    save_array(
        OUTPUT_DIR / "val_y.npy",
        val_y
    )

    save_array(
        OUTPUT_DIR / "test_X.npy",
        test_X
    )

    save_array(
        OUTPUT_DIR / "test_y.npy",
        test_y
    )

    # ----------------------------------------------------------------------
    # Save multiclass labels
    # ----------------------------------------------------------------------

    print("\nSaving multiclass forecasting labels ...")

    save_array(
        OUTPUT_DIR / "train_y_multiclass.npy",
        train_y_multi
    )

    save_array(
        OUTPUT_DIR / "val_y_multiclass.npy",
        val_y_multi
    )

    save_array(
        OUTPUT_DIR / "test_y_multiclass.npy",
        test_y_multi
    )

    # ----------------------------------------------------------------------
    # Metadata
    # ----------------------------------------------------------------------

    metadata = {

        "sequence_length":
            SEQUENCE_LENGTH,

        "forecast_horizon_steps":
            FORECAST_HORIZON_STEPS,

        "forecast_horizon_seconds":
            FORECAST_HORIZON_STEPS * TIME_DELTA,

        "feature_count":
            len(feature_columns),

        "feature_columns":
            feature_columns,

        "train_shape":
            list(train_X.shape),

        "val_shape":
            list(val_X.shape),

        "test_shape":
            list(test_X.shape),
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_DIR / "dataset_metadata.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(metadata, f, indent=4)

    with open(
        OUTPUT_DIR / "feature_columns.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(feature_columns, f, indent=4)

    # ----------------------------------------------------------------------

    print("\n" + "=" * 72)
    print("LSTM FORECASTING DATASET COMPLETE")
    print("=" * 72)

    print("\nSaved:")
    print("  data/lstm/")

    print("\nNext step:")
    print("  07_train_lstm_forecaster.py")

    print("\nDONE.\n")


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Phase 6 — Create LSTM Forecast Dataset"
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

    args = parser.parse_args()

    main(args.input)