"""
Phase 8 — Create Rich Spatiotemporal Graph Dataset
==================================================

Creates LOCALIZED spatiotemporal graph samples for ST-GNN training.

Dataset Design
--------------
Each graph sample contains:

Temporal sequence:
    [T, N, F]

Where:
    T = sequence length
    N = local graph nodes
    F = feature count

Goal:
    Forecast future hazard state.

Output
------
data/graph/
├── train_graphs.pt
├── val_graphs.pt
├── test_graphs.pt
├── graph_metadata.json
├── feature_columns.json
└── scaler.pkl
"""

import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import torch
from torch_geometric.data import Data

# ============================================================================
# Paths
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parent

OUTPUT_DIR = REPO_ROOT / "data" / "graph"

# ============================================================================
# Configuration
# ============================================================================

RANDOM_STATE = 42

SEQUENCE_LENGTH = 4

FORECAST_HORIZON_STEPS = 2

TIME_DELTA = 15.0

# Bigger neighborhood for richer spatial learning
K_NEIGHBORS = 24

# Actual graph node count
LOCAL_SUBGRAPH_SIZE = 25

# Larger scalable dataset
MAX_SAMPLES_PER_SCENARIO = 40000

MAX_CENTER_NODES = 8000

PRINT_EVERY = 5000

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

# ============================================================================
# Columns
# ============================================================================

DROP_COLUMNS = [

    "Risk",

    "hazard_binary",
    "hazard_3class",

    "future_hazard_binary",
    "future_hazard_3class",

    "Scenario",
]

# ============================================================================
# Helpers
# ============================================================================

def optimize_memory(df):

    float_cols = df.select_dtypes(
        include=["float64"]
    ).columns

    int_cols = df.select_dtypes(
        include=["int64"]
    ).columns

    if len(float_cols):

        df[float_cols] = (
            df[float_cols]
            .astype("float32")
        )

    if len(int_cols):

        df[int_cols] = (
            df[int_cols]
            .astype("int32")
        )

    return df

# ============================================================================
# Future targets
# ============================================================================

def create_future_targets(df):

    print("\nCreating future forecasting targets ...")

    df = df.sort_values(
        ["Scenario", "x", "y", "z", "Time"]
    ).reset_index(drop=True)

    grouped = df.groupby(
        ["Scenario", "x", "y", "z"],
        sort=False
    )

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
        f"  Removed incomplete rows: "
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
# Encode categoricals
# ============================================================================

def encode_categoricals(df):

    if "zone" in df.columns:

        df = pd.get_dummies(
            df,
            columns=["zone"],
            drop_first=True
        )

    return df

# ============================================================================
# Feature columns
# ============================================================================

def get_feature_columns(df):

    drop_cols = [
        c for c in DROP_COLUMNS
        if c in df.columns
    ]

    feature_df = df.drop(columns=drop_cols)

    return feature_df.columns.tolist()

# ============================================================================
# Normalize
# ============================================================================

def normalize_features(
    df,
    feature_columns,
):

    print("\nFitting feature scaler ...")

    train_df = df[
        df["Scenario"] != 5
    ]

    scaler = StandardScaler()

    scaler.fit(
        train_df[feature_columns]
    )

    df[feature_columns] = scaler.transform(
        df[feature_columns]
    )

    return df, scaler

# ============================================================================
# Spatial neighbors
# ============================================================================

def build_spatial_neighbors(cell_df):

    print("\nBuilding spatial neighborhoods ...")

    coords = cell_df[
        ["x", "y", "z"]
    ].values

    knn = NearestNeighbors(

        n_neighbors=K_NEIGHBORS + 1,

        algorithm="ball_tree"
    )

    knn.fit(coords)

    _, indices = knn.kneighbors(coords)

    print(f"  Cells: {len(coords):,}")

    return indices

# ============================================================================
# Build local graph edges
# ============================================================================

def build_local_edge_index(
    local_nodes,
    neighbor_indices,
):

    # ----------------------------------------------------------------------
    # LOCAL REMAPPING
    # ----------------------------------------------------------------------

    node_mapping = {

        old_id: new_id

        for new_id, old_id
        in enumerate(local_nodes)
    }

    edge_list = []

    # ----------------------------------------------------------------------
    # Build local-only edges
    # ----------------------------------------------------------------------

    for src_global in local_nodes:

        src_local = node_mapping[src_global]

        for dst_global in neighbor_indices[src_global]:

            if dst_global in node_mapping:

                dst_local = node_mapping[dst_global]

                edge_list.append([
                    src_local,
                    dst_local
                ])

    edge_index = np.array(
        edge_list,
        dtype=np.int64
    ).T

    # ----------------------------------------------------------------------
    # Validation
    # ----------------------------------------------------------------------

    if edge_index.size == 0:
        return None

    assert edge_index.max() < len(local_nodes)
    assert edge_index.min() >= 0

    edge_index = torch.tensor(
        edge_index,
        dtype=torch.long
    )

    return edge_index

# ============================================================================
# Build graph samples
# ============================================================================

def build_graph_samples(
    df,
    feature_columns,
    neighbor_indices,
):

    print(
        "\nBuilding local temporal graph samples ..."
    )

    graph_samples = []

    # ----------------------------------------------------------------------
    # Unique CFD cells
    # ----------------------------------------------------------------------

    cell_df = (
        df[
            ["x", "y", "z"]
        ]
        .drop_duplicates()
        .sort_values(["x", "y", "z"])
        .reset_index(drop=True)
    )

    scenarios = sorted(
        df["Scenario"].unique()
    )

    # ----------------------------------------------------------------------
    # Random center nodes
    # ----------------------------------------------------------------------

    selected_centers = np.random.choice(

        len(cell_df),

        size=min(
            MAX_CENTER_NODES,
            len(cell_df)
        ),

        replace=False,
    )

    print(
        f"\nUsing {len(selected_centers):,} "
        f"center nodes"
    )

    # ----------------------------------------------------------------------

    for scenario in scenarios:

        print(f"\nScenario {scenario}")

        sdf = df[
            df["Scenario"] == scenario
        ].copy()

        time_steps = sorted(
            sdf["Time"].unique()
        )

        scenario_samples = 0

        stop_scenario = False

        # ------------------------------------------------------------------

        for center_idx in selected_centers:

            if stop_scenario:
                break

            # --------------------------------------------------------------
            # LOCAL NEIGHBORHOOD
            # --------------------------------------------------------------

            local_nodes = np.concatenate([

                [center_idx],

                neighbor_indices[
                    center_idx
                ][:K_NEIGHBORS]

            ])

            local_nodes = np.unique(
                local_nodes
            )

            # --------------------------------------------------------------
            # LOCAL GRAPH
            # --------------------------------------------------------------

            edge_index = build_local_edge_index(

                local_nodes,

                neighbor_indices,
            )

            if edge_index is None:
                continue

            # --------------------------------------------------------------
            # Local cell coordinates
            # --------------------------------------------------------------

            local_cells = cell_df.iloc[
                local_nodes
            ]

            # --------------------------------------------------------------
            # Temporal windows
            # --------------------------------------------------------------

            for start_idx in range(

                len(time_steps)
                - SEQUENCE_LENGTH
                - FORECAST_HORIZON_STEPS
                + 1
            ):

                sequence_times = time_steps[
                    start_idx:
                    start_idx + SEQUENCE_LENGTH
                ]

                target_time = time_steps[
                    start_idx
                    + SEQUENCE_LENGTH
                    + FORECAST_HORIZON_STEPS
                    - 1
                ]

                x_sequence = []

                valid = True

                # ----------------------------------------------------------
                # Temporal sequence
                # ----------------------------------------------------------

                for t in sequence_times:

                    tdf = sdf[
                        sdf["Time"] == t
                    ]

                    merged = local_cells.merge(

                        tdf,

                        on=["x", "y", "z"],

                        how="left"
                    )

                    if merged.isnull().any().any():

                        valid = False
                        break

                    x = merged[
                        feature_columns
                    ].values.astype(np.float32)

                    x_sequence.append(x)

                if not valid:
                    continue

                # ----------------------------------------------------------
                # Future targets
                # ----------------------------------------------------------

                target_df = sdf[
                    sdf["Time"] == target_time
                ]

                target_merge = local_cells.merge(

                    target_df,

                    on=["x", "y", "z"],

                    how="left"
                )

                y_binary = target_merge[
                    "future_hazard_binary"
                ].values.astype(np.int8)

                y_multi = target_merge[
                    "future_hazard_3class"
                ].values.astype(np.int8)

                # ----------------------------------------------------------
                # Create graph sample
                # ----------------------------------------------------------

                graph = Data(

                    x=torch.tensor(
                        np.stack(
                            x_sequence,
                            axis=0
                        ),
                        dtype=torch.float32
                    ),

                    edge_index=edge_index,

                    y=torch.tensor(
                        y_binary,
                        dtype=torch.float32
                    )
                )

                graph.scenario = int(scenario)

                graph.num_nodes = len(local_nodes)

                graph.center_node = 0

                graph.multiclass_target = torch.tensor(
                    y_multi,
                    dtype=torch.long
                )

                graph_samples.append(graph)

                scenario_samples += 1

                # ----------------------------------------------------------
                # Progress logging
                # ----------------------------------------------------------

                if scenario_samples % PRINT_EVERY == 0:

                    print(
                        f"  Samples: "
                        f"{scenario_samples:,}"
                    )

                # ----------------------------------------------------------
                # Scenario limit
                # ----------------------------------------------------------

                if (
                    scenario_samples
                    >= MAX_SAMPLES_PER_SCENARIO
                ):

                    print(
                        f"  Reached scenario limit "
                        f"({scenario_samples:,})"
                    )

                    stop_scenario = True
                    break

        print(
            f"  Final scenario samples: "
            f"{scenario_samples:,}"
        )

    return graph_samples

# ============================================================================
# Split
# ============================================================================

def split_graphs(graph_samples):

    print("\nSplitting graph datasets ...")

    train_graphs = []
    val_graphs = []
    test_graphs = []

    for graph in graph_samples:

        scenario = graph.scenario

        if scenario == 5:

            test_graphs.append(graph)

        elif scenario == 4:

            val_graphs.append(graph)

        else:

            train_graphs.append(graph)

    print(f"  Train graphs: {len(train_graphs):,}")
    print(f"  Val graphs  : {len(val_graphs):,}")
    print(f"  Test graphs : {len(test_graphs):,}")

    return (
        train_graphs,
        val_graphs,
        test_graphs,
    )

# ============================================================================
# Main
# ============================================================================

def main(input_path):

    print("=" * 72)
    print(
        "  GreenMining — Phase 8: ST-GNN Graph Dataset"
    )
    print("=" * 72)

    print("\nLoading engineered dataset ...")

    df = pd.read_csv(input_path)

    df = optimize_memory(df)

    print(f"  Rows loaded: {len(df):,}")

    # ----------------------------------------------------------------------
    # Targets
    # ----------------------------------------------------------------------

    df = create_future_targets(df)

    # ----------------------------------------------------------------------
    # Categoricals
    # ----------------------------------------------------------------------

    df = encode_categoricals(df)

    # ----------------------------------------------------------------------
    # Features
    # ----------------------------------------------------------------------

    feature_columns = get_feature_columns(df)

    print(
        f"\nFeature count: "
        f"{len(feature_columns)}"
    )

    # ----------------------------------------------------------------------
    # Normalize
    # ----------------------------------------------------------------------

    df, scaler = normalize_features(
        df,
        feature_columns,
    )

    # ----------------------------------------------------------------------
    # Spatial cells
    # ----------------------------------------------------------------------

    cell_df = (
        df[
            ["x", "y", "z"]
        ]
        .drop_duplicates()
        .sort_values(["x", "y", "z"])
        .reset_index(drop=True)
    )

    neighbor_indices = build_spatial_neighbors(
        cell_df
    )

    # ----------------------------------------------------------------------
    # Graph samples
    # ----------------------------------------------------------------------

    graph_samples = build_graph_samples(

        df,

        feature_columns,

        neighbor_indices,
    )

    # ----------------------------------------------------------------------
    # Split
    # ----------------------------------------------------------------------

    (
        train_graphs,
        val_graphs,
        test_graphs,
    ) = split_graphs(graph_samples)

    # ----------------------------------------------------------------------
    # Save
    # ----------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nSaving graph datasets ...")

    torch.save(

        train_graphs,

        OUTPUT_DIR / "train_graphs.pt"
    )

    torch.save(

        val_graphs,

        OUTPUT_DIR / "val_graphs.pt"
    )

    torch.save(

        test_graphs,

        OUTPUT_DIR / "test_graphs.pt"
    )

    joblib.dump(

        scaler,

        OUTPUT_DIR / "scaler.pkl"
    )

    with open(

        OUTPUT_DIR / "feature_columns.json",

        "w",

        encoding="utf-8"
    ) as f:

        json.dump(
            feature_columns,
            f,
            indent=4
        )

    metadata = {

        "sequence_length":
            SEQUENCE_LENGTH,

        "forecast_horizon_steps":
            FORECAST_HORIZON_STEPS,

        "forecast_horizon_seconds":
            FORECAST_HORIZON_STEPS
            * TIME_DELTA,

        "k_neighbors":
            K_NEIGHBORS,

        "local_subgraph_size":
            LOCAL_SUBGRAPH_SIZE,

        "max_samples_per_scenario":
            MAX_SAMPLES_PER_SCENARIO,

        "max_center_nodes":
            MAX_CENTER_NODES,

        "feature_count":
            len(feature_columns),

        "train_graphs":
            len(train_graphs),

        "val_graphs":
            len(val_graphs),

        "test_graphs":
            len(test_graphs),
    }

    with open(

        OUTPUT_DIR / "graph_metadata.json",

        "w",

        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )

    # ----------------------------------------------------------------------

    print("\n" + "=" * 72)
    print("ST-GNN GRAPH DATASET COMPLETE")
    print("=" * 72)

    print("\nSaved:")
    print("  data/graph/")

    print("\nFinal dataset sizes:")

    print(f"  Train graphs: {len(train_graphs):,}")
    print(f"  Val graphs  : {len(val_graphs):,}")
    print(f"  Test graphs : {len(test_graphs):,}")

    print("\nNext step:")
    print("  09_train_stgnn.py")

    print("\nDONE.\n")

# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Phase 8 — ST-GNN Dataset"
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=(
            REPO_ROOT
            / "data"
            / "processed"
            / "greenmining_features.csv"
        ),
    )

    args = parser.parse_args()

    main(args.input)