"""
Phase 8 — Create Rich Spatiotemporal Graph Dataset (v2)
========================================================

Key improvements over v1
-------------------------
1. Time-based split within EACH scenario (70 / 15 / 15):
   - v1 split by scenario: 1-3 train, 4 val, 5 test.
     Every scenario is a different CFD sim → model never saw val/test spatial
     configurations → hard domain shift from epoch 1.
   - v2: all 5 scenarios contribute to every split, separated by time.
     Model sees all spatial configurations in training; only extrapolates
     in time, which is the actual forecasting task.

2. Relative spatial encoding appended to every node:
   (dx, dy, dz, distance) relative to the subgraph centre node.
   Gives the model explicit spatial context without leaking absolute coords.

3. Hazard-balanced centre selection:
   50 % centre nodes sampled from cells that are ever hazardous.
   Prevents training graphs from being dominated by all-SAFE subgraphs.

4. Time dropped from node features:
   Absolute simulation time causes distribution shift (train = early times,
   val/test = later times). Removed so the model reads the physical state,
   not the clock.

5. No normalisation here — 09_train_stgnn.py normalises using training
   graph statistics so there is no double-normalisation and no scaler
   fitted on val/test data.

6. Longer sequence: T = 6 timesteps (90 s of history vs 60 s before).

Outputs (same paths — pipeline unchanged)
------------------------------------------
data/graph/
├── train_graphs.pt
├── val_graphs.pt
├── test_graphs.pt
├── graph_metadata.json
└── feature_columns.json
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors

import torch
from torch_geometric.data import Data

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parent
OUTPUT_DIR = REPO_ROOT / "data" / "graph"

# ── Configuration ─────────────────────────────────────────────────────────────
RANDOM_STATE = 42

SEQUENCE_LENGTH        = 6    # timesteps of history (was 4; 90 s at 15 s/step)
FORECAST_HORIZON_STEPS = 2    # predict 2 steps ahead (30 s)
TIME_DELTA             = 15.0 # seconds per step

K_NEIGHBORS        = 24   # neighbours for KNN graph
MAX_CENTER_NODES   = 8000 # unique centre cells to sample

# Per-scenario sample caps
MAX_TRAIN_PER_SCENARIO   = 40_000
MAX_VALTEST_PER_SCENARIO = 8_000

TRAIN_FRAC = 0.70   # first 70 % of each scenario's time → train
VAL_FRAC   = 0.85   # next 15 % → val, last 15 % → test

PRINT_EVERY = 5_000

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

# ── Columns ───────────────────────────────────────────────────────────────────
# Time removed: causes distribution shift (train=early times, val/test=late)
DROP_COLUMNS = [
    "Risk",
    "hazard_binary",
    "hazard_3class",
    "future_hazard_binary",
    "future_hazard_3class",
    "Scenario",
    "Time",        # absolute sim time — leaks temporal position, causes shift
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def optimize_memory(df):
    for col in df.select_dtypes("float64").columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes("int64").columns:
        df[col] = df[col].astype("int32")
    return df


# ── Future targets ────────────────────────────────────────────────────────────

def create_future_targets(df):
    print("\nCreating future forecasting targets ...")
    df = df.sort_values(["Scenario", "x", "y", "z", "Time"]).reset_index(drop=True)

    grouped = df.groupby(["Scenario", "x", "y", "z"], sort=False)
    df["future_hazard_binary"] = grouped["hazard_binary"].shift(-FORECAST_HORIZON_STEPS)
    df["future_hazard_3class"] = grouped["hazard_3class"].shift(-FORECAST_HORIZON_STEPS)

    before = len(df)
    df = df.dropna(subset=["future_hazard_binary", "future_hazard_3class"]).reset_index(drop=True)
    print(f"  Dropped {before - len(df):,} rows with missing future targets")

    df["future_hazard_binary"] = df["future_hazard_binary"].astype("int8")
    df["future_hazard_3class"] = df["future_hazard_3class"].astype("int8")
    return df


# ── Categoricals ──────────────────────────────────────────────────────────────

def encode_categoricals(df):
    if "zone" in df.columns:
        df = pd.get_dummies(df, columns=["zone"], drop_first=True)
    return df


# ── Feature columns ───────────────────────────────────────────────────────────

def get_feature_columns(df):
    drop = [c for c in DROP_COLUMNS if c in df.columns]
    return df.drop(columns=drop).columns.tolist()


# ── Per-scenario time splits ──────────────────────────────────────────────────

def get_scenario_time_splits(df):
    """Return {scenario: {train: set, val: set, test: set}} split by time."""
    print("\nComputing per-scenario time splits ...")
    splits = {}
    for scenario in sorted(df["Scenario"].unique()):
        times = sorted(df[df["Scenario"] == scenario]["Time"].unique())
        n        = len(times)
        n_train  = int(TRAIN_FRAC * n)
        n_val    = int(VAL_FRAC   * n)
        splits[int(scenario)] = {
            "train": set(times[:n_train]),
            "val":   set(times[n_train:n_val]),
            "test":  set(times[n_val:]),
        }
        print(
            f"  Scenario {scenario}: "
            f"{n_train} train | {n_val-n_train} val | {n-n_val} test timesteps"
        )
    return splits


# ── Spatial neighbours ────────────────────────────────────────────────────────

def build_spatial_neighbors(cell_df):
    print("\nBuilding KNN spatial neighbourhood ...")
    coords = cell_df[["x", "y", "z"]].values
    knn = NearestNeighbors(n_neighbors=K_NEIGHBORS + 1, algorithm="ball_tree")
    knn.fit(coords)
    _, indices = knn.kneighbors(coords)
    print(f"  Unique cells: {len(coords):,}")
    return indices


# ── Local edge index ──────────────────────────────────────────────────────────

def build_local_edge_index(local_nodes, neighbor_indices):
    node_map = {old: new for new, old in enumerate(local_nodes)}
    edges = []
    for src_g in local_nodes:
        src_l = node_map[src_g]
        for dst_g in neighbor_indices[src_g]:
            if dst_g in node_map:
                edges.append([src_l, node_map[dst_g]])
    if not edges:
        return None
    ei = np.array(edges, dtype=np.int64).T
    assert ei.max() < len(local_nodes) and ei.min() >= 0
    return torch.tensor(ei, dtype=torch.long)


# ── Hazard-balanced centre selection ─────────────────────────────────────────

def select_balanced_centers(df, cell_df, n_centers):
    """Sample 50 % from ever-hazardous cells, 50 % from always-safe cells."""
    print("\nSelecting hazard-balanced centre nodes ...")

    hazard_coords = set(
        df[df["future_hazard_binary"] == 1][["x", "y", "z"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    hazard_idx, safe_idx = [], []
    for i, row in enumerate(cell_df[["x", "y", "z"]].itertuples(index=False, name=None)):
        (hazard_idx if row in hazard_coords else safe_idx).append(i)

    n_h = min(n_centers // 2, len(hazard_idx))
    n_s = min(n_centers - n_h, len(safe_idx))

    h_sel = np.random.choice(hazard_idx, n_h, replace=False) if n_h else np.array([], int)
    s_sel = np.random.choice(safe_idx,   n_s, replace=False) if n_s else np.array([], int)

    centers = np.concatenate([h_sel, s_sel]).astype(int)
    np.random.shuffle(centers)
    print(f"  Hazardous centres: {n_h:,}  Safe centres: {n_s:,}  Total: {len(centers):,}")
    return centers


# ── Graph sample builder ──────────────────────────────────────────────────────

def build_split_graphs(
    df,
    feature_columns,
    neighbor_indices,
    cell_df,
    selected_centers,
    split_times_per_scenario,
    split_name,
    max_per_scenario,
):
    """Build graph samples for one split (train / val / test)."""
    print(f"\nBuilding {split_name} graphs ...")
    all_graphs = []

    for scenario in sorted(df["Scenario"].unique()):
        sdf        = df[df["Scenario"] == scenario]
        times_set  = split_times_per_scenario[int(scenario)]
        time_steps = sorted(times_set)

        if len(time_steps) < SEQUENCE_LENGTH + FORECAST_HORIZON_STEPS:
            print(f"  Scenario {scenario}: not enough timesteps — skipping")
            continue

        n_samples = 0

        for center_idx in selected_centers:
            if n_samples >= max_per_scenario:
                break

            # ── Local subgraph ────────────────────────────────────────────────
            local_nodes = np.unique(np.concatenate([
                [center_idx],
                neighbor_indices[center_idx][:K_NEIGHBORS],
            ]))

            edge_index = build_local_edge_index(local_nodes, neighbor_indices)
            if edge_index is None:
                continue

            local_cells = cell_df.iloc[local_nodes]

            # ── Relative spatial encoding (constant per subgraph) ─────────────
            centre_xyz  = cell_df.iloc[center_idx][["x", "y", "z"]].values.astype(np.float32)
            node_xyz    = local_cells[["x", "y", "z"]].values.astype(np.float32)
            rel_xyz     = node_xyz - centre_xyz                          # [N, 3]
            rel_dist    = np.linalg.norm(rel_xyz, axis=1, keepdims=True) # [N, 1]
            spatial_enc = np.hstack([rel_xyz, rel_dist])                 # [N, 4]

            # ── Time windows ──────────────────────────────────────────────────
            n_windows = len(time_steps) - SEQUENCE_LENGTH - FORECAST_HORIZON_STEPS + 1
            for start_idx in range(n_windows):
                if n_samples >= max_per_scenario:
                    break

                seq_times   = time_steps[start_idx : start_idx + SEQUENCE_LENGTH]
                target_time = time_steps[start_idx + SEQUENCE_LENGTH + FORECAST_HORIZON_STEPS - 1]

                # ── Sequence features ─────────────────────────────────────────
                x_seq  = []
                valid  = True

                for t in seq_times:
                    tdf    = sdf[sdf["Time"] == t]
                    merged = local_cells.merge(tdf, on=["x", "y", "z"], how="left")
                    if merged.isnull().any().any():
                        valid = False
                        break
                    feats = merged[feature_columns].values.astype(np.float32)
                    feats = np.hstack([feats, spatial_enc])   # [N, F+4]
                    x_seq.append(feats)

                if not valid:
                    continue

                # ── Target ────────────────────────────────────────────────────
                t_df   = sdf[sdf["Time"] == target_time]
                t_mrg  = local_cells.merge(t_df, on=["x", "y", "z"], how="left")
                if t_mrg.isnull().any().any():
                    continue

                y_bin   = t_mrg["future_hazard_binary"].values.astype(np.int8)
                y_multi = t_mrg["future_hazard_3class"].values.astype(np.int8)

                # ── Graph object ──────────────────────────────────────────────
                graph = Data(
                    x          = torch.tensor(np.stack(x_seq, axis=0), dtype=torch.float32),
                    edge_index = edge_index,
                    y          = torch.tensor(y_bin, dtype=torch.float32),
                )
                graph.scenario          = int(scenario)
                graph.num_nodes         = len(local_nodes)
                graph.center_node       = 0
                graph.multiclass_target = torch.tensor(y_multi, dtype=torch.long)

                all_graphs.append(graph)
                n_samples += 1

                if n_samples % PRINT_EVERY == 0:
                    print(f"  Scenario {scenario}: {n_samples:,} samples")

        print(f"  Scenario {scenario} → {n_samples:,} {split_name} samples")

    print(f"  Total {split_name}: {len(all_graphs):,} graphs")
    return all_graphs


# ── Main ──────────────────────────────────────────────────────────────────────

def main(input_path):
    print("=" * 72)
    print("  GreenMining — Phase 8 v2: ST-GNN Graph Dataset")
    print("=" * 72)

    print(f"\nLoading {input_path} ...")
    df = pd.read_csv(input_path)
    df = optimize_memory(df)
    print(f"  Rows: {len(df):,}")

    # ── Future targets ────────────────────────────────────────────────────────
    df = create_future_targets(df)
    df = encode_categoricals(df)

    feature_columns = get_feature_columns(df)
    # +4 relative spatial dims will be appended per node during graph building
    total_features = len(feature_columns) + 4
    print(f"\nBase feature columns: {len(feature_columns)}  (+4 spatial → {total_features} total)")

    # ── Time-based splits ─────────────────────────────────────────────────────
    time_splits = get_scenario_time_splits(df)

    # ── Spatial structure ─────────────────────────────────────────────────────
    cell_df = (
        df[["x", "y", "z"]]
        .drop_duplicates()
        .sort_values(["x", "y", "z"])
        .reset_index(drop=True)
    )
    neighbor_indices = build_spatial_neighbors(cell_df)

    # ── Hazard-balanced centre nodes ──────────────────────────────────────────
    selected_centers = select_balanced_centers(df, cell_df, MAX_CENTER_NODES)

    # ── Build each split ──────────────────────────────────────────────────────
    train_graphs = build_split_graphs(
        df, feature_columns, neighbor_indices, cell_df, selected_centers,
        {s: time_splits[s]["train"] for s in time_splits},
        "train", MAX_TRAIN_PER_SCENARIO,
    )
    val_graphs = build_split_graphs(
        df, feature_columns, neighbor_indices, cell_df, selected_centers,
        {s: time_splits[s]["val"] for s in time_splits},
        "val", MAX_VALTEST_PER_SCENARIO,
    )
    test_graphs = build_split_graphs(
        df, feature_columns, neighbor_indices, cell_df, selected_centers,
        {s: time_splits[s]["test"] for s in time_splits},
        "test", MAX_VALTEST_PER_SCENARIO,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nSaving ...")
    torch.save(train_graphs, OUTPUT_DIR / "train_graphs.pt")
    torch.save(val_graphs,   OUTPUT_DIR / "val_graphs.pt")
    torch.save(test_graphs,  OUTPUT_DIR / "test_graphs.pt")

    with open(OUTPUT_DIR / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_columns, f, indent=4)

    metadata = {
        "sequence_length":            SEQUENCE_LENGTH,
        "forecast_horizon_steps":     FORECAST_HORIZON_STEPS,
        "forecast_horizon_seconds":   FORECAST_HORIZON_STEPS * TIME_DELTA,
        "k_neighbors":                K_NEIGHBORS,
        "base_feature_count":         len(feature_columns),
        "total_feature_count":        total_features,
        "spatial_encoding_dims":      4,
        "split_strategy":             "time_based_within_scenario_70_15_15",
        "centre_selection":           "hazard_balanced_50_50",
        "normalisation":              "none_handled_by_training_script",
        "train_graphs":               len(train_graphs),
        "val_graphs":                 len(val_graphs),
        "test_graphs":                len(test_graphs),
    }
    with open(OUTPUT_DIR / "graph_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("ST-GNN GRAPH DATASET v2 COMPLETE")
    print("=" * 72)
    print(f"  Train: {len(train_graphs):,}  Val: {len(val_graphs):,}  Test: {len(test_graphs):,}")
    print(f"  Node features: {total_features}  (T={SEQUENCE_LENGTH} timesteps)")
    print(f"  Split: time-based 70/15/15 within each of 5 scenarios")
    print("\nNext step: python 09_train_stgnn.py")
    print("\nDONE.\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 8 v2 — ST-GNN Dataset")
    parser.add_argument(
        "--input", type=Path,
        default=REPO_ROOT / "data" / "processed" / "greenmining_features.csv",
    )
    args = parser.parse_args()
    main(args.input)
