"""
Fast resplit utility for existing ST-GNN graphs.

Purpose
-------
Reuse the already-built `data/graph/train_graphs.pt` and carve out
validation/test subsets without regenerating all graphs from the raw CFD data.

Important limitation
--------------------
This script cannot reconstruct the original temporal split because the saved
graph objects do not store the source timestep/window identifiers. It creates
deterministic holdout splits from the existing training graphs instead.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parent
GRAPH_DIR = REPO_ROOT / "data" / "graph"

RANDOM_STATE = 42


def split_indices(count, train_frac, val_frac, rng):
    indices = np.arange(count)
    rng.shuffle(indices)

    n_train = int(count * train_frac)
    n_val = int(count * val_frac)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return train_idx, val_idx, test_idx


def main(train_frac, val_frac):
    if train_frac <= 0 or val_frac <= 0 or train_frac + val_frac >= 1:
        raise ValueError("Require 0 < train_frac, val_frac and train_frac + val_frac < 1")

    train_path = GRAPH_DIR / "train_graphs.pt"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing graph file: {train_path}")

    print("=" * 72)
    print("  GreenMining - Fast Graph Resplit")
    print("=" * 72)

    print(f"\nLoading {train_path} ...")
    graphs = torch.load(train_path, weights_only=False)
    print(f"  Loaded graphs: {len(graphs):,}")

    by_scenario = {}
    for graph in graphs:
        scenario = int(getattr(graph, "scenario", -1))
        by_scenario.setdefault(scenario, []).append(graph)

    rng = np.random.default_rng(RANDOM_STATE)

    new_train = []
    new_val = []
    new_test = []
    per_scenario_counts = {}

    print("\nResplitting within each scenario ...")

    for scenario in sorted(by_scenario):
        scenario_graphs = by_scenario[scenario]
        train_idx, val_idx, test_idx = split_indices(
            len(scenario_graphs),
            train_frac=train_frac,
            val_frac=val_frac,
            rng=rng,
        )

        new_train.extend(scenario_graphs[i] for i in train_idx)
        new_val.extend(scenario_graphs[i] for i in val_idx)
        new_test.extend(scenario_graphs[i] for i in test_idx)

        per_scenario_counts[str(scenario)] = {
            "train": len(train_idx),
            "val": len(val_idx),
            "test": len(test_idx),
        }

        print(
            f"  Scenario {scenario}: "
            f"{len(train_idx):,} train | {len(val_idx):,} val | {len(test_idx):,} test"
        )

    print("\nSaving resplit graph files ...")
    torch.save(new_train, GRAPH_DIR / "train_graphs.pt")
    torch.save(new_val, GRAPH_DIR / "val_graphs.pt")
    torch.save(new_test, GRAPH_DIR / "test_graphs.pt")

    metadata_path = GRAPH_DIR / "graph_metadata.json"
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    metadata.update({
        "train_graphs": len(new_train),
        "val_graphs": len(new_val),
        "test_graphs": len(new_test),
        "split_strategy": "posthoc_random_within_existing_train_graphs",
        "posthoc_resplit": True,
        "posthoc_resplit_note": (
            "Validation/test graphs were carved from the original train_graphs.pt "
            "because temporal window metadata was not stored in the saved graphs."
        ),
        "posthoc_per_scenario_counts": per_scenario_counts,
        "posthoc_random_state": RANDOM_STATE,
    })
    metadata_path.write_text(json.dumps(metadata, indent=4), encoding="utf-8")

    print("\n" + "=" * 72)
    print("FAST RESPLIT COMPLETE")
    print("=" * 72)
    print(
        f"  Train: {len(new_train):,}  "
        f"Val: {len(new_val):,}  "
        f"Test: {len(new_test):,}"
    )
    print("\nDone.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast resplit existing ST-GNN graphs")
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    args = parser.parse_args()
    main(args.train_frac, args.val_frac)
