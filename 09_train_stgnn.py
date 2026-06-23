"""
Phase 9 — Optimized Batched ST-GNN Training
===========================================

GPU-optimized spatiotemporal graph neural network
for underground mine hazard forecasting.

Architecture
------------
GCN -> GCN -> GRU -> Hazard Forecast

Optimized Features
------------------
- Fully batched graph processing
- GPU-efficient PyG DataLoader
- Vectorized temporal learning
- No Python graph loops
- CUDA accelerated
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np

# ============================================================================
# PyTorch
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# PyG
# ============================================================================

from torch_geometric.nn import GCNConv

from torch_geometric.loader import DataLoader

# ============================================================================
# Metrics
# ============================================================================

from sklearn.metrics import (

    accuracy_score,

    classification_report,

    confusion_matrix,

    roc_auc_score,
)

# ============================================================================
# Matplotlib
# ============================================================================

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

# ============================================================================
# Paths
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parent

GRAPH_DIR = REPO_ROOT / "data" / "graph"

MODEL_DIR = REPO_ROOT / "models_stgnn"

REPORT_DIR = REPO_ROOT / "reports_stgnn"

PLOT_DIR = REPORT_DIR / "plots"

METRIC_DIR = REPORT_DIR / "metrics"

# ============================================================================
# Config
# ============================================================================

RANDOM_STATE = 42

BATCH_SIZE = 128

LEARNING_RATE = 1e-3

EPOCHS = 30

PATIENCE = 6

GRAPH_HIDDEN = 64

TEMPORAL_HIDDEN = 64

DROPOUT = 0.2

GRAD_CLIP = 1.0

# ============================================================================
# Reproducibility
# ============================================================================

random.seed(RANDOM_STATE)

np.random.seed(RANDOM_STATE)

torch.manual_seed(RANDOM_STATE)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(RANDOM_STATE)

# ============================================================================
# Device
# ============================================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# ============================================================================
# Fast ST-GNN
# ============================================================================

class FastSTGNN(nn.Module):

    def __init__(

        self,

        input_dim,

        graph_hidden=64,

        temporal_hidden=64,

        dropout=0.2,
    ):

        super().__init__()

        # --------------------------------------------------------------
        # Spatial Graph Layers
        # --------------------------------------------------------------

        self.gcn1 = GCNConv(
            input_dim,
            graph_hidden
        )

        self.gcn2 = GCNConv(
            graph_hidden,
            graph_hidden
        )

        # --------------------------------------------------------------
        # Temporal GRU
        # --------------------------------------------------------------

        self.gru = nn.GRU(

            input_size=graph_hidden,

            hidden_size=temporal_hidden,

            batch_first=True,
        )

        # --------------------------------------------------------------
        # Forecast Head
        # --------------------------------------------------------------

        self.fc = nn.Sequential(

            nn.Linear(
                temporal_hidden,
                64,
            ),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(
                64,
                1,
            )
        )

        self.dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------

    def forward(self, data):

        x = data.x

        edge_index = data.edge_index

        # --------------------------------------------------------------
        # x shape:
        # [N_total, T, F]
        # --------------------------------------------------------------

        seq_len = x.shape[1]

        temporal_embeddings = []

        # --------------------------------------------------------------
        # Vectorized spatial processing
        # --------------------------------------------------------------

        for t in range(seq_len):

            x_t = x[:, t, :]

            x_t = self.gcn1(
                x_t,
                edge_index
            )

            x_t = F.relu(x_t)

            x_t = self.dropout(x_t)

            x_t = self.gcn2(
                x_t,
                edge_index
            )

            x_t = F.relu(x_t)

            temporal_embeddings.append(x_t)

        # --------------------------------------------------------------
        # [N_total, T, F]
        # --------------------------------------------------------------

        temporal_embeddings = torch.stack(
            temporal_embeddings,
            dim=1
        )

        # --------------------------------------------------------------
        # Temporal learning
        # --------------------------------------------------------------

        gru_out, _ = self.gru(
            temporal_embeddings
        )

        final_hidden = gru_out[:, -1]

        out = self.fc(
            final_hidden
        )

        return out.squeeze(1)

# ============================================================================
# Train
# ============================================================================

def train_epoch(

    model,

    loader,

    optimizer,

    criterion,
):

    model.train()

    running_loss = 0.0

    for batch in loader:

        batch = batch.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(batch)

        y = batch.y.float()

        loss = criterion(
            outputs,
            y
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP
        )

        optimizer.step()

        running_loss += (
            loss.item()
            * batch.num_graphs
        )

    return (
        running_loss
        / len(loader.dataset)
    )

# ============================================================================
# Validation
# ============================================================================

def evaluate_loss(

    model,

    loader,

    criterion,
):

    model.eval()

    running_loss = 0.0

    with torch.no_grad():

        for batch in loader:

            batch = batch.to(DEVICE)

            outputs = model(batch)

            y = batch.y.float()

            loss = criterion(
                outputs,
                y
            )

            running_loss += (
                loss.item()
                * batch.num_graphs
            )

    return (
        running_loss
        / len(loader.dataset)
    )

# ============================================================================
# Evaluation
# ============================================================================

def evaluate_model(

    model,

    loader,
):

    model.eval()

    y_true = []

    y_prob = []

    with torch.no_grad():

        for batch in loader:

            batch = batch.to(DEVICE)

            outputs = model(batch)

            probs = torch.sigmoid(
                outputs
            )

            y_true.extend(
                batch.y.cpu()
                .numpy()
                .flatten()
            )

            y_prob.extend(
                probs.cpu()
                .numpy()
                .flatten()
            )

    y_true = np.array(y_true)

    y_prob = np.array(y_prob)

    y_pred = (
        y_prob >= 0.5
    ).astype(int)

    report = classification_report(

        y_true,

        y_pred,

        output_dict=True,

        zero_division=0,
    )

    metrics = {

        "accuracy":
            accuracy_score(
                y_true,
                y_pred,
            ),

        "precision_macro":
            report["macro avg"]["precision"],

        "recall_macro":
            report["macro avg"]["recall"],

        "f1_macro":
            report["macro avg"]["f1-score"],

        "class_0_recall":
            report.get(
                "0",
                {}
            ).get(
                "recall",
                0.0
            ),

        "class_0_precision":
            report.get(
                "0",
                {}
            ).get(
                "precision",
                0.0
            ),

        "class_1_recall":
            report.get(
                "1",
                {}
            ).get(
                "recall",
                0.0
            ),

        "class_1_precision":
            report.get(
                "1",
                {}
            ).get(
                "precision",
                0.0
            ),

        "roc_auc":
            roc_auc_score(
                y_true,
                y_prob,
            ),
    }

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    return metrics, cm

# ============================================================================
# Main
# ============================================================================

def main():

    print("=" * 72)
    print(
        "  GreenMining — Optimized ST-GNN"
    )
    print("=" * 72)

    print(f"\nDevice: {DEVICE}")

    if DEVICE.type == "cuda":

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # ----------------------------------------------------------------------
    # Load datasets
    # ----------------------------------------------------------------------

    print("\nLoading graph datasets ...")

    train_graphs = torch.load(
        GRAPH_DIR / "train_graphs.pt"
    )

    val_graphs = torch.load(
        GRAPH_DIR / "val_graphs.pt"
    )

    test_graphs = torch.load(
        GRAPH_DIR / "test_graphs.pt"
    )

    # ----------------------------------------------------------------------
    # OPTIONAL DEBUG SUBSET
    # ----------------------------------------------------------------------

    # train_graphs = train_graphs[:5000]
    # val_graphs = val_graphs[:1000]
    # test_graphs = test_graphs[:1000]

    print(
        f"  Train graphs: "
        f"{len(train_graphs):,}"
    )

    print(
        f"  Val graphs  : "
        f"{len(val_graphs):,}"
    )

    print(
        f"  Test graphs : "
        f"{len(test_graphs):,}"
    )

    # ----------------------------------------------------------------------
    # Convert x shape
    # ----------------------------------------------------------------------

    print("\nPreparing temporal graph tensors ...")

    for graph in train_graphs:

        graph.x = graph.x.permute(
            1,
            0,
            2
        ).contiguous()

    for graph in val_graphs:

        graph.x = graph.x.permute(
            1,
            0,
            2
        ).contiguous()

    for graph in test_graphs:

        graph.x = graph.x.permute(
            1,
            0,
            2
        ).contiguous()

    # ----------------------------------------------------------------------
    # DataLoaders
    # ----------------------------------------------------------------------

    train_loader = DataLoader(

        train_graphs,

        batch_size=BATCH_SIZE,

        shuffle=True,

        pin_memory=True,
    )

    val_loader = DataLoader(

        val_graphs,

        batch_size=BATCH_SIZE,

        shuffle=False,

        pin_memory=True,
    )

    test_loader = DataLoader(

        test_graphs,

        batch_size=BATCH_SIZE,

        shuffle=False,

        pin_memory=True,
    )

    # ----------------------------------------------------------------------
    # Feature dimension
    # ----------------------------------------------------------------------

    input_dim = train_graphs[0].x.shape[-1]

    print(f"\nInput features: {input_dim}")

    # ----------------------------------------------------------------------
    # Model
    # ----------------------------------------------------------------------

    model = FastSTGNN(

        input_dim=input_dim,

        graph_hidden=GRAPH_HIDDEN,

        temporal_hidden=TEMPORAL_HIDDEN,

        dropout=DROPOUT,
    ).to(DEVICE)

    # ----------------------------------------------------------------------
    # Loss
    # ----------------------------------------------------------------------

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=LEARNING_RATE,
    )

    # ----------------------------------------------------------------------
    # Output dirs
    # ----------------------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    METRIC_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ----------------------------------------------------------------------
    # Training
    # ----------------------------------------------------------------------

    print("\nTraining optimized ST-GNN ...")

    best_val_loss = np.inf

    patience_counter = 0

    train_losses = []

    val_losses = []

    for epoch in range(EPOCHS):

        train_loss = train_epoch(

            model,

            train_loader,

            optimizer,

            criterion,
        )

        val_loss = evaluate_loss(

            model,

            val_loader,

            criterion,
        )

        train_losses.append(train_loss)

        val_losses.append(val_loss)

        print(
            f"Epoch {epoch+1:02d}/{EPOCHS} | "
            f"Train: {train_loss:.5f} | "
            f"Val: {val_loss:.5f}"
        )

        # --------------------------------------------------------------
        # Save best
        # --------------------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            patience_counter = 0

            torch.save(

                model.state_dict(),

                MODEL_DIR / "best_stgnn.pt"
            )

        else:

            patience_counter += 1

            if patience_counter >= PATIENCE:

                print(
                    "\nEarly stopping triggered."
                )

                break

    # ----------------------------------------------------------------------
    # Load best
    # ----------------------------------------------------------------------

    print("\nLoading best model ...")

    model.load_state_dict(

        torch.load(

            MODEL_DIR / "best_stgnn.pt",

            map_location=DEVICE,
        )
    )

    # ----------------------------------------------------------------------
    # Evaluation
    # ----------------------------------------------------------------------

    print("\nEvaluating ST-GNN ...")

    metrics, cm = evaluate_model(

        model,

        test_loader,
    )

    # ----------------------------------------------------------------------
    # Save metrics
    # ----------------------------------------------------------------------

    with open(

        METRIC_DIR / "stgnn_metrics.json",

        "w",

        encoding="utf-8"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

    # ----------------------------------------------------------------------
    # Loss plot
    # ----------------------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        train_losses,
        label="Train"
    )

    plt.plot(
        val_losses,
        label="Validation"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title("Optimized ST-GNN Training")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "stgnn_loss.png"
    )

    plt.close()

    # ----------------------------------------------------------------------
    # Final
    # ----------------------------------------------------------------------

    print("\n" + "=" * 72)
    print("OPTIMIZED ST-GNN COMPLETE")
    print("=" * 72)

    print("\nMetrics:")

    for key, value in metrics.items():

        print(
            f"  {key:<25} "
            f"{value:.6f}"
        )

    print("\nSaved:")
    print("  models_stgnn/")
    print("  reports_stgnn/")

    print("\nDONE.\n")

# ============================================================================
# Entry
# ============================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Optimized ST-GNN"
    )

    args = parser.parse_args()

    main()