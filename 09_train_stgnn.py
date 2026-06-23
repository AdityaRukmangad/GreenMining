"""
Phase 9 — Improved ST-GNN Training
====================================

Architecture upgrade: GAT + BatchNorm + Multi-layer GRU + Temporal Attention
-----------------------------------------------------------------------------

Improvements over v1
--------------------
- GCNConv  →  GATConv (4 heads): learns which neighbours matter
- BatchNorm1d after each GAT layer for stable training
- GRU upgraded to 2 layers
- Multi-head temporal self-attention (residual + LayerNorm)
- pos_weight in loss to penalise false negatives
- AMP mixed-precision training on CUDA
- CosineAnnealing LR schedule
- F2-based threshold tuning on validation set
- Fixed per-class metric extraction (int cast before report)
- Explicit false-negative count in saved metrics

Outputs
-------
models_stgnn/
reports_stgnn/
"""

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

from torch_geometric.nn import GATConv
from torch_geometric.loader import DataLoader

# ============================================================================
# Metrics
# ============================================================================

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    fbeta_score,
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

REPO_ROOT  = Path(__file__).resolve().parent
GRAPH_DIR  = REPO_ROOT / "data" / "graph"
MODEL_DIR  = REPO_ROOT / "models_stgnn"
REPORT_DIR = REPO_ROOT / "reports_stgnn"
PLOT_DIR   = REPORT_DIR / "plots"
METRIC_DIR = REPORT_DIR / "metrics"

# ============================================================================
# Config
# ============================================================================

RANDOM_STATE    = 42
BATCH_SIZE      = 128
LEARNING_RATE   = 5e-4
EPOCHS          = 50
PATIENCE        = 10
GRAPH_HIDDEN    = 128     # GAT output per head × heads
GAT_HEADS       = 4
TEMPORAL_HIDDEN = 128
GRU_LAYERS      = 2
ATTN_HEADS      = 4       # temporal attention heads
DROPOUT         = 0.3
GRAD_CLIP       = 1.0
FBETA           = 2.0     # β for threshold search

# ============================================================================
# Reproducibility
# ============================================================================

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# Improved ST-GNN  (GAT + BatchNorm + multi-layer GRU + temporal attention)
# ============================================================================

class ImprovedSTGNN(nn.Module):
    """
    Spatiotemporal GNN for node-level hazard forecasting.

    Spatial:  2 × GATConv (multi-head) + BatchNorm
    Temporal: 2-layer GRU + multi-head self-attention (residual)
    Head:     FC → sigmoid
    """

    def __init__(
        self,
        input_dim,
        graph_hidden=GRAPH_HIDDEN,
        gat_heads=GAT_HEADS,
        temporal_hidden=TEMPORAL_HIDDEN,
        gru_layers=GRU_LAYERS,
        attn_heads=ATTN_HEADS,
        dropout=DROPOUT,
    ):
        super().__init__()

        # ----------------------------------------------------------
        # GAT layer 1:  F → graph_hidden  (concat heads)
        # ----------------------------------------------------------
        self.gat1 = GATConv(
            input_dim,
            graph_hidden // gat_heads,   # per-head dim
            heads=gat_heads,
            dropout=dropout,
            concat=True,                 # output: graph_hidden
        )
        self.bn1 = nn.BatchNorm1d(graph_hidden)

        # ----------------------------------------------------------
        # GAT layer 2:  graph_hidden → graph_hidden  (single head)
        # ----------------------------------------------------------
        self.gat2 = GATConv(
            graph_hidden,
            graph_hidden,
            heads=1,
            dropout=dropout,
            concat=False,
        )
        self.bn2 = nn.BatchNorm1d(graph_hidden)

        # ----------------------------------------------------------
        # Temporal: GRU
        # ----------------------------------------------------------
        self.gru = nn.GRU(
            input_size=graph_hidden,
            hidden_size=temporal_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        # ----------------------------------------------------------
        # Temporal self-attention
        # ----------------------------------------------------------
        self.time_attn = nn.MultiheadAttention(
            embed_dim=temporal_hidden,
            num_heads=attn_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.time_norm = nn.LayerNorm(temporal_hidden)

        # ----------------------------------------------------------
        # Forecast head
        # ----------------------------------------------------------
        self.fc = nn.Sequential(
            nn.Linear(temporal_hidden, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        self.drop = nn.Dropout(dropout)

    def forward(self, data):
        x          = data.x          # [N_total, T, F]
        edge_index = data.edge_index

        seq_len = x.shape[1]
        temporal_embeddings = []

        for t in range(seq_len):
            x_t = x[:, t, :]                       # [N, F]

            x_t = self.gat1(x_t, edge_index)       # [N, H]
            x_t = F.elu(x_t)
            x_t = self.bn1(x_t)
            x_t = self.drop(x_t)

            x_t = self.gat2(x_t, edge_index)       # [N, H]
            x_t = F.elu(x_t)
            x_t = self.bn2(x_t)

            temporal_embeddings.append(x_t)

        # [N_total, T, H]
        temporal_embeddings = torch.stack(temporal_embeddings, dim=1)

        gru_out, _ = self.gru(temporal_embeddings) # [N, T, H_t]

        # Temporal attention (residual)
        attn_out, _ = self.time_attn(gru_out, gru_out, gru_out)
        out = self.time_norm(gru_out + attn_out)   # [N, T, H_t]

        final = out[:, -1, :]                       # [N, H_t]

        return self.fc(final).squeeze(1)            # [N]

# ============================================================================
# Training helpers
# ============================================================================

def train_epoch(model, loader, optimizer, criterion, scaler_amp):
    model.train()
    running_loss = 0.0

    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()

        if scaler_amp is not None:
            with torch.cuda.amp.autocast():
                outputs = model(batch)
                loss    = criterion(outputs, batch.y.float())
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler_amp.step(optimizer)
            scaler_amp.update()
        else:
            outputs = model(batch)
            loss    = criterion(outputs, batch.y.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        # Normalise by total nodes in batch (not graphs)
        running_loss += loss.item() * batch.num_nodes

    return running_loss / sum(
        g.num_nodes for g in loader.dataset
    )


def evaluate_loss(model, loader, criterion, scaler_amp):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            if DEVICE.type == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = model(batch)
                    loss    = criterion(outputs, batch.y.float())
            else:
                outputs = model(batch)
                loss    = criterion(outputs, batch.y.float())
            running_loss += loss.item() * batch.num_nodes

    return running_loss / sum(
        g.num_nodes for g in loader.dataset
    )


def collect_predictions(model, loader):
    """Collect all (y_true, y_prob) pairs for a split."""
    model.eval()
    y_true_all, y_prob_all = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            if DEVICE.type == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = model(batch)
            else:
                outputs = model(batch)

            probs = torch.sigmoid(outputs)
            y_true_all.extend(batch.y.cpu().numpy().flatten())
            y_prob_all.extend(probs.cpu().numpy().flatten())

    return (
        np.array(y_true_all, dtype=np.int32),
        np.array(y_prob_all, dtype=np.float32),
    )

# ============================================================================
# Threshold search
# ============================================================================

def find_optimal_threshold(y_true, y_prob, beta=FBETA):
    best_t, best_score = 0.5, -1.0
    for t in np.arange(0.10, 0.90, 0.005):
        y_hat = (y_prob >= t).astype(int)
        score = fbeta_score(y_true, y_hat, beta=beta, pos_label=1, zero_division=0)
        if score > best_score:
            best_score = score
            best_t     = float(t)
    return best_t, best_score

# ============================================================================
# Full evaluation
# ============================================================================

def evaluate_model(model, loader, threshold=0.5):
    y_true, y_prob = collect_predictions(model, loader)
    y_pred = (y_prob >= threshold).astype(np.int32)

    print("\nPrediction distribution:")
    u, c = np.unique(y_pred, return_counts=True)
    for ui, ci in zip(u, c):
        print(f"  Pred {ui}: {ci:,}")
    print("Target distribution:")
    u, c = np.unique(y_true, return_counts=True)
    for ui, ci in zip(u, c):
        print(f"  True {ui}: {ci:,}")

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm     = confusion_matrix(y_true, y_pred)

    metrics = {
        "threshold":         threshold,
        "accuracy":          float(accuracy_score(y_true, y_pred)),
        "precision_macro":   float(report["macro avg"]["precision"]),
        "recall_macro":      float(report["macro avg"]["recall"]),
        "f1_macro":          float(report["macro avg"]["f1-score"]),
        # INT keys are safe now (y_true is int32)
        "class_0_recall":    float(report.get("0", {}).get("recall",    0.0)),
        "class_0_precision": float(report.get("0", {}).get("precision", 0.0)),
        "class_1_recall":    float(report.get("1", {}).get("recall",    0.0)),
        "class_1_precision": float(report.get("1", {}).get("precision", 0.0)),
        "false_negatives":   int(cm[1, 0]) if cm.shape == (2, 2) else -1,
        "false_positives":   int(cm[0, 1]) if cm.shape == (2, 2) else -1,
    }

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["roc_auc"] = 0.0

    return metrics, cm

# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 72)
    print("  GreenMining — Improved ST-GNN (GAT + Attention)")
    print("=" * 72)

    print(f"\nDevice: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ------------------------------------------------------------------
    # Load datasets
    # ------------------------------------------------------------------

    print("\nLoading graph datasets ...")
    train_graphs = torch.load(GRAPH_DIR / "train_graphs.pt", weights_only=False)
    val_graphs   = torch.load(GRAPH_DIR / "val_graphs.pt",   weights_only=False)
    test_graphs  = torch.load(GRAPH_DIR / "test_graphs.pt",  weights_only=False)

    print(f"  Train: {len(train_graphs):,}  Val: {len(val_graphs):,}  Test: {len(test_graphs):,}")

    # ------------------------------------------------------------------
    # Permute x from [T, N, F] → [N, T, F]  (as in original script)
    # ------------------------------------------------------------------

    print("\nPreparing temporal graph tensors ...")
    for g in train_graphs:
        g.x = g.x.permute(1, 0, 2).contiguous()
    for g in val_graphs:
        g.x = g.x.permute(1, 0, 2).contiguous()
    for g in test_graphs:
        g.x = g.x.permute(1, 0, 2).contiguous()

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------

    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True,  pin_memory=True)
    val_loader   = DataLoader(val_graphs,   batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    test_loader  = DataLoader(test_graphs,  batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    # ------------------------------------------------------------------
    # Feature dimension
    # ------------------------------------------------------------------

    input_dim = train_graphs[0].x.shape[-1]
    print(f"\nInput features: {input_dim}")

    # ------------------------------------------------------------------
    # Compute pos_weight from ALL labels in train graphs
    # ------------------------------------------------------------------

    all_train_labels = np.concatenate([g.y.numpy().flatten() for g in train_graphs])
    n_pos = all_train_labels.sum()
    n_neg = len(all_train_labels) - n_pos
    pos_weight_val = float(n_neg / max(n_pos, 1))
    print(f"pos_weight: {pos_weight_val:.3f}  (neg/pos in train nodes)")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    model = ImprovedSTGNN(input_dim=input_dim).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # ------------------------------------------------------------------
    # Loss + optimiser + scheduler
    # ------------------------------------------------------------------

    pos_weight_tensor = torch.tensor(pos_weight_val, dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6,
    )

    use_amp    = (DEVICE.type == "cuda")
    scaler_amp = torch.cuda.amp.GradScaler() if use_amp else None

    # ------------------------------------------------------------------
    # Output dirs
    # ------------------------------------------------------------------

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "best_stgnn.pt"

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    print("\nTraining Improved ST-GNN ...")

    best_val_loss    = np.inf
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler_amp)
        val_loss   = evaluate_loss(model, val_loader, criterion, scaler_amp)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(epoch + val_loss)

        print(
            f"Epoch {epoch+1:02d}/{EPOCHS} | "
            f"Train: {train_loss:.5f} | Val: {val_loss:.5f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("\nEarly stopping triggered.")
                break

    # ------------------------------------------------------------------
    # Load best, tune threshold on val, evaluate on test
    # ------------------------------------------------------------------

    print("\nLoading best model ...")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))

    print("\nFinding optimal threshold on validation set ...")
    val_true, val_prob = collect_predictions(model, val_loader)
    threshold, fbeta_val = find_optimal_threshold(val_true, val_prob)
    print(f"  Threshold: {threshold:.3f}  (F{FBETA:.0f} on val = {fbeta_val:.4f})")

    print("\nEvaluating ST-GNN on test set ...")
    metrics, cm = evaluate_model(model, test_loader, threshold)

    # ------------------------------------------------------------------
    # Save metrics
    # ------------------------------------------------------------------

    with open(METRIC_DIR / "stgnn_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    # ------------------------------------------------------------------
    # Loss plot
    # ------------------------------------------------------------------

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses,   label="Validation")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Improved ST-GNN Training")
    plt.legend(); plt.tight_layout()
    plt.savefig(PLOT_DIR / "stgnn_loss.png")
    plt.close()

    # ------------------------------------------------------------------
    # Confusion matrix plot
    # ------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["SAFE", "HAZARD"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["SAFE", "HAZARD"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("ST-GNN Confusion Matrix")
    fig.savefig(PLOT_DIR / "stgnn_cm.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------

    print("\n" + "=" * 72)
    print("IMPROVED ST-GNN COMPLETE")
    print("=" * 72)
    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k:<25} {v}")
    print(f"\n  class_1_recall  (hazard recall) : {metrics.get('class_1_recall'):.4f}")
    print(f"  false_negatives                 : {metrics.get('false_negatives')}")
    print("\nSaved:  models_stgnn/  reports_stgnn/")
    print("\nDONE.\n")


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 9 — Improved ST-GNN")
    parser.parse_args()
    main()
