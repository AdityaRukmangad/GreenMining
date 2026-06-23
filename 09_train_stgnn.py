"""
Phase 9 — SpatioTemporal Net  (GraphSAGE + BiLSTM + Attention)
===============================================================

Why this replaces the GAT/GRU version
--------------------------------------
- Graph data is split temporally: train = early CFD timesteps, val/test = later.
  Gas concentrations build up over time, so raw feature distributions shift.
  GAT learned to memorize training distributions → val loss diverged immediately.

- Fix 1: z-score normalise ALL splits using TRAINING stats only. The model
  always sees zero-mean unit-variance features regardless of CFD timestep.

- Fix 2: GraphSAGE uses fixed mean aggregation (no learned edge weights).
  Less capacity → far less prone to memorising training graphs.

- Temporal backbone: BiLSTM + multi-head self-attention, identical to the
  LSTM forecaster that achieves 97 %+ — proven to generalise on this data.

- AdamW + weight_decay=1e-3 + LayerNorm everywhere (no BatchNorm, which
  accumulates running stats that diverge between train and val splits).

Outputs (same paths as before — pipeline unchanged)
-------
models_stgnn/best_stgnn.pt
reports_stgnn/metrics/stgnn_metrics.json
reports_stgnn/plots/stgnn_loss.png
reports_stgnn/plots/stgnn_cm.png
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv
from torch_geometric.loader import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    fbeta_score,
    roc_auc_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parent
GRAPH_DIR  = REPO_ROOT / "data" / "graph"
MODEL_DIR  = REPO_ROOT / "models_stgnn"
REPORT_DIR = REPO_ROOT / "reports_stgnn"
PLOT_DIR   = REPORT_DIR / "plots"
METRIC_DIR = REPORT_DIR / "metrics"

# ── Hyper-parameters ──────────────────────────────────────────────────────────
RANDOM_STATE = 42
BATCH_SIZE   = 64
LR           = 3e-4
EPOCHS       = 80
PATIENCE     = 15
HIDDEN       = 128
LSTM_HIDDEN  = 128
LSTM_LAYERS  = 2
ATTN_HEADS   = 4
DROPOUT      = 0.35
GRAD_CLIP    = 1.0
FBETA        = 2.0

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model ─────────────────────────────────────────────────────────────────────

class SpatioTemporalNet(nn.Module):
    """
    Spatial: 2 x SAGEConv (shared across T timesteps) + LayerNorm
    Temporal: 2-layer BiLSTM + multi-head self-attention (residual)
    Head: 3-layer FC with GELU

    SAGEConv formula: h_v = W1*h_v + W2*mean(h_neighbours)
    Fixed aggregation — no learned edge weights — prevents overfitting to
    specific training graph topologies.
    """

    def __init__(
        self,
        input_dim,
        hidden=HIDDEN,
        lstm_hidden=LSTM_HIDDEN,
        lstm_layers=LSTM_LAYERS,
        attn_heads=ATTN_HEADS,
        dropout=DROPOUT,
    ):
        super().__init__()

        # ── Spatial (shared weights applied at each timestep) ─────────────────
        self.sage1  = SAGEConv(input_dim, hidden)
        self.norm1  = nn.LayerNorm(hidden)
        self.sage2  = SAGEConv(hidden, hidden)
        self.norm2  = nn.LayerNorm(hidden)
        self.s_drop = nn.Dropout(dropout)

        # ── Temporal: BiLSTM ──────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            hidden, lstm_hidden, lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # ── Temporal self-attention (residual) ────────────────────────────────
        self.t_attn = nn.MultiheadAttention(
            lstm_hidden * 2, attn_heads, dropout=dropout, batch_first=True,
        )
        self.t_norm = nn.LayerNorm(lstm_hidden * 2)

        # ── Head ──────────────────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, data):
        x          = data.x          # [N_total, T, F]  (after permute)
        edge_index = data.edge_index

        T = x.shape[1]
        sage_outs = []

        for t in range(T):
            x_t = x[:, t, :]                                    # [N, F]
            x_t = F.gelu(self.norm1(self.sage1(x_t, edge_index)))
            x_t = self.s_drop(x_t)
            x_t = F.gelu(self.norm2(self.sage2(x_t, edge_index)))
            sage_outs.append(x_t)

        te = torch.stack(sage_outs, dim=1)                      # [N, T, H]

        lstm_out, _ = self.lstm(te)                             # [N, T, 2H]

        attn_out, _ = self.t_attn(lstm_out, lstm_out, lstm_out)
        out = self.t_norm(lstm_out + attn_out)                  # [N, T, 2H]

        return self.head(out[:, -1, :]).squeeze(1)              # [N]


# ── Feature normalisation ─────────────────────────────────────────────────────

def fit_normalise(train_graphs):
    """Compute z-score statistics from all training node features."""
    all_x = torch.cat(
        [g.x.reshape(-1, g.x.shape[-1]) for g in train_graphs], dim=0
    )
    mean = all_x.mean(dim=0)
    std  = all_x.std(dim=0).clamp(min=1e-6)
    return mean, std


def apply_normalise(graphs, mean, std):
    """Apply training z-score in-place to a list of graphs."""
    for g in graphs:
        g.x = (g.x - mean) / std


# ── Training helpers ──────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, scaler_amp):
    model.train()
    running_loss = 0.0
    total_nodes  = 0

    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()

        if scaler_amp is not None:
            with torch.cuda.amp.autocast():
                out  = model(batch)
                loss = criterion(out, batch.y.float())
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler_amp.step(optimizer)
            scaler_amp.update()
        else:
            out  = model(batch)
            loss = criterion(out, batch.y.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        n = batch.num_nodes
        running_loss += loss.item() * n
        total_nodes  += n

    return running_loss / total_nodes


def evaluate_loss(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    total_nodes  = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            if DEVICE.type == "cuda":
                with torch.cuda.amp.autocast():
                    out  = model(batch)
                    loss = criterion(out, batch.y.float())
            else:
                out  = model(batch)
                loss = criterion(out, batch.y.float())
            n = batch.num_nodes
            running_loss += loss.item() * n
            total_nodes  += n

    return running_loss / total_nodes


def collect_predictions(model, loader):
    model.eval()
    y_true_all, y_prob_all = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            if DEVICE.type == "cuda":
                with torch.cuda.amp.autocast():
                    out = model(batch)
            else:
                out = model(batch)
            probs = torch.sigmoid(out)
            y_true_all.extend(batch.y.cpu().numpy().flatten())
            y_prob_all.extend(probs.cpu().numpy().flatten())

    return (
        np.array(y_true_all, dtype=np.int32),
        np.array(y_prob_all, dtype=np.float32),
    )


def find_optimal_threshold(y_true, y_prob, beta=FBETA):
    best_t, best_score = 0.5, -1.0
    for t in np.arange(0.10, 0.90, 0.005):
        y_hat = (y_prob >= t).astype(int)
        score = fbeta_score(y_true, y_hat, beta=beta, pos_label=1, zero_division=0)
        if score > best_score:
            best_score = score
            best_t     = float(t)
    return best_t, best_score


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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  GreenMining — SpatioTemporal Net (GraphSAGE + BiLSTM + Attention)")
    print("=" * 72)
    print(f"\nDevice: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Load graphs ───────────────────────────────────────────────────────────
    print("\nLoading graph datasets ...")
    train_graphs = torch.load(GRAPH_DIR / "train_graphs.pt", weights_only=False)
    val_graphs   = torch.load(GRAPH_DIR / "val_graphs.pt",   weights_only=False)
    test_graphs  = torch.load(GRAPH_DIR / "test_graphs.pt",  weights_only=False)
    print(f"  Train: {len(train_graphs):,}  Val: {len(val_graphs):,}  Test: {len(test_graphs):,}")

    # ── Permute x: [T, N, F] → [N, T, F] ────────────────────────────────────
    print("\nPreparing temporal tensors ...")
    for g in train_graphs: g.x = g.x.permute(1, 0, 2).contiguous()
    for g in val_graphs:   g.x = g.x.permute(1, 0, 2).contiguous()
    for g in test_graphs:  g.x = g.x.permute(1, 0, 2).contiguous()

    # ── Z-score normalise (training stats applied to all splits) ──────────────
    print("\nNormalising features using training statistics ...")
    feat_mean, feat_std = fit_normalise(train_graphs)
    apply_normalise(train_graphs, feat_mean, feat_std)
    apply_normalise(val_graphs,   feat_mean, feat_std)
    apply_normalise(test_graphs,  feat_mean, feat_std)
    print(f"  Feature mean range:  [{feat_mean.min():.3f}, {feat_mean.max():.3f}]")
    print(f"  Feature std  range:  [{feat_std.min():.3f},  {feat_std.max():.3f}]")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True,  pin_memory=True)
    val_loader   = DataLoader(val_graphs,   batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    test_loader  = DataLoader(test_graphs,  batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    input_dim = train_graphs[0].x.shape[-1]
    print(f"\nInput features per node per timestep: {input_dim}")

    # ── Class weight (capped to avoid loss explosion) ─────────────────────────
    all_labels = np.concatenate([g.y.numpy().flatten() for g in train_graphs])
    n_pos = all_labels.sum()
    n_neg = len(all_labels) - n_pos
    pos_weight_val = float(min(n_neg / max(n_pos, 1), 6.0))
    print(f"Class balance — SAFE: {n_neg:,}  HAZARD: {n_pos:,}  pos_weight: {pos_weight_val:.3f}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model    = SpatioTemporalNet(input_dim=input_dim).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # ── Loss / optimiser / scheduler ──────────────────────────────────────────
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight_val, dtype=torch.float32).to(DEVICE)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6, verbose=True,
    )
    use_amp    = (DEVICE.type == "cuda")
    scaler_amp = torch.cuda.amp.GradScaler() if use_amp else None

    # ── Output dirs ───────────────────────────────────────────────────────────
    for d in (MODEL_DIR, PLOT_DIR, METRIC_DIR):
        d.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "best_stgnn.pt"

    # ── Training loop ─────────────────────────────────────────────────────────
    print("\nTraining SpatioTemporal Net ...")
    best_val_loss    = np.inf
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler_amp)
        val_loss   = evaluate_loss(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

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

    # ── Load best checkpoint ──────────────────────────────────────────────────
    print("\nLoading best model ...")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))

    # ── Threshold tuning on val ───────────────────────────────────────────────
    print("\nFinding optimal threshold on validation set ...")
    val_true, val_prob = collect_predictions(model, val_loader)
    threshold, fbeta_val = find_optimal_threshold(val_true, val_prob)
    print(f"  Threshold: {threshold:.3f}  (F{FBETA:.0f} on val = {fbeta_val:.4f})")

    # ── Test evaluation ───────────────────────────────────────────────────────
    print("\nEvaluating on test set ...")
    metrics, cm = evaluate_model(model, test_loader, threshold)

    # ── Save metrics ──────────────────────────────────────────────────────────
    with open(METRIC_DIR / "stgnn_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    # ── Loss curve ────────────────────────────────────────────────────────────
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses,   label="Validation")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("SpatioTemporal Net (GraphSAGE + BiLSTM)")
    plt.legend(); plt.tight_layout()
    plt.savefig(PLOT_DIR / "stgnn_loss.png")
    plt.close()

    # ── Confusion matrix ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["SAFE", "HAZARD"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["SAFE", "HAZARD"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("SpatioTemporal Net Confusion Matrix")
    fig.savefig(PLOT_DIR / "stgnn_cm.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("SPATIOTEMPORAL NET — COMPLETE")
    print("=" * 72)
    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k:<25} {v}")
    print(f"\n  class_1_recall  (hazard recall) : {metrics.get('class_1_recall'):.4f}")
    print(f"  false_negatives                 : {metrics.get('false_negatives')}")
    print("\nSaved:  models_stgnn/  reports_stgnn/")
    print("\nDONE.\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 9 — SpatioTemporal Net")
    parser.parse_args()
    main()
