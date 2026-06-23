"""
Phase 7 — Train Improved LSTM Hazard Forecaster
================================================

Architecture upgrade: BiLSTM + Multi-Head Temporal Attention
-------------------------------------------------------------
Input  : [t-45, t-30, t-15, t]  →  4 timesteps × 38 features
Predict: hazard at t+30s

Improvements over v1
--------------------
- Bidirectional LSTM (256 hidden × 3 layers)
- Multi-head self-attention over time dimension
- Layer-norm residual connection
- GELU activations in classifier head
- CosineAnnealingWarmRestarts LR schedule
- F2-based threshold tuning on validation set (recall-prioritised)
- Explicit false-negative count in metrics

Outputs
-------
models_lstm/
reports_lstm/
"""

import json
import random
from pathlib import Path

import joblib
import numpy as np

# ============================================================================
# PyTorch
# ============================================================================

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

# ============================================================================
# Sklearn metrics
# ============================================================================

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    fbeta_score,
    roc_auc_score,
    roc_curve,
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
DATA_DIR   = REPO_ROOT / "data" / "lstm"
MODELS_DIR = REPO_ROOT / "models_lstm"
REPORTS_DIR = REPO_ROOT / "reports_lstm"
METRICS_DIR = REPORTS_DIR / "metrics"
PLOTS_DIR   = REPORTS_DIR / "plots"

# ============================================================================
# Hyper-parameters
# ============================================================================

RANDOM_STATE   = 42
BATCH_SIZE     = 2048
LEARNING_RATE  = 5e-4
EPOCHS         = 100
PATIENCE       = 15
HIDDEN_SIZE    = 256
NUM_LAYERS     = 3
N_HEADS        = 8       # attention heads (must divide HIDDEN_SIZE*2)
DROPOUT        = 0.3
GRAD_CLIP      = 1.0
FBETA          = 2.0     # β for threshold search (recall-weighted)

# ============================================================================
# Reproducibility
# ============================================================================

def set_seed(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# Model — BiLSTM + Temporal Attention
# ============================================================================

class HazardTemporalNet(nn.Module):
    """
    Bidirectional LSTM with multi-head self-attention over the time axis.

    Residual + LayerNorm stabilise training and help gradients flow.
    """

    def __init__(
        self,
        input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        n_heads=N_HEADS,
        dropout=DROPOUT,
    ):
        super().__init__()

        # Project raw features into model dimension
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )

        # Bidirectional LSTM — output dim = hidden_size * 2
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )

        lstm_out_dim = hidden_size * 2

        # Multi-head self-attention over time steps
        self.attn = nn.MultiheadAttention(
            embed_dim=lstm_out_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(lstm_out_dim)
        self.drop = nn.Dropout(dropout)

        # Classifier head
        self.fc = nn.Sequential(
            nn.Linear(lstm_out_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        # x: [B, T, F]
        x = self.input_proj(x)          # [B, T, H]
        lstm_out, _ = self.lstm(x)       # [B, T, 2H]

        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
        out = self.norm(lstm_out + attn_out)   # residual

        out = out[:, -1, :]             # last timestep  [B, 2H]
        out = self.drop(out)

        return self.fc(out).squeeze(1)  # [B]

# ============================================================================
# Normalisation (fit on train, apply to all)
# ============================================================================

def normalize_data(train_X, val_X, test_X):
    print("\nNormalising datasets ...")
    mean = train_X.mean(axis=(0, 1), keepdims=True)
    std  = train_X.std(axis=(0, 1),  keepdims=True)
    std  = np.where(std < 1e-8, 1.0, std)
    train_X = (train_X - mean) / std
    val_X   = (val_X   - mean) / std
    test_X  = (test_X  - mean) / std
    return train_X, val_X, test_X, {"mean": mean, "std": std}

# ============================================================================
# Threshold optimisation (F-beta on validation)
# ============================================================================

def find_optimal_threshold(y_true, y_prob, beta=FBETA):
    best_t, best_score = 0.5, -1.0
    for t in np.arange(0.10, 0.90, 0.005):
        y_hat = (y_prob >= t).astype(int)
        score = fbeta_score(y_true, y_hat, beta=beta, pos_label=1, zero_division=0)
        if score > best_score:
            best_score = score
            best_t = float(t)
    return best_t, best_score

# ============================================================================
# Evaluation
# ============================================================================

def evaluate_model(model, loader, threshold=0.5):
    model.eval()
    all_probs, all_targets = [], []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(DEVICE, non_blocking=True)
            logits = model(X)
            probs  = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(y.numpy())

    all_probs   = np.array(all_probs,   dtype=np.float32)
    all_targets = np.array(all_targets, dtype=np.int32)
    all_preds   = (all_probs >= threshold).astype(np.int32)

    print("\nPrediction distribution:")
    unique, counts = np.unique(all_preds, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  Pred {u}: {c:,}")
    print("Target distribution:")
    unique, counts = np.unique(all_targets, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  True {u}: {c:,}")

    report = classification_report(all_targets, all_preds, output_dict=True, zero_division=0)
    cm     = confusion_matrix(all_targets, all_preds)

    metrics = {
        "threshold":        threshold,
        "accuracy":         float(accuracy_score(all_targets, all_preds)),
        "precision_macro":  float(report["macro avg"]["precision"]),
        "recall_macro":     float(report["macro avg"]["recall"]),
        "f1_macro":         float(report["macro avg"]["f1-score"]),
        "class_0_recall":   float(report.get("0", {}).get("recall",    0.0)),
        "class_0_precision":float(report.get("0", {}).get("precision", 0.0)),
        "class_1_recall":   float(report.get("1", {}).get("recall",    0.0)),
        "class_1_precision":float(report.get("1", {}).get("precision", 0.0)),
        "false_negatives":  int(cm[1, 0]) if cm.shape == (2, 2) else -1,
        "false_positives":  int(cm[0, 1]) if cm.shape == (2, 2) else -1,
    }

    try:
        metrics["roc_auc"] = float(roc_auc_score(all_targets, all_probs))
    except ValueError:
        metrics["roc_auc"] = 0.0

    return metrics, all_targets, all_preds, all_probs

# ============================================================================
# Plotting
# ============================================================================

def save_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["SAFE", "HAZARD"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["SAFE", "HAZARD"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("BiLSTM-Attn Confusion Matrix")
    path = PLOTS_DIR / "confusion_matrix.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_roc_curve(y_true, y_probs):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc = roc_auc_score(y_true, y_probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("BiLSTM-Attn ROC Curve")
    ax.legend()
    fig.savefig(PLOTS_DIR / "roc_curve.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_training_curve(train_losses, val_losses):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(train_losses, label="Train Loss")
    ax.plot(val_losses,   label="Val Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("BiLSTM-Attn Training Curve")
    ax.legend()
    fig.savefig(PLOTS_DIR / "training_curve.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 72)
    print("  GreenMining — Phase 7: BiLSTM + Attention Hazard Forecaster")
    print("=" * 72)

    set_seed()
    print(f"\nDevice: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------

    print("\nLoading sequence datasets ...")
    train_X = np.load(DATA_DIR / "train_X.npy")
    train_y = np.load(DATA_DIR / "train_y.npy")
    val_X   = np.load(DATA_DIR / "val_X.npy")
    val_y   = np.load(DATA_DIR / "val_y.npy")
    test_X  = np.load(DATA_DIR / "test_X.npy")
    test_y  = np.load(DATA_DIR / "test_y.npy")

    print(f"  Train X: {train_X.shape}  |  Val X: {val_X.shape}  |  Test X: {test_X.shape}")

    # ------------------------------------------------------------------
    # Normalise
    # ------------------------------------------------------------------

    train_X, val_X, test_X, scaler = normalize_data(train_X, val_X, test_X)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

    # ------------------------------------------------------------------
    # Tensor datasets
    # ------------------------------------------------------------------

    def make_loader(X, y, shuffle):
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        return DataLoader(
            ds, batch_size=BATCH_SIZE, shuffle=shuffle,
            num_workers=4, pin_memory=True, persistent_workers=True,
        )

    train_loader = make_loader(train_X, train_y, shuffle=True)
    val_loader   = make_loader(val_X,   val_y,   shuffle=False)
    test_loader  = make_loader(test_X,  test_y,  shuffle=False)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    input_size = train_X.shape[2]
    model = HazardTemporalNet(input_size=input_size).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {n_params:,}")

    # ------------------------------------------------------------------
    # Loss — weighted to penalise false negatives
    # ------------------------------------------------------------------

    pos_weight = torch.tensor(
        (len(train_y) - train_y.sum()) / train_y.sum(),
        dtype=torch.float32,
    ).to(DEVICE)
    print(f"BCEWithLogitsLoss pos_weight: {pos_weight.item():.3f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    # CosineAnnealing: restarts every T0 epochs, double period each restart
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6,
    )

    use_amp   = torch.cuda.is_available()
    scaler_amp = torch.cuda.amp.GradScaler() if use_amp else None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    model_path  = MODELS_DIR / "best_lstm.pt"

    print("\nTraining BiLSTM + Attention ...")

    best_val_loss    = np.inf
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE, non_blocking=True)
            y_batch = y_batch.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()

            if use_amp:
                with torch.cuda.amp.autocast():
                    loss = criterion(model(X_batch), y_batch)
                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler_amp.step(optimizer)
                scaler_amp.update()
            else:
                loss = criterion(model(X_batch), y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

            running_loss += loss.item() * X_batch.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE, non_blocking=True)
                y_batch = y_batch.to(DEVICE, non_blocking=True)
                if use_amp:
                    with torch.cuda.amp.autocast():
                        val_loss_b = criterion(model(X_batch), y_batch)
                else:
                    val_loss_b = criterion(model(X_batch), y_batch)
                val_running += val_loss_b.item() * X_batch.size(0)

        val_loss = val_running / len(val_loader.dataset)
        val_losses.append(val_loss)

        scheduler.step(epoch + val_loss)   # CosineAnnealing step

        print(
            f"Epoch {epoch+1:03d}/{EPOCHS} | "
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

    save_training_curve(train_losses, val_losses)

    # ------------------------------------------------------------------
    # Load best, tune threshold on val, evaluate on test
    # ------------------------------------------------------------------

    print("\nLoading best model ...")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))

    # Collect val probabilities for threshold search
    print("\nFinding optimal decision threshold on validation set ...")
    model.eval()
    val_probs, val_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(DEVICE, non_blocking=True)
            probs   = torch.sigmoid(model(X_batch))
            val_probs.extend(probs.cpu().numpy())
            val_true.extend(y_batch.numpy())

    val_probs = np.array(val_probs, dtype=np.float32)
    val_true  = np.array(val_true,  dtype=np.int32)

    threshold, fbeta_val = find_optimal_threshold(val_true, val_probs)
    print(f"  Threshold: {threshold:.3f}  (F{FBETA:.0f} on val = {fbeta_val:.4f})")

    # ------------------------------------------------------------------
    # Test evaluation
    # ------------------------------------------------------------------

    print("\nEvaluating on test set ...")
    metrics, y_true, y_pred, y_probs = evaluate_model(model, test_loader, threshold)

    with open(METRICS_DIR / "forecast_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    save_confusion_matrix(y_true, y_pred)
    save_roc_curve(y_true, y_probs)

    # ------------------------------------------------------------------

    print("\n" + "=" * 72)
    print("BiLSTM + ATTENTION FORECASTING COMPLETE")
    print("=" * 72)
    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k:<24} {v}")
    print("\nKey metrics:")
    print(f"  class_1_recall (hazard recall) : {metrics.get('class_1_recall'):.4f}")
    print(f"  false_negatives                : {metrics.get('false_negatives')}")
    print("\nSaved:  models_lstm/  reports_lstm/")
    print("\nDONE.\n")


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 7 — BiLSTM Attention Forecaster")
    parser.parse_args()
    main()
