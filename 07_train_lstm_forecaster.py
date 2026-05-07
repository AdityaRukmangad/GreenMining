"""
Phase 7 — Train LSTM Hazard Forecaster
======================================

Train a temporal LSTM model for future hazard prediction.

Task
----
Input:
    [t-45, t-30, t-15, t]

Predict:
    hazard at t+30s

Outputs
-------
models_lstm/
reports_lstm/
"""

import argparse
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

from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

# ============================================================================
# Performance
# ============================================================================

torch.backends.cudnn.benchmark = True

torch.set_float32_matmul_precision("high")

# ============================================================================
# Sklearn metrics
# ============================================================================

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
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
# Repository paths
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parent

DATA_DIR = REPO_ROOT / "data" / "lstm"

MODELS_DIR = REPO_ROOT / "models_lstm"

REPORTS_DIR = REPO_ROOT / "reports_lstm"

METRICS_DIR = REPORTS_DIR / "metrics"

PLOTS_DIR = REPORTS_DIR / "plots"

# ============================================================================
# Configuration
# ============================================================================

RANDOM_STATE = 42

BATCH_SIZE = 4096

LEARNING_RATE = 1e-3

EPOCHS = 50

PATIENCE = 8

HIDDEN_SIZE = 128

NUM_LAYERS = 2

DROPOUT = 0.2

GRAD_CLIP = 1.0

# ============================================================================
# Reproducibility
# ============================================================================

def set_seed(seed=RANDOM_STATE):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

# ============================================================================
# Device
# ============================================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# ============================================================================
# LSTM Model
# ============================================================================

class HazardLSTM(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            batch_first=True,
        )

        self.fc = nn.Sequential(

            nn.Linear(hidden_size, hidden_size),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):

        out, _ = self.lstm(x)

        out = out[:, -1, :]

        out = self.fc(out)

        return out.squeeze(1)

# ============================================================================
# Normalization
# ============================================================================

def normalize_data(
    train_X,
    val_X,
    test_X,
):

    print("\nNormalizing datasets ...")

    mean = train_X.mean(
        axis=(0, 1),
        keepdims=True
    )

    std = train_X.std(
        axis=(0, 1),
        keepdims=True
    )

    std = np.where(
        std < 1e-8,
        1.0,
        std
    )

    train_X = (
        train_X - mean
    ) / std

    val_X = (
        val_X - mean
    ) / std

    test_X = (
        test_X - mean
    ) / std

    scaler = {
        "mean": mean,
        "std": std,
    }

    return (
        train_X,
        val_X,
        test_X,
        scaler,
    )

# ============================================================================
# Evaluation
# ============================================================================

def evaluate_model(
    model,
    loader,
):

    model.eval()

    all_probs = []

    all_preds = []

    all_targets = []

    with torch.no_grad():

        for X, y in loader:

            X = X.to(
                DEVICE,
                non_blocking=True
            )

            logits = model(X)

            probs = torch.sigmoid(logits)

            preds = (
                probs >= 0.5
            ).long()

            all_probs.extend(
                probs.cpu().numpy()
            )

            all_preds.extend(
                preds.cpu().numpy()
            )

            all_targets.extend(
                y.numpy()
            )

    all_probs = np.array(all_probs)

    all_preds = np.array(all_preds)

    all_targets = np.array(all_targets)

    # ----------------------------------------------------------------------
    # Diagnostics
    # ----------------------------------------------------------------------

    print("\nPrediction distribution:")

    unique, counts = np.unique(
        all_preds,
        return_counts=True
    )

    for u, c in zip(unique, counts):

        print(f"  Pred {u}: {c:,}")

    print("\nTarget distribution:")

    unique, counts = np.unique(
        all_targets,
        return_counts=True
    )

    for u, c in zip(unique, counts):

        print(f"  True {u}: {c:,}")

    # ----------------------------------------------------------------------
    # Metrics
    # ----------------------------------------------------------------------

    metrics = {}

    metrics["accuracy"] = float(
        accuracy_score(
            all_targets,
            all_preds
        )
    )
    
    all_targets = all_targets.astype(int)
    all_preds = all_preds.astype(int)

    report = classification_report(
        all_targets,
        all_preds,
        output_dict=True,
        zero_division=0,
    )

    metrics["precision_macro"] = float(
        report["macro avg"]["precision"]
    )

    metrics["recall_macro"] = float(
        report["macro avg"]["recall"]
    )

    metrics["f1_macro"] = float(
        report["macro avg"]["f1-score"]
    )

    metrics["class_0_recall"] = float(
        report.get("0", {}).get("recall", 0.0)
    )

    metrics["class_0_precision"] = float(
        report.get("0", {}).get("precision", 0.0)
    )

    metrics["class_1_recall"] = float(
        report.get("1", {}).get("recall", 0.0)
    )

    metrics["class_1_precision"] = float(
        report.get("1", {}).get("precision", 0.0)
    )

    try:

        metrics["roc_auc"] = float(
            roc_auc_score(
                all_targets,
                all_probs
            )
        )

    except ValueError:

        metrics["roc_auc"] = 0.0

        print(
            "\nWARNING: ROC-AUC could not be computed."
        )

    return (
        metrics,
        all_targets,
        all_preds,
        all_probs,
    )

# ============================================================================
# Plotting
# ============================================================================

def save_confusion_matrix(
    y_true,
    y_pred,
):

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.imshow(cm)

    ax.set_xticks([0, 1])

    ax.set_yticks([0, 1])

    ax.set_xticklabels([
        "SAFE",
        "HAZARD"
    ])

    ax.set_yticklabels([
        "SAFE",
        "HAZARD"
    ])

    ax.set_xlabel("Predicted")

    ax.set_ylabel("Actual")

    for i in range(2):
        for j in range(2):

            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
            )

    ax.set_title(
        "LSTM Forecast Confusion Matrix"
    )

    path = (
        PLOTS_DIR /
        "confusion_matrix.png"
    )

    fig.savefig(
        path,
        dpi=120,
        bbox_inches="tight"
    )

    plt.close(fig)

def save_roc_curve(
    y_true,
    y_probs,
):

    fpr, tpr, _ = roc_curve(
        y_true,
        y_probs
    )

    auc = roc_auc_score(
        y_true,
        y_probs
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(
        fpr,
        tpr,
        label=f"AUC = {auc:.4f}"
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.set_title(
        "LSTM Forecast ROC Curve"
    )

    ax.legend()

    path = (
        PLOTS_DIR /
        "roc_curve.png"
    )

    fig.savefig(
        path,
        dpi=120,
        bbox_inches="tight"
    )

    plt.close(fig)

def save_training_curve(
    train_losses,
    val_losses,
):

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(
        train_losses,
        label="Train Loss"
    )

    ax.plot(
        val_losses,
        label="Val Loss"
    )

    ax.set_xlabel("Epoch")

    ax.set_ylabel("Loss")

    ax.set_title(
        "LSTM Training Curve"
    )

    ax.legend()

    path = (
        PLOTS_DIR /
        "training_curve.png"
    )

    fig.savefig(
        path,
        dpi=120,
        bbox_inches="tight"
    )

    plt.close(fig)

# ============================================================================
# Main
# ============================================================================

def main():

    print("=" * 72)
    print("  GreenMining — Phase 7: LSTM Hazard Forecasting")
    print("=" * 72)

    set_seed()

    print(f"\nDevice: {DEVICE}")

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # ----------------------------------------------------------------------
    # Load data
    # ----------------------------------------------------------------------

    print("\nLoading sequence datasets ...")

    train_X = np.load(
        DATA_DIR / "train_X.npy"
    )

    train_y = np.load(
        DATA_DIR / "train_y.npy"
    )

    val_X = np.load(
        DATA_DIR / "val_X.npy"
    )

    val_y = np.load(
        DATA_DIR / "val_y.npy"
    )

    test_X = np.load(
        DATA_DIR / "test_X.npy"
    )

    test_y = np.load(
        DATA_DIR / "test_y.npy"
    )

    print(f"  Train X: {train_X.shape}")
    print(f"  Val X  : {val_X.shape}")
    print(f"  Test X : {test_X.shape}")

    # ----------------------------------------------------------------------
    # Normalize
    # ----------------------------------------------------------------------

    (
        train_X,
        val_X,
        test_X,
        scaler,
    ) = normalize_data(
        train_X,
        val_X,
        test_X,
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        scaler,
        MODELS_DIR / "scaler.pkl"
    )

    # ----------------------------------------------------------------------
    # Tensor datasets
    # ----------------------------------------------------------------------

    train_ds = TensorDataset(

        torch.tensor(
            train_X,
            dtype=torch.float32
        ),

        torch.tensor(
            train_y,
            dtype=torch.float32
        ),
    )

    val_ds = TensorDataset(

        torch.tensor(
            val_X,
            dtype=torch.float32
        ),

        torch.tensor(
            val_y,
            dtype=torch.float32
        ),
    )

    test_ds = TensorDataset(

        torch.tensor(
            test_X,
            dtype=torch.float32
        ),

        torch.tensor(
            test_y,
            dtype=torch.float32
        ),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    # ----------------------------------------------------------------------
    # Model
    # ----------------------------------------------------------------------

    input_size = train_X.shape[2]

    model = HazardLSTM(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)

    # ----------------------------------------------------------------------
    # Loss
    # ----------------------------------------------------------------------

    pos_weight = (
        len(train_y) - train_y.sum()
    ) / train_y.sum()

    pos_weight = torch.tensor(
        pos_weight,
        dtype=torch.float32
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    scaler_amp = torch.cuda.amp.GradScaler()

    # ----------------------------------------------------------------------
    # Skip training if model exists
    # ----------------------------------------------------------------------

    model_path = (
        MODELS_DIR /
        "best_lstm.pt"
    )

    if model_path.exists():

        print(
            "\nFound existing trained model."
        )

        print(
            "Skipping training phase ..."
        )

    else:

        print("\nTraining LSTM ...")

        best_val_loss = np.inf

        patience_counter = 0

        train_losses = []

        val_losses = []

        for epoch in range(EPOCHS):

            model.train()

            running_loss = 0.0

            for X_batch, y_batch in train_loader:

                X_batch = X_batch.to(
                    DEVICE,
                    non_blocking=True
                )

                y_batch = y_batch.to(
                    DEVICE,
                    non_blocking=True
                )

                optimizer.zero_grad()

                with torch.cuda.amp.autocast():

                    logits = model(X_batch)

                    loss = criterion(
                        logits,
                        y_batch
                    )

                scaler_amp.scale(
                    loss
                ).backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRAD_CLIP
                )

                scaler_amp.step(
                    optimizer
                )

                scaler_amp.update()

                running_loss += (
                    loss.item() *
                    X_batch.size(0)
                )

            train_loss = (
                running_loss /
                len(train_loader.dataset)
            )

            train_losses.append(train_loss)

            # --------------------------------------------------------------
            # Validation
            # --------------------------------------------------------------

            model.eval()

            val_running = 0.0

            with torch.no_grad():

                for X_batch, y_batch in val_loader:

                    X_batch = X_batch.to(
                        DEVICE,
                        non_blocking=True
                    )

                    y_batch = y_batch.to(
                        DEVICE,
                        non_blocking=True
                    )

                    with torch.cuda.amp.autocast():

                        logits = model(X_batch)

                        loss = criterion(
                            logits,
                            y_batch
                        )

                    val_running += (
                        loss.item() *
                        X_batch.size(0)
                    )

            val_loss = (
                val_running /
                len(val_loader.dataset)
            )

            val_losses.append(val_loss)

            print(
                f"Epoch {epoch+1:02d}/{EPOCHS} | "
                f"Train: {train_loss:.5f} | "
                f"Val: {val_loss:.5f}"
            )

            # --------------------------------------------------------------
            # Early stopping
            # --------------------------------------------------------------

            if val_loss < best_val_loss:

                best_val_loss = val_loss

                patience_counter = 0

                torch.save(
                    model.state_dict(),
                    model_path
                )

            else:

                patience_counter += 1

                if patience_counter >= PATIENCE:

                    print(
                        "\nEarly stopping triggered."
                    )

                    break

        save_training_curve(
            train_losses,
            val_losses,
        )

    # ----------------------------------------------------------------------
    # Load best model
    # ----------------------------------------------------------------------

    print("\nLoading best model ...")

    model.load_state_dict(

        torch.load(
            model_path,
            map_location=DEVICE,
            weights_only=True
        )
    )

    # ----------------------------------------------------------------------
    # Evaluation
    # ----------------------------------------------------------------------

    print("\nEvaluating forecasting model ...")

    (
        metrics,
        y_true,
        y_pred,
        y_probs,
    ) = evaluate_model(
        model,
        test_loader,
    )

    # ----------------------------------------------------------------------
    # Save outputs
    # ----------------------------------------------------------------------

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        METRICS_DIR / "forecast_metrics.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

    save_confusion_matrix(
        y_true,
        y_pred,
    )

    save_roc_curve(
        y_true,
        y_probs,
    )

    # ----------------------------------------------------------------------

    print("\n" + "=" * 72)
    print("LSTM FORECASTING COMPLETE")
    print("=" * 72)

    print("\nMetrics:")

    for k, v in metrics.items():

        print(
            f"  {k:<24} {v:.6f}"
        )

    print("\nSaved:")
    print("  models_lstm/")
    print("  reports_lstm/")

    print("\nMost important metric:")
    print("  class_1_recall")

    print("\nDONE.\n")

# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Phase 7 — LSTM Forecasting"
    )

    args = parser.parse_args()

    main()