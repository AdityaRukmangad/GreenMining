"""
Phase 5B — Physics-Driven Ablation Experiment (Improved)
=========================================================
Train baseline ML models WITHOUT direct gas concentration features.

Improvements over baseline ablation
-------------------------------------
- Physics-informed interaction features (no gas data needed)
- Scale-pos-weight / balanced class weighting to minimise false negatives
- Early stopping with validation set for XGB and LGB
- F2-score-based threshold tuning on val set (recall-prioritised)
- Reports false-negative count explicitly

Outputs
-------
models_ablation/
reports_ablation/
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ============================================================================
# Matplotlib
# ============================================================================

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================================
# Sklearn
# ============================================================================

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# ============================================================================
# Gradient boosting
# ============================================================================

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ============================================================================
# Paths
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parent

MODELS_DIR  = REPO_ROOT / "models_ablation"
REPORTS_DIR = REPO_ROOT / "reports_ablation"
METRICS_DIR = REPORTS_DIR / "metrics"
PLOTS_DIR   = REPORTS_DIR / "plots"

# ============================================================================
# Configuration
# ============================================================================

RANDOM_STATE = 42

# Recall weight in F-beta threshold search (beta=2 → recall 4× more important)
FBETA = 2.0

# ============================================================================
# Gas-state ablation columns (everything directly related to concentration)
# ============================================================================

ABLATION_DROP = [
    "CH4", "CO", "H2",
    "CH4_frac", "CO_frac", "H2_frac",
    "CH4_log", "CO_log", "H2_log",
    "total_gas", "gas_LEL_equiv", "co_toxicity_ratio",
    "dCH4_dt", "dCO_dt", "dH2_dt", "dCH4_dt_abs",
    "accumulating",
]

DROP_COLUMNS = ["Risk", "hazard_binary", "hazard_3class", "Scenario"]

# ============================================================================
# Helpers
# ============================================================================

def optimize_memory(df):
    df[df.select_dtypes("float64").columns] = (
        df.select_dtypes("float64").astype("float32")
    )
    df[df.select_dtypes("int64").columns] = (
        df.select_dtypes("int64").astype("int32")
    )
    return df


def add_interaction_features(X):
    """Physics-informed interactions: infer accumulation risk from flow/geometry."""
    if "low_velocity" in X.columns and "dist_source" in X.columns:
        X = X.copy()
        X["lv_x_dist_src"]   = X["low_velocity"] * X["dist_source"]
        X["lv_x_dist_outlet"] = X["low_velocity"] * X.get("dist_outlet", 0)
        X["vel_x_dist_outlet"] = X["Velocity"] * X.get("dist_outlet", 0)
        if "recirculation_proxy" in X.columns:
            X["recirc_x_lv"]     = X["recirculation_proxy"] * X["low_velocity"]
            if "in_chamber" in X.columns:
                X["recirc_x_chamber"] = X["recirculation_proxy"] * X["in_chamber"]
        if "time_norm" in X.columns:
            X["time_x_lv"] = X["time_norm"] * X["low_velocity"]
        if "Pressure" in X.columns:
            X["pressure_x_vel"] = X["Pressure"] * X["Velocity"]
    return X


def prepare_features(df, target_col):
    drop_cols = [c for c in (DROP_COLUMNS + ABLATION_DROP) if c in df.columns]
    X = df.drop(columns=drop_cols)

    if "zone" in X.columns:
        X = pd.get_dummies(X, columns=["zone"], drop_first=True)

    for col in X.select_dtypes("bool").columns:
        X[col] = X[col].astype("int8")

    X = add_interaction_features(X)

    y = df[target_col]
    return X, y


def compute_pos_weight(y):
    """neg / pos ratio for class-imbalance weighting."""
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    if n_pos == 0:
        return 1.0
    return float(n_neg / n_pos)


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4)


def save_model(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"    Saved {path.name:<40}{path.stat().st_size/(1024**2):8.1f} MB")

# ============================================================================
# Threshold optimisation (F-beta on validation set)
# ============================================================================

def find_optimal_threshold(model, X_val, y_val, beta=FBETA):
    """Search threshold that maximises F-beta (recall-weighted) on val set."""
    y_prob = model.predict_proba(X_val)[:, 1]
    best_t, best_score = 0.5, -1.0
    for t in np.arange(0.10, 0.90, 0.01):
        y_hat = (y_prob >= t).astype(int)
        score = fbeta_score(y_val, y_hat, beta=beta, pos_label=1, zero_division=0)
        if score > best_score:
            best_score = score
            best_t = float(t)
    return best_t, best_score

# ============================================================================
# Evaluation
# ============================================================================

def evaluate_model(model, X_test, y_test, task_name, model_name, labels, threshold=0.5):
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1] if len(labels) == 2 else None
        y_pred = (y_prob >= threshold).astype(int) if y_prob is not None else model.predict(X_test)
    else:
        y_pred = model.predict(X_test)
        y_prob = None

    metrics = {
        "threshold": threshold,
        "accuracy":         float(accuracy_score(y_test, y_pred)),
        "precision_macro":  float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro":     float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro":         float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
    }

    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    for label in labels:
        key = str(label)
        if key in report_dict:
            metrics[f"class_{label}_recall"]    = report_dict[key]["recall"]
            metrics[f"class_{label}_precision"] = report_dict[key]["precision"]
            metrics[f"class_{label}_f1"]        = report_dict[key]["f1-score"]

    # Explicit FN count
    if len(labels) == 2:
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        metrics["false_negatives"] = int(cm[1, 0])
        metrics["false_positives"] = int(cm[0, 1])
        if y_prob is not None:
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))

    return metrics, y_pred

# ============================================================================
# Plotting
# ============================================================================

def save_confusion_matrix(y_true, y_pred, labels, task_name, model_name):
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(ax=ax)
    ax.set_title(f"{model_name} — {task_name} (ablation)")
    path = PLOTS_DIR / f"cm_{task_name}_{model_name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_feature_importance(model, feature_names, task_name, model_name):
    if not hasattr(model, "feature_importances_"):
        return
    importance = model.feature_importances_
    idx = np.argsort(importance)[::-1][:20]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(np.array(feature_names)[idx][::-1], importance[idx][::-1])
    ax.set_title(f"Ablation Feature Importance — {model_name} ({task_name})")
    ax.set_xlabel("Importance")
    path = PLOTS_DIR / f"importance_{task_name}_{model_name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_roc_curve(model, X_test, y_test, task_name, model_name):
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name} ({task_name})")
    ax.legend()
    path = PLOTS_DIR / f"roc_{task_name}_{model_name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)

# ============================================================================
# Model factory
# ============================================================================

def build_models(task_type, pos_weight):
    pw = max(1.0, pos_weight)

    return {
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=22,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            class_weight="balanced",       # equal weight to every sample in each class
        ),

        "xgboost": XGBClassifier(
            objective=(
                "binary:logistic"
                if task_type == "binary"
                else "multi:softprob"
            ),
            n_estimators=800,
            max_depth=9,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.05,
            reg_lambda=1.0,
            # penalise missed positives
            scale_pos_weight=(pw if task_type == "binary" else 1.0),
            eval_metric=(
                "logloss" if task_type == "binary" else "mlogloss"
            ),
            early_stopping_rounds=30,
            random_state=RANDOM_STATE,
            tree_method="hist",
            n_jobs=-1,
            verbosity=0,
        ),

        "lightgbm": LGBMClassifier(
            objective=(
                "binary" if task_type == "binary" else "multiclass"
            ),
            n_estimators=800,
            learning_rate=0.05,
            max_depth=9,
            num_leaves=63,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_samples=20,
            reg_alpha=0.05,
            reg_lambda=1.0,
            scale_pos_weight=(pw if task_type == "binary" else 1.0),
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
    }

# ============================================================================
# Task runner
# ============================================================================

def run_task(task_name, target_col, labels, train_df, val_df, test_df):
    print("\n" + "=" * 72)
    print(f"  TASK: {task_name} (ABLATION — improved)")
    print("=" * 72)

    X_train, y_train = prepare_features(train_df, target_col)
    X_val,   y_val   = prepare_features(val_df,   target_col)
    X_test,  y_test  = prepare_features(test_df,  target_col)

    # Align columns after interaction features / dummies
    X_val  = X_val.reindex(columns=X_train.columns, fill_value=0)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    print(f"\nFeature count : {X_train.shape[1]}")
    print(f"Train  : {X_train.shape}  |  Val : {X_val.shape}  |  Test : {X_test.shape}")

    pos_weight = compute_pos_weight(y_train)
    print(f"\nPos-weight (neg/pos): {pos_weight:.2f}")

    models   = build_models(task_name, pos_weight)
    all_metrics = {}

    for model_name, model in models.items():
        print(f"\n── Training: {model_name}")

        # Early stopping for boosting models
        if model_name == "xgboost":
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        elif model_name == "lightgbm":
            from lightgbm import early_stopping, log_evaluation
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    early_stopping(stopping_rounds=30, verbose=False),
                    log_evaluation(period=-1),
                ],
            )
        else:
            model.fit(X_train, y_train)

        # Threshold tuning on validation set (binary only)
        if len(labels) == 2:
            threshold, fbeta_val = find_optimal_threshold(model, X_val, y_val)
            print(f"   Optimal threshold: {threshold:.2f}  (F{FBETA:.0f}={fbeta_val:.4f})")
        else:
            threshold = 0.5

        metrics, y_pred = evaluate_model(
            model, X_test, y_test,
            task_name, model_name, labels, threshold,
        )
        all_metrics[model_name] = metrics

        print(f"   Accuracy  : {metrics['accuracy']:.4f}")
        print(f"   Recall_1  : {metrics.get('class_1_recall', 'N/A')}")
        if "false_negatives" in metrics:
            print(f"   False Neg : {metrics['false_negatives']:,}")

        save_json(metrics, METRICS_DIR / f"{task_name}_{model_name}.json")
        save_model(model, MODELS_DIR / f"{task_name}_{model_name}.pkl")
        save_confusion_matrix(y_test, y_pred, labels, task_name, model_name)
        save_feature_importance(model, X_train.columns, task_name, model_name)

        if len(labels) == 2:
            save_roc_curve(model, X_test, y_test, task_name, model_name)

    return all_metrics

# ============================================================================
# Main
# ============================================================================

def main(train_path, val_path, test_path):
    print("=" * 72)
    print("  GreenMining — Phase 5B: Physics-Driven Ablation (Improved)")
    print("=" * 72)

    for path in [train_path, val_path, test_path]:
        if not path.exists():
            print(f"\nERROR: Missing file: {path}")
            sys.exit(1)

    print("\nLoading datasets ...")
    train_df = optimize_memory(pd.read_csv(train_path))
    val_df   = optimize_memory(pd.read_csv(val_path))
    test_df  = optimize_memory(pd.read_csv(test_path))

    print(f"  Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")

    binary_metrics = run_task(
        task_name="binary",
        target_col="hazard_binary",
        labels=[0, 1],
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )

    multiclass_metrics = run_task(
        task_name="multiclass",
        target_col="hazard_3class",
        labels=[0, 1, 2],
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )

    save_json(
        {"binary": binary_metrics, "multiclass": multiclass_metrics},
        METRICS_DIR / "summary_metrics.json",
    )

    print("\n" + "=" * 72)
    print("ABLATION EXPERIMENT COMPLETE")
    print("=" * 72)
    print("\nSaved:  models_ablation/  reports_ablation/")
    print("\nDONE.\n")


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 5B — Physics-Driven Ablation (Improved)")
    parser.add_argument("--train", type=Path,
        default=REPO_ROOT / "data" / "final" / "train.csv")
    parser.add_argument("--val",   type=Path,
        default=REPO_ROOT / "data" / "final" / "val.csv")
    parser.add_argument("--test",  type=Path,
        default=REPO_ROOT / "data" / "final" / "test.csv")
    args = parser.parse_args()
    main(args.train, args.val, args.test)
