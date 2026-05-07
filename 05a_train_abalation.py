"""
Phase 5B — Physics-Driven Ablation Experiment
=============================================
Train baseline ML models WITHOUT direct gas concentration features.

Goal:
------
Test whether hazard emergence can be inferred from:
- geometry
- airflow structure
- recirculation behavior
- accumulation dynamics
- spatial topology

WITHOUT direct concentration-state inputs.

This script DOES NOT overwrite previous baseline results.

Outputs:
--------
models_ablation/
reports_ablation/

Tasks:
------
1. Binary hazard classification
2. 3-class hazard classification

Usage:
------
python 05b_train_ablation_models.py
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
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# ============================================================================
# Gradient Boosting
# ============================================================================

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# ============================================================================
# Repository paths
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parent

MODELS_DIR = REPO_ROOT / "models_ablation"

REPORTS_DIR = REPO_ROOT / "reports_ablation"

METRICS_DIR = REPORTS_DIR / "metrics"
PLOTS_DIR = REPORTS_DIR / "plots"

# ============================================================================
# Configuration
# ============================================================================

RANDOM_STATE = 42

RISK_ORDER = ["SAFE", "WARNING", "DANGER"]

# ============================================================================
# FULL GAS-STATE ABLATION
# ============================================================================

ABLATION_DROP = [

    # ----------------------------------------------------------------------
    # Direct concentrations
    # ----------------------------------------------------------------------

    "CH4",
    "CO",
    "H2",

    # ----------------------------------------------------------------------
    # Fractions
    # ----------------------------------------------------------------------

    "CH4_frac",
    "CO_frac",
    "H2_frac",

    # ----------------------------------------------------------------------
    # Logs
    # ----------------------------------------------------------------------

    "CH4_log",
    "CO_log",
    "H2_log",

    # ----------------------------------------------------------------------
    # Aggregate gas metrics
    # ----------------------------------------------------------------------

    "total_gas",
    "gas_LEL_equiv",
    "co_toxicity_ratio",

    # ----------------------------------------------------------------------
    # Temporal concentration derivatives
    # ----------------------------------------------------------------------

    "dCH4_dt",
    "dCO_dt",
    "dH2_dt",
    "dCH4_dt_abs",

    # ----------------------------------------------------------------------
    # Derived accumulation indicators
    # ----------------------------------------------------------------------

    "accumulating",
]

# ============================================================================
# Labels / leakage
# ============================================================================

DROP_COLUMNS = [
    "Risk",
    "hazard_binary",
    "hazard_3class",
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


def prepare_features(df, target_col):

    drop_cols = (
        DROP_COLUMNS +
        ABLATION_DROP
    )

    drop_cols = [
        c for c in drop_cols
        if c in df.columns
    ]

    X = df.drop(columns=drop_cols)

    # ----------------------------------------------------------------------
    # Categorical encoding
    # ----------------------------------------------------------------------

    if "zone" in X.columns:

        X = pd.get_dummies(
            X,
            columns=["zone"],
            drop_first=True
        )

    # ----------------------------------------------------------------------
    # Bool → int
    # ----------------------------------------------------------------------

    bool_cols = X.select_dtypes(include=["bool"]).columns

    for col in bool_cols:
        X[col] = X[col].astype("int8")

    y = df[target_col]

    return X, y


def save_json(obj, path):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4)


def save_model(model, path):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(model, path)

    size_mb = path.stat().st_size / (1024**2)

    print(
        f"    Saved {path.name:<40}"
        f"{size_mb:8.1f} MB"
    )


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_model(
    model,
    X_test,
    y_test,
    task_name,
    model_name,
    labels,
):

    y_pred = model.predict(X_test)

    metrics = {}

    metrics["accuracy"] = float(
        accuracy_score(y_test, y_pred)
    )

    metrics["precision_macro"] = float(
        precision_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    metrics["recall_macro"] = float(
        recall_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    metrics["f1_macro"] = float(
        f1_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    report_dict = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    for label in labels:

        key = str(label)

        if key in report_dict:

            metrics[f"class_{label}_recall"] = (
                report_dict[key]["recall"]
            )

            metrics[f"class_{label}_precision"] = (
                report_dict[key]["precision"]
            )

    # ROC-AUC only for binary
    if len(labels) == 2:

        y_prob = model.predict_proba(X_test)[:, 1]

        metrics["roc_auc"] = float(
            roc_auc_score(y_test, y_prob)
        )

    return metrics, y_pred


# ============================================================================
# Plotting
# ============================================================================

def save_confusion_matrix(
    y_true,
    y_pred,
    labels,
    task_name,
    model_name,
):

    fig, ax = plt.subplots(figsize=(6, 5))

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )

    disp.plot(ax=ax)

    ax.set_title(
        f"{model_name} — {task_name}"
    )

    path = (
        PLOTS_DIR /
        f"cm_{task_name}_{model_name}.png"
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fig.savefig(
        path,
        dpi=120,
        bbox_inches="tight"
    )

    plt.close(fig)


def save_feature_importance(
    model,
    feature_names,
    task_name,
    model_name,
):

    if not hasattr(model, "feature_importances_"):
        return

    importance = model.feature_importances_

    idx = np.argsort(importance)[::-1][:20]

    top_features = np.array(feature_names)[idx]
    top_importance = importance[idx]

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.barh(
        top_features[::-1],
        top_importance[::-1]
    )

    ax.set_title(
        f"Ablation Feature Importance — "
        f"{model_name} ({task_name})"
    )

    ax.set_xlabel("Importance")

    path = (
        PLOTS_DIR /
        f"importance_{task_name}_{model_name}.png"
    )

    fig.savefig(
        path,
        dpi=120,
        bbox_inches="tight"
    )

    plt.close(fig)


def save_roc_curve(
    model,
    X_test,
    y_test,
    task_name,
    model_name,
):

    y_prob = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(
        y_test,
        y_prob
    )

    auc = roc_auc_score(
        y_test,
        y_prob
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(
        fpr,
        tpr,
        label=f"AUC = {auc:.3f}"
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    ax.set_title(
        f"ROC Curve — {model_name} ({task_name})"
    )

    ax.legend()

    path = (
        PLOTS_DIR /
        f"roc_{task_name}_{model_name}.png"
    )

    fig.savefig(
        path,
        dpi=120,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================================
# Models
# ============================================================================

def build_models(task_type):

    objective = (
        "binary"
        if task_type == "binary"
        else "multiclass"
    )

    return {

        "random_forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=18,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            class_weight="balanced_subsample",
        ),

        "xgboost": XGBClassifier(
            objective=(
                "binary:logistic"
                if task_type == "binary"
                else "multi:softprob"
            ),
            n_estimators=200,
            max_depth=8,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric=(
                "logloss"
                if task_type == "binary"
                else "mlogloss"
            ),
            random_state=RANDOM_STATE,
            tree_method="hist",
            n_jobs=-1,
        ),

        "lightgbm": LGBMClassifier(
            objective=objective,
            n_estimators=200,
            learning_rate=0.08,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


# ============================================================================
# Task runner
# ============================================================================

def run_task(
    task_name,
    target_col,
    labels,
    train_df,
    val_df,
    test_df,
):

    print("\n" + "=" * 72)
    print(f"  TASK: {task_name} (ABLATION)")
    print("=" * 72)

    X_train, y_train = prepare_features(
        train_df,
        target_col
    )

    X_val, y_val = prepare_features(
        val_df,
        target_col
    )

    X_test, y_test = prepare_features(
        test_df,
        target_col
    )

    print("\nFeatures used:")

    for col in X_train.columns:
        print(f"  {col}")

    print("\nFeature matrix:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val  : {X_val.shape}")
    print(f"  Test : {X_test.shape}")

    models = build_models(task_name)

    all_metrics = {}

    for model_name, model in models.items():

        print(f"\nTraining: {model_name}")

        model.fit(X_train, y_train)

        metrics, y_pred = evaluate_model(
            model,
            X_test,
            y_test,
            task_name,
            model_name,
            labels,
        )

        all_metrics[model_name] = metrics

        save_json(
            metrics,
            METRICS_DIR /
            f"{task_name}_{model_name}.json"
        )

        save_model(
            model,
            MODELS_DIR /
            f"{task_name}_{model_name}.pkl"
        )

        save_confusion_matrix(
            y_test,
            y_pred,
            labels,
            task_name,
            model_name,
        )

        save_feature_importance(
            model,
            X_train.columns,
            task_name,
            model_name,
        )

        if len(labels) == 2:

            save_roc_curve(
                model,
                X_test,
                y_test,
                task_name,
                model_name,
            )

    return all_metrics


# ============================================================================
# Main
# ============================================================================

def main(train_path, val_path, test_path):

    print("=" * 72)
    print("  GreenMining — Phase 5B: Physics-Driven Ablation")
    print("=" * 72)

    for path in [
        train_path,
        val_path,
        test_path,
    ]:

        if not path.exists():

            print(f"\nERROR: Missing file:")
            print(f"  {path}")

            sys.exit(1)

    print("\nLoading datasets ...")

    train_df = pd.read_csv(train_path)
    val_df   = pd.read_csv(val_path)
    test_df  = pd.read_csv(test_path)

    train_df = optimize_memory(train_df)
    val_df   = optimize_memory(val_df)
    test_df  = optimize_memory(test_df)

    print(f"\nTrain rows: {len(train_df):,}")
    print(f"Val rows  : {len(val_df):,}")
    print(f"Test rows : {len(test_df):,}")

    # ----------------------------------------------------------------------
    # Binary
    # ----------------------------------------------------------------------

    binary_metrics = run_task(
        task_name="binary",
        target_col="hazard_binary",
        labels=[0, 1],
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )

    # ----------------------------------------------------------------------
    # Multiclass
    # ----------------------------------------------------------------------

    multiclass_metrics = run_task(
        task_name="multiclass",
        target_col="hazard_3class",
        labels=[0, 1, 2],
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )

    # ----------------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------------

    summary = {
        "binary": binary_metrics,
        "multiclass": multiclass_metrics,
    }

    save_json(
        summary,
        METRICS_DIR /
        "summary_metrics.json"
    )

    print("\n" + "=" * 72)
    print("ABLATION EXPERIMENT COMPLETE")
    print("=" * 72)

    print("\nSaved:")
    print("  models_ablation/")
    print("  reports_ablation/")

    print("\nThis experiment evaluates:")
    print("  Flow + geometry driven hazard inference")
    print("  WITHOUT direct concentration features")

    print("\nDONE.\n")


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Phase 5B — Physics-Driven Ablation"
    )

    parser.add_argument(
        "--train",
        type=Path,
        default=(
            REPO_ROOT /
            "data" /
            "final" /
            "train.csv"
        ),
    )

    parser.add_argument(
        "--val",
        type=Path,
        default=(
            REPO_ROOT /
            "data" /
            "final" /
            "val.csv"
        ),
    )

    parser.add_argument(
        "--test",
        type=Path,
        default=(
            REPO_ROOT /
            "data" /
            "final" /
            "test.csv"
        ),
    )

    args = parser.parse_args()

    main(
        args.train,
        args.val,
        args.test,
    )