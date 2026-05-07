"""
Phase 4 — Train / Test Split
==============================
Creates physically meaningful ML splits from the feature-engineered dataset.

Recommended split (primary):
  Train : Scenarios 1–4  (~80% of data)
  Test  : Scenario  5    (~20% of data, emergency blowout)

Rationale: Testing on Scenario 5 evaluates model generalisation to the most
extreme hazard regime — conditions that were NOT seen during training.
A model that fails on S5 cannot be trusted for safety-critical deployment.

Additional splits also created (for ablation studies):
  Temporal split  : Train on t=[0,225 s], Test on t=[225,300 s]
                    Tests whether the model predicts late-stage plume evolution.
  Spatial split   : Train on main tunnel, Test on dead-ends/stub
                    Tests whether the model generalises to accumulation zones.

Outputs:
  data/final/train.csv                — primary train split (S1-S4)
  data/final/test.csv                 — primary test split  (S5)
  data/final/train_temporal.csv       — temporal train (t < 225 s)
  data/final/test_temporal.csv        — temporal test  (t >= 225 s)
  data/final/train_spatial.csv        — spatial train (main tunnel)
  data/final/test_spatial.csv         — spatial test  (dead-ends + stub)
  reports/summaries/split_report.txt  — class distribution in each split

Usage:
  python 04_train_test_split.py
  python 04_train_test_split.py --input data/processed/greenmining_features.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Matplotlib for split visualisation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent

RISK_ORDER       = ["SAFE", "WARNING", "DANGER"]
RISK_COLOURS     = {"SAFE": "#2ecc71", "WARNING": "#f39c12", "DANGER": "#e74c3c"}

TRAIN_SCENARIOS  = [1, 2, 3, 4]
TEST_SCENARIOS   = [5]

TEMPORAL_CUTOFF  = 225.0   # seconds — last 25% (75s) of simulation used for test
SPATIAL_DEAD_END_COL = "in_chamber"   # 0 = main tunnel, 1 = dead-end/stub


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

def distribution_str(df: pd.DataFrame, label: str) -> list[str]:
    """Return a list of formatted lines describing class distribution."""
    lines = [f"  {label}:  {len(df):,} rows"]

    if "Risk" in df.columns:
        counts = df["Risk"].astype(str).value_counts()
        total  = len(df)
        for cls in RISK_ORDER:
            n   = counts.get(cls, 0)
            pct = 100 * n / total
            lines.append(f"    {cls:8s}: {n:>8,}  ({pct:5.1f}%)")

    if "Scenario" in df.columns:
        sc_counts = df["Scenario"].value_counts().sort_index()
        lines.append(f"  Scenarios present: {sorted(sc_counts.index.tolist())}")
        for sid, n in sc_counts.items():
            lines.append(f"    S{sid}: {n:,} rows")

    if "Time" in df.columns:
        lines.append(f"  Time range: [{df['Time'].min():.0f}s, {df['Time'].max():.0f}s]")

    return lines


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------

def verify_no_leakage(train: pd.DataFrame, test: pd.DataFrame, split_name: str) -> None:
    """Confirm there is no row-level overlap between train and test."""
    key_cols = [c for c in ["Scenario", "Time", "x", "y", "z"] if c in train.columns]
    train_keys = set(train[key_cols].itertuples(index=False, name=None))
    test_keys  = set(test[key_cols].itertuples(index=False, name=None))
    overlap = train_keys & test_keys
    if overlap:
        print(f"  [ERROR] {split_name}: {len(overlap):,} rows appear in both train and test!")
    else:
        print(f"  [{split_name}] No data leakage — train/test sets are disjoint ✓")


def print_class_warnings(train: pd.DataFrame, test: pd.DataFrame, split_name: str) -> list[str]:
    """Warn if DANGER or WARNING class is absent from train or test."""
    warnings_out = []
    for name, df in [("train", train), ("test", test)]:
        if "Risk" not in df.columns:
            continue
        risk_str = df["Risk"].astype(str)
        for cls in ["WARNING", "DANGER"]:
            count = (risk_str == cls).sum()
            if count == 0:
                msg = f"  [WARN] {split_name} {name}: '{cls}' class has 0 rows — cannot train/evaluate this class"
                print(msg)
                warnings_out.append(msg)
            elif count < 100:
                msg = f"  [WARN] {split_name} {name}: '{cls}' class has only {count} rows — consider oversampling"
                print(msg)
                warnings_out.append(msg)
    return warnings_out


# ---------------------------------------------------------------------------
# Split functions
# ---------------------------------------------------------------------------

def scenario_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Primary split: Train S1-S4, Test S5."""
    train = df[df["Scenario"].isin(TRAIN_SCENARIOS)].copy()
    test  = df[df["Scenario"].isin(TEST_SCENARIOS)].copy()
    return train, test


def temporal_split(df: pd.DataFrame, cutoff: float = TEMPORAL_CUTOFF) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Temporal split: Train on early time steps, Test on late time steps."""
    train = df[df["Time"] < cutoff].copy()
    test  = df[df["Time"] >= cutoff].copy()
    return train, test


def spatial_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Spatial split: Train on main tunnel, Test on dead-ends and stub."""
    if SPATIAL_DEAD_END_COL not in df.columns:
        raise ValueError(
            f"Column '{SPATIAL_DEAD_END_COL}' not found. "
            "Run 03_feature_engineering.py first to add geometry features."
        )
    train = df[df[SPATIAL_DEAD_END_COL] == 0].copy()   # main tunnel
    test  = df[df[SPATIAL_DEAD_END_COL] == 1].copy()   # dead-ends / stub
    return train, test


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _save_fig(fig: plt.Figure, name: str) -> None:
    path = REPO_ROOT / "reports" / "plots" / f"split_{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved → reports/plots/split_{name}.png")


def plot_split_class_comparison(
    splits: dict[str, tuple[pd.DataFrame, pd.DataFrame]]
) -> None:
    """Side-by-side bar charts comparing risk class distribution in train vs test for each split."""

    n_splits = len(splits)
    fig, axes = plt.subplots(1, n_splits, figsize=(6 * n_splits, 5))
    if n_splits == 1:
        axes = [axes]

    for ax, (split_name, (train, test)) in zip(axes, splits.items()):
        width = 0.35
        x = np.arange(len(RISK_ORDER))
        total_train = max(len(train), 1)
        total_test  = max(len(test), 1)

        train_risk = train["Risk"].astype(str).value_counts() if "Risk" in train.columns else {}
        test_risk  = test["Risk"].astype(str).value_counts()  if "Risk" in test.columns  else {}

        train_pcts = [100 * train_risk.get(r, 0) / total_train for r in RISK_ORDER]
        test_pcts  = [100 * test_risk.get(r, 0)  / total_test  for r in RISK_ORDER]

        b1 = ax.bar(x - width/2, train_pcts, width, label="Train",
                    color=[RISK_COLOURS[r] for r in RISK_ORDER], alpha=0.8, edgecolor="white")
        b2 = ax.bar(x + width/2, test_pcts,  width, label="Test",
                    color=[RISK_COLOURS[r] for r in RISK_ORDER], alpha=0.4, edgecolor="black",
                    linewidth=0.8, hatch="//")

        ax.set_xticks(x)
        ax.set_xticklabels(RISK_ORDER, fontsize=10)
        ax.set_ylabel("% of split")
        ax.set_title(f"{split_name}\nTrain: {len(train):,}  Test: {len(test):,}")
        ax.set_ylim(0, 105)
        ax.legend(fontsize=9)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))

    fig.suptitle("Risk Class Distribution — Train vs Test", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, "class_comparison")


def plot_scenario_split_overview(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Timeline plot: which scenario goes to train vs test."""
    fig, ax = plt.subplots(figsize=(10, 4))

    all_scenarios = sorted(pd.concat([train, test])["Scenario"].unique())
    y_pos = {s: i for i, s in enumerate(all_scenarios)}
    height = 0.6

    for df, colour, label in [(train, "#3498db", "Train"), (test, "#e74c3c", "Test")]:
        for sid in df["Scenario"].unique():
            sub = df[df["Scenario"] == sid]
            t_min, t_max = sub["Time"].min(), sub["Time"].max()
            ax.barh(y_pos[sid], t_max - t_min, left=t_min,
                    height=height, color=colour, alpha=0.8,
                    edgecolor="white", label=label if sid == list(df["Scenario"].unique())[0] else "")

    ax.set_yticks(list(y_pos.values()))
    ax.set_yticklabels([f"Scenario {s}" for s in all_scenarios])
    ax.set_xlabel("Simulation time [s]")
    ax.set_title("Scenario-Based Split Overview")
    ax.axvline(300, color="gray", linewidth=0.8, linestyle=":")

    # Legend (deduplicate)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=10)

    fig.tight_layout()
    _save_fig(fig, "scenario_timeline")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(input_path: Path, output_dir: Path, report_path: Path) -> None:
    print("=" * 64)
    print("  GreenMining — Phase 4: Train / Test Split")
    print("=" * 64)

    if not input_path.exists():
        print(f"\n  ERROR: {input_path} not found.")
        print("  Run 03_feature_engineering.py first.")
        sys.exit(1)

    print(f"\n  Loading {input_path.relative_to(REPO_ROOT)} ...")
    df = pd.read_csv(input_path)
    print(f"  Rows: {len(df):,}   Columns: {len(df.columns)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report_lines = [
        "=" * 64,
        "  GreenMining — Split Report",
        "=" * 64,
        f"  Input: {input_path.relative_to(REPO_ROOT)}",
        f"  Total rows: {len(df):,}",
        "",
    ]
    all_warnings = []

    # ------------------------------------------------------------------ #
    # 1. Primary split: Scenario-based (recommended)
    # ------------------------------------------------------------------ #
    print("\n  [1/3] Scenario-based split (RECOMMENDED) ...")
    print(f"    Train: Scenarios {TRAIN_SCENARIOS}")
    print(f"    Test:  Scenarios {TEST_SCENARIOS}")

    train_sc, test_sc = scenario_split(df)
    verify_no_leakage(train_sc, test_sc, "SCENARIO")
    w = print_class_warnings(train_sc, test_sc, "SCENARIO")
    all_warnings.extend(w)

    train_sc.to_csv(output_dir / "train.csv", index=False)
    test_sc.to_csv(output_dir  / "test.csv",  index=False)
    print(f"    Train rows: {len(train_sc):,}  → data/final/train.csv")
    print(f"    Test rows:  {len(test_sc):,}   → data/final/test.csv")

    report_lines += ["  SPLIT 1: Scenario-based (Primary — Recommended)", "  " + "-" * 60]
    report_lines += distribution_str(train_sc, "Train (S1-S4)")
    report_lines += distribution_str(test_sc,  "Test  (S5)")
    report_lines += [
        "",
        "  Rationale: S5 (Emergency Blowout) was NOT seen during training.",
        "  A model that generalises to S5 is trustworthy for extreme conditions.",
        "",
    ]

    # ------------------------------------------------------------------ #
    # 2. Temporal split
    # ------------------------------------------------------------------ #
    print(f"\n  [2/3] Temporal split (cutoff = {TEMPORAL_CUTOFF} s) ...")
    train_t, test_t = temporal_split(df)
    verify_no_leakage(train_t, test_t, "TEMPORAL")
    w = print_class_warnings(train_t, test_t, "TEMPORAL")
    all_warnings.extend(w)

    train_t.to_csv(output_dir / "train_temporal.csv", index=False)
    test_t.to_csv(output_dir  / "test_temporal.csv",  index=False)
    print(f"    Train (t<{TEMPORAL_CUTOFF}s): {len(train_t):,} rows")
    print(f"    Test  (t≥{TEMPORAL_CUTOFF}s): {len(test_t):,} rows")

    report_lines += ["  SPLIT 2: Temporal", "  " + "-" * 60]
    report_lines += distribution_str(train_t, f"Train (t < {TEMPORAL_CUTOFF} s)")
    report_lines += distribution_str(test_t,  f"Test  (t >= {TEMPORAL_CUTOFF} s)")
    report_lines += [
        "",
        "  Rationale: Tests whether model predicts late-stage plume evolution",
        "  (t=225-300 s) from early-stage training data (t=0-225 s).",
        "",
    ]

    # ------------------------------------------------------------------ #
    # 3. Spatial split
    # ------------------------------------------------------------------ #
    print("\n  [3/3] Spatial split (main tunnel → train, dead-ends → test) ...")
    try:
        train_sp, test_sp = spatial_split(df)
        verify_no_leakage(train_sp, test_sp, "SPATIAL")
        w = print_class_warnings(train_sp, test_sp, "SPATIAL")
        all_warnings.extend(w)

        train_sp.to_csv(output_dir / "train_spatial.csv", index=False)
        test_sp.to_csv(output_dir  / "test_spatial.csv",  index=False)
        print(f"    Train (main tunnel): {len(train_sp):,} rows")
        print(f"    Test  (dead-ends):   {len(test_sp):,} rows")

        report_lines += ["  SPLIT 3: Spatial", "  " + "-" * 60]
        report_lines += distribution_str(train_sp, "Train (main tunnel)")
        report_lines += distribution_str(test_sp,  "Test  (dead-ends + stub)")
        report_lines += [
            "",
            "  Rationale: Tests generalisation to accumulation zones where",
            "  gas concentration dynamics differ from through-flow regions.",
            "  CAUTION: Most DANGER cells are in dead-ends — test set will be",
            "  DANGER-heavy while train set may be SAFE-dominated.",
            "",
        ]
    except ValueError as exc:
        print(f"    SKIP: {exc}")
        train_sp, test_sp = None, None

    # ------------------------------------------------------------------ #
    # Warnings summary in report
    # ------------------------------------------------------------------ #
    if all_warnings:
        report_lines += ["  WARNINGS", "  " + "-" * 60]
        report_lines.extend(all_warnings)
        report_lines += [""]

    # ------------------------------------------------------------------ #
    # ML recommendations
    # ------------------------------------------------------------------ #
    report_lines += [
        "  ML RECOMMENDATIONS",
        "  " + "-" * 60,
        "",
        "  Primary split (scenario-based):",
        "    - Use class_weight='balanced' in all classifiers",
        "    - Evaluate on DANGER recall first — false negatives are critical",
        "    - Report per-class F1, not just accuracy",
        "    - Confusion matrix: expect high SAFE precision, check WARNING/DANGER recall",
        "",
        "  Class imbalance handling:",
        "    - DO NOT use SMOTE or random oversampling — synthetic CFD data is",
        "      physically meaningless and will leak geometry information",
        "    - USE class_weight='balanced' or adjust decision threshold post-training",
        "    - USE stratified cross-validation within the training set",
        "",
        "  Features to prioritise:",
        "    - dist_source: strongest predictor of gas presence",
        "    - in_chamber: dead-end cells have 3-10x higher DANGER rate",
        "    - dCH4_dt: rising CH4 is an early warning signal",
        "    - gas_LEL_equiv: direct explosion risk metric",
        "    - low_velocity: stagnation correlates with accumulation",
        "",
        "  Recommended first models (Phase 6):",
        "    1. Random Forest  (class_weight='balanced', n_estimators=500)",
        "    2. XGBoost        (scale_pos_weight tuned to DANGER:SAFE ratio)",
        "    3. LightGBM       (is_unbalance=True)",
        "",
    ]

    # ------------------------------------------------------------------ #
    # Output files summary
    # ------------------------------------------------------------------ #
    report_lines += [
        "  OUTPUT FILES",
        "  " + "-" * 60,
        f"    data/final/train.csv            {len(train_sc):>9,} rows  (primary — use this first)",
        f"    data/final/test.csv             {len(test_sc):>9,} rows  (primary — S5 only)",
        f"    data/final/train_temporal.csv   {len(train_t):>9,} rows  (ablation)",
        f"    data/final/test_temporal.csv    {len(test_t):>9,} rows  (ablation)",
    ]
    if train_sp is not None:
        report_lines += [
            f"    data/final/train_spatial.csv    {len(train_sp):>9,} rows  (ablation)",
            f"    data/final/test_spatial.csv     {len(test_sp):>9,} rows  (ablation)",
        ]

    # Save report
    report_path.write_text("\n".join(report_lines))
    print(f"\n  Report → {report_path.relative_to(REPO_ROOT)}")

    # ------------------------------------------------------------------ #
    # Plots
    # ------------------------------------------------------------------ #
    print("\n  Generating split visualisations ...")
    active_splits = {"Scenario": (train_sc, test_sc), "Temporal": (train_t, test_t)}
    if train_sp is not None:
        active_splits["Spatial"] = (train_sp, test_sp)
    plot_split_class_comparison(active_splits)
    plot_scenario_split_overview(train_sc, test_sc)

    # ------------------------------------------------------------------ #
    # Final summary
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 64)
    print("  SPLIT SUMMARY")
    print("=" * 64)
    print(f"\n  PRIMARY SPLIT (for Phase 6 model training):")
    print(f"    Train : {len(train_sc):,} rows — Scenarios {TRAIN_SCENARIOS}")
    print(f"    Test  : {len(test_sc):,} rows  — Scenarios {TEST_SCENARIOS} (emergency blowout)")
    print(f"\n  Start training with:  data/final/train.csv")
    print(f"  Evaluate models on:   data/final/test.csv")
    print(f"\n  Key metric to watch:  DANGER class recall")
    print(f"  (a model that misses DANGER cells is dangerous in production)\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4: Train/test split.")
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "greenmining_features.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "final",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "reports" / "summaries" / "split_report.txt",
    )
    args = parser.parse_args()
    main(args.input, args.output_dir, args.report)
