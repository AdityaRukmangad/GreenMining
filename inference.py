"""
GreenMining — Inference Module
==============================

Loads all trained models and runs predictions on raw CFD data.

Supported models
----------------
  baseline   RF / XGBoost / LightGBM  (binary + multiclass)
  ablation   same, without gas-concentration features
  lstm       BiLSTM + Attention        (binary, t+30 s forecast)
  stgnn      GraphSAGE + BiLSTM        (binary, t+30 s forecast)
             → requires models_stgnn/best_stgnn.pt
             → requires models_stgnn/norm_stats.pt  (save from 09 training)

Quick start
-----------
    from inference import GreenMiningPredictor

    predictor = GreenMiningPredictor()
    print(predictor.status())

    results = predictor.predict_all(df_raw)
    # df_raw must have: Time, x, y, z, CH4, CO, H2,
    #                   Temperature, Velocity, Pressure
    # Scenario column is optional but improves dist_source + recirculation.

    # Or for a single model:
    preds = predictor.predict_baseline(df_raw, task="binary", algo="lightgbm")
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent

# ─────────────────────────────────────────────────────────────────────────────
# Physics constants  (must stay in sync with 03_feature_engineering.py)
# ─────────────────────────────────────────────────────────────────────────────

_EPS               = 1e-9
_LEL_CH4           = 0.050
_LEL_H2            = 0.040
_CO_DANGER         = 0.0005
_OUTLET_X          = 80.0
_STAGNATION_VEL    = 0.3
_F32               = np.float32

_INLET_VELOCITY = {1: 2.0, 2: 1.5, 3: 0.5, 4: 1.0, 5: 2.0}

_SOURCE_POINTS = {
    1: [],
    2: [(18.0,  2.0, 1.5)],
    3: [(25.0, 13.0, 1.5), (56.0, 17.0, 1.5), (39.0, -4.0, 1.5)],
    4: [(25.0, 14.0, 1.5)],
    5: [(25.0, 13.0, 1.5), (56.0, 17.0, 1.5), (39.0, -4.0, 1.5)],
}

_ZONE_X = {
    "INLET_SECTION":  ( 0.0, 20.0),
    "JUNCTION_1":     (20.0, 30.0),
    "MID_TUNNEL":     (30.0, 50.0),
    "JUNCTION_2_3":   (50.0, 62.0),
    "OUTLET_SECTION": (62.0, 80.0),
}

_DEAD_END_ZONES = [
    (20.0, 30.0,  4.0, 16.0, 0.0, 3.0, "CHAMBER_1"),
    (50.0, 62.0,  4.0, 20.0, 0.0, 3.0, "CHAMBER_2"),
    (35.0, 43.0, -6.0,  0.0, 0.0, 3.0, "SOUTH_STUB"),
]

# All zone dummy columns produced after one-hot encoding (drop_first=False).
# Columns absent in the data after engineering are filled with 0.
_ZONE_DUMMIES = [
    "zone_CHAMBER_2",
    "zone_INLET_SECTION",
    "zone_JUNCTION_1",
    "zone_JUNCTION_2_3",
    "zone_MID_TUNNEL",
    "zone_OUTLET_SECTION",
    "zone_SOUTH_STUB",
]

# Columns to drop before passing to any model
_LABEL_COLS = [
    "Risk", "hazard_binary", "hazard_3class",
    "future_hazard_binary", "future_hazard_3class",
    "Scenario", "zone",
]

# Gas-state columns removed in the ablation experiment
_ABLATION_DROP = [
    "CH4", "CO", "H2",
    "CH4_frac", "CO_frac", "H2_frac",
    "CH4_log",  "CO_log",  "H2_log",
    "total_gas", "gas_LEL_equiv", "co_toxicity_ratio",
    "dCH4_dt", "dCO_dt", "dH2_dt", "dCH4_dt_abs", "accumulating",
]


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering  (mirrors 03_feature_engineering.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the full Phase-03 pipeline to a raw CFD DataFrame.

    Required columns : Time, x, y, z, CH4, CO, H2, Temperature, Velocity, Pressure
    Optional columns : Scenario  (improves dist_source and recirculation_proxy)

    Returns the input DataFrame extended with all engineered features.
    The original columns are preserved so callers can still access raw values.
    """
    df = df.copy()
    _cast_dtypes(df)
    df = _geometry(df)
    df = _concentrations(df)
    df = _velocity(df)
    df = _time_norm(df)
    df = _temporal_gradients(df)
    df = _one_hot_zone(df)
    _cast_dtypes(df)
    return df


def _cast_dtypes(df: pd.DataFrame) -> None:
    for col in df.select_dtypes("float64").columns:
        df[col] = df[col].astype(_F32)
    for col in df.select_dtypes("int64").columns:
        df[col] = df[col].astype(np.int32)


def _geometry(df: pd.DataFrame) -> pd.DataFrame:
    x, y, z = df["x"].values, df["y"].values, df["z"].values

    df["dist_inlet"]  = np.sqrt(x**2         + (y - 2.0)**2 + (z - 1.5)**2).astype(_F32)
    df["dist_outlet"] = np.sqrt((x - _OUTLET_X)**2 + (y - 2.0)**2 + (z - 1.5)**2).astype(_F32)

    dist_src = np.full(len(df), _OUTLET_X, dtype=_F32)

    if "Scenario" in df.columns:
        for sid, pts in _SOURCE_POINTS.items():
            mask = df["Scenario"].values == sid
            if not np.any(mask) or not pts:
                continue
            best = np.full(mask.sum(), np.inf, dtype=_F32)
            for px, py, pz in pts:
                d = np.sqrt((x[mask]-px)**2 + (y[mask]-py)**2 + (z[mask]-pz)**2)
                best = np.minimum(best, d)
            dist_src[mask] = best
    else:
        all_pts = [p for pts in _SOURCE_POINTS.values() for p in pts]
        if all_pts:
            best = np.full(len(df), np.inf, dtype=_F32)
            for px, py, pz in all_pts:
                d = np.sqrt((x-px)**2 + (y-py)**2 + (z-pz)**2)
                best = np.minimum(best, d)
            dist_src = best

    df["dist_source"] = dist_src

    in_ch = np.zeros(len(df), dtype=np.int8)
    zone  = np.full(len(df), "MAIN_TUNNEL", dtype=object)

    for x0, x1, y0, y1, z0, z1, name in _DEAD_END_ZONES:
        mask = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1) & (z >= z0) & (z <= z1)
        in_ch[mask] = 1
        zone[mask]  = name

    main = in_ch == 0
    for name, (x0, x1) in _ZONE_X.items():
        mask = main & (x >= x0) & (x < x1)
        zone[mask] = name

    df["in_chamber"] = in_ch
    df["zone"]       = pd.Categorical(zone)
    return df


def _concentrations(df: pd.DataFrame) -> pd.DataFrame:
    total = df["CH4"] + df["CO"] + df["H2"]
    df["total_gas"]        = total.astype(_F32)
    df["CH4_frac"]         = (df["CH4"] / (total + _EPS)).astype(_F32)
    df["CO_frac"]          = (df["CO"]  / (total + _EPS)).astype(_F32)
    df["H2_frac"]          = (df["H2"]  / (total + _EPS)).astype(_F32)
    df["gas_LEL_equiv"]    = (df["CH4"] / _LEL_CH4 + df["H2"] / _LEL_H2).astype(_F32)
    df["co_toxicity_ratio"]= (df["CO"]  / _CO_DANGER).astype(_F32)
    for gas in ("CH4", "CO", "H2"):
        df[f"{gas}_log"]   = np.log10(df[gas] + 1e-6).astype(_F32)
    return df


def _velocity(df: pd.DataFrame) -> pd.DataFrame:
    vel = df["Velocity"]
    df["low_velocity"] = (vel < _STAGNATION_VEL).astype(np.int8)

    inlet = (
        df["Scenario"].map(_INLET_VELOCITY)
        if "Scenario" in df.columns
        else pd.Series(np.full(len(df), 1.5), index=df.index)
    )
    df["recirculation_proxy"] = (
        (df["in_chamber"] == 1) & (vel < 0.5 * inlet)
    ).astype(np.int8)
    return df


def _time_norm(df: pd.DataFrame) -> pd.DataFrame:
    df["time_norm"] = (df["Time"] / 300.0).astype(_F32)
    return df


def _temporal_gradients(df: pd.DataFrame) -> pd.DataFrame:
    grp_cols = (["Scenario", "x", "y", "z"] if "Scenario" in df.columns
                else ["x", "y", "z"])
    df = df.sort_values(grp_cols + ["Time"]).reset_index(drop=True)
    grp = df.groupby(grp_cols, sort=False)
    dt  = grp["Time"].diff()

    for gas in ("CH4", "CO", "H2"):
        df[f"d{gas}_dt"] = (grp[gas].diff() / dt).fillna(0.0).astype(_F32)

    df["dCH4_dt_abs"] = np.abs(df["dCH4_dt"]).astype(_F32)
    df["accumulating"] = (df["dCH4_dt"] > 1e-5).astype(np.int8)
    return df


def _one_hot_zone(df: pd.DataFrame) -> pd.DataFrame:
    df = pd.get_dummies(df, columns=["zone"], drop_first=False)
    for col in _ZONE_DUMMIES:
        if col not in df.columns:
            df[col] = np.int8(0)
    for col in df.select_dtypes("bool").columns:
        df[col] = df[col].astype(np.int8)
    return df


def _drop_labels(df: pd.DataFrame, extra: list[str] | None = None) -> pd.DataFrame:
    cols = [c for c in (_LABEL_COLS + (extra or [])) if c in df.columns]
    return df.drop(columns=cols)


def _align_to_model(X: pd.DataFrame, model) -> pd.DataFrame:
    """Reorder / fill columns to match what the sklearn model was trained on."""
    if hasattr(model, "feature_names_in_"):
        return X.reindex(columns=model.feature_names_in_, fill_value=0)
    return X


# ─────────────────────────────────────────────────────────────────────────────
# Model architectures  (must match the training scripts exactly)
# ─────────────────────────────────────────────────────────────────────────────

class _HazardTemporalNet(nn.Module):
    """BiLSTM + multi-head self-attention (Phase 7 — upgraded architecture)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int  = 3,
        n_heads: int     = 8,
        dropout: float   = 0.3,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True, bidirectional=True,
        )
        D = hidden_size * 2
        self.attn = nn.MultiheadAttention(D, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(D)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Sequential(
            nn.Linear(D,           hidden_size),     nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x  = self.input_proj(x)
        lo, _ = self.lstm(x)
        ao, _ = self.attn(lo, lo, lo)
        out   = self.norm(lo + ao)
        return self.fc(self.drop(out[:, -1, :])).squeeze(1)


class _SimpleLSTM(nn.Module):
    """Plain multi-layer LSTM + FC head (earlier/lighter architecture)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int  = 2,
        dropout: float   = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(1)


def _load_lstm_net(state_dict: dict, device: torch.device) -> tuple[nn.Module, int]:
    """
    Auto-detect LSTM architecture from checkpoint keys/shapes and return
    (model, input_size).  Handles both the simple and BiLSTM variants.
    """
    has_proj  = "input_proj.0.weight" in state_dict
    is_bidir  = "lstm.weight_ih_l0_reverse" in state_dict

    # Count LSTM layers (forward only)
    n_layers = sum(
        1 for k in state_dict
        if k.startswith("lstm.weight_ih_l") and "_reverse" not in k
    )

    # Hidden size (4 gates × hidden_size = first dim of weight_ih)
    hidden = state_dict["lstm.weight_ih_l0"].shape[0] // 4

    if has_proj:
        raw_input = state_dict["input_proj.0.weight"].shape[1]
        n_heads   = 8  # not stored; use training default
        net       = _HazardTemporalNet(raw_input, hidden, n_layers, n_heads)
    elif is_bidir:
        raw_input = state_dict["lstm.weight_ih_l0"].shape[1]
        net       = _HazardTemporalNet(raw_input, hidden, n_layers)
    else:
        raw_input = state_dict["lstm.weight_ih_l0"].shape[1]
        net       = _SimpleLSTM(raw_input, hidden, n_layers)

    net.load_state_dict(state_dict, strict=True)
    net.to(device).eval()
    return net, raw_input


def _build_stgnn_class():
    """
    Deferred import: only built when torch_geometric is present.
    Returns the SpatioTemporalNet class or None.
    """
    try:
        import torch.nn.functional as F
        from torch_geometric.nn import SAGEConv
    except ImportError:
        return None

    class SpatioTemporalNet(nn.Module):
        """GraphSAGE + BiLSTM + Attention (Phase 9)."""

        def __init__(
            self,
            input_dim: int,
            hidden:      int   = 128,
            lstm_hidden: int   = 128,
            lstm_layers: int   = 2,
            attn_heads:  int   = 4,
            dropout:     float = 0.35,
        ):
            super().__init__()
            self.sage1 = SAGEConv(input_dim, hidden)
            self.norm1 = nn.LayerNorm(hidden)
            self.sage2 = SAGEConv(hidden, hidden)
            self.norm2 = nn.LayerNorm(hidden)
            self.s_drop = nn.Dropout(dropout)

            self.lstm = nn.LSTM(
                hidden, lstm_hidden, lstm_layers,
                batch_first=True, bidirectional=True,
                dropout=dropout if lstm_layers > 1 else 0.0,
            )
            D = lstm_hidden * 2
            self.t_attn = nn.MultiheadAttention(D, attn_heads, dropout=dropout, batch_first=True)
            self.t_norm = nn.LayerNorm(D)
            self.head   = nn.Sequential(
                nn.Linear(D,      hidden),      nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden // 2, 1),
            )

        def forward(self, data):
            x, ei = data.x, data.edge_index  # x: [N, T, F]
            outs = []
            for t in range(x.shape[1]):
                xt = F.gelu(self.norm1(self.sage1(x[:, t, :], ei)))
                xt = self.s_drop(xt)
                xt = F.gelu(self.norm2(self.sage2(xt, ei)))
                outs.append(xt)
            te = torch.stack(outs, dim=1)           # [N, T, H]
            lo, _ = self.lstm(te)                   # [N, T, 2H]
            ao, _ = self.t_attn(lo, lo, lo)
            out   = self.t_norm(lo + ao)
            return self.head(out[:, -1, :]).squeeze(1)

    return SpatioTemporalNet


# ─────────────────────────────────────────────────────────────────────────────
# Main predictor
# ─────────────────────────────────────────────────────────────────────────────

class GreenMiningPredictor:
    """
    Single entry point for all GreenMining model predictions.

    All models are loaded once at construction time; subsequent calls to
    predict_* are fast (no disk I/O).

    Parameters
    ----------
    repo_root : Path, optional
        Root of the GreenMining repository.  Defaults to the directory
        that contains inference.py.
    """

    _LSTM_SEQ_LEN   = 4
    _LSTM_THRESHOLD = 0.535   # tuned on val set (Phase 7)
    _STGNN_SEQ_LEN  = 6
    _STGNN_THRESHOLD= 0.445   # tuned on val set (Phase 9)
    _STGNN_K        = 24      # KNN neighbours for graph construction
    _BATCH           = 2048   # inference batch size

    def __init__(self, repo_root: Optional[Path] = None):
        self.root   = Path(repo_root) if repo_root else REPO_ROOT
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._baseline: dict = {}
        self._ablation: dict = {}
        self._lstm:     Optional[dict] = None
        self._stgnn:    Optional[dict] = None

        self._load_all()

    # ── Loaders ───────────────────────────────────────────────────────────────

    def _load_all(self) -> None:
        self._baseline = self._load_sklearn("models")
        self._ablation = self._load_sklearn("models_ablation")
        self._lstm     = self._load_lstm()
        self._stgnn    = self._load_stgnn()

    def _load_sklearn(self, subdir: str) -> dict:
        d = self.root / subdir
        if not d.exists():
            return {}
        store: dict = {}
        for task in ("binary", "multiclass"):
            store[task] = {}
            for algo in ("random_forest", "xgboost", "lightgbm"):
                p = d / f"{task}_{algo}.pkl"
                if not p.exists():
                    continue
                try:
                    store[task][algo] = joblib.load(p)
                except Exception as exc:
                    warnings.warn(f"Could not load {p.name}: {exc}")
        return store

    def _load_lstm(self) -> Optional[dict]:
        model_path  = self.root / "models_lstm" / "best_lstm.pt"
        scaler_path = self.root / "models_lstm" / "scaler.pkl"
        feat_path   = self.root / "data" / "lstm" / "feature_columns.json"

        if not model_path.exists():
            return None

        try:
            feat_cols  = (json.loads(feat_path.read_text(encoding="utf-8"))
                          if feat_path.exists() else None)
            scaler     = joblib.load(scaler_path) if scaler_path.exists() else None
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            net, _     = _load_lstm_net(state_dict, self.device)
        except Exception as exc:
            warnings.warn(f"Could not load LSTM model: {exc}")
            return None

        return {"net": net, "scaler": scaler, "feature_cols": feat_cols}

    def _load_stgnn(self) -> Optional[dict]:
        model_path = self.root / "models_stgnn" / "best_stgnn.pt"
        stats_path = self.root / "models_stgnn" / "norm_stats.pt"
        feat_path  = self.root / "data" / "graph" / "feature_columns.json"

        if not model_path.exists():
            return None

        STGNNClass = _build_stgnn_class()
        if STGNNClass is None:
            warnings.warn(
                "torch_geometric not installed — STGNN model found but cannot be loaded."
            )
            return None

        feat_cols  = (json.loads(feat_path.read_text(encoding="utf-8"))
                      if feat_path.exists() else None)
        norm_stats = (torch.load(stats_path, map_location="cpu", weights_only=True)
                      if stats_path.exists() else None)

        if norm_stats is None:
            warnings.warn(
                "models_stgnn/norm_stats.pt not found — STGNN predictions will be "
                "un-normalised and unreliable.  Re-run 09_train_stgnn.py after adding "
                "a torch.save({'mean': feat_mean, 'std': feat_std}, ...) call."
            )

        n_feat = len(feat_cols) + 4 if feat_cols else 42
        net    = STGNNClass(input_dim=n_feat)
        try:
            net.load_state_dict(
                torch.load(model_path, map_location=self.device, weights_only=True)
            )
            net.to(self.device).eval()
        except Exception as exc:
            warnings.warn(f"STGNN failed to load state dict: {exc}")
            return None

        return {"net": net, "norm_stats": norm_stats, "feature_cols": feat_cols}

    # ── Public API ────────────────────────────────────────────────────────────

    def predict_all(self, df_raw: pd.DataFrame) -> dict:
        """
        Run all loaded models on raw CFD data.

        Parameters
        ----------
        df_raw : pd.DataFrame
            Raw CFD data with columns:
            Time, x, y, z, CH4, CO, H2, Temperature, Velocity, Pressure
            Optionally: Scenario, Risk (ignored during inference)

        Returns
        -------
        dict with keys:
            baseline   — {binary: {algo: {pred, proba}}, multiclass: {...}}
            ablation   — same structure
            lstm       — {pred, proba, indices, threshold}  or None
            stgnn      — {pred, proba, indices, threshold}  or None
            engineered — the feature-engineered DataFrame
        """
        df_eng = engineer_features(df_raw)
        return {
            "baseline":   self._run_sklearn(df_eng, self._baseline, ablation=False),
            "ablation":   self._run_sklearn(df_eng, self._ablation, ablation=True),
            "lstm":       self._run_lstm(df_eng),
            "stgnn":      self._run_stgnn(df_eng),
            "engineered": df_eng,
        }

    def predict_baseline(
        self,
        df_raw: pd.DataFrame,
        task: str  = "binary",
        algo: str  = "lightgbm",
        ablation: bool = False,
    ) -> np.ndarray:
        """
        Predict using a single baseline model.

        Parameters
        ----------
        task    : "binary" or "multiclass"
        algo    : "lightgbm", "xgboost", or "random_forest"
        ablation: use the gas-feature-free ablation models

        Returns
        -------
        np.ndarray of integer class labels, shape (n_samples,)
        """
        df_eng  = engineer_features(df_raw)
        store   = self._ablation if ablation else self._baseline
        model   = store.get(task, {}).get(algo)
        if model is None:
            raise ValueError(
                f"Model not loaded: {'ablation' if ablation else 'baseline'}/{task}/{algo}"
            )
        X = self._sklearn_features(df_eng, ablation=ablation)
        X = _align_to_model(X, model)
        return model.predict(X)

    def predict_proba_baseline(
        self,
        df_raw: pd.DataFrame,
        task: str  = "binary",
        algo: str  = "lightgbm",
        ablation: bool = False,
    ) -> np.ndarray:
        """
        Return class probabilities from a single baseline model.
        Shape: (n_samples, n_classes)
        """
        df_eng = engineer_features(df_raw)
        store  = self._ablation if ablation else self._baseline
        model  = store.get(task, {}).get(algo)
        if model is None:
            raise ValueError(
                f"Model not loaded: {'ablation' if ablation else 'baseline'}/{task}/{algo}"
            )
        X = self._sklearn_features(df_eng, ablation=ablation)
        X = _align_to_model(X, model)
        return model.predict_proba(X)

    def status(self) -> dict:
        """Report which models are loaded and on which device."""
        return {
            "baseline_binary":     sorted(self._baseline.get("binary",     {}).keys()),
            "baseline_multiclass": sorted(self._baseline.get("multiclass", {}).keys()),
            "ablation_binary":     sorted(self._ablation.get("binary",     {}).keys()),
            "ablation_multiclass": sorted(self._ablation.get("multiclass", {}).keys()),
            "lstm":                self._lstm  is not None,
            "stgnn":               self._stgnn is not None,
            "device":              str(self.device),
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _sklearn_features(self, df_eng: pd.DataFrame, ablation: bool) -> pd.DataFrame:
        extra = _ABLATION_DROP if ablation else []
        X = _drop_labels(df_eng, extra=extra)
        return X

    def _run_sklearn(
        self,
        df_eng: pd.DataFrame,
        store:  dict,
        ablation: bool,
    ) -> dict:
        if not store:
            return {}
        X = self._sklearn_features(df_eng, ablation=ablation)
        out: dict = {}
        for task, models in store.items():
            out[task] = {}
            for algo, model in models.items():
                Xa = _align_to_model(X, model)
                try:
                    out[task][algo] = {
                        "pred":  model.predict(Xa),
                        "proba": (model.predict_proba(Xa)
                                  if hasattr(model, "predict_proba") else None),
                    }
                except Exception as exc:
                    out[task][algo] = {"error": str(exc)}
        return out

    def _run_lstm(self, df_eng: pd.DataFrame) -> Optional[dict]:
        if self._lstm is None:
            return None

        net       = self._lstm["net"]
        scaler    = self._lstm["scaler"]
        feat_cols = self._lstm["feature_cols"]
        seq_len   = self._LSTM_SEQ_LEN

        grp_cols = (["Scenario", "x", "y", "z"] if "Scenario" in df_eng.columns
                    else ["x", "y", "z"])

        df_s = df_eng.sort_values(grp_cols + ["Time"]).reset_index(drop=True)

        # Build [N, F] feature matrix aligned to the saved column list
        X_full = df_s.reindex(columns=feat_cols, fill_value=0.0).values.astype(_F32)

        sequences, row_indices = [], []
        pos = 0
        for _, grp in df_s.groupby(grp_cols, sort=False):
            n    = len(grp)
            Xg   = X_full[pos : pos + n]
            for start in range(n - seq_len + 1):
                sequences.append(Xg[start : start + seq_len])
                row_indices.append(pos + start + seq_len - 1)
            pos += n

        if not sequences:
            return {
                "error": (
                    f"LSTM requires ≥ {seq_len} timesteps per spatial cell. "
                    f"Input has only {len(df_eng)} rows."
                )
            }

        X_arr = np.stack(sequences, axis=0)  # [S, T, F]

        if scaler is not None:
            mean = scaler["mean"]  # [1, 1, F]
            std  = scaler["std"]   # [1, 1, F]
            X_arr = (X_arr - mean) / std

        proba = self._torch_infer(net, X_arr)
        pred  = (proba >= self._LSTM_THRESHOLD).astype(np.int8)

        return {
            "pred":      pred,
            "proba":     proba,
            "indices":   np.array(row_indices, dtype=np.int64),
            "threshold": self._LSTM_THRESHOLD,
        }

    def _run_stgnn(self, df_eng: pd.DataFrame) -> Optional[dict]:
        if self._stgnn is None:
            return None

        try:
            from sklearn.neighbors import NearestNeighbors
            from torch_geometric.data import Data
        except ImportError:
            return {"error": "sklearn or torch_geometric not available."}

        net        = self._stgnn["net"]
        norm_stats = self._stgnn["norm_stats"]
        feat_cols  = self._stgnn["feature_cols"]
        seq_len    = self._STGNN_SEQ_LEN
        K          = self._STGNN_K

        grp_cols = (["Scenario", "x", "y", "z"] if "Scenario" in df_eng.columns
                    else ["x", "y", "z"])

        df_s = df_eng.sort_values(grp_cols + ["Time"]).reset_index(drop=True)
        X_full = df_s.reindex(columns=feat_cols, fill_value=0.0).values.astype(_F32)

        # ── Collect per-cell sequences ─────────────────────────────────────────
        # Each cell contributes one sequence of shape [T, F].
        # We need N cells × T timesteps × F features → [N, T, F]

        cell_seqs, cell_coords, original_indices = [], [], []
        pos = 0
        for _, grp in df_s.groupby(grp_cols, sort=False):
            n = len(grp)
            if n >= seq_len:
                # Use the last `seq_len` timesteps available for this cell
                cell_seqs.append(X_full[pos + n - seq_len : pos + n])
                # Spatial coords from the last row of the group
                row = df_s.iloc[pos + n - 1]
                cell_coords.append([row["x"], row["y"], row["z"]])
                original_indices.append(pos + n - 1)
            pos += n

        if not cell_seqs:
            return {
                "error": (
                    f"STGNN requires ≥ {seq_len} timesteps per spatial cell."
                )
            }

        node_x   = np.stack(cell_seqs,  axis=0)              # [N, T, F]
        coords   = np.array(cell_coords, dtype=np.float32)    # [N, 3]

        # ── Build KNN spatial graph ───────────────────────────────────────────
        k_actual = min(K, len(coords) - 1)
        nbrs     = NearestNeighbors(n_neighbors=k_actual + 1, algorithm="auto").fit(coords)
        _, idx   = nbrs.kneighbors(coords)
        # idx[:, 0] is the cell itself → drop it
        src = np.repeat(np.arange(len(coords)), k_actual)
        dst = idx[:, 1 : k_actual + 1].ravel()
        edge_index = torch.tensor(
            np.vstack([np.concatenate([src, dst]),
                       np.concatenate([dst, src])]),
            dtype=torch.long,
        )

        # ── Assemble graph ────────────────────────────────────────────────────
        x_t = torch.tensor(node_x, dtype=torch.float32)  # [N, T, F]
        
        # Append 4 spatial dimensions (zeros, approximating each node as its own centre)
        N_nodes, T_steps, F_dim = x_t.shape
        spatial_zeros = torch.zeros((N_nodes, T_steps, 4), dtype=torch.float32)
        x_t = torch.cat([x_t, spatial_zeros], dim=-1)

        if norm_stats is not None:
            mean = norm_stats["mean"].unsqueeze(0).unsqueeze(0)  # [1, 1, F+4]
            std  = norm_stats["std"].unsqueeze(0).unsqueeze(0)
            x_t  = (x_t - mean) / std

        data = Data(
            x=x_t,
            edge_index=edge_index,
            y=torch.zeros(len(cell_seqs)),
        ).to(self.device)

        # ── Inference ─────────────────────────────────────────────────────────
        with torch.no_grad():
            logits = net(data)
            proba  = torch.sigmoid(logits).cpu().numpy().astype(np.float32)

        pred = (proba >= self._STGNN_THRESHOLD).astype(np.int8)

        return {
            "pred":      pred,
            "proba":     proba,
            "indices":   np.array(original_indices, dtype=np.int64),
            "threshold": self._STGNN_THRESHOLD,
        }

    def _torch_infer(self, net: nn.Module, X_arr: np.ndarray) -> np.ndarray:
        """Run batched sigmoid inference; returns float32 proba array."""
        proba = []
        with torch.no_grad():
            for i in range(0, len(X_arr), self._BATCH):
                batch = torch.tensor(X_arr[i : i + self._BATCH],
                                     dtype=torch.float32).to(self.device)
                proba.extend(torch.sigmoid(net(batch)).cpu().numpy())
        return np.array(proba, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# CLI smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading GreenMining models ...")
    p = GreenMiningPredictor()
    print("\nModel status:")
    for k, v in p.status().items():
        print(f"  {k:<26} {v}")

    # Quick self-test against the saved test split
    test_csv = REPO_ROOT / "data" / "final" / "test.csv"
    if test_csv.exists():
        print(f"\nRunning predict_all on {test_csv.name} (first 500 rows) ...")
        df = pd.read_csv(test_csv, nrows=500)
        results = p.predict_all(df)

        for model_group in ("baseline", "ablation"):
            for task, algos in results[model_group].items():
                for algo, out in algos.items():
                    if "error" not in out:
                        n_hazard = int((out["pred"] != 0).sum())
                        print(f"  {model_group}/{task}/{algo}: {n_hazard} hazard predictions")

        if results["lstm"] and "error" not in results["lstm"]:
            n = int(results["lstm"]["pred"].sum())
            print(f"  lstm: {n} hazard predictions  "
                  f"(threshold={results['lstm']['threshold']})")

        if results["stgnn"] and "error" not in results["stgnn"]:
            n = int(results["stgnn"]["pred"].sum())
            print(f"  stgnn: {n} hazard predictions  "
                  f"(threshold={results['stgnn']['threshold']})")
    else:
        print("\ndata/final/test.csv not found — skipping self-test.")

    print("\nDone.")
