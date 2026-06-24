"""
GreenMining Safety Intelligence Dashboard
==========================================
Run:  streamlit run dashboard.py
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent

# ─────────────────────────────────────────────────────────────────────────────
# Page config  — must be the very first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GreenMining · Safety Intelligence",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Design tokens
# ─────────────────────────────────────────────────────────────────────────────
C = {
    "bg":       "#0d1117",
    "card":     "#161b22",
    "border":   "#30363d",
    "green":    "#3fb950",
    "red":      "#f85149",
    "orange":   "#d29922",
    "blue":     "#58a6ff",
    "purple":   "#bc8cff",
    "text":     "#c9d1d9",
    "sub":      "#8b949e",
    "glow_g":   "rgba(63,185,80,0.10)",
    "glow_r":   "rgba(248,81,73,0.12)",
    "glow_o":   "rgba(210,153,34,0.10)",
}

HAZARD_CS = [
    [0.0,  "#3fb950"],
    [0.35, "#d29922"],
    [0.65, "#ff7b24"],
    [1.0,  "#f85149"],
]

PT = "plotly_dark"   # Plotly template

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  /* ── Base ── */
  .stApp {{ background-color:{C['bg']}; color:{C['text']}; }}
  .stApp > header {{ background-color:transparent !important; }}
  .block-container {{ padding-top:1.2rem; padding-bottom:0; max-width:1400px; }}
  #MainMenu, footer {{ visibility:hidden; }}

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {{
    background-color:{C['card']};
    border-right:1px solid {C['border']};
  }}

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {{
    gap:4px;
    background:{C['card']};
    border-radius:8px;
    padding:4px;
    border:1px solid {C['border']};
  }}
  .stTabs [data-baseweb="tab"] {{
    background:transparent;
    color:{C['sub']};
    border-radius:6px;
    padding:6px 18px;
    font-size:13px;
    font-weight:500;
  }}
  .stTabs [aria-selected="true"] {{
    background:{C['border']};
    color:{C['text']};
  }}

  /* ── Metric cards ── */
  .kpi {{ background:{C['card']}; border:1px solid {C['border']};
           border-radius:12px; padding:18px 16px; text-align:center; }}
  .kpi .lbl {{ font-size:11px; color:{C['sub']}; text-transform:uppercase;
               letter-spacing:1.2px; margin-bottom:8px; }}
  .kpi .val {{ font-size:34px; font-weight:700; line-height:1; }}
  .kpi .suf {{ font-size:12px; color:{C['sub']}; margin-top:6px; }}

  /* ── Status banners ── */
  .status-safe {{
    background:{C['glow_g']}; border:1px solid {C['green']};
    border-radius:12px; padding:16px 24px; text-align:center;
    color:{C['green']}; font-size:24px; font-weight:700; letter-spacing:3px;
  }}
  .status-warn {{
    background:{C['glow_o']}; border:1px solid {C['orange']};
    border-radius:12px; padding:16px 24px; text-align:center;
    color:{C['orange']}; font-size:24px; font-weight:700; letter-spacing:3px;
  }}
  .status-danger {{
    background:{C['glow_r']}; border:1px solid {C['red']};
    border-radius:12px; padding:16px 24px; text-align:center;
    color:{C['red']}; font-size:24px; font-weight:700; letter-spacing:3px;
    animation:pulse 2s infinite;
  }}
  @keyframes pulse {{
    0%,100% {{ box-shadow:0 0 0 0 rgba(248,81,73,.4); }}
    50%      {{ box-shadow:0 0 18px 6px rgba(248,81,73,.12); }}
  }}

  /* ── Section dividers ── */
  .sec {{ font-size:11px; font-weight:600; color:{C['sub']};
          text-transform:uppercase; letter-spacing:1.5px;
          border-bottom:1px solid {C['border']}; padding-bottom:6px;
          margin:24px 0 14px; }}

  /* ── Alert rows ── */
  .alert-row {{ background:rgba(248,81,73,.07); border-left:3px solid {C['red']};
                border-radius:0 8px 8px 0; padding:9px 14px;
                margin-bottom:7px; font-size:13px; color:{C['text']}; }}

  /* ── Plotly card wrapper ── */
  .chart-card {{ background:{C['card']}; border:1px solid {C['border']};
                 border-radius:12px; padding:4px; }}

  /* ── Streamlit widget overrides ── */
  .stSelectbox label, .stSlider label, .stRadio label,
  .stFileUploader label {{ color:{C['sub']} !important; font-size:12px !important; }}
  div[data-baseweb="select"] > div {{ background:{C['card']} !important;
    border-color:{C['border']} !important; }}
  .stDataFrame {{ border-radius:8px; overflow:hidden; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Cached resources
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="⚙️ Loading ML models …")
def load_predictor():
    try:
        from inference import GreenMiningPredictor
        return GreenMiningPredictor()
    except Exception as exc:
        return None


@st.cache_data(show_spinner=False)
def load_test_sample(n_rows: int = 6000) -> pd.DataFrame:
    """Load a balanced sample from Scenario 5 test data."""
    path = REPO_ROOT / "data" / "final" / "test.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    safe  = df[df["hazard_binary"] == 0].sample(
        min(n_rows // 2, (df["hazard_binary"] == 0).sum()), random_state=42)
    haz   = df[df["hazard_binary"] == 1].sample(
        min(n_rows // 2, (df["hazard_binary"] == 1).sum()), random_state=42)
    return (pd.concat([safe, haz])
              .sample(frac=1, random_state=42)
              .reset_index(drop=True))


@st.cache_data(show_spinner="🔮 Running predictions …")
def run_predictions(_predictor, data_key: str, df_json: str):
    """Cached inference; data_key is a hash of the data for cache busting."""
    if _predictor is None:
        return None
    df = pd.read_json(df_json, orient="split")
    return _predictor.predict_all(df)


@st.cache_data
def load_saved_metrics() -> dict:
    """Load all saved training metrics from the reports directories."""
    out = {}
    mapping = {
        "Baseline (full)":    REPO_ROOT / "reports" / "metrics" / "summary_metrics.json",
        "Ablation (no gas)":  REPO_ROOT / "reports_ablation" / "metrics" / "summary_metrics.json",
        "LSTM Forecast":      REPO_ROOT / "reports_lstm" / "metrics" / "forecast_metrics.json",
        "STGNN":              REPO_ROOT / "reports_stgnn" / "metrics" / "stgnn_metrics.json",
    }
    for name, path in mapping.items():
        if path.exists():
            out[name] = json.loads(path.read_text(encoding="utf-8"))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# HTML helpers
# ─────────────────────────────────────────────────────────────────────────────

def kpi_html(label: str, value: str, suffix: str = "", color: str = "") -> str:
    col = f"color:{color}" if color else f"color:{C['text']}"
    return f"""
    <div class="kpi">
      <div class="lbl">{label}</div>
      <div class="val" style="{col}">{value}</div>
      <div class="suf">{suffix}</div>
    </div>"""


def status_html(hazard_pct: float) -> str:
    if hazard_pct < 0.15:
        return f'<div class="status-safe">✅ &nbsp; SAFE &nbsp; — &nbsp; {hazard_pct:.1%} hazard coverage</div>'
    if hazard_pct < 0.40:
        return f'<div class="status-warn">⚠️ &nbsp; WARNING &nbsp; — &nbsp; {hazard_pct:.1%} hazard coverage</div>'
    return f'<div class="status-danger">🚨 &nbsp; DANGER &nbsp; — &nbsp; {hazard_pct:.1%} hazard coverage</div>'


# ─────────────────────────────────────────────────────────────────────────────
# Plotly chart builders
# ─────────────────────────────────────────────────────────────────────────────

def _base_layout(**kw):
    return dict(
        template=PT,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=C["text"], family="Inter, sans-serif", size=12),
        margin=dict(t=40, b=30, l=10, r=10),
        **kw,
    )


def fig_gauge(value: float, title: str = "Hazard Coverage") -> go.Figure:
    val = round(value * 100, 1)
    bar_color = C["green"] if val < 15 else (C["orange"] if val < 40 else C["red"])
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=val,
        number={"suffix": "%", "font": {"size": 44, "color": bar_color}},
        delta={"reference": 15, "valueformat": ".1f",
               "increasing": {"color": C["red"]},
               "decreasing": {"color": C["green"]}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1,
                     "tickcolor": C["sub"], "tickfont": {"size": 10}},
            "bar":  {"color": bar_color, "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)",
            "bordercolor": C["border"],
            "steps": [
                {"range": [0,  15], "color": "rgba(63,185,80,.12)"},
                {"range": [15, 40], "color": "rgba(210,153,34,.12)"},
                {"range": [40,100], "color": "rgba(248,81,73,.12)"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 2},
                "thickness": 0.8, "value": val,
            },
        },
        title={"text": title, "font": {"size": 13, "color": C["sub"]}},
    ))
    fig.update_layout(**_base_layout(height=240, margin=dict(t=30, b=0, l=30, r=30)))
    return fig


def fig_3d_mine(df: pd.DataFrame, color_col: str) -> go.Figure:
    """3-D scatter of every spatial cell coloured by hazard probability."""
    n = min(len(df), 8000)
    d = df.sample(n, random_state=42) if len(df) > n else df

    hover = (
        "<b>(%{x:.1f}, %{y:.1f}, %{z:.1f})</b><br>"
        "Hazard prob: %{marker.color:.3f}<br>"
        "CH₄: %{customdata[0]:.4f}<br>"
        "CO: %{customdata[1]:.5f}<br>"
        "H₂: %{customdata[2]:.4f}<br>"
        "Zone: %{customdata[3]}"
        "<extra></extra>"
    )

    cd_cols = ["CH4", "CO", "H2"]
    zone_col = next(
        (c for c in ["zone", "zone_CHAMBER_2"] if c in d.columns), None
    )
    custom = d[cd_cols].values if all(c in d.columns else False for c in cd_cols) else None

    # Annotate zone from dummy columns
    zone_labels = _decode_zone(d)

    fig = go.Figure(go.Scatter3d(
        x=d["x"], y=d["y"], z=d["z"],
        mode="markers",
        marker=dict(
            size=3.5,
            color=d[color_col].clip(0, 1),
            colorscale=HAZARD_CS,
            cmin=0, cmax=1,
            opacity=0.75,
            colorbar=dict(
                title=dict(text="Hazard<br>Prob", side="right", font=dict(size=11)),
                tickfont=dict(size=10),
                len=0.7, thickness=12,
            ),
            line=dict(width=0),
        ),
        customdata=np.column_stack([
            d.get("CH4", np.zeros(len(d))),
            d.get("CO",  np.zeros(len(d))),
            d.get("H2",  np.zeros(len(d))),
            zone_labels,
        ]),
        hovertemplate=hover,
    ))

    fig.update_layout(
        **_base_layout(height=540),
        scene=dict(
            xaxis=dict(title="X (m)", gridcolor=C["border"],
                       backgroundcolor="rgba(0,0,0,0)"),
            yaxis=dict(title="Y (m)", gridcolor=C["border"],
                       backgroundcolor="rgba(0,0,0,0)"),
            zaxis=dict(title="Z (m)", gridcolor=C["border"],
                       backgroundcolor="rgba(0,0,0,0)"),
            bgcolor="rgba(0,0,0,0)",
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9)),
            aspectmode="data",
        ),
        margin=dict(t=10, b=0, l=0, r=0),
    )
    return fig


def fig_2d_heatmap(df: pd.DataFrame, color_col: str) -> go.Figure:
    """XY-plane density heatmap of hazard probability."""
    fig = go.Figure(go.Histogram2dContour(
        x=df["x"], y=df["y"],
        z=df[color_col],
        colorscale=HAZARD_CS,
        histfunc="avg",
        contours=dict(showlabels=True, labelfont=dict(size=9, color="white")),
        colorbar=dict(title="Avg Hazard Prob",
                      tickfont=dict(size=10), len=0.8, thickness=12),
        line=dict(width=0.5),
        ncontours=18,
        hovertemplate="X: %{x:.1f} m<br>Y: %{y:.1f} m<br>Avg: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        **_base_layout(height=380),
        xaxis=dict(title="X (m)  — Tunnel Length",
                   gridcolor=C["border"], zeroline=False),
        yaxis=dict(title="Y (m)  — Width",
                   gridcolor=C["border"], zeroline=False),
        title=dict(text="XY Hazard Heatmap  (averaged over Z)",
                   font=dict(size=13), x=0.01),
    )
    return fig


def fig_temporal(df: pd.DataFrame, color_col: str) -> go.Figure:
    """Hazard fraction evolution over simulation time."""
    ts = (df.groupby("Time")[color_col]
            .agg(["mean", "max", lambda x: (x >= 0.5).mean()])
            .reset_index())
    ts.columns = ["Time", "avg_prob", "max_prob", "hazard_frac"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts["Time"], y=ts["hazard_frac"],
        name="Hazard cell fraction", mode="lines",
        line=dict(color=C["red"], width=2.5),
        fill="tozeroy", fillcolor="rgba(248,81,73,.10)",
    ))
    fig.add_trace(go.Scatter(
        x=ts["Time"], y=ts["avg_prob"],
        name="Avg hazard probability", mode="lines",
        line=dict(color=C["orange"], width=2, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=ts["Time"], y=ts["max_prob"],
        name="Peak hazard probability", mode="lines",
        line=dict(color=C["red"], width=1.5, dash="dot"),
    ))
    fig.add_hline(y=0.15, line_dash="dot", line_color=C["orange"],
                  annotation_text="Warning threshold", annotation_position="right",
                  annotation_font_size=10)
    fig.add_hline(y=0.40, line_dash="dot", line_color=C["red"],
                  annotation_text="Danger threshold", annotation_position="right",
                  annotation_font_size=10)
    fig.update_layout(
        **_base_layout(height=380),
        xaxis=dict(title="Simulation Time (s)", gridcolor=C["border"], zeroline=False),
        yaxis=dict(title="Fraction / Probability", gridcolor=C["border"],
                   zeroline=False, range=[0, 1]),
        legend=dict(x=0.01, y=0.98, bgcolor="rgba(0,0,0,0)",
                    bordercolor=C["border"], borderwidth=1),
        title=dict(text="Hazard Evolution Over Time", font=dict(size=13), x=0.01),
    )
    return fig


def fig_zone_bar(df: pd.DataFrame, color_col: str) -> go.Figure:
    """Hazard probability by mine zone."""
    zone_col = _primary_zone_col(df)
    if zone_col is None:
        return go.Figure().update_layout(**_base_layout(),
            title="Zone data not available")

    z = (df.groupby(zone_col)[color_col]
           .agg(["mean", "count", lambda x: (x >= 0.5).sum()])
           .reset_index())
    z.columns = [zone_col, "avg_prob", "n_cells", "n_hazard"]
    z["hazard_pct"] = z["n_hazard"] / z["n_cells"]
    z = z.sort_values("avg_prob", ascending=True)

    colors_bar = [
        C["red"] if v >= 0.5 else (C["orange"] if v >= 0.25 else C["green"])
        for v in z["avg_prob"]
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=z[zone_col], x=z["avg_prob"],
        orientation="h",
        name="Avg Hazard Prob",
        marker_color=colors_bar,
        marker_line_width=0,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Avg prob: %{x:.3f}<br>"
            "Cells: %{customdata[0]:,}<br>"
            "Hazard cells: %{customdata[1]:,}<extra></extra>"
        ),
        customdata=list(zip(z["n_cells"], z["n_hazard"])),
    ))
    fig.add_vline(x=0.5, line_dash="dot", line_color=C["orange"],
                  annotation_text="Decision threshold",
                  annotation_font_size=10)
    fig.update_layout(
        **_base_layout(height=360),
        xaxis=dict(title="Average Hazard Probability",
                   gridcolor=C["border"], range=[0, 1]),
        yaxis=dict(title="", gridcolor=C["border"]),
        title=dict(text="Hazard Risk by Mine Zone", font=dict(size=13), x=0.01),
        showlegend=False,
    )
    return fig


def fig_gas_violin(df: pd.DataFrame, pred_col: str) -> go.Figure:
    """Violin plot of gas concentrations split by predicted class."""
    gases = [g for g in ["CH4", "CO", "H2"] if g in df.columns]
    if not gases or pred_col not in df.columns:
        return go.Figure().update_layout(**_base_layout())

    df2 = df.copy()
    df2["Class"] = np.where(df2[pred_col] >= 0.5, "Hazard", "Safe")

    fig = make_subplots(rows=1, cols=len(gases),
                        subplot_titles=[f"{g} Concentration" for g in gases])
    for i, gas in enumerate(gases, 1):
        for cls, clr in [("Safe", C["green"]), ("Hazard", C["red"])]:
            sub = df2[df2["Class"] == cls][gas]
            fig.add_trace(go.Violin(
                y=sub, name=cls, legendgroup=cls,
                showlegend=(i == 1),
                line_color=clr,
                fillcolor=f"rgba({','.join(str(int(clr[j:j+2],16)) for j in (1,3,5))},.2)",
                box_visible=True, meanline_visible=True,
                hoverinfo="none",
            ), row=1, col=i)
    fig.update_layout(
        **_base_layout(height=350),
        title=dict(text="Gas Concentrations  ·  Safe vs Hazard Cells",
                   font=dict(size=13), x=0.01),
        violingap=0.15, violingroupgap=0.05,
    )
    return fig


def fig_benchmark_bar(metrics: dict) -> go.Figure:
    """Grouped bar chart comparing all saved model metrics."""
    rows = []
    for model_name, data in metrics.items():
        if model_name in ("LSTM Forecast", "STGNN"):
            m = data
            rows.append({"Model": model_name,
                         "Accuracy": m.get("accuracy", 0),
                         "Recall (Hazard)": m.get("class_1_recall", 0),
                         "Precision (Hazard)": m.get("class_1_precision", 0),
                         "F1 (Macro)": m.get("f1_macro", 0),
                         "ROC-AUC": m.get("roc_auc", 0)})
        else:
            binary = data.get("binary", {})
            for algo, m in binary.items():
                label = f"{model_name} · {algo.replace('_', ' ').title()}"
                rows.append({"Model": label,
                             "Accuracy": m.get("accuracy", 0),
                             "Recall (Hazard)": m.get("class_1_recall", 0),
                             "Precision (Hazard)": m.get("class_1_precision", 0),
                             "F1 (Macro)": m.get("f1_macro", 0),
                             "ROC-AUC": m.get("roc_auc", 0)})
    if not rows:
        return go.Figure().update_layout(**_base_layout())

    df_b = pd.DataFrame(rows)
    metric_cols = ["Accuracy", "Recall (Hazard)", "Precision (Hazard)",
                   "F1 (Macro)", "ROC-AUC"]
    palette = [C["blue"], C["green"], C["purple"], C["orange"], C["red"]]

    fig = go.Figure()
    for col, clr in zip(metric_cols, palette):
        fig.add_trace(go.Bar(
            name=col, x=df_b["Model"], y=df_b[col],
            marker_color=clr, marker_line_width=0,
            hovertemplate=f"<b>%{{x}}</b><br>{col}: %{{y:.4f}}<extra></extra>",
        ))
    fig.update_layout(
        **_base_layout(height=480),
        barmode="group",
        xaxis=dict(tickangle=-30, tickfont=dict(size=10), gridcolor=C["border"]),
        yaxis=dict(title="Score", gridcolor=C["border"], range=[0, 1.05]),
        legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center",
                    bgcolor="rgba(0,0,0,0)"),
        title=dict(text="Model Performance Benchmark  (Test Set)",
                   font=dict(size=13), x=0.01),
    )
    return fig


def fig_benchmark_radar(metrics: dict) -> go.Figure:
    """Radar chart for the top-performing models."""
    cats = ["Accuracy", "Recall", "Precision", "F1", "ROC-AUC"]
    angles = cats + [cats[0]]
    fig = go.Figure()
    palette = [C["green"], C["blue"], C["orange"], C["purple"], C["red"]]

    idx = 0
    for model_name, data in metrics.items():
        if model_name in ("LSTM Forecast", "STGNN"):
            m    = data
            vals = [m.get("accuracy", 0), m.get("class_1_recall", 0),
                    m.get("class_1_precision", 0), m.get("f1_macro", 0),
                    m.get("roc_auc", 0)]
            vals += [vals[0]]
            fig.add_trace(go.Scatterpolar(
                r=vals, theta=angles,
                fill="toself", name=model_name,
                line_color=palette[idx % len(palette)],
                fillcolor=palette[idx % len(palette)].replace("#", "rgba(").rstrip(")")
                          .replace("rgba(", "rgba(")[:20] + ", .12)",
            ))
            idx += 1
        else:
            binary = data.get("binary", {})
            for algo, m in binary.items():
                if idx >= 5:
                    break
                vals = [m.get("accuracy", 0), m.get("class_1_recall", 0),
                        m.get("class_1_precision", 0), m.get("f1_macro", 0),
                        m.get("roc_auc", 0)]
                vals += [vals[0]]
                label = f"{model_name} · {algo.replace('_',' ').title()}"
                fig.add_trace(go.Scatterpolar(
                    r=vals, theta=angles, fill="toself", name=label,
                    line_color=palette[idx % len(palette)],
                ))
                idx += 1

    fig.update_layout(
        **_base_layout(height=420),
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor=C["border"],
                            tickfont=dict(size=9)),
            angularaxis=dict(gridcolor=C["border"]),
        ),
        legend=dict(x=1.05, y=0.5, bgcolor="rgba(0,0,0,0)"),
        title=dict(text="Performance Radar  (all models)", font=dict(size=13), x=0.01),
        margin=dict(t=50, b=20, l=20, r=160),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────────

def _decode_zone(df: pd.DataFrame) -> np.ndarray:
    """Reconstruct zone name from one-hot dummy columns."""
    zone_map = {
        "zone_CHAMBER_2":     "Chamber 2",
        "zone_INLET_SECTION": "Inlet",
        "zone_JUNCTION_1":    "Junction 1",
        "zone_JUNCTION_2_3":  "Junction 2/3",
        "zone_MID_TUNNEL":    "Mid Tunnel",
        "zone_OUTLET_SECTION":"Outlet",
        "zone_SOUTH_STUB":    "South Stub",
    }
    labels = np.full(len(df), "Main Tunnel", dtype=object)
    for col, name in zone_map.items():
        if col in df.columns:
            labels[df[col].astype(bool).values] = name
    return labels


def _primary_zone_col(df: pd.DataFrame) -> str | None:
    zone_dummies = [c for c in df.columns if c.startswith("zone_")]
    if zone_dummies:
        df = df.copy()
        df["_zone"] = _decode_zone(df)
        df["_zone_src"] = df["_zone"]
        return "_zone_src"
    return None


def _attach_zone(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_zone"] = _decode_zone(df)
    return df


def _top_hazard_cells(df: pd.DataFrame, color_col: str, n: int = 8):
    return (df.nlargest(n, color_col)[["x", "y", "z", color_col]]
              .rename(columns={color_col: "Hazard Prob"})
              .reset_index(drop=True))


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div style="padding:12px 0 8px;">
          <span style="font-size:22px;">⛏️</span>
          <span style="font-size:17px; font-weight:700; color:{C['green']};
                       margin-left:6px;">GreenMining</span>
          <div style="font-size:10px; color:{C['sub']}; margin-top:2px;
                      letter-spacing:.8px; text-transform:uppercase;">
            Safety Intelligence
          </div>
        </div>
        <hr style="border-color:{C['border']}; margin:8px 0 16px;">
        """, unsafe_allow_html=True)

        st.markdown(f'<div class="sec">📂 Data Source</div>', unsafe_allow_html=True)
        source = st.radio("", ["Built-in Test Data", "Upload CSV"],
                          label_visibility="collapsed")

        uploaded = None
        n_rows = 6000
        if source == "Upload CSV":
            uploaded = st.file_uploader(
                "CFD CSV (Time, x, y, z, CH4, CO, H2, Temperature, Velocity, Pressure)",
                type=["csv"],
            )
        else:
            n_rows = st.slider("Sample size", 1000, 12000, 6000, 500)

        st.markdown(f'<div class="sec">🤖 Prediction Model</div>', unsafe_allow_html=True)
        primary_model = st.selectbox(
            "Primary model for visualisation",
            ["Random Forest", "LSTM Forecast"],
        )

        lstm_threshold = 0.535
        if primary_model == "LSTM Forecast":
            lstm_threshold = st.slider(
                "LSTM decision threshold", 0.10, 0.90, 0.535, 0.005,
                help="Lower = more sensitive (fewer missed hazards, more false alarms)"
            )

        st.markdown(f'<div class="sec">ℹ️ System</div>', unsafe_allow_html=True)
        predictor = load_predictor()
        st.markdown(f"""
        <div style="font-size:11px; color:{C['sub']}; line-height:1.8;">
          {'✅' if predictor else '❌'} Predictor loaded<br>
        """, unsafe_allow_html=True)
        if predictor:
            s = predictor.status()
            for k in ["baseline_binary", "lstm", "stgnn"]:
                v = s.get(k, [])
                icon = "✅" if v else "⬜"
                st.markdown(
                    f'<span style="font-size:11px; color:{C["sub"]};">'
                    f'{icon} {k.replace("_"," ").title()}: '
                    f'{"loaded" if v else "not available"}</span><br>',
                    unsafe_allow_html=True,
                )
            st.markdown(
                f'<span style="font-size:11px; color:{C["sub"]};">'
                f'Device: {s.get("device","cpu")}</span>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    return source, uploaded, n_rows, primary_model, lstm_threshold


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    source, uploaded, n_rows, primary_model, lstm_threshold = render_sidebar()

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:14px; padding:0 0 14px;">
      <div>
        <h1 style="color:{C['green']}; margin:0; font-size:24px; font-weight:700;
                   letter-spacing:.5px;">
          ⛏️ &nbsp; GreenMining Safety Intelligence
        </h1>
        <p style="color:{C['sub']}; margin:3px 0 0; font-size:12px;">
          Underground mine hazard prediction &nbsp;·&nbsp; CFD + ML
          &nbsp;·&nbsp; OpenFOAM · RF · BiLSTM · STGNN
        </p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    if source == "Upload CSV" and uploaded is not None:
        df_raw = pd.read_csv(uploaded)
    elif source == "Built-in Test Data":
        df_raw = load_test_sample(n_rows)
    else:
        st.info("👈 Upload a CSV file or switch to Built-in Test Data to begin.")
        _show_welcome_metrics()
        return

    if df_raw.empty:
        st.error("No data found. Ensure `data/final/test.csv` exists.")
        return

    # ── Run inference ─────────────────────────────────────────────────────────
    predictor = load_predictor()
    data_key  = str(hash(df_raw.to_json()))
    results   = run_predictions(predictor, data_key, df_raw.to_json(orient="split"))

    # ── Build prediction dataframe ────────────────────────────────────────────
    df = results["engineered"].copy() if results else df_raw.copy()

    # RF predictions (all rows)
    rf_proba = None
    if results and results.get("baseline", {}).get("binary", {}).get("random_forest"):
        rf_out   = results["baseline"]["binary"]["random_forest"]
        rf_proba = rf_out["proba"][:, 1] if rf_out.get("proba") is not None else None
        if rf_proba is not None:
            df["rf_proba"] = rf_proba
            df["rf_pred"]  = rf_out["pred"]

    # LSTM predictions (sequence rows)
    lstm_proba_col = None
    if results and results.get("lstm") and "error" not in results["lstm"]:
        lstm_r    = results["lstm"]
        idx_arr   = lstm_r["indices"]
        prob_arr  = lstm_r["proba"]
        df["lstm_proba"] = np.nan
        df.iloc[idx_arr, df.columns.get_loc("lstm_proba")] = prob_arr
        df["lstm_pred"] = np.where(df["lstm_proba"] >= lstm_threshold, 1, 0)
        lstm_proba_col = "lstm_proba"

    # Choose the primary colour column for charts
    if primary_model == "LSTM Forecast" and lstm_proba_col:
        color_col = "lstm_proba"
        df_plot   = df.dropna(subset=["lstm_proba"]).copy()
    elif "rf_proba" in df.columns:
        color_col = "rf_proba"
        df_plot   = df.copy()
    else:
        st.warning("No predictions available. Models may not be loaded.")
        return

    df_plot = _attach_zone(df_plot)

    hazard_pct = float((df_plot[color_col] >= 0.5).mean())
    n_hazard   = int((df_plot[color_col] >= 0.5).sum())
    n_total    = len(df_plot)

    # ── Status banner ─────────────────────────────────────────────────────────
    st.markdown(status_html(hazard_pct), unsafe_allow_html=True)
    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)

    # ── KPI row ───────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    kpi_color = C["red"] if hazard_pct >= 0.40 else (C["orange"] if hazard_pct >= 0.15 else C["green"])

    with k1:
        st.markdown(kpi_html("Analysed Cells", f"{n_total:,}", "spatial points"), unsafe_allow_html=True)
    with k2:
        st.markdown(kpi_html("Hazard Cells", f"{n_hazard:,}",
                              f"{hazard_pct:.1%} of total", color=kpi_color), unsafe_allow_html=True)
    with k3:
        if "Time" in df_plot.columns:
            t_range = f"{df_plot['Time'].min():.0f}s – {df_plot['Time'].max():.0f}s"
            st.markdown(kpi_html("Time Window", t_range, "simulation seconds"), unsafe_allow_html=True)
        else:
            st.markdown(kpi_html("Primary Model", primary_model, "active"), unsafe_allow_html=True)
    with k4:
        rf_acc = ""
        saved  = load_saved_metrics()
        if "Baseline (full)" in saved:
            rf_m   = saved["Baseline (full)"].get("binary", {}).get("random_forest", {})
            rf_acc = f"{rf_m.get('accuracy', 0):.4f}"
        st.markdown(kpi_html("RF Test Accuracy", rf_acc or "—", "from saved metrics"), unsafe_allow_html=True)
    with k5:
        lstm_m = saved.get("LSTM Forecast", {})
        lstm_recall = f"{lstm_m.get('class_1_recall', 0):.4f}" if lstm_m else "—"
        st.markdown(kpi_html("LSTM Hazard Recall", lstm_recall, "BiLSTM + Attention"), unsafe_allow_html=True)

    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tabs = st.tabs(["🗺 Overview", "🌐 Mine Map 3D", "📈 Temporal", "🏗 Zones", "🏆 Benchmarks", "🗃 Data"])

    # ── Tab 1: Overview ────────────────────────────────────────────────────────
    with tabs[0]:
        c1, c2 = st.columns([1, 1])

        with c1:
            st.plotly_chart(fig_gauge(hazard_pct, "Hazard Zone Coverage"),
                            use_container_width=True)

            # Top danger cells
            st.markdown(f'<div class="sec">🚨 Highest Risk Cells</div>', unsafe_allow_html=True)
            top = _top_hazard_cells(df_plot, color_col, n=8)
            if not top.empty:
                for _, row in top.iterrows():
                    prob  = row["Hazard Prob"]
                    clr   = C["red"] if prob >= 0.8 else (C["orange"] if prob >= 0.5 else C["sub"])
                    st.markdown(
                        f'<div class="alert-row">'
                        f'📍 ({row["x"]:.1f}, {row["y"]:.1f}, {row["z"]:.1f}) m &nbsp;·&nbsp; '
                        f'<span style="color:{clr}; font-weight:600;">{prob:.3f}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        with c2:
            # Zone pie chart
            zone_counts = df_plot.groupby("_zone")[color_col].agg(
                ["mean", "count"]).reset_index()
            zone_counts.columns = ["Zone", "avg_prob", "count"]

            fig_pie = go.Figure(go.Pie(
                labels=zone_counts["Zone"],
                values=zone_counts["count"],
                hole=0.52,
                marker=dict(
                    colors=[
                        C["red"] if p >= 0.5 else (C["orange"] if p >= 0.25 else C["green"])
                        for p in zone_counts["avg_prob"]
                    ],
                    line=dict(color=C["bg"], width=2),
                ),
                textinfo="percent+label",
                textfont=dict(size=11),
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "Cells: %{value:,}<br>"
                    "Share: %{percent}<extra></extra>"
                ),
            ))
            fig_pie.update_layout(
                **_base_layout(height=280),
                title=dict(text="Cell Distribution by Zone", font=dict(size=13), x=0.01),
                showlegend=True,
                legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
                margin=dict(t=40, b=10, l=10, r=10),
                annotations=[dict(text=f"{hazard_pct:.0%}<br>hazard",
                                  font_size=14, showarrow=False,
                                  font_color=kpi_color)],
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            # Gas violin
            if all(g in df_plot.columns for g in ["CH4", "CO", "H2"]):
                st.plotly_chart(fig_gas_violin(df_plot.sample(min(3000, len(df_plot)), random_state=42), color_col),
                                use_container_width=True)

    # ── Tab 2: Mine Map 3D ────────────────────────────────────────────────────
    with tabs[1]:
        st.plotly_chart(fig_3d_mine(df_plot, color_col),
                        use_container_width=True, key="3d")
        st.markdown(f'<div class="sec">2D Plan View  (XY cross-section)</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(fig_2d_heatmap(df_plot, color_col),
                        use_container_width=True, key="hm")

    # ── Tab 3: Temporal ───────────────────────────────────────────────────────
    with tabs[2]:
        if "Time" not in df_plot.columns:
            st.info("No Time column found in the data.")
        else:
            st.plotly_chart(fig_temporal(df_plot, color_col),
                            use_container_width=True)

            # Gas over time
            st.markdown(f'<div class="sec">Gas Concentrations Over Time</div>',
                        unsafe_allow_html=True)
            gas_ts = df_plot.groupby("Time")[
                [g for g in ["CH4", "CO", "H2"] if g in df_plot.columns]
            ].mean().reset_index()

            fig_gas = go.Figure()
            gas_colors = {"CH4": C["blue"], "CO": C["red"], "H2": C["orange"]}
            for gas in [g for g in ["CH4", "CO", "H2"] if g in gas_ts.columns]:
                fig_gas.add_trace(go.Scatter(
                    x=gas_ts["Time"], y=gas_ts[gas],
                    name=gas, mode="lines",
                    line=dict(color=gas_colors[gas], width=2),
                ))
            fig_gas.update_layout(
                **_base_layout(height=320),
                xaxis=dict(title="Time (s)", gridcolor=C["border"]),
                yaxis=dict(title="Mean Concentration (vol. fraction)",
                           gridcolor=C["border"]),
                legend=dict(bgcolor="rgba(0,0,0,0)"),
                title=dict(text="Average Gas Concentrations vs Time",
                           font=dict(size=13), x=0.01),
            )
            st.plotly_chart(fig_gas, use_container_width=True)

    # ── Tab 4: Zones ──────────────────────────────────────────────────────────
    with tabs[3]:
        c1, c2 = st.columns([1.1, 1])
        with c1:
            st.plotly_chart(fig_zone_bar(_remap_zone_col(df_plot), color_col),
                            use_container_width=True)
        with c2:
            # Zone table
            zt = (df_plot.groupby("_zone")
                  .agg(
                      Cells=pd.NamedAgg(column=color_col, aggfunc="count"),
                      Avg_Prob=pd.NamedAgg(column=color_col, aggfunc="mean"),
                      Max_Prob=pd.NamedAgg(column=color_col, aggfunc="max"),
                      Hazard_Cells=pd.NamedAgg(
                          column=color_col, aggfunc=lambda x: (x >= 0.5).sum()),
                  )
                  .reset_index()
                  .rename(columns={"_zone": "Zone"})
                  .sort_values("Avg_Prob", ascending=False))
            zt["Hazard %"] = (zt["Hazard_Cells"] / zt["Cells"] * 100).round(1)
            zt[["Avg_Prob", "Max_Prob"]] = zt[["Avg_Prob", "Max_Prob"]].round(4)
            st.markdown(f'<div class="sec">Zone Risk Summary</div>', unsafe_allow_html=True)
            st.dataframe(zt, use_container_width=True, hide_index=True)

        # XZ profile
        st.markdown(f'<div class="sec">Hazard Profile Along Tunnel (X axis)</div>',
                    unsafe_allow_html=True)
        x_bins = pd.cut(df_plot["x"], bins=40)
        xp = (df_plot.groupby(x_bins, observed=True)[color_col]
              .mean().reset_index())
        xp["x_mid"] = xp["x"].apply(lambda i: i.mid)

        fig_xp = go.Figure(go.Bar(
            x=xp["x_mid"], y=xp[color_col],
            marker=dict(
                color=xp[color_col],
                colorscale=HAZARD_CS,
                cmin=0, cmax=1,
                line_width=0,
            ),
            hovertemplate="X: %{x:.1f} m<br>Avg prob: %{y:.3f}<extra></extra>",
        ))
        fig_xp.add_hline(y=0.5, line_dash="dot", line_color=C["orange"])
        fig_xp.update_layout(
            **_base_layout(height=280),
            xaxis=dict(title="Tunnel Position X (m)", gridcolor=C["border"]),
            yaxis=dict(title="Avg Hazard Probability", gridcolor=C["border"], range=[0, 1]),
            showlegend=False,
            title=dict(text="Hazard Risk Profile  (averaged over Y, Z)",
                       font=dict(size=13), x=0.01),
        )
        st.plotly_chart(fig_xp, use_container_width=True)

    # ── Tab 5: Benchmarks ─────────────────────────────────────────────────────
    with tabs[4]:
        saved = load_saved_metrics()
        if not saved:
            st.info("No saved metrics found. Run the training scripts first.")
        else:
            st.markdown(f'<div class="sec">Comprehensive Model Comparison</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(fig_benchmark_bar(saved), use_container_width=True)

            c1, c2 = st.columns([1, 1])
            with c1:
                st.plotly_chart(fig_benchmark_radar(saved), use_container_width=True)
            with c2:
                st.markdown(f'<div class="sec">Detailed Metrics Table</div>',
                            unsafe_allow_html=True)
                _show_metrics_table(saved)

    # ── Tab 6: Data ───────────────────────────────────────────────────────────
    with tabs[5]:
        st.markdown(f'<div class="sec">Prediction Results</div>', unsafe_allow_html=True)

        show_cols = ["x", "y", "z"]
        if "Time" in df_plot.columns:
            show_cols = ["Time"] + show_cols
        for g in ["CH4", "CO", "H2", "Temperature", "Velocity"]:
            if g in df_plot.columns:
                show_cols.append(g)
        if "rf_proba" in df_plot.columns:
            show_cols += ["rf_proba", "rf_pred"]
        if "lstm_proba" in df_plot.columns:
            show_cols += ["lstm_proba", "lstm_pred"]
        if "hazard_binary" in df_plot.columns:
            show_cols.append("hazard_binary")
        show_cols += ["_zone"]

        show_cols = [c for c in show_cols if c in df_plot.columns]

        filter_col, download_col = st.columns([2, 1])
        with filter_col:
            filt = st.selectbox("Filter",
                                ["All cells", "Hazard only", "Safe only"])
        disp = df_plot[show_cols].copy()
        if filt == "Hazard only":
            disp = disp[df_plot[color_col] >= 0.5]
        elif filt == "Safe only":
            disp = disp[df_plot[color_col] < 0.5]

        with download_col:
            csv_bytes = disp.to_csv(index=False).encode()
            st.download_button("⬇ Download CSV", csv_bytes,
                               "greenmining_predictions.csv", "text/csv")

        st.dataframe(
            disp.round(5).reset_index(drop=True),
            use_container_width=True,
            height=480,
        )
        st.caption(f"Showing {len(disp):,} of {len(df_plot):,} cells")


# ─────────────────────────────────────────────────────────────────────────────
# Helper renders
# ─────────────────────────────────────────────────────────────────────────────

def _show_welcome_metrics():
    """Show benchmark metrics when no data is loaded yet."""
    saved = load_saved_metrics()
    if not saved:
        return
    st.markdown(f'<div class="sec">📊 Saved Model Benchmarks</div>', unsafe_allow_html=True)
    st.plotly_chart(fig_benchmark_bar(saved), use_container_width=True)


def _show_metrics_table(saved: dict):
    rows = []
    for model_name, data in saved.items():
        if model_name in ("LSTM Forecast", "STGNN"):
            m = data
            rows.append({
                "Model": model_name,
                "Accuracy": f"{m.get('accuracy',0):.4f}",
                "Hazard Recall": f"{m.get('class_1_recall',0):.4f}",
                "Hazard Precision": f"{m.get('class_1_precision',0):.4f}",
                "F1 Macro": f"{m.get('f1_macro',0):.4f}",
                "ROC-AUC": f"{m.get('roc_auc',0):.4f}",
                "False Neg": str(m.get("false_negatives", "—")),
            })
        else:
            for task in ["binary"]:
                for algo, m in data.get(task, {}).items():
                    rows.append({
                        "Model": f"{model_name}·{algo.replace('_',' ').title()}",
                        "Accuracy": f"{m.get('accuracy',0):.4f}",
                        "Hazard Recall": f"{m.get('class_1_recall',0):.4f}",
                        "Hazard Precision": f"{m.get('class_1_precision',0):.4f}",
                        "F1 Macro": f"{m.get('f1_macro',0):.4f}",
                        "ROC-AUC": f"{m.get('roc_auc',0):.4f}",
                        "False Neg": str(m.get("false_negatives", "—")),
                    })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _remap_zone_col(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the zone bar chart can find its column."""
    d = df.copy()
    if "_zone" in d.columns:
        d["_zone_src"] = d["_zone"]
    return d


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
main()
