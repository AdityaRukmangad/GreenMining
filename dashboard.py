"""
GreenMining Safety Intelligence — Dashboard
Run: streamlit run dashboard.py
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent

# ─── Page config (must be FIRST) ─────────────────────────────────────────────
st.set_page_config(
    page_title="GreenMining · Safety",
    page_icon="⛏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Design tokens ────────────────────────────────────────────────────────────
BG      = "#0a0c10"
CARD    = "#12151c"
BORDER  = "#1e2330"
GREEN   = "#00e5a0"
RED     = "#ff4757"
ORANGE  = "#ffa502"
BLUE    = "#3d8ef0"
PURPLE  = "#a78bfa"
TEXT    = "#e2e8f0"
MUTED   = "#64748b"

HAZARD_CS = [[0.0, GREEN], [0.4, ORANGE], [0.7, "#ff6b35"], [1.0, RED]]

# ─── CSS (minimal, targeted) ──────────────────────────────────────────────────
st.markdown(f"""
<style>
  /* Hide Streamlit chrome */
  #MainMenu, footer, header {{ visibility: hidden; }}
  .block-container {{ padding: 1.5rem 2rem 1rem; max-width: 1440px; }}

  /* Sidebar */
  section[data-testid="stSidebar"] > div:first-child {{
    background: {CARD};
    border-right: 1px solid {BORDER};
    padding: 1.5rem 1rem;
  }}

  /* Tab strip */
  .stTabs [data-baseweb="tab-list"] {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 4px;
    gap: 2px;
  }}
  .stTabs [data-baseweb="tab"] {{
    border-radius: 7px;
    padding: 6px 20px;
    color: {MUTED};
    font-size: 13px;
    font-weight: 500;
    background: transparent;
  }}
  .stTabs [aria-selected="true"] {{
    background: {BORDER};
    color: {TEXT};
  }}

  /* Download button */
  .stDownloadButton > button {{
    background: {CARD};
    border: 1px solid {BORDER};
    color: {TEXT};
    border-radius: 8px;
    font-size: 13px;
  }}
  .stDownloadButton > button:hover {{
    border-color: {GREEN};
    color: {GREEN};
  }}
</style>
""", unsafe_allow_html=True)


# ─── Zone decoder ─────────────────────────────────────────────────────────────
_ZONE_DUMMY_MAP = {
    "zone_CHAMBER_2":     "Chamber 2",
    "zone_INLET_SECTION": "Inlet",
    "zone_JUNCTION_1":    "Junction 1",
    "zone_JUNCTION_2_3":  "Junction 2/3",
    "zone_MID_TUNNEL":    "Mid Tunnel",
    "zone_OUTLET_SECTION":"Outlet",
    "zone_SOUTH_STUB":    "South Stub",
}

def decode_zones(df: pd.DataFrame) -> np.ndarray:
    """Reconstruct zone name from one-hot dummies + in_chamber flag."""
    if "in_chamber" in df.columns:
        names = np.where(df["in_chamber"].values == 1, "Chamber 1", "Main Tunnel")
    else:
        names = np.full(len(df), "Main Tunnel", dtype=object)

    for col, label in _ZONE_DUMMY_MAP.items():
        if col in df.columns:
            names = np.where(df[col].astype(bool).values, label, names)

    return names.astype(object)


# ─── Cached resources ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_predictor():
    try:
        from inference import GreenMiningPredictor
        return GreenMiningPredictor()
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_test_data(n: int) -> pd.DataFrame:
    path = ROOT / "data" / "final" / "test.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    safe = df[df["hazard_binary"] == 0].sample(
        min(n // 2, int((df["hazard_binary"] == 0).sum())), random_state=42)
    haz  = df[df["hazard_binary"] == 1].sample(
        min(n // 2, int((df["hazard_binary"] == 1).sum())), random_state=42)
    return (pd.concat([safe, haz])
              .sample(frac=1, random_state=42)
              .reset_index(drop=True))


@st.cache_data(show_spinner=False)
def build_pred_df(_pred, cache_key: str, df_json: str,
                  model: str, thresh: float) -> pd.DataFrame:
    """
    Run inference and return a single clean DataFrame with:
      zone_name, hazard_prob, hazard_pred  (plus all engineered columns)
    """
    df_raw = pd.read_json(df_json, orient="split")

    if _pred is None:
        df_raw["zone_name"]  = "Unknown"
        df_raw["hazard_prob"] = 0.5
        df_raw["hazard_pred"] = 0
        return df_raw

    results = _pred.predict_all(df_raw)
    df      = results["engineered"].copy()

    # ── Decode zones ──────────────────────────────────────────────────────────
    df["zone_name"] = decode_zones(df)

    # ── Attach RF predictions ─────────────────────────────────────────────────
    rf_out = (results.get("baseline", {})
                     .get("binary", {})
                     .get("random_forest", {}))
    if rf_out and rf_out.get("proba") is not None:
        df["rf_prob"] = rf_out["proba"][:, 1].astype(np.float32)
        df["rf_pred"] = rf_out["pred"].astype(np.int8)
    else:
        df["rf_prob"] = 0.5
        df["rf_pred"] = 0

    # ── Attach LSTM predictions ───────────────────────────────────────────────
    df["lstm_prob"] = np.nan
    lstm = results.get("lstm")
    if lstm and "error" not in (lstm or {}):
        df.iloc[lstm["indices"],
                df.columns.get_loc("lstm_prob")] = lstm["proba"].astype(np.float32)

    df["lstm_pred"] = np.where(df["lstm_prob"].fillna(0) >= thresh, 1, 0)

    # ── Choose primary probability for all charts ─────────────────────────────
    if model == "LSTM" and df["lstm_prob"].notna().any():
        df["hazard_prob"] = df["lstm_prob"].fillna(df["rf_prob"])
        df["hazard_pred"] = np.where(df["hazard_prob"] >= thresh, 1, 0).astype(np.int8)
    else:
        df["hazard_prob"] = df["rf_prob"]
        df["hazard_pred"] = df["rf_pred"]

    return df


@st.cache_data
def load_saved_metrics() -> dict:
    paths = {
        "Baseline":  ROOT / "reports"        / "metrics" / "summary_metrics.json",
        "Ablation":  ROOT / "reports_ablation"/ "metrics" / "summary_metrics.json",
        "LSTM":      ROOT / "reports_lstm"   / "metrics" / "forecast_metrics.json",
        "STGNN":     ROOT / "reports_stgnn"  / "metrics" / "stgnn_metrics.json",
    }
    return {k: json.loads(p.read_text()) for k, p in paths.items() if p.exists()}


# ─── Chart helpers ────────────────────────────────────────────────────────────
def _layout(**kw):
    base = dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", color=TEXT, size=12),
        margin=dict(t=44, b=28, l=12, r=12),
    )
    base.update(kw)
    return base


def chart_gauge(pct: float) -> go.Figure:
    clr = RED if pct >= .40 else (ORANGE if pct >= .15 else GREEN)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(pct * 100, 1),
        number=dict(suffix="%", font=dict(size=46, color=clr)),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1,
                      tickcolor=MUTED, tickfont=dict(size=10)),
            bar=dict(color=clr, thickness=0.22),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                dict(range=[0,  15], color=f"rgba(0,229,160,.08)"),
                dict(range=[15, 40], color=f"rgba(255,165,2,.08)"),
                dict(range=[40,100], color=f"rgba(255,71,87,.08)"),
            ],
            threshold=dict(line=dict(color="white", width=2),
                           thickness=0.78, value=round(pct*100,1)),
        ),
        title=dict(text="HAZARD ZONE COVERAGE",
                   font=dict(size=11, color=MUTED)),
    ))
    fig.update_layout(**_layout(height=230, margin=dict(t=30, b=0, l=30, r=30)))
    return fig


def chart_3d(df: pd.DataFrame) -> go.Figure:
    n   = min(len(df), 7000)
    d   = df.sample(n, random_state=42) if len(df) > n else df
    col = d["hazard_prob"].clip(0, 1)

    hover_extra = []
    for g in ["CH4", "CO", "H2"]:
        if g in d.columns:
            hover_extra.append(f"{g}: %{{customdata[{len(hover_extra)}]:.5f}}")

    customdata = np.column_stack([
        d[g].values if g in d.columns else np.zeros(len(d))
        for g in ["CH4", "CO", "H2"]
    ] + [d["zone_name"].values])

    hover = (
        "<b style='font-size:13px'>(%{x:.1f}, %{y:.1f}, %{z:.1f}) m</b><br>"
        "Hazard prob: <b>%{marker.color:.3f}</b><br>"
        + "<br>".join(
            f"{g}: %{{customdata[{i}]:.5f}}"
            for i, g in enumerate(["CH4", "CO", "H2"])
            if g in d.columns
        )
        + "<br>Zone: %{customdata[3]}<extra></extra>"
    )

    fig = go.Figure(go.Scatter3d(
        x=d["x"], y=d["y"], z=d["z"],
        mode="markers",
        marker=dict(
            size=3,
            color=col,
            colorscale=HAZARD_CS,
            cmin=0, cmax=1,
            opacity=0.80,
            colorbar=dict(
                title=dict(text="Hazard Prob", side="right", font=dict(size=11)),
                tickfont=dict(size=10),
                len=0.65, thickness=12,
            ),
            line=dict(width=0),
        ),
        customdata=customdata,
        hovertemplate=hover,
    ))
    fig.update_layout(
        **_layout(height=560),
        scene=dict(
            xaxis=dict(title="X — Length (m)",
                       backgroundcolor="rgba(0,0,0,0)", gridcolor=BORDER),
            yaxis=dict(title="Y — Width (m)",
                       backgroundcolor="rgba(0,0,0,0)", gridcolor=BORDER),
            zaxis=dict(title="Z — Height (m)",
                       backgroundcolor="rgba(0,0,0,0)", gridcolor=BORDER),
            bgcolor="rgba(0,0,0,0)",
            aspectmode="data",
            camera=dict(eye=dict(x=1.7, y=-1.5, z=0.85)),
        ),
        margin=dict(t=8, b=0, l=0, r=0),
    )
    return fig


def chart_heatmap(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Histogram2dContour(
        x=df["x"], y=df["y"],
        z=df["hazard_prob"],
        colorscale=HAZARD_CS,
        histfunc="avg",
        ncontours=20,
        contours=dict(showlabels=True, labelfont=dict(size=9, color="white")),
        colorbar=dict(title="Avg Hazard",
                      tickfont=dict(size=10), len=0.75, thickness=11),
        line=dict(width=0.5),
        hovertemplate="X: %{x:.1f} m  Y: %{y:.1f} m  Avg: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        **_layout(height=340),
        xaxis=dict(title="X — Tunnel Length (m)", gridcolor=BORDER, zeroline=False),
        yaxis=dict(title="Y — Width (m)", gridcolor=BORDER, zeroline=False),
        title=dict(text="Top-View Hazard Map (XY plane, averaged over height)",
                   font=dict(size=13), x=0),
    )
    return fig


def chart_temporal(df: pd.DataFrame) -> go.Figure:
    if "Time" not in df.columns:
        return go.Figure().update_layout(**_layout(height=340),
            title=dict(text="No Time column available"))

    ts = (df.groupby("Time")["hazard_prob"]
            .agg(frac=lambda x: (x >= .5).mean(),
                 avg="mean",
                 peak="max")
            .reset_index())

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts["Time"], y=ts["frac"],
        name="Hazard cell fraction", mode="lines",
        line=dict(color=RED, width=2.5),
        fill="tozeroy",
        fillcolor="rgba(255,71,87,.10)",
        hovertemplate="t=%{x:.0f}s  frac=%{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=ts["Time"], y=ts["avg"],
        name="Avg probability", mode="lines",
        line=dict(color=ORANGE, width=2, dash="dash"),
        hovertemplate="t=%{x:.0f}s  avg=%{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=ts["Time"], y=ts["peak"],
        name="Peak probability", mode="lines",
        line=dict(color=RED, width=1.5, dash="dot"),
        hovertemplate="t=%{x:.0f}s  peak=%{y:.3f}<extra></extra>",
    ))
    for y_val, lbl, clr in [(0.15, "Warning", ORANGE), (0.40, "Danger", RED)]:
        fig.add_hline(y=y_val, line_dash="dot", line_color=clr,
                      annotation_text=f" {lbl}", annotation_font_size=10,
                      annotation_position="right")
    fig.update_layout(
        **_layout(height=360),
        xaxis=dict(title="Simulation Time (s)", gridcolor=BORDER, zeroline=False),
        yaxis=dict(title="Fraction / Probability", gridcolor=BORDER,
                   zeroline=False, range=[0, 1]),
        legend=dict(bgcolor="rgba(0,0,0,0)", x=0.01, y=0.98,
                    bordercolor=BORDER, borderwidth=1, font=dict(size=11)),
        title=dict(text="Hazard Evolution Over Time", font=dict(size=13), x=0),
    )
    return fig


def chart_gas_time(df: pd.DataFrame) -> go.Figure:
    if "Time" not in df.columns:
        return go.Figure().update_layout(**_layout())

    gases   = [g for g in ["CH4", "CO", "H2"] if g in df.columns]
    palette = [BLUE, RED, ORANGE]
    gas_ts  = df.groupby("Time")[gases].mean().reset_index()

    fig = go.Figure()
    for gas, clr in zip(gases, palette):
        fig.add_trace(go.Scatter(
            x=gas_ts["Time"], y=gas_ts[gas],
            name=gas, mode="lines",
            line=dict(color=clr, width=2),
            hovertemplate=f"t=%{{x:.0f}}s  {gas}=%{{y:.6f}}<extra></extra>",
        ))
    fig.update_layout(
        **_layout(height=300),
        xaxis=dict(title="Simulation Time (s)", gridcolor=BORDER, zeroline=False),
        yaxis=dict(title="Mean Concentration (vol. fraction)",
                   gridcolor=BORDER, zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        title=dict(text="Average Gas Concentrations vs Time",
                   font=dict(size=13), x=0),
    )
    return fig


def chart_zone_bar(df: pd.DataFrame) -> go.Figure:
    z = (df.groupby("zone_name")["hazard_prob"]
           .agg(avg="mean", n="count",
                hazard=lambda x: int((x >= .5).sum()))
           .reset_index()
           .sort_values("avg", ascending=True))

    clrs = [RED if v >= .5 else (ORANGE if v >= .25 else GREEN) for v in z["avg"]]

    fig = go.Figure(go.Bar(
        y=z["zone_name"], x=z["avg"],
        orientation="h",
        marker=dict(color=clrs, line_width=0),
        customdata=np.column_stack([z["n"], z["hazard"]]),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Avg prob: %{x:.3f}<br>"
            "Total cells: %{customdata[0]:,}<br>"
            "Hazard cells: %{customdata[1]:,}<extra></extra>"
        ),
    ))
    fig.add_vline(x=0.5, line_dash="dot", line_color=ORANGE,
                  annotation_text=" threshold", annotation_font_size=10)
    fig.update_layout(
        **_layout(height=340),
        xaxis=dict(title="Average Hazard Probability",
                   gridcolor=BORDER, range=[0, 1], zeroline=False),
        yaxis=dict(gridcolor=BORDER),
        title=dict(text="Hazard Risk by Mine Zone", font=dict(size=13), x=0),
        showlegend=False,
    )
    return fig


def chart_x_profile(df: pd.DataFrame) -> go.Figure:
    bins     = pd.cut(df["x"], bins=40)
    xp       = df.groupby(bins, observed=True)["hazard_prob"].mean().reset_index()
    xp["xm"] = xp["x"].apply(lambda i: i.mid)

    fig = go.Figure(go.Bar(
        x=xp["xm"], y=xp["hazard_prob"],
        marker=dict(
            color=xp["hazard_prob"],
            colorscale=HAZARD_CS, cmin=0, cmax=1, line_width=0,
        ),
        hovertemplate="X: %{x:.1f} m<br>Avg prob: %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=0.5, line_dash="dot", line_color=ORANGE)
    fig.update_layout(
        **_layout(height=260),
        xaxis=dict(title="Tunnel Position X (m)", gridcolor=BORDER, zeroline=False),
        yaxis=dict(title="Avg Hazard Probability", gridcolor=BORDER,
                   zeroline=False, range=[0, 1]),
        title=dict(text="Hazard Profile Along Tunnel Axis",
                   font=dict(size=13), x=0),
        showlegend=False,
    )
    return fig


def chart_benchmark(saved: dict) -> go.Figure:
    """Grouped bar comparing all saved model metrics."""
    rows = []
    for name, data in saved.items():
        if name in ("LSTM", "STGNN"):
            m = data
            rows.append({
                "Model": name,
                "Accuracy":    m.get("accuracy", 0),
                "Recall":      m.get("class_1_recall", 0),
                "Precision":   m.get("class_1_precision", 0),
                "F1 Macro":    m.get("f1_macro", 0),
                "ROC-AUC":     m.get("roc_auc", 0),
            })
        else:
            for algo, m in data.get("binary", {}).items():
                label = f"{name} · {algo.replace('_',' ').title()}"
                rows.append({
                    "Model":     label,
                    "Accuracy":  m.get("accuracy", 0),
                    "Recall":    m.get("class_1_recall", 0),
                    "Precision": m.get("class_1_precision", 0),
                    "F1 Macro":  m.get("f1_macro", 0),
                    "ROC-AUC":   m.get("roc_auc", 0),
                })

    if not rows:
        return go.Figure().update_layout(**_layout())

    df_b   = pd.DataFrame(rows)
    cols   = ["Accuracy", "Recall", "Precision", "F1 Macro", "ROC-AUC"]
    pal    = [BLUE, GREEN, PURPLE, ORANGE, RED]

    fig = go.Figure()
    for c, clr in zip(cols, pal):
        fig.add_trace(go.Bar(
            name=c, x=df_b["Model"], y=df_b[c],
            marker=dict(color=clr, line_width=0),
            hovertemplate=f"<b>%{{x}}</b><br>{c}: %{{y:.4f}}<extra></extra>",
        ))
    fig.update_layout(
        **_layout(height=460),
        barmode="group",
        xaxis=dict(tickangle=-28, tickfont=dict(size=10), gridcolor=BORDER),
        yaxis=dict(title="Score", gridcolor=BORDER, range=[0, 1.05]),
        legend=dict(orientation="h", y=1.06, x=0.5, xanchor="center",
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        title=dict(text="Model Performance Benchmark — Binary Hazard Classification (Test Set)",
                   font=dict(size=13), x=0),
    )
    return fig


def chart_radar(saved: dict) -> go.Figure:
    cats   = ["Accuracy", "Recall", "Precision", "F1", "ROC-AUC"]
    angles = cats + [cats[0]]
    pal    = [GREEN, BLUE, ORANGE, PURPLE, RED]
    fig    = go.Figure()
    idx    = 0

    for name, data in saved.items():
        if name in ("LSTM", "STGNN"):
            m = data
            vals = [m.get("accuracy",0), m.get("class_1_recall",0),
                    m.get("class_1_precision",0), m.get("f1_macro",0),
                    m.get("roc_auc",0)] + [m.get("accuracy",0)]
            clr = pal[idx % len(pal)]
            fig.add_trace(go.Scatterpolar(
                r=vals, theta=angles, fill="toself",
                name=name, line_color=clr,
                fillcolor=clr.replace("#", "rgba(")[:-1] + ", .08)" if clr.startswith("#") else clr,
            ))
            idx += 1
        else:
            for algo, m in list(data.get("binary", {}).items())[:2]:
                clr  = pal[idx % len(pal)]
                vals = [m.get("accuracy",0), m.get("class_1_recall",0),
                        m.get("class_1_precision",0), m.get("f1_macro",0),
                        m.get("roc_auc",0)] + [m.get("accuracy",0)]
                fig.add_trace(go.Scatterpolar(
                    r=vals, theta=angles, fill="toself",
                    name=f"{name} · {algo.replace('_',' ').title()}",
                    line_color=clr,
                ))
                idx += 1
                if idx >= 6:
                    break

    fig.update_layout(
        **_layout(height=400),
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1],
                            gridcolor=BORDER, tickfont=dict(size=8)),
            angularaxis=dict(gridcolor=BORDER),
        ),
        legend=dict(x=1.08, y=0.5, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        title=dict(text="Performance Radar — All Models", font=dict(size=13), x=0),
        margin=dict(t=44, b=20, l=20, r=150),
    )
    return fig


# ─── HTML helpers ─────────────────────────────────────────────────────────────
def kpi(label: str, value: str, sub: str = "", color: str = TEXT) -> str:
    return f"""
<div style="background:{CARD};border:1px solid {BORDER};border-radius:12px;
            padding:20px 16px;text-align:center;height:100%;">
  <div style="color:{MUTED};font-size:10.5px;text-transform:uppercase;
              letter-spacing:1.3px;margin-bottom:10px;">{label}</div>
  <div style="color:{color};font-size:32px;font-weight:700;
              line-height:1.1;">{value}</div>
  <div style="color:{MUTED};font-size:11px;margin-top:7px;">{sub}</div>
</div>"""


def status_bar(pct: float, n_haz: int, n_total: int) -> str:
    if pct < 0.15:
        bg, border, color, icon, label = \
            "rgba(0,229,160,.08)", GREEN, GREEN, "✅", "SAFE"
    elif pct < 0.40:
        bg, border, color, icon, label = \
            "rgba(255,165,2,.10)", ORANGE, ORANGE, "⚠️", "WARNING"
    else:
        bg, border, color, icon, label = \
            "rgba(255,71,87,.12)", RED, RED, "🚨", "DANGER"

    return f"""
<div style="background:{bg};border:1.5px solid {border};border-radius:12px;
            padding:18px 28px;text-align:center;margin-bottom:20px;">
  <span style="color:{color};font-size:26px;font-weight:800;
               letter-spacing:4px;">{icon} &nbsp; {label}</span>
  <span style="color:{MUTED};font-size:14px;margin-left:20px;">
    {n_haz:,} hazard cells &nbsp;/&nbsp; {n_total:,} total
    &nbsp;·&nbsp; <b style="color:{color}">{pct:.1%}</b> coverage
  </span>
</div>"""


def section(title: str) -> None:
    st.markdown(
        f'<div style="color:{MUTED};font-size:11px;font-weight:600;'
        f'text-transform:uppercase;letter-spacing:1.5px;'
        f'border-bottom:1px solid {BORDER};padding-bottom:6px;'
        f'margin:24px 0 14px;">{title}</div>',
        unsafe_allow_html=True,
    )


# ─── Sidebar ──────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div style="margin-bottom:24px;">
          <div style="font-size:20px;font-weight:700;color:{GREEN};
                      letter-spacing:.5px;">⛏ GreenMining</div>
          <div style="font-size:11px;color:{MUTED};letter-spacing:1px;
                      text-transform:uppercase;margin-top:3px;">
            Safety Intelligence
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            f'<div style="color:{MUTED};font-size:10.5px;font-weight:600;'
            f'text-transform:uppercase;letter-spacing:1.2px;'
            f'border-bottom:1px solid {BORDER};padding-bottom:5px;'
            f'margin-bottom:12px;">Data Source</div>',
            unsafe_allow_html=True,
        )

        source = st.radio("Data source", ["Built-in Test Data", "Upload CSV"],
                          label_visibility="collapsed")

        uploaded = None
        n_rows   = 6000

        if source == "Upload CSV":
            uploaded = st.file_uploader(
                "CSV with CFD columns",
                type=["csv"],
                help="Required: Time, x, y, z, CH4, CO, H2, Temperature, "
                     "Velocity, Pressure  |  Optional: Scenario, Risk",
            )
        else:
            n_rows = st.slider("Sample size", 1000, 12000, 6000, 1000)

        st.markdown(
            f'<div style="color:{MUTED};font-size:10.5px;font-weight:600;'
            f'text-transform:uppercase;letter-spacing:1.2px;'
            f'border-bottom:1px solid {BORDER};padding-bottom:5px;'
            f'margin:20px 0 12px;">Prediction Model</div>',
            unsafe_allow_html=True,
        )

        model = st.selectbox(
            "Primary model", ["Random Forest", "LSTM"],
            label_visibility="collapsed",
        )
        thresh = st.slider(
            "Decision threshold", 0.10, 0.90, 0.535, 0.005,
            help="Lower = more sensitive (fewer missed hazards, more false alarms)",
        )

        # Model status
        pred = get_predictor()
        st.markdown(
            f'<div style="color:{MUTED};font-size:10.5px;font-weight:600;'
            f'text-transform:uppercase;letter-spacing:1.2px;'
            f'border-bottom:1px solid {BORDER};padding-bottom:5px;'
            f'margin:20px 0 12px;">Model Status</div>',
            unsafe_allow_html=True,
        )
        if pred:
            s = pred.status()
            items = [
                ("Random Forest",  bool(s.get("baseline_binary"))),
                ("LSTM Forecast",  s.get("lstm", False)),
                ("STGNN",          s.get("stgnn", False)),
            ]
            for name, ok in items:
                ico = f'<span style="color:{GREEN}">●</span>' if ok \
                      else f'<span style="color:{BORDER}">●</span>'
                clr = TEXT if ok else MUTED
                st.markdown(
                    f'<div style="font-size:12px;color:{clr};'
                    f'margin-bottom:6px;">{ico} &nbsp;{name}</div>',
                    unsafe_allow_html=True,
                )
            dev = s.get("device", "cpu")
            st.markdown(
                f'<div style="font-size:11px;color:{MUTED};margin-top:8px;">'
                f'Device: {dev}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div style="color:{RED};font-size:12px;">⚠ Predictor unavailable</div>',
                unsafe_allow_html=True,
            )

    return source, uploaded, n_rows, model, thresh


# ─── Tab renderers ────────────────────────────────────────────────────────────
def tab_overview(df: pd.DataFrame, pct: float, n_haz: int):
    c_gauge, c_pie = st.columns([1, 1], gap="medium")

    with c_gauge:
        st.plotly_chart(chart_gauge(pct), use_container_width=True, key="gauge")

        section("Top Risk Cells")
        top = df.nlargest(8, "hazard_prob")[["x", "y", "z", "hazard_prob", "zone_name"]]
        for _, r in top.iterrows():
            p = r["hazard_prob"]
            c = RED if p >= .8 else (ORANGE if p >= .5 else MUTED)
            st.markdown(
                f'<div style="background:{CARD};border-left:3px solid {c};'
                f'border-radius:0 8px 8px 0;padding:9px 14px;'
                f'margin-bottom:6px;font-size:12.5px;">'
                f'📍 ({r["x"]:.1f}, {r["y"]:.1f}, {r["z"]:.1f}) m'
                f'  ·  {r["zone_name"]}'
                f'  <span style="color:{c};font-weight:600;float:right;">'
                f'{p:.3f}</span></div>',
                unsafe_allow_html=True,
            )

    with c_pie:
        zone_g = (df.groupby("zone_name")["hazard_prob"]
                    .agg(avg="mean", n="count").reset_index())

        pie_colors = [
            RED if v >= .5 else (ORANGE if v >= .25 else GREEN)
            for v in zone_g["avg"]
        ]
        fig_pie = go.Figure(go.Pie(
            labels=zone_g["zone_name"],
            values=zone_g["n"],
            hole=0.55,
            marker=dict(colors=pie_colors,
                        line=dict(color=BG, width=2)),
            textinfo="percent+label",
            textfont=dict(size=11),
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Cells: %{value:,}<br>"
                "Share: %{percent}<extra></extra>"
            ),
        ))
        clr_label = RED if pct >= .4 else (ORANGE if pct >= .15 else GREEN)
        fig_pie.update_layout(
            **_layout(height=300),
            title=dict(text="Cell Distribution by Zone", font=dict(size=13), x=0),
            showlegend=True,
            legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=44, b=10, l=10, r=10),
            annotations=[dict(
                text=f"<b style='color:{clr_label}'>{pct:.0%}</b><br>hazard",
                font=dict(size=14, color=clr_label),
                showarrow=False,
            )],
        )
        st.plotly_chart(fig_pie, use_container_width=True, key="pie")

        # Gas violin (safe vs hazard)
        gases = [g for g in ["CH4", "CO", "H2"] if g in df.columns]
        if gases:
            section("Gas Distribution  ·  Safe vs Hazard Cells")
            sample = df.sample(min(4000, len(df)), random_state=42)
            sample["_class"] = np.where(sample["hazard_prob"] >= .5,
                                         "Hazard", "Safe")
            fig_v = make_subplots(
                rows=1, cols=len(gases),
                subplot_titles=gases,
            )
            for i, gas in enumerate(gases, 1):
                for cls, clr in [("Safe", GREEN), ("Hazard", RED)]:
                    sub = sample[sample["_class"] == cls][gas]
                    fig_v.add_trace(
                        go.Violin(
                            y=sub, name=cls,
                            legendgroup=cls, showlegend=(i == 1),
                            line_color=clr,
                            fillcolor=clr + "20",
                            box_visible=True,
                            meanline_visible=True,
                        ),
                        row=1, col=i,
                    )
            fig_v.update_layout(
                **_layout(height=300),
                title=dict(text="", font=dict(size=13)),
                violingap=0.15, violingroupgap=0.05,
            )
            st.plotly_chart(fig_v, use_container_width=True, key="violin")


def tab_minemap(df: pd.DataFrame):
    st.plotly_chart(chart_3d(df), use_container_width=True, key="3d")
    section("Top-View Plan (XY Heatmap)")
    st.plotly_chart(chart_heatmap(df), use_container_width=True, key="hm")


def tab_temporal(df: pd.DataFrame):
    if "Time" not in df.columns:
        st.info("No Time column in the data.")
        return
    st.plotly_chart(chart_temporal(df), use_container_width=True, key="temp")
    section("Gas Concentrations Over Time")
    st.plotly_chart(chart_gas_time(df), use_container_width=True, key="gas_t")


def tab_zones(df: pd.DataFrame):
    c1, c2 = st.columns([1.1, 1], gap="medium")
    with c1:
        st.plotly_chart(chart_zone_bar(df), use_container_width=True, key="zbar")
    with c2:
        section("Zone Risk Summary")
        zt = (df.groupby("zone_name")["hazard_prob"]
                .agg(Cells="count",
                     Avg_Prob="mean",
                     Max_Prob="max",
                     Hazard_Cells=lambda x: int((x >= .5).sum()))
                .reset_index()
                .rename(columns={"zone_name": "Zone"})
                .sort_values("Avg_Prob", ascending=False))
        zt["Hazard %"] = (zt["Hazard_Cells"] / zt["Cells"] * 100).round(1)
        zt[["Avg_Prob", "Max_Prob"]] = zt[["Avg_Prob","Max_Prob"]].round(4)
        st.dataframe(zt, use_container_width=True, hide_index=True,
                     column_config={
                         "Avg_Prob": st.column_config.ProgressColumn(
                             "Avg Prob", format="%.4f", min_value=0, max_value=1),
                         "Hazard %": st.column_config.NumberColumn(
                             "Hazard %", format="%.1f %%"),
                     })

    section("Hazard Profile Along Tunnel Axis")
    st.plotly_chart(chart_x_profile(df), use_container_width=True, key="xprofile")


def tab_benchmarks():
    saved = load_saved_metrics()
    if not saved:
        st.info("No saved metrics found. Run the training scripts first.")
        return

    st.plotly_chart(chart_benchmark(saved), use_container_width=True, key="bench")

    c1, c2 = st.columns([1, 1.1], gap="medium")
    with c1:
        st.plotly_chart(chart_radar(saved), use_container_width=True, key="radar")
    with c2:
        section("Detailed Metrics Table")
        rows = []
        for name, data in saved.items():
            if name in ("LSTM", "STGNN"):
                m = data
                rows.append({
                    "Model":     name,
                    "Accuracy":  round(m.get("accuracy", 0), 4),
                    "Recall":    round(m.get("class_1_recall", 0), 4),
                    "Precision": round(m.get("class_1_precision", 0), 4),
                    "F1 Macro":  round(m.get("f1_macro", 0), 4),
                    "ROC-AUC":   round(m.get("roc_auc", 0), 4),
                    "False Neg": m.get("false_negatives", "—"),
                })
            else:
                for algo, m in data.get("binary", {}).items():
                    rows.append({
                        "Model":     f"{name}·{algo.replace('_',' ').title()}",
                        "Accuracy":  round(m.get("accuracy", 0), 4),
                        "Recall":    round(m.get("class_1_recall", 0), 4),
                        "Precision": round(m.get("class_1_precision", 0), 4),
                        "F1 Macro":  round(m.get("f1_macro", 0), 4),
                        "ROC-AUC":   round(m.get("roc_auc", 0), 4),
                        "False Neg": m.get("false_negatives", "—"),
                    })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True,
                         hide_index=True,
                         column_config={
                             "Recall": st.column_config.ProgressColumn(
                                 "Recall", format="%.4f", min_value=0, max_value=1),
                             "ROC-AUC": st.column_config.ProgressColumn(
                                 "ROC-AUC", format="%.4f", min_value=0, max_value=1),
                         })


def tab_data(df: pd.DataFrame):
    keep = ["x", "y", "z"]
    if "Time"        in df.columns: keep = ["Time"] + keep
    if "zone_name"   in df.columns: keep.append("zone_name")
    for g in ["CH4","CO","H2","Temperature","Velocity","Pressure"]:
        if g in df.columns:
            keep.append(g)
    keep.append("hazard_prob")
    keep.append("hazard_pred")
    if "rf_prob"   in df.columns: keep.append("rf_prob")
    if "lstm_prob" in df.columns: keep.append("lstm_prob")
    if "hazard_binary" in df.columns: keep.append("hazard_binary")

    c1, c2, _ = st.columns([2, 1, 2])
    with c1:
        filt = st.selectbox("Filter rows",
                            ["All", "Hazard only (prob ≥ 0.5)",
                             "Safe only (prob < 0.5)"])
    disp = df[keep].copy()
    if "Hazard"  in filt: disp = disp[disp["hazard_prob"] >= 0.5]
    if "Safe"    in filt: disp = disp[disp["hazard_prob"] <  0.5]

    with c2:
        st.download_button(
            "⬇  Download predictions CSV",
            disp.round(6).to_csv(index=False).encode(),
            "greenmining_predictions.csv", "text/csv",
        )

    st.dataframe(
        disp.round(5).reset_index(drop=True),
        use_container_width=True,
        height=500,
        column_config={
            "hazard_prob": st.column_config.ProgressColumn(
                "hazard_prob", format="%.4f", min_value=0, max_value=1),
        },
    )
    st.caption(f"Showing {len(disp):,} of {len(df):,} cells")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    source, uploaded, n_rows, model, thresh = render_sidebar()

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:16px;margin-bottom:4px;">
      <div>
        <h1 style="color:{GREEN};margin:0;font-size:22px;font-weight:800;
                   letter-spacing:.3px;line-height:1.2;">
          ⛏ &nbsp; GreenMining Safety Intelligence
        </h1>
        <p style="color:{MUTED};margin:4px 0 0;font-size:12px;letter-spacing:.3px;">
          Underground mine hazard prediction &nbsp;·&nbsp;
          OpenFOAM CFD &nbsp;·&nbsp; RF &nbsp;·&nbsp; BiLSTM &nbsp;·&nbsp; STGNN
        </p>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # ── Load data ─────────────────────────────────────────────────────────────
    if source == "Upload CSV":
        if uploaded is None:
            st.info("👈 Upload a CSV to begin, or switch to Built-in Test Data.")
            _show_benchmark_only()
            return
        df_raw = pd.read_csv(uploaded)
    else:
        with st.spinner("Loading test data …"):
            df_raw = load_test_data(n_rows)

    if df_raw.empty:
        st.error("No data available. Ensure `data/final/test.csv` exists.")
        return

    # ── Run predictions ───────────────────────────────────────────────────────
    pred  = get_predictor()
    cache = str(hash(df_raw.to_json())) + model + str(thresh)

    with st.spinner("Running predictions …"):
        df = build_pred_df(
            pred, cache,
            df_raw.to_json(orient="split"),
            model, thresh,
        )

    if df.empty or "hazard_prob" not in df.columns:
        st.error("Prediction failed. Check that inference.py is in the same directory.")
        return

    pct   = float((df["hazard_prob"] >= 0.5).mean())
    n_haz = int((df["hazard_prob"] >= 0.5).sum())
    n_tot = len(df)

    # ── Status banner ─────────────────────────────────────────────────────────
    st.markdown(status_bar(pct, n_haz, n_tot), unsafe_allow_html=True)

    # ── KPI row ───────────────────────────────────────────────────────────────
    saved    = load_saved_metrics()
    rf_acc   = saved.get("Baseline",{}).get("binary",{}).get("random_forest",{}).get("accuracy", None)
    lstm_rec = saved.get("LSTM",{}).get("class_1_recall", None)

    kpi_clr = RED if pct >= .4 else (ORANGE if pct >= .15 else GREEN)
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, html in zip(
        [c1, c2, c3, c4, c5],
        [
            kpi("Total Cells",       f"{n_tot:,}",      "analysed"),
            kpi("Hazard Cells",      f"{n_haz:,}",      f"{pct:.1%} of total", kpi_clr),
            kpi("Safe Cells",        f"{n_tot - n_haz:,}", f"{1-pct:.1%} of total", GREEN),
            kpi("RF Test Accuracy",  f"{rf_acc:.4f}" if rf_acc else "—",
                "saved metric"),
            kpi("LSTM Hazard Recall", f"{lstm_rec:.4f}" if lstm_rec else "—",
                "saved metric"),
        ],
    ):
        col.markdown(html, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    t1, t2, t3, t4, t5, t6 = st.tabs([
        "🗺  Overview",
        "🌐  Mine Map 3D",
        "📈  Temporal",
        "🏗  Zones",
        "🏆  Benchmarks",
        "🗃  Data",
    ])

    with t1: tab_overview(df, pct, n_haz)
    with t2: tab_minemap(df)
    with t3: tab_temporal(df)
    with t4: tab_zones(df)
    with t5: tab_benchmarks()
    with t6: tab_data(df)


def _show_benchmark_only():
    """Fallback when no data is loaded yet — still show saved metrics."""
    saved = load_saved_metrics()
    if saved:
        st.markdown(f"""
        <div style="color:{MUTED};font-size:12px;margin:12px 0 20px;">
          No data loaded — showing saved training metrics below.
        </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(chart_benchmark(saved), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
main()
