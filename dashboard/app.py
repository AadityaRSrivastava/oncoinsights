"""Module 7: OncoInsights KPI dashboard (Streamlit).

Five tabs: Overview, Clinical & Demographic, Expression, Mutation, Survival.
Sidebar filters (age, gender, stage, mutation status, gene selector) apply
across all tabs via a single filtered `patients` dataframe.

Visual layer: a dark "genomics-console" theme — custom CSS for the hero,
KPI cards, section headers and callouts, plus one shared Plotly template so
every chart is styled consistently. Presentation only; all computations are
unchanged and stay consistent with the offline pipeline (Module 6).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless server: avoid GUI backend init crashes from lifelines' matplotlib import

import pandas as pd
import plotly.express as px
import streamlit as st
import yaml
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "config.yaml"

st.set_page_config(
    page_title="OncoInsights — TCGA-LUAD",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Design system
# --------------------------------------------------------------------------- #
INK = "#e2e8f0"        # primary text
MUTED = "#94a3b8"      # secondary text
GRID = "rgba(148,163,184,0.12)"
AXIS = "rgba(148,163,184,0.28)"
FONT = "Inter, ui-sans-serif, system-ui, sans-serif"

# Qualitative palette (cyan → violet → teal → pink → amber → green …)
PALETTE = ["#22d3ee", "#818cf8", "#2dd4bf", "#f472b6", "#fbbf24",
           "#34d399", "#60a5fa", "#fb7185", "#a78bfa", "#38bdf8"]

# Stage colour ramp encodes clinical severity (I low → IV high).
STAGE_COLORS = {"I": "#2dd4bf", "II": "#38bdf8", "III": "#fbbf24", "IV": "#f472b6"}
STAGE_ORDER = ["I", "II", "III", "IV"]

PLOT_CONFIG = {"displayModeBar": False, "responsive": True}


def _css() -> str:
    return """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');

:root {
  --bg:#0a0f1e; --panel:#111a2e; --border:rgba(148,163,184,0.14);
  --ink:#e2e8f0; --muted:#94a3b8;
  --cyan:#22d3ee; --violet:#818cf8; --teal:#2dd4bf; --pink:#f472b6; --amber:#fbbf24;
}

html, body, [class*="css"], .stApp, [data-testid="stSidebar"] {
  font-family: 'Inter', ui-sans-serif, system-ui, sans-serif;
}

.stApp {
  background:
    radial-gradient(1200px 620px at 82% -12%, rgba(34,211,238,0.10), transparent 60%),
    radial-gradient(1000px 520px at -5% 0%, rgba(129,140,248,0.09), transparent 55%),
    var(--bg);
}
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1400px; }

/* ---------- Hero ---------- */
.hero {
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 22px 26px;
  margin-bottom: 18px;
  background:
    linear-gradient(180deg, rgba(17,26,46,0.85), rgba(10,15,30,0.55)),
    radial-gradient(600px 200px at 0% 0%, rgba(34,211,238,0.12), transparent 70%);
  box-shadow: 0 20px 50px -30px rgba(34,211,238,0.35);
  position: relative;
  overflow: hidden;
}
.hero::after {
  content:""; position:absolute; inset:0 0 auto 0; height:2px;
  background: linear-gradient(90deg, var(--cyan), var(--violet), transparent);
}
.hero-eyebrow {
  font-size: 0.72rem; letter-spacing: 0.22em; text-transform: uppercase;
  color: var(--cyan); font-weight: 700; margin-bottom: 6px;
  display:flex; align-items:center; gap:8px;
}
.hero-eyebrow .dot { width:7px; height:7px; border-radius:50%; background:var(--cyan);
  box-shadow:0 0 10px 2px rgba(34,211,238,0.7); }
.hero-title {
  font-size: 2.05rem; font-weight: 800; line-height: 1.1; margin: 0 0 8px 0;
  background: linear-gradient(92deg, #f8fafc 10%, #7dd3fc 55%, #a5b4fc 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.hero-sub { color: var(--muted); font-size: 0.95rem; max-width: 820px; line-height: 1.5; }
.hero-chips { margin-top: 14px; display:flex; flex-wrap:wrap; gap:8px; }
.chip {
  font-size: 0.74rem; font-weight: 600; color: #cbd5e1;
  border: 1px solid var(--border); border-radius: 999px; padding: 4px 12px;
  background: rgba(148,163,184,0.06);
}
.chip b { color: var(--cyan); font-family:'JetBrains Mono', monospace; }

/* ---------- KPI cards ---------- */
.kpi {
  border: 1px solid var(--border); border-radius: 16px; padding: 16px 18px 14px;
  background: linear-gradient(180deg, rgba(17,26,46,0.9), rgba(12,19,35,0.75));
  position: relative; overflow: hidden; height: 100%;
  transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease;
}
.kpi:hover { transform: translateY(-2px); border-color: rgba(148,163,184,0.28);
  box-shadow: 0 18px 40px -28px var(--accent); }
.kpi::before {
  content:""; position:absolute; top:0; left:0; right:0; height:3px;
  background: linear-gradient(90deg, var(--accent), transparent 85%);
}
.kpi-label {
  font-size: 0.72rem; letter-spacing: 0.13em; text-transform: uppercase;
  color: var(--muted); font-weight: 700; display:flex; align-items:center; gap:7px;
}
.kpi-label .ic { font-size: 0.95rem; filter: saturate(1.2); }
.kpi-value {
  font-family: 'JetBrains Mono', monospace; font-weight: 600;
  font-size: 1.72rem; color: var(--ink); line-height: 1.15; margin-top: 6px;
  white-space: nowrap;
}
.kpi-value .accent { color: var(--accent); font-size: 0.82em; }
.kpi-sub { font-size: 0.76rem; color: var(--muted); margin-top: 3px; }

/* ---------- Section header ---------- */
.sec { display:flex; align-items:baseline; gap:12px; margin: 6px 0 2px; }
.sec .bar { width:4px; height:20px; border-radius:4px; background: linear-gradient(180deg,var(--cyan),var(--violet)); align-self:center; }
.sec h3 { margin:0; font-size:1.12rem; font-weight:700; color:#f1f5f9; }
.sec p { margin:0; color: var(--muted); font-size:0.82rem; }

/* ---------- Callout ---------- */
.callout {
  border:1px solid var(--border); border-left:3px solid var(--accent);
  border-radius:12px; padding:12px 16px; margin-top:6px;
  background: rgba(148,163,184,0.05);
}
.callout .k { font-size:0.72rem; text-transform:uppercase; letter-spacing:0.12em; color:var(--muted); font-weight:700; }
.callout .v { font-family:'JetBrains Mono',monospace; font-size:1.15rem; color:var(--ink); font-weight:600; margin:2px 0; }
.callout .v b { color: var(--accent); }
.callout .d { font-size:0.82rem; color:var(--muted); }

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] {
  height: 42px; padding: 0 16px; background: transparent; border-radius: 10px 10px 0 0;
  color: var(--muted); font-weight: 600; font-size: 0.9rem;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--ink); background: rgba(148,163,184,0.06); }
.stTabs [aria-selected="true"] { color: #ffffff !important; background: rgba(34,211,238,0.08); }
.stTabs [data-baseweb="tab-highlight"] { background: linear-gradient(90deg,var(--cyan),var(--violet)); height:3px; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(14,22,38,0.96), rgba(10,15,30,0.96));
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }
.side-brand { display:flex; align-items:center; gap:10px; margin-bottom: 4px; }
.side-brand .logo {
  width:34px; height:34px; border-radius:10px; display:grid; place-items:center; font-size:1.1rem;
  background: linear-gradient(135deg, rgba(34,211,238,0.22), rgba(129,140,248,0.22));
  border:1px solid var(--border);
}
.side-brand .t { font-weight:800; font-size:1.02rem; color:#f1f5f9; letter-spacing:-0.01em; }
.side-brand .s { font-size:0.72rem; color:var(--muted); }
.side-badge {
  margin-top: 4px; border:1px solid var(--border); border-radius:12px; padding:10px 12px;
  background: rgba(34,211,238,0.07);
}
.side-badge .n { font-family:'JetBrains Mono',monospace; font-size:1.35rem; font-weight:600; color:var(--cyan); }
.side-badge .l { font-size:0.72rem; color:var(--muted); }

hr { border-color: var(--border); }
</style>
"""


def style_fig(fig, height: int = 340, showlegend: bool = True):
    """Apply the shared premium template to any Plotly figure."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=13, color=INK),
        colorway=PALETTE,
        title=dict(text=""),
        legend_title_text="",
        height=height,
        margin=dict(l=8, r=12, t=10, b=8),
        showlegend=showlegend,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            font=dict(color=MUTED, size=12), bgcolor="rgba(0,0,0,0)",
        ),
        hoverlabel=dict(bgcolor="#0e1626", font=dict(family=FONT, size=12, color=INK),
                        bordercolor="rgba(148,163,184,0.3)"),
        bargap=0.18,
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=AXIS,
                     tickfont=dict(color=MUTED), title_font=dict(color=MUTED, size=12))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=AXIS,
                     tickfont=dict(color=MUTED), title_font=dict(color=MUTED, size=12))
    return fig


def show(fig, height: int = 340, showlegend: bool = True):
    st.plotly_chart(style_fig(fig, height, showlegend), use_container_width=True, config=PLOT_CONFIG)


def section(title: str, subtitle: str = "") -> None:
    st.markdown(
        f'<div class="sec"><span class="bar"></span><h3>{title}</h3>'
        f'<p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def kpi(col, label: str, value: str, sub: str, accent: str, icon: str = "") -> None:
    ic = f'<span class="ic">{icon}</span>' if icon else ""
    col.markdown(
        f'<div class="kpi" style="--accent:{accent}">'
        f'<div class="kpi-label">{ic}{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def callout(col, k: str, value_html: str, desc: str, accent: str) -> None:
    col.markdown(
        f'<div class="callout" style="--accent:{accent}">'
        f'<div class="k">{k}</div><div class="v">{value_html}</div>'
        f'<div class="d">{desc}</div></div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@st.cache_data
def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@st.cache_data
def load_data(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    conn = sqlite3.connect(ROOT / cfg["paths"]["processed_files"]["sqlite_db"])
    patients = pd.read_sql("SELECT * FROM patients", conn)
    mutations = pd.read_sql("SELECT * FROM mutations", conn)
    expression = pd.read_sql("SELECT * FROM expression", conn)
    conn.close()
    return patients, mutations, expression


def apply_filters(patients: pd.DataFrame, mutations: pd.DataFrame,
                   age_range, sexes, stages, mutation_gene, mutation_status) -> pd.DataFrame:
    df = patients[
        (patients["AGE"] >= age_range[0]) & (patients["AGE"] <= age_range[1])
        & (patients["SEX"].isin(sexes))
        & (patients["STAGE_GROUP"].isin(stages))
    ]
    if mutation_gene != "(any)":
        mutated_patients = set(mutations.loc[mutations["hugoGeneSymbol"] == mutation_gene, "patientId"])
        if mutation_status == "Mutant":
            df = df[df["patientId"].isin(mutated_patients)]
        elif mutation_status == "Wild-type":
            df = df[~df["patientId"].isin(mutated_patients)]
    return df


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
def main() -> None:
    cfg = load_config()
    patients, mutations, expression = load_data(cfg)
    gene_panel = sorted(cfg["gene_panel"])

    st.markdown(_css(), unsafe_allow_html=True)

    # ---- Sidebar ----
    st.sidebar.markdown(
        '<div class="side-brand"><div class="logo">🧬</div>'
        '<div><div class="t">OncoInsights</div>'
        '<div class="s">TCGA-LUAD cohort explorer</div></div></div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Filters**")
    age_min, age_max = int(patients["AGE"].min()), int(patients["AGE"].max())
    age_range = st.sidebar.slider("Age range", age_min, age_max, (age_min, age_max))
    sexes = st.sidebar.multiselect("Gender", sorted(patients["SEX"].dropna().unique()),
                                    default=sorted(patients["SEX"].dropna().unique()))
    stages = st.sidebar.multiselect("Stage", STAGE_ORDER, default=STAGE_ORDER)
    mutation_gene = st.sidebar.selectbox("Mutation filter — gene", ["(any)"] + gene_panel)
    mutation_status = st.sidebar.radio("Mutation status", ["Any", "Mutant", "Wild-type"],
                                        disabled=(mutation_gene == "(any)"))
    expr_gene = st.sidebar.selectbox("Expression gene (Expression tab)", gene_panel,
                                      index=gene_panel.index("EGFR") if "EGFR" in gene_panel else 0)

    filtered = apply_filters(patients, mutations, age_range, sexes, stages, mutation_gene, mutation_status)
    pct_match = 100 * len(filtered) / len(patients) if len(patients) else 0
    st.sidebar.markdown(
        f'<div class="side-badge"><div class="n">{len(filtered)}<span style="color:#64748b;font-size:0.9rem"> / {len(patients)}</span></div>'
        f'<div class="l">patients match filters ({pct_match:.0f}%)</div></div>',
        unsafe_allow_html=True,
    )

    # ---- Hero ----
    st.markdown(
        '<div class="hero">'
        '<div class="hero-eyebrow"><span class="dot"></span>Clinical &amp; Genomic Decision Intelligence</div>'
        '<div class="hero-title">OncoInsights — TCGA-LUAD Analytics Platform</div>'
        '<div class="hero-sub">Interactive exploration of the TCGA Lung Adenocarcinoma PanCancer Atlas cohort: '
        'demographics, a 40-gene curated driver panel, expression profiles, and a survival-validated '
        'composite risk score — every view responds to the sidebar filters.</div>'
        '<div class="hero-chips">'
        '<span class="chip">Source <b>cBioPortal · luad_tcga_pan_can_atlas_2018</b></span>'
        f'<span class="chip">Cohort <b>{len(patients)}</b> patients</span>'
        '<span class="chip">Panel <b>40</b> driver genes</span>'
        '<span class="chip">Risk model <b>0.4·mut + 0.3·age + 0.3·stage</b></span>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    if len(filtered) == 0:
        st.warning("No patients match the current filters. Widen the age range or add stages/genders in the sidebar.")
        return

    tab_overview, tab_clinical, tab_expression, tab_mutation, tab_survival = st.tabs(
        ["  Overview  ", "  Clinical & Demographic  ", "  Expression  ", "  Mutation  ", "  Survival  "]
    )

    # ===================== OVERVIEW =====================
    with tab_overview:
        section("Headline KPIs", "Live figures for the filtered cohort")
        c1, c2, c3, c4 = st.columns(4)
        n_high = int(filtered["HIGH_RISK_FLAG"].sum())
        pct_high = 100 * n_high / len(filtered)
        kpi(c1, "Patients", f"{len(filtered)}", "matching active filters", "#22d3ee", "👥")
        kpi(c2, "Median age", f"{filtered['AGE'].median():.0f}<span class='accent'> yr</span>",
            f"mean {filtered['AGE'].mean():.1f} · at diagnosis", "#818cf8", "🎂")
        kpi(c3, "Avg risk score", f"{filtered['RISK_SCORE'].mean():.3f}",
            "composite z-scored index", "#fbbf24", "⚖️")
        kpi(c4, "High-risk", f"{n_high}<span class='accent'> ({pct_high:.0f}%)</span>",
            "top-quartile RISK_SCORE", "#f472b6", "🚩")

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            section("Stage distribution", "AJCC pathologic stage · colour encodes severity")
            stage_counts = filtered["STAGE_GROUP"].value_counts().reindex(STAGE_ORDER).fillna(0)
            fig = px.bar(x=stage_counts.index, y=stage_counts.values,
                         labels={"x": "Stage", "y": "Patients"},
                         color=stage_counts.index, color_discrete_map=STAGE_COLORS)
            fig.update_traces(marker_line_width=0, hovertemplate="Stage %{x}<br>%{y} patients<extra></extra>")
            show(fig, height=330, showlegend=False)
        with col2:
            section("Risk score distribution", "Dashed line = high-risk threshold (top quartile)")
            thr = patients["RISK_SCORE"].quantile(cfg["features"]["high_risk_threshold_percentile"])
            fig = px.histogram(filtered, x="RISK_SCORE", nbins=28,
                               color_discrete_sequence=["#22d3ee"])
            fig.update_traces(marker_line_width=0, opacity=0.9,
                              hovertemplate="Risk %{x:.2f}<br>%{y} patients<extra></extra>")
            fig.add_vline(x=thr, line_dash="dash", line_color="#f472b6",
                          annotation_text="high-risk", annotation_position="top",
                          annotation_font_color="#f472b6")
            fig.update_layout(xaxis_title="RISK_SCORE", yaxis_title="Patients")
            show(fig, height=330, showlegend=False)

    # ===================== CLINICAL =====================
    with tab_clinical:
        section("Clinical & Demographic Analytics", "Age, sex and stage structure of the filtered cohort")
        col1, col2 = st.columns([1.3, 1])
        with col1:
            st.caption("Age distribution")
            fig = px.histogram(filtered, x="AGE", nbins=22, color_discrete_sequence=["#818cf8"])
            fig.update_traces(marker_line_width=0, opacity=0.9,
                              hovertemplate="Age %{x}<br>%{y} patients<extra></extra>")
            fig.update_layout(xaxis_title="Age (years)", yaxis_title="Patients")
            show(fig, height=320, showlegend=False)
        with col2:
            st.caption("Gender ratio")
            sex_counts = filtered["SEX"].value_counts()
            fig = px.pie(names=sex_counts.index, values=sex_counts.values, hole=0.6,
                         color_discrete_sequence=["#22d3ee", "#f472b6", "#818cf8"])
            fig.update_traces(textinfo="percent+label", textfont_size=13,
                              marker=dict(line=dict(color="#0a0f1e", width=2)))
            show(fig, height=320, showlegend=False)

        section("Age by stage", "Do later-stage patients skew older at diagnosis?")
        fig = px.box(filtered, x="STAGE_GROUP", y="AGE", color="STAGE_GROUP",
                     category_orders={"STAGE_GROUP": STAGE_ORDER}, color_discrete_map=STAGE_COLORS,
                     points="outliers")
        fig.update_layout(xaxis_title="Stage", yaxis_title="Age (years)")
        show(fig, height=340, showlegend=False)

    # ===================== EXPRESSION =====================
    with tab_expression:
        section(f"Expression Analytics — {expr_gene}", "RNA-seq (RSEM), log2(x+1) transformed")
        gene_expr_long = expression[expression["hugoGeneSymbol"] == expr_gene]
        merged = filtered.merge(gene_expr_long[["patientId", "log2_expression"]], on="patientId", how="left")

        col1, col2 = st.columns(2)
        with col1:
            st.caption(f"{expr_gene} expression distribution")
            fig = px.histogram(merged, x="log2_expression", nbins=26,
                               color_discrete_sequence=["#2dd4bf"])
            fig.update_traces(marker_line_width=0, opacity=0.9,
                              hovertemplate="log2 %{x:.2f}<br>%{y} patients<extra></extra>")
            fig.update_layout(xaxis_title=f"{expr_gene} log2(RSEM+1)", yaxis_title="Patients")
            show(fig, height=330, showlegend=False)
        with col2:
            st.caption(f"{expr_gene} expression by stage")
            fig = px.box(merged, x="STAGE_GROUP", y="log2_expression", color="STAGE_GROUP",
                         category_orders={"STAGE_GROUP": STAGE_ORDER}, color_discrete_map=STAGE_COLORS,
                         points="outliers")
            fig.update_layout(xaxis_title="Stage", yaxis_title=f"{expr_gene} log2(RSEM+1)")
            show(fig, height=330, showlegend=False)

        med = merged["log2_expression"].median()
        st.caption(f"Median {expr_gene} expression in the filtered cohort: "
                   f"**{med:.2f}** log2(RSEM+1)" if pd.notna(med) else
                   f"No expression values available for {expr_gene} in the filtered cohort.")

    # ===================== MUTATION =====================
    with tab_mutation:
        section("Mutation Analytics", "Driver-panel mutation frequency and burden per patient")
        cohort_mutations = mutations[mutations["patientId"].isin(filtered["patientId"])]

        # Headline mutation KPIs
        n = len(filtered)
        top_counts = cohort_mutations.groupby("hugoGeneSymbol")["patientId"].nunique().sort_values(ascending=False)
        c1, c2, c3 = st.columns(3)
        if len(top_counts):
            kpi(c1, "Top mutated gene", f"{top_counts.index[0]}",
                f"{100*top_counts.iloc[0]/n:.0f}% of cohort mutated", "#22d3ee", "🧬")
        else:
            kpi(c1, "Top mutated gene", "—", "no mutations in filter", "#22d3ee", "🧬")
        kpi(c2, "Mean burden", f"{filtered['PANEL_MUTATION_COUNT'].mean():.1f}",
            "panel mutations / patient", "#818cf8", "📊")
        pct_any = 100 * (filtered["PANEL_MUTATION_COUNT"] > 0).mean()
        kpi(c3, "Any panel mutation", f"{pct_any:.0f}%",
            "≥1 of 40 driver genes", "#fbbf24", "✳️")

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns([1.25, 1])
        with col1:
            section("Most frequently mutated genes", "Top 15 by patients carrying ≥1 mutation")
            top_genes = top_counts.head(15)
            fig = px.bar(x=top_genes.values, y=top_genes.index, orientation="h",
                         labels={"x": "Patients mutated", "y": "Gene"},
                         color=top_genes.values, color_continuous_scale=["#164e63", "#22d3ee"])
            fig.update_traces(marker_line_width=0,
                              hovertemplate="%{y}<br>%{x} patients<extra></extra>")
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
            show(fig, height=430, showlegend=False)
        with col2:
            section("Mutation burden", "Panel mutation count per patient")
            fig = px.histogram(filtered, x="PANEL_MUTATION_COUNT", nbins=15,
                               color_discrete_sequence=["#818cf8"])
            fig.update_traces(marker_line_width=0, opacity=0.9,
                              hovertemplate="%{x} mutations<br>%{y} patients<extra></extra>")
            fig.update_layout(xaxis_title="Panel mutations", yaxis_title="Patients")
            show(fig, height=430, showlegend=False)

    # ===================== SURVIVAL =====================
    with tab_survival:
        section("Survival Analytics", "Kaplan-Meier overall survival with log-rank testing")
        km_group_by = st.radio("Stratify Kaplan-Meier curve by",
                               ["HIGH_RISK_FLAG", "STAGE_GROUP"], horizontal=True)

        surv_df = filtered.dropna(subset=["SURVIVAL_MONTHS", "EVENT_OCCURRED"])
        fig = px.line()
        groups = sorted(surv_df[km_group_by].dropna().unique())
        label_map = {0: "Low risk", 1: "High risk"} if km_group_by == "HIGH_RISK_FLAG" else {}
        color_for = (lambda g: {0: "#2dd4bf", 1: "#f472b6"}.get(g, "#22d3ee")) \
            if km_group_by == "HIGH_RISK_FLAG" else (lambda g: STAGE_COLORS.get(str(g), "#22d3ee"))
        for g in groups:
            sub = surv_df[surv_df[km_group_by] == g]
            if len(sub) < 2:
                continue
            kmf = KaplanMeierFitter()
            label = label_map.get(g, f"Stage {g}" if km_group_by == "STAGE_GROUP" else str(g))
            kmf.fit(sub["SURVIVAL_MONTHS"], event_observed=sub["EVENT_OCCURRED"], label=label)
            km_curve = kmf.survival_function_.reset_index()
            fig.add_scatter(x=km_curve["timeline"], y=km_curve[label], mode="lines", name=label,
                            line=dict(width=2.6, shape="hv", color=color_for(g)),
                            hovertemplate=f"{label}<br>%{{x:.0f}} mo<br>S(t)=%{{y:.2f}}<extra></extra>")
        fig.update_layout(xaxis_title="Months", yaxis_title="Survival probability",
                          yaxis=dict(range=[0, 1.02]))
        show(fig, height=420, showlegend=True)

        if len(groups) == 2:
            a = surv_df[surv_df[km_group_by] == groups[0]]
            b = surv_df[surv_df[km_group_by] == groups[1]]
            if len(a) > 1 and len(b) > 1:
                result = logrank_test(a["SURVIVAL_MONTHS"], b["SURVIVAL_MONTHS"],
                                       event_observed_A=a["EVENT_OCCURRED"], event_observed_B=b["EVENT_OCCURRED"])
                p = result.p_value
                sig = p < 0.05
                accent = "#2dd4bf" if sig else "#fbbf24"
                verdict = "Statistically significant separation" if sig else "No significant separation"
                cc1, cc2 = st.columns([1, 2])
                callout(cc1, "Log-rank p-value",
                        f"<b>{p:.4f}</b>", verdict, accent)
                cc2.markdown(
                    f'<div class="callout" style="--accent:{accent}">'
                    f'<div class="k">Interpretation</div>'
                    f'<div class="d" style="margin-top:4px">The two survival curves '
                    f'{"differ more than expected by chance" if sig else "are statistically indistinguishable"} '
                    f'at α=0.05. {"The stratification meaningfully separates outcomes." if sig else "Treat any visual gap with caution."}</div></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("Log-rank test is shown when exactly two strata are present "
                       "(e.g. High vs Low risk, or filter Stage down to two groups).")


if __name__ == "__main__":
    main()
