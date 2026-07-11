"""Module 7: OncoInsights KPI dashboard (Streamlit).

Five tabs: Overview, Clinical & Demographic, Expression, Mutation, Survival.
Sidebar filters (age, gender, stage, mutation status, gene selector) apply
across all tabs via a single filtered `patients` dataframe.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import yaml
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "config.yaml"

st.set_page_config(page_title="OncoInsights", layout="wide")


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


STAGE_ORDER = ["I", "II", "III", "IV"]


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


def main() -> None:
    cfg = load_config()
    patients, mutations, expression = load_data(cfg)
    gene_panel = sorted(cfg["gene_panel"])

    st.title("OncoInsights — TCGA-LUAD Analytics Platform")
    st.caption(
        "Clinical & genomic decision intelligence for the TCGA Lung Adenocarcinoma PanCancer Atlas "
        "cohort (503 patients, 40-gene curated driver panel)."
    )

    st.sidebar.header("Filters")
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
    st.sidebar.markdown(f"**{len(filtered)} / {len(patients)} patients match filters**")

    tab_overview, tab_clinical, tab_expression, tab_mutation, tab_survival = st.tabs(
        ["Overview", "Clinical & Demographic", "Expression", "Mutation", "Survival"]
    )

    with tab_overview:
        st.subheader("Headline KPIs")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Patients", len(filtered))
        c2.metric("Avg. age", f"{filtered['AGE'].mean():.1f}" if len(filtered) else "-")
        c3.metric("Avg. risk score", f"{filtered['RISK_SCORE'].mean():.3f}" if len(filtered) else "-")
        c4.metric("High-risk patients", int(filtered["HIGH_RISK_FLAG"].sum()))

        stage_counts = filtered["STAGE_GROUP"].value_counts().reindex(STAGE_ORDER).fillna(0)
        fig = px.bar(x=stage_counts.index, y=stage_counts.values,
                     labels={"x": "Stage", "y": "Patients"}, title="Stage distribution (filtered cohort)")
        st.plotly_chart(fig, use_container_width=True)

    with tab_clinical:
        st.subheader("Clinical & Demographic Analytics")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(filtered, x="AGE", nbins=20, title="Age distribution")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            sex_counts = filtered["SEX"].value_counts()
            fig = px.pie(names=sex_counts.index, values=sex_counts.values, title="Gender ratio")
            st.plotly_chart(fig, use_container_width=True)

        fig = px.box(filtered, x="STAGE_GROUP", y="AGE", category_orders={"STAGE_GROUP": STAGE_ORDER},
                     title="Age by stage")
        st.plotly_chart(fig, use_container_width=True)

    with tab_expression:
        st.subheader(f"Expression Analytics — {expr_gene}")
        expr_col = f"{expr_gene}_EXPR" if f"{expr_gene}_EXPR" in filtered.columns else None
        if expr_col:
            expr_vals = filtered[expr_col]
        else:
            gene_expr_long = expression[expression["hugoGeneSymbol"] == expr_gene]
            merged = filtered.merge(gene_expr_long[["patientId", "log2_expression"]], on="patientId", how="left")
            expr_vals = merged["log2_expression"]
            filtered_expr_df = merged
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(x=expr_vals, nbins=25,
                                labels={"x": f"{expr_gene} log2(RSEM+1)"},
                                title=f"{expr_gene} expression distribution")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            gene_expr_long = expression[expression["hugoGeneSymbol"] == expr_gene]
            merged = filtered.merge(gene_expr_long[["patientId", "log2_expression"]], on="patientId", how="left")
            fig = px.box(merged, x="STAGE_GROUP", y="log2_expression",
                         category_orders={"STAGE_GROUP": STAGE_ORDER},
                         title=f"{expr_gene} expression by stage")
            st.plotly_chart(fig, use_container_width=True)

    with tab_mutation:
        st.subheader("Mutation Analytics")
        cohort_mutations = mutations[mutations["patientId"].isin(filtered["patientId"])]
        top_genes = (cohort_mutations.groupby("hugoGeneSymbol")["patientId"]
                     .nunique().sort_values(ascending=False).head(15))
        fig = px.bar(x=top_genes.values, y=top_genes.index, orientation="h",
                     labels={"x": "Patients mutated", "y": "Gene"},
                     title="Most frequently mutated genes (filtered cohort)")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

        fig = px.histogram(filtered, x="PANEL_MUTATION_COUNT", nbins=15,
                            title="Panel mutation count per patient")
        st.plotly_chart(fig, use_container_width=True)

    with tab_survival:
        st.subheader("Survival Analytics")
        km_group_by = st.radio("Stratify Kaplan-Meier curve by", ["HIGH_RISK_FLAG", "STAGE_GROUP"], horizontal=True)

        surv_df = filtered.dropna(subset=["SURVIVAL_MONTHS", "EVENT_OCCURRED"])
        fig = px.line(title="Kaplan-Meier overall survival")
        groups = sorted(surv_df[km_group_by].dropna().unique())
        for g in groups:
            sub = surv_df[surv_df[km_group_by] == g]
            if len(sub) < 2:
                continue
            kmf = KaplanMeierFitter()
            kmf.fit(sub["SURVIVAL_MONTHS"], event_observed=sub["EVENT_OCCURRED"], label=str(g))
            km_curve = kmf.survival_function_.reset_index()
            fig.add_scatter(x=km_curve["timeline"], y=km_curve[str(g)], mode="lines", name=str(g))
        fig.update_layout(xaxis_title="Months", yaxis_title="Survival probability")
        st.plotly_chart(fig, use_container_width=True)

        if len(groups) == 2:
            a = surv_df[surv_df[km_group_by] == groups[0]]
            b = surv_df[surv_df[km_group_by] == groups[1]]
            if len(a) > 1 and len(b) > 1:
                result = logrank_test(a["SURVIVAL_MONTHS"], b["SURVIVAL_MONTHS"],
                                       event_observed_A=a["EVENT_OCCURRED"], event_observed_B=b["EVENT_OCCURRED"])
                st.info(f"Log-rank test p-value: {result.p_value:.4f}")


if __name__ == "__main__":
    main()
