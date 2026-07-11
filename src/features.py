"""Module 4: feature engineering on the cleaned TCGA-LUAD cohort.

Every engineered feature is documented in reports/feature_dictionary.md
(name, definition, rationale) so it can be defended in an interview.
Output: data/processed/features.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
ROOT = CONFIG_PATH.parent.parent

STAGE_WEIGHT = {"I": 0.0, "II": 1 / 3, "III": 2 / 3, "IV": 1.0}


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()


def build_features(clinical: pd.DataFrame, expr: pd.DataFrame, mutations: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = clinical.copy()

    # --- Age bucket ---
    edges = cfg["features"]["age_bucket_edges"]
    labels = cfg["features"]["age_bucket_labels"]
    df["AGE_BUCKET"] = pd.cut(df["AGE"], bins=edges, labels=labels, right=False)

    # --- Expression quartile group for genes of interest ---
    for gene in cfg["genes_of_interest"]:
        gene_expr = expr.loc[gene]
        df[f"{gene}_EXPR"] = df["sampleId"].map(gene_expr)
        df[f"{gene}_EXPR_QUARTILE"] = pd.qcut(
            df[f"{gene}_EXPR"], q=4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"]
        )

    # --- Panel mutation count per patient ---
    panel_mut_count = mutations.groupby("patientId").size()
    df["PANEL_MUTATION_COUNT"] = df["patientId"].map(panel_mut_count).fillna(0).astype(int)

    # --- Survival time + binary event ---
    df["SURVIVAL_MONTHS"] = df["OS_MONTHS"]
    df["EVENT_OCCURRED"] = df["OS_STATUS"].str.startswith("1").astype(int)  # 1 = deceased, 0 = censored

    # --- Composite risk score ---
    w = cfg["features"]["risk_score"]["weights"]
    z_mut = zscore(df["PANEL_MUTATION_COUNT"])
    z_age = zscore(df["AGE"])
    stage_w = df["STAGE_GROUP"].map(STAGE_WEIGHT)
    df["RISK_SCORE"] = w["mutation_count"] * z_mut + w["age"] * z_age + w["stage"] * stage_w

    # --- High-risk flag ---
    threshold = df["RISK_SCORE"].quantile(cfg["features"]["high_risk_threshold_percentile"])
    df["HIGH_RISK_FLAG"] = (df["RISK_SCORE"] >= threshold).astype(int)

    return df


FEATURE_DOC = """# OncoInsights — Engineered Feature Dictionary

| Feature | Definition | Rationale |
|---|---|---|
| `AGE_BUCKET` | Age binned into `<50`, `50-59`, `60-69`, `70+` | Coarser age groups are easier to stratify and plot than raw continuous age, and match common clinical reporting conventions. |
| `{gene}_EXPR` | log2(RSEM + 1) expression value for {gene} (EGFR, KRAS) | Carried over from the cleaned expression matrix for the two most clinically actionable LUAD driver genes (both have approved/investigational targeted therapies). |
| `{gene}_EXPR_QUARTILE` | Patient's expression for {gene}, binned into within-cohort quartiles (Q1 low - Q4 high) | Converts a continuous value into a categorical group for boxplots, cohort filters, and readable "high vs low expressers" comparisons. |
| `PANEL_MUTATION_COUNT` | Count of mutation calls in `mutations.csv` (40-gene driver panel) per patient, 0 if none | Engineered directly from the mutation data pulled for this project (distinct from the pre-existing genome-wide `MUTATION_COUNT` field from cBioPortal) — reflects driver-gene mutation burden specifically. |
| `SURVIVAL_MONTHS` | `OS_MONTHS` renamed for clarity | Time-to-event/censoring in months, required for Kaplan-Meier estimation. |
| `EVENT_OCCURRED` | Binary: 1 if `OS_STATUS` = "1:DECEASED", 0 if "0:LIVING" (censored) | Standard binary event indicator for survival analysis (lifelines/KaplanMeierFitter expects this exact encoding). |
| `RISK_SCORE` | `0.4 * z(PANEL_MUTATION_COUNT) + 0.3 * z(AGE) + 0.3 * stage_weight` where `stage_weight` maps stage I/II/III/IV to 0 / 0.33 / 0.67 / 1.0 | A simple, fully transparent composite score combining three independent risk signals (mutation burden, age, disease stage) into one number for cohort ranking. Weights are a judgment call (mutation burden weighted highest as the most direct biological signal) documented here, not learned from data — this is intentionally a scoring rule, not a black-box model, so every value is traceable to its inputs. |
| `HIGH_RISK_FLAG` | 1 if `RISK_SCORE` is in the top 25% of the cohort, else 0 | Threshold-based binary flag for dashboard filtering and the SQL "high-risk cohort" view; 75th percentile chosen to flag roughly a quarter of patients as a manageable "watch list" size. |
"""


def main() -> None:
    cfg = load_config()

    clinical = pd.read_csv(ROOT / cfg["paths"]["processed_files"]["clean_clinical"])
    expr = pd.read_csv(ROOT / cfg["paths"]["processed_files"]["clean_expression"], index_col=0)
    mutations = pd.read_csv(ROOT / cfg["paths"]["raw_files"]["mutations"])

    features = build_features(clinical, expr, mutations, cfg)

    out_path = ROOT / cfg["paths"]["processed_files"]["features"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out_path, index=False)

    doc_path = ROOT / cfg["paths"]["reports_dir"] / "feature_dictionary.md"
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(FEATURE_DOC, encoding="utf-8")

    print(f"Wrote {out_path} ({features.shape[0]} rows, {features.shape[1]} columns)")
    print(f"Wrote {doc_path}")
    print("\nRisk score summary:")
    print(features["RISK_SCORE"].describe())
    print(f"\nHigh-risk patients: {features['HIGH_RISK_FLAG'].sum()} / {len(features)}")


if __name__ == "__main__":
    main()
