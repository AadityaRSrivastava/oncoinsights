"""Module 6: statistical analysis on the cleaned/engineered TCGA-LUAD cohort.

Methods (each chosen because it's explainable end-to-end, not because it's
exhaustive):
  - Welch's t-test: gene expression, mutated vs wild-type, for 8 driver genes
    (+ Benjamini-Hochberg FDR correction since we ran 8 tests)
  - Chi-square test of independence: tumor stage (early I/II vs late III/IV)
    vs TP53 mutation presence
  - Pearson correlation: patient age vs panel mutation count
  - Kaplan-Meier + log-rank test: overall survival, high-risk vs low-risk
    (per the engineered RISK_SCORE / HIGH_RISK_FLAG)

Every result gets a one-paragraph plain-English interpretation, printed and
written to reports/statistical_results.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from statsmodels.stats.multitest import multipletests
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
ROOT = CONFIG_PATH.parent.parent

DRIVER_GENES_FOR_TTEST = ["TP53", "KRAS", "EGFR", "STK11", "KEAP1", "SMARCA4", "ATM", "NF1"]


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def welchs_ttests(features: pd.DataFrame, expr: pd.DataFrame, mutations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gene in DRIVER_GENES_FOR_TTEST:
        mutated_patients = set(mutations.loc[mutations["hugoGeneSymbol"] == gene, "patientId"])
        sample_to_patient = features.set_index("sampleId")["patientId"]
        gene_expr = expr.loc[gene]

        mutant_vals, wt_vals = [], []
        for sample_id, value in gene_expr.items():
            if sample_id not in sample_to_patient.index:
                continue
            patient_id = sample_to_patient.loc[sample_id]
            (mutant_vals if patient_id in mutated_patients else wt_vals).append(value)

        t_stat, p_val = stats.ttest_ind(mutant_vals, wt_vals, equal_var=False)  # Welch's t-test
        rows.append({
            "gene": gene,
            "n_mutant": len(mutant_vals),
            "n_wildtype": len(wt_vals),
            "mean_expr_mutant": np.mean(mutant_vals),
            "mean_expr_wildtype": np.mean(wt_vals),
            "t_stat": t_stat,
            "p_value": p_val,
        })

    result = pd.DataFrame(rows)
    # Benjamini-Hochberg FDR correction across the 8 tests.
    reject, p_adj, _, _ = multipletests(result["p_value"], alpha=0.05, method="fdr_bh")
    result["p_value_fdr_bh"] = p_adj
    result["significant_fdr_0.05"] = reject
    return result


def chi_square_stage_vs_tp53(features: pd.DataFrame, mutations: pd.DataFrame) -> dict:
    tp53_patients = set(mutations.loc[mutations["hugoGeneSymbol"] == "TP53", "patientId"])
    df = features.copy()
    df["STAGE_BINARY"] = df["STAGE_GROUP"].map({"I": "Early (I/II)", "II": "Early (I/II)",
                                                  "III": "Late (III/IV)", "IV": "Late (III/IV)"})
    df["TP53_STATUS"] = df["patientId"].apply(lambda p: "Mutant" if p in tp53_patients else "Wild-type")

    contingency = pd.crosstab(df["STAGE_BINARY"], df["TP53_STATUS"])
    chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
    return {"contingency": contingency, "chi2": chi2, "p_value": p_val, "dof": dof}


def correlation_age_vs_mutation_count(features: pd.DataFrame) -> dict:
    r, p_val = stats.pearsonr(features["AGE"], features["PANEL_MUTATION_COUNT"])
    return {"r": r, "p_value": p_val, "n": len(features)}


def kaplan_meier_high_risk(features: pd.DataFrame, fig_dir: Path) -> dict:
    high = features[features["HIGH_RISK_FLAG"] == 1]
    low = features[features["HIGH_RISK_FLAG"] == 0]

    kmf_high = KaplanMeierFitter()
    kmf_low = KaplanMeierFitter()

    fig, ax = plt.subplots(figsize=(8, 6))
    kmf_high.fit(high["SURVIVAL_MONTHS"], event_observed=high["EVENT_OCCURRED"], label="High risk")
    kmf_high.plot_survival_function(ax=ax, color="#C44E52")
    kmf_low.fit(low["SURVIVAL_MONTHS"], event_observed=low["EVENT_OCCURRED"], label="Low risk")
    kmf_low.plot_survival_function(ax=ax, color="#4C72B0")
    ax.set_title("Overall survival — high-risk vs low-risk (RISK_SCORE top quartile)")
    ax.set_xlabel("Months")
    ax.set_ylabel("Survival probability")
    fig.tight_layout()
    fig.savefig(fig_dir / "kaplan_meier_risk_group.png", dpi=150)
    plt.close(fig)

    result = logrank_test(
        high["SURVIVAL_MONTHS"], low["SURVIVAL_MONTHS"],
        event_observed_A=high["EVENT_OCCURRED"], event_observed_B=low["EVENT_OCCURRED"],
    )
    return {
        "median_survival_high": kmf_high.median_survival_time_,
        "median_survival_low": kmf_low.median_survival_time_,
        "logrank_p_value": result.p_value,
        "n_high": len(high),
        "n_low": len(low),
    }


def write_report(ttest_df, chi2_result, corr_result, km_result, out_path: Path) -> None:
    lines = ["# OncoInsights — Statistical Analysis Results\n"]

    lines.append("## 1. Welch's t-test — gene expression by mutation status\n")
    lines.append(
        "**Why Welch's, not Student's:** mutant and wild-type groups have very different sample "
        "sizes and no reason to assume equal variance, so Welch's t-test (which doesn't assume "
        "equal variance) is the safer default.\n"
    )
    lines.append(ttest_df.round(4).to_markdown(index=False))
    lines.append("\n**Interpretation:** ")
    sig_genes = ttest_df.loc[ttest_df["significant_fdr_0.05"], "gene"].tolist()
    if sig_genes:
        lines.append(
            f"After Benjamini-Hochberg FDR correction across all 8 genes tested, "
            f"{', '.join(sig_genes)} show a statistically significant difference in expression "
            f"between mutated and wild-type patients (FDR-adjusted p < 0.05). This is a sanity "
            f"check as much as a finding: a truncating/damaging mutation often does shift a gene's "
            f"own expression (e.g. nonsense-mediated decay, altered transcriptional feedback), so "
            f"significant genes here validate that the mutation calls are behaviorally consistent "
            f"with the expression data.\n"
        )
    else:
        lines.append(
            "None of the 8 genes remain significant after FDR correction, meaning any raw p<0.05 "
            "hits were likely false positives given the number of tests run.\n"
        )

    lines.append("\n## 2. Chi-square test — tumor stage vs TP53 mutation status\n")
    lines.append(chi2_result["contingency"].to_markdown())
    lines.append(
        f"\n\nChi2 = {chi2_result['chi2']:.3f}, dof = {chi2_result['dof']}, "
        f"p = {chi2_result['p_value']:.4f}\n"
    )
    verdict = "is" if chi2_result["p_value"] < 0.05 else "is not"
    lines.append(
        f"**Interpretation:** TP53 mutation status {verdict} significantly associated with "
        f"early- vs late-stage disease at diagnosis (p {'<' if chi2_result['p_value']<0.05 else '>='} 0.05). "
        f"TP53 is the most frequently mutated gene in this cohort (52% of patients), so this test "
        f"asks whether losing TP53 function tracks with more advanced disease at the time of "
        f"diagnosis, or whether it's roughly evenly distributed across stages.\n"
    )

    lines.append("\n## 3. Pearson correlation — age vs panel mutation count\n")
    lines.append(
        f"r = {corr_result['r']:.3f}, p = {corr_result['p_value']:.4f}, n = {corr_result['n']}\n\n"
    )
    strength = "weak" if abs(corr_result["r"]) < 0.3 else ("moderate" if abs(corr_result["r"]) < 0.6 else "strong")
    lines.append(
        f"**Interpretation:** There is a {strength} {'positive' if corr_result['r']>0 else 'negative'} "
        f"correlation (r={corr_result['r']:.3f}) between patient age and driver-panel mutation count. "
        f"{'This is consistent with the biological expectation that mutation burden accumulates with age.' if corr_result['r']>0 else 'This runs counter to the usual expectation that mutation burden rises with age, and would be worth flagging for follow-up.'}\n"
    )

    lines.append("\n## 4. Kaplan-Meier + log-rank test — high-risk vs low-risk survival\n")
    lines.append(
        f"n(high-risk) = {km_result['n_high']}, n(low-risk) = {km_result['n_low']}\n\n"
        f"Median survival, high-risk group: {km_result['median_survival_high']:.1f} months\n\n"
        f"Median survival, low-risk group: {km_result['median_survival_low']:.1f} months\n\n"
        f"Log-rank test p-value: {km_result['logrank_p_value']:.6f}\n\n"
    )
    verdict = "does" if km_result["logrank_p_value"] < 0.05 else "does not"
    lines.append(
        f"**Interpretation:** The engineered `RISK_SCORE` (built from mutation burden, age, and "
        f"stage) {verdict} produce a statistically significant separation in overall survival "
        f"between the top-quartile high-risk group and everyone else (log-rank p "
        f"{'<' if km_result['logrank_p_value']<0.05 else '>='} 0.05). "
        f"{'This validates the risk score as a meaningful stratification, not just an arbitrary composite.' if km_result['logrank_p_value']<0.05 else 'This suggests the composite score, as weighted, is not yet capturing the dominant driver of survival differences in this cohort — a useful, honest negative result.'}\n"
    )

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    cfg = load_config()

    features = pd.read_csv(ROOT / cfg["paths"]["processed_files"]["features"])
    expr = pd.read_csv(ROOT / cfg["paths"]["processed_files"]["clean_expression"], index_col=0)
    mutations = pd.read_csv(ROOT / cfg["paths"]["raw_files"]["mutations"])
    mutations = mutations[mutations["patientId"].isin(features["patientId"])]

    fig_dir = ROOT / cfg["paths"]["figures_dir"]
    fig_dir.mkdir(parents=True, exist_ok=True)

    ttest_df = welchs_ttests(features, expr, mutations)
    chi2_result = chi_square_stage_vs_tp53(features, mutations)
    corr_result = correlation_age_vs_mutation_count(features)
    km_result = kaplan_meier_high_risk(features, fig_dir)

    out_path = ROOT / cfg["paths"]["reports_dir"] / "statistical_results.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_report(ttest_df, chi2_result, corr_result, km_result, out_path)

    print(ttest_df.round(4))
    print(f"\nChi-square: chi2={chi2_result['chi2']:.3f}, p={chi2_result['p_value']:.4f}")
    print(f"Correlation: r={corr_result['r']:.3f}, p={corr_result['p_value']:.4f}")
    print(f"Log-rank: p={km_result['logrank_p_value']:.6f}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
