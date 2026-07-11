"""Module 3: exploratory data analysis on the cleaned TCGA-LUAD cohort.

Generates and exports (to figures/ and reports/):
  - demographics (age distribution, stage distribution, gender ratio)
  - expression distribution for genes of interest
  - correlation heatmap across the gene panel
  - missing-value heatmap (pre-cleaning, raw clinical)
  - PCA of the expression matrix
  - boxplots of expression by stage for genes of interest
  - a summary statistics table
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
ROOT = CONFIG_PATH.parent.parent

sns.set_theme(style="whitegrid")
STAGE_ORDER = ["I", "II", "III", "IV"]


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def plot_demographics(clinical: pd.DataFrame, fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].hist(clinical["AGE"], bins=20, color="#4C72B0", edgecolor="white")
    axes[0].set_title("Age distribution")
    axes[0].set_xlabel("Age (years)")
    axes[0].set_ylabel("Patients")

    stage_counts = clinical["STAGE_GROUP"].value_counts().reindex(STAGE_ORDER)
    axes[1].bar(stage_counts.index, stage_counts.values, color="#55A868")
    axes[1].set_title("Tumor stage distribution")
    axes[1].set_xlabel("AJCC stage group")
    axes[1].set_ylabel("Patients")

    sex_counts = clinical["SEX"].value_counts()
    axes[2].pie(sex_counts.values, labels=sex_counts.index, autopct="%1.0f%%",
                colors=["#C44E52", "#4C72B0"])
    axes[2].set_title("Gender ratio")

    fig.tight_layout()
    fig.savefig(fig_dir / "demographics.png", dpi=150)
    plt.close(fig)


def plot_expression_distributions(expr: pd.DataFrame, genes: list[str], fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, len(genes), figsize=(6 * len(genes), 4.5))
    if len(genes) == 1:
        axes = [axes]
    for ax, gene in zip(axes, genes):
        sns.histplot(expr.loc[gene], kde=True, ax=ax, color="#4C72B0")
        ax.set_title(f"{gene} expression (log2 RSEM)")
        ax.set_xlabel("log2(RSEM + 1)")
    fig.tight_layout()
    fig.savefig(fig_dir / "expression_distribution.png", dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(expr: pd.DataFrame, fig_dir: Path) -> None:
    corr = expr.T.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, cmap="RdBu_r", center=0, square=True, ax=ax,
                cbar_kws={"label": "Pearson r"})
    ax.set_title("Gene-gene expression correlation (40-gene LUAD driver panel)")
    fig.tight_layout()
    fig.savefig(fig_dir / "correlation_heatmap.png", dpi=150)
    plt.close(fig)


def plot_missing_value_heatmap(clinical_raw: pd.DataFrame, cfg: dict, fig_dir: Path) -> None:
    cols = cfg["validation"]["required_clinical_columns"] + [
        "DFS_MONTHS", "DFS_STATUS", "ETHNICITY", "RACE", "SUBTYPE", "PATH_T_STAGE",
        "PATH_N_STAGE", "PATH_M_STAGE",
    ]
    cols = [c for c in dict.fromkeys(cols) if c in clinical_raw.columns]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(clinical_raw[cols].isna(), cbar=False, cmap=["#4C72B0", "#C44E52"], ax=ax)
    ax.set_title("Missing-value map — raw clinical data (red = missing)")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(fig_dir / "missing_value_heatmap.png", dpi=150)
    plt.close(fig)


def plot_pca(expr: pd.DataFrame, clinical: pd.DataFrame, fig_dir: Path) -> None:
    X = expr.T  # samples x genes
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)
    pca_df = pd.DataFrame(coords, columns=["PC1", "PC2"], index=expr.columns)
    pca_df = pca_df.merge(clinical.set_index("sampleId")[["STAGE_GROUP"]],
                           left_index=True, right_index=True, how="left")

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="STAGE_GROUP",
                     hue_order=STAGE_ORDER, palette="viridis", ax=ax, s=50)
    ax.set_title(
        f"PCA of expression panel "
        f"(PC1 {pca.explained_variance_ratio_[0]:.1%}, "
        f"PC2 {pca.explained_variance_ratio_[1]:.1%} variance explained)"
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "pca_expression.png", dpi=150)
    plt.close(fig)


def plot_boxplots_by_stage(expr: pd.DataFrame, clinical: pd.DataFrame, genes: list[str], fig_dir: Path) -> None:
    long_rows = []
    clin_idx = clinical.set_index("sampleId")
    for gene in genes:
        for sample_id, value in expr.loc[gene].items():
            if sample_id not in clin_idx.index:
                continue
            long_rows.append({
                "gene": gene,
                "expression": value,
                "STAGE_GROUP": clin_idx.loc[sample_id, "STAGE_GROUP"],
            })
    long_df = pd.DataFrame(long_rows)

    fig, axes = plt.subplots(1, len(genes), figsize=(6 * len(genes), 5))
    if len(genes) == 1:
        axes = [axes]
    for ax, gene in zip(axes, genes):
        sub = long_df[long_df["gene"] == gene]
        sns.boxplot(data=sub, x="STAGE_GROUP", y="expression", order=STAGE_ORDER, ax=ax,
                    hue="STAGE_GROUP", palette="Set2", legend=False)
        sns.stripplot(data=sub, x="STAGE_GROUP", y="expression", order=STAGE_ORDER, ax=ax, color="black", alpha=0.3, size=3)
        ax.set_title(f"{gene} expression by tumor stage")
        ax.set_xlabel("AJCC stage group")
        ax.set_ylabel("log2(RSEM + 1)")
    fig.tight_layout()
    fig.savefig(fig_dir / "expression_by_stage_boxplot.png", dpi=150)
    plt.close(fig)


def write_summary_stats(clinical: pd.DataFrame, expr: pd.DataFrame, reports_dir: Path) -> None:
    numeric_clin = clinical[["AGE", "OS_MONTHS", "MUTATION_COUNT"]].describe().T
    expr_stats = expr.T.describe().T[["mean", "std", "min", "max"]]
    expr_stats.columns = [f"expr_{c}" for c in expr_stats.columns]

    numeric_clin.to_csv(reports_dir / "summary_statistics_clinical.csv")
    expr_stats.to_csv(reports_dir / "summary_statistics_expression.csv")


def main() -> None:
    cfg = load_config()
    fig_dir = ROOT / cfg["paths"]["figures_dir"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = ROOT / cfg["paths"]["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)

    clinical = pd.read_csv(ROOT / cfg["paths"]["processed_files"]["clean_clinical"])
    expr = pd.read_csv(ROOT / cfg["paths"]["processed_files"]["clean_expression"], index_col=0)
    clinical_raw = pd.read_csv(ROOT / cfg["paths"]["raw_files"]["clinical"])

    genes_of_interest = cfg["genes_of_interest"]

    plot_demographics(clinical, fig_dir)
    plot_expression_distributions(expr, genes_of_interest, fig_dir)
    plot_correlation_heatmap(expr, fig_dir)
    plot_missing_value_heatmap(clinical_raw, cfg, fig_dir)
    plot_pca(expr, clinical, fig_dir)
    plot_boxplots_by_stage(expr, clinical, genes_of_interest, fig_dir)
    write_summary_stats(clinical, expr, reports_dir)

    print(f"Figures written to {fig_dir}")
    print(f"Summary statistics written to {reports_dir}")


if __name__ == "__main__":
    main()
