"""Module 2: clean the validated raw extracts.

Missing-value strategy is documented per column (see config.yaml ->
cleaning) rather than blanket-imputed. Outputs clean_clinical.csv,
clean_expression.csv, and a quality_report.html summarizing every
transformation applied.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
ROOT = CONFIG_PATH.parent.parent

CORE_CLINICAL_FIELDS = ["AGE", "SEX", "AJCC_PATHOLOGIC_TUMOR_STAGE", "OS_MONTHS", "OS_STATUS"]


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def clean_clinical(df: pd.DataFrame, cfg: dict, notes: list[str]) -> pd.DataFrame:
    n0 = len(df)

    # 1. Duplicates: one row per patientId expected.
    n_dupe = df["patientId"].duplicated().sum()
    df = df.drop_duplicates(subset="patientId", keep="first")
    notes.append(f"Removed {n_dupe} duplicate patientId row(s).")

    # 2. Drop patients with zero usable clinical data (missing every core
    #    field: AGE, SEX, stage, and both survival fields). These are blank
    #    follow-up forms, not analyzable, and imputing them would fabricate
    #    an entire patient record.
    missing_all_core = df[CORE_CLINICAL_FIELDS].isna().all(axis=1)
    n_blank = missing_all_core.sum()
    df = df[~missing_all_core].copy()
    notes.append(
        f"Dropped {n_blank} patient(s) with no clinical follow-up data at all "
        f"(all of AGE/SEX/stage/OS_MONTHS/OS_STATUS missing)."
    )

    # 3. AGE: median-impute remaining nulls (~{pct}% of survivors), per
    #    config `age_missing_strategy: median_impute`. Age is a covariate,
    #    not an outcome, so imputation doesn't bias downstream survival
    #    analysis the way imputing OS_MONTHS would.
    n_age_missing = df["AGE"].isna().sum()
    age_median = df["AGE"].median()
    df["AGE"] = df["AGE"].fillna(age_median)
    notes.append(f"Median-imputed {n_age_missing} missing AGE value(s) with median={age_median:.0f}.")

    # 4. Stage: required for stratified analysis (boxplots by stage, SQL
    #    cohort views) — rows without it are dropped rather than imputed,
    #    since there's no defensible way to guess a tumor stage.
    n_stage_missing = df["AJCC_PATHOLOGIC_TUMOR_STAGE"].isna().sum()
    df = df[df["AJCC_PATHOLOGIC_TUMOR_STAGE"].notna()].copy()
    notes.append(f"Dropped {n_stage_missing} row(s) missing AJCC_PATHOLOGIC_TUMOR_STAGE (required for stratification).")

    # 5. Survival: OS_MONTHS/OS_STATUS are required for Kaplan-Meier and
    #    log-rank in Module 6 — can't impute a survival time, so drop.
    n_os_missing = df["OS_MONTHS"].isna().sum()
    df = df[df["OS_MONTHS"].notna()].copy()
    notes.append(f"Dropped {n_os_missing} row(s) missing OS_MONTHS (required for survival analysis).")

    # 6. Normalize stage to a coarse group (I/II/III/IV) for readable charts.
    def stage_group(s: str) -> str:
        if pd.isna(s):
            return np.nan
        s = s.replace("STAGE ", "")
        for grp in ("IV", "III", "II", "I"):
            if s.startswith(grp):
                return grp
        return np.nan

    df["STAGE_GROUP"] = df["AJCC_PATHOLOGIC_TUMOR_STAGE"].apply(stage_group)

    # 7. Outlier flagging (not removal): AGE beyond 3 SD from mean.
    age_mean, age_std = df["AGE"].mean(), df["AGE"].std()
    df["AGE_OUTLIER_FLAG"] = (df["AGE"] - age_mean).abs() > 3 * age_std
    n_age_outliers = df["AGE_OUTLIER_FLAG"].sum()
    notes.append(f"Flagged {n_age_outliers} AGE outlier(s) (>3 SD from mean) - kept, not dropped.")

    notes.append(f"clean_clinical: {n0} raw rows -> {len(df)} clean rows.")
    return df


def clean_expression(expr: pd.DataFrame, clinical_clean: pd.DataFrame, cfg: dict, notes: list[str]) -> pd.DataFrame:
    expr = expr.set_index(expr.columns[0])
    expr.index.name = "hugoGeneSymbol"
    n_genes0 = expr.shape[0]

    # Keep only samples that survived clinical cleaning.
    valid_samples = set(clinical_clean["sampleId"])
    kept_cols = [c for c in expr.columns if c in valid_samples]
    expr = expr[kept_cols]
    notes.append(f"Restricted expression matrix to {len(kept_cols)} samples with clean clinical records.")

    # log2(x + 1) transform — RSEM counts are heavily right-skewed.
    expr_log = np.log2(expr.astype(float) + 1)

    # Near-zero-variance gene filtering.
    variances = expr_log.var(axis=1)
    threshold = cfg["cleaning"]["near_zero_variance_threshold"]
    low_var_genes = variances[variances < threshold].index.tolist()
    expr_log = expr_log.drop(index=low_var_genes)
    notes.append(
        f"Dropped {len(low_var_genes)} near-zero-variance gene(s) "
        f"(log2 variance < {threshold}): {low_var_genes}"
    )
    notes.append(f"clean_expression: {n_genes0} raw genes -> {expr_log.shape[0]} clean genes, {expr_log.shape[1]} samples.")
    return expr_log


def write_quality_report(notes: list[str], clinical: pd.DataFrame, expr: pd.DataFrame, out_path: Path) -> None:
    rows = "".join(f"<li>{n}</li>" for n in notes)
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>OncoInsights Quality Report</title>
<style>
body {{ font-family: -apple-system, Segoe UI, sans-serif; max-width: 800px; margin: 2rem auto; line-height: 1.5; }}
h1 {{ font-size: 1.4rem; }}
table {{ border-collapse: collapse; margin-top: 1rem; }}
td, th {{ border: 1px solid #ccc; padding: 4px 10px; text-align: left; }}
</style></head>
<body>
<h1>OncoInsights — Data Quality Report</h1>
<p>Generated {dt.datetime.now().isoformat(timespec='seconds')}</p>
<h2>Cleaning steps applied</h2>
<ol>{rows}</ol>
<h2>Final dataset shape</h2>
<table>
<tr><th>Table</th><th>Rows</th><th>Columns</th></tr>
<tr><td>clean_clinical.csv</td><td>{clinical.shape[0]}</td><td>{clinical.shape[1]}</td></tr>
<tr><td>clean_expression.csv</td><td>{expr.shape[0]} genes</td><td>{expr.shape[1]} samples</td></tr>
</table>
</body></html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")


def main() -> None:
    cfg = load_config()
    notes: list[str] = []

    clinical_raw = pd.read_csv(ROOT / cfg["paths"]["raw_files"]["clinical"])
    expr_raw = pd.read_csv(ROOT / cfg["paths"]["raw_files"]["expression"])

    clinical_clean = clean_clinical(clinical_raw, cfg, notes)
    expr_clean = clean_expression(expr_raw, clinical_clean, cfg, notes)

    processed_dir = ROOT / cfg["paths"]["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    clinical_out = ROOT / cfg["paths"]["processed_files"]["clean_clinical"]
    expr_out = ROOT / cfg["paths"]["processed_files"]["clean_expression"]
    clinical_clean.to_csv(clinical_out, index=False)
    expr_clean.to_csv(expr_out)

    report_path = ROOT / cfg["paths"]["processed_files"]["quality_report"]
    write_quality_report(notes, clinical_clean, expr_clean, report_path)

    print("\n".join(notes))
    print(f"\nWrote {clinical_out}")
    print(f"Wrote {expr_out}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
