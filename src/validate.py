"""Module 1: load and validate the raw TCGA-LUAD extracts.

Simulates the first step of a real analyst workflow: data has already been
handed off (here, pulled from cBioPortal) and the job starts at validation,
not acquisition. Performs schema checks, row/column counts, and a
missing-value summary for each raw file, and logs everything to
data/acquisition_log.txt.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _log(lines: list[str], msg: str) -> None:
    lines.append(msg)
    print(msg)


def validate_clinical(df: pd.DataFrame, cfg: dict, log: list[str]) -> None:
    _log(log, f"\n[clinical.csv] shape = {df.shape}")
    required = cfg["validation"]["required_clinical_columns"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        _log(log, f"  SCHEMA ERROR: missing required columns: {missing_cols}")
    else:
        _log(log, f"  Schema OK: all {len(required)} required columns present.")

    n = len(df)
    if n < cfg["validation"]["min_expected_samples"]:
        _log(log, f"  WARNING: only {n} rows, below expected minimum "
                   f"{cfg['validation']['min_expected_samples']}.")
    else:
        _log(log, f"  Row count OK: {n} samples.")

    dupe_patients = df["patientId"].duplicated().sum() if "patientId" in df else None
    _log(log, f"  Duplicate patientId rows: {dupe_patients}")

    threshold = cfg["validation"]["max_missing_fraction_clinical"]
    miss = df.isna().mean().sort_values(ascending=False)
    flagged = miss[miss > threshold]
    _log(log, f"  Missing-value summary (top 15 columns by null fraction):")
    for col, frac in miss.head(15).items():
        flag = " <-- FLAGGED (>{:.0%} missing)".format(threshold) if col in flagged.index else ""
        _log(log, f"    {col:35s} {frac:6.1%}{flag}")


def validate_mutations(df: pd.DataFrame, cfg: dict, log: list[str]) -> None:
    _log(log, f"\n[mutations.csv] shape = {df.shape}")
    required = ["patientId", "sampleId", "hugoGeneSymbol", "mutationType"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        _log(log, f"  SCHEMA ERROR: missing required columns: {missing_cols}")
    else:
        _log(log, f"  Schema OK: all required columns present.")

    genes = set(df["hugoGeneSymbol"].unique())
    panel = set(cfg["gene_panel"])
    off_panel = genes - panel
    _log(log, f"  Distinct genes mutated: {len(genes)} (panel size: {len(panel)})")
    if off_panel:
        _log(log, f"  WARNING: {len(off_panel)} gene symbol(s) outside configured panel: {sorted(off_panel)}")

    _log(log, f"  Rows with null proteinChange: {df['proteinChange'].isna().sum()}")
    _log(log, f"  Mutation type breakdown:")
    for mtype, count in df["mutationType"].value_counts().items():
        _log(log, f"    {mtype:25s} {count}")


def validate_expression(df: pd.DataFrame, cfg: dict, log: list[str]) -> None:
    _log(log, f"\n[expression_matrix.csv] shape = {df.shape} (genes x samples)")
    panel = set(cfg["gene_panel"])
    genes = set(df.iloc[:, 0]) if df.columns[0].lower() != "hugogenesymbol" else set(df["hugoGeneSymbol"])
    missing_genes = panel - genes
    if missing_genes:
        _log(log, f"  WARNING: {len(missing_genes)} panel gene(s) absent from expression matrix: {sorted(missing_genes)}")
    else:
        _log(log, f"  Schema OK: all {len(panel)} panel genes present.")

    numeric = df.select_dtypes(include="number")
    n_missing = numeric.isna().sum().sum()
    total = numeric.size
    _log(log, f"  Missing expression values: {n_missing} / {total} ({n_missing / total:.2%})")
    _log(log, f"  Value range: min={numeric.min().min():.2f}, max={numeric.max().max():.2f}")


def validate_samples(df: pd.DataFrame, log: list[str]) -> None:
    _log(log, f"\n[samples.csv] shape = {df.shape}")
    _log(log, f"  Unique patients: {df['patientId'].nunique()}")
    _log(log, f"  Sample type breakdown: {dict(df['sampleType'].value_counts())}")


def main() -> None:
    cfg = load_config()
    root = CONFIG_PATH.parent.parent
    log: list[str] = [f"OncoInsights acquisition/validation log - {dt.datetime.now().isoformat(timespec='seconds')}"]
    log.append(f"Cohort: {cfg['project']['cohort']}")

    clinical = pd.read_csv(root / cfg["paths"]["raw_files"]["clinical"])
    mutations = pd.read_csv(root / cfg["paths"]["raw_files"]["mutations"])
    expression = pd.read_csv(root / cfg["paths"]["raw_files"]["expression"])
    samples = pd.read_csv(root / cfg["paths"]["raw_files"]["samples"])

    validate_clinical(clinical, cfg, log)
    validate_mutations(mutations, cfg, log)
    validate_expression(expression, cfg, log)
    validate_samples(samples, log)

    log_path = root / cfg["paths"]["log_file"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log) + "\n")
    print(f"\nLog written to {log_path}")


if __name__ == "__main__":
    main()
