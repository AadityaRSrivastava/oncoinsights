"""Module 5 (part 1): load cleaned/engineered data into SQLite for querying.

Creates three tables:
  - patients: one row per patient, clinical + engineered features
  - mutations: one row per mutation call (40-gene driver panel)
  - expression: long-format (patientId, sampleId, gene, log2_expression)
"""

from __future__ import annotations
import sqlite3
from pathlib import Path

import pandas as pd
import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
ROOT = CONFIG_PATH.parent.parent


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def main() -> None:
    cfg = load_config()

    features = pd.read_csv(ROOT / cfg["paths"]["processed_files"]["features"])
    expr = pd.read_csv(ROOT / cfg["paths"]["processed_files"]["clean_expression"], index_col=0)
    mutations = pd.read_csv(ROOT / cfg["paths"]["raw_files"]["mutations"])
    mutations = mutations[mutations["patientId"].isin(features["patientId"])].copy()

    sample_to_patient = features.set_index("sampleId")["patientId"]
    expr_long = expr.T.reset_index().rename(columns={"index": "sampleId"})
    expr_long = expr_long.melt(id_vars="sampleId", var_name="hugoGeneSymbol", value_name="log2_expression")
    expr_long["patientId"] = expr_long["sampleId"].map(sample_to_patient)
    expr_long = expr_long.dropna(subset=["patientId"])

    db_path = ROOT / cfg["paths"]["processed_files"]["sqlite_db"]
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    features.to_sql("patients", conn, index=False)
    mutations.to_sql("mutations", conn, index=False)
    expr_long.to_sql("expression", conn, index=False)
    conn.execute("CREATE INDEX idx_patients_id ON patients(patientId)")
    conn.execute("CREATE INDEX idx_mutations_patient ON mutations(patientId)")
    conn.execute("CREATE INDEX idx_mutations_gene ON mutations(hugoGeneSymbol)")
    conn.execute("CREATE INDEX idx_expression_patient ON expression(patientId)")
    conn.execute("CREATE INDEX idx_expression_gene ON expression(hugoGeneSymbol)")
    conn.commit()

    print(f"patients: {len(features)} rows")
    print(f"mutations: {len(mutations)} rows")
    print(f"expression: {len(expr_long)} rows")
    print(f"Wrote {db_path}")
    conn.close()


if __name__ == "__main__":
    main()
