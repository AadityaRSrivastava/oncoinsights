# OncoInsights — Analytics Platform for Clinical & Genomic Decision Intelligence

A reproducible analytics pipeline that turns raw genomic and clinical data into actionable
insights through statistical analysis, an interactive dashboard, and an executive summary.

Built as a portfolio project demonstrating the analytics skillset expected from a Data Analyst /
Analytics Engineer role — SQL, statistical testing, data cleaning judgment calls, dashboarding,
and business-framed communication — using a real TCGA cancer genomics cohort as the data source.

## Problem

Clinical teams reviewing a cancer cohort need a fast way to answer three questions: who is
highest-risk, which mutations are driving that risk, and whether a simple composite risk score
actually separates patients by outcome. OncoInsights builds that answer end to end: raw data →
validated → cleaned → engineered features → SQL cohort views → statistical tests → dashboard →
executive summary.

## Dataset

**TCGA Lung Adenocarcinoma (LUAD), PanCancer Atlas** (`luad_tcga_pan_can_atlas_2018`), pulled live
from the [cBioPortal REST API](https://www.cbioportal.org/api) — real, published TCGA data, not
synthetic.

- **566 samples / ~500 patients** pulled; **503 patients** survive data cleaning
- **Clinical**: age, sex, AJCC stage, overall survival time + status, disease-free survival,
  mutation count, TMB, MSI scores (59 fields)
- **Mutations**: 2,817 calls across a **curated 40-gene LUAD driver panel** (EGFR, KRAS, TP53,
  STK11, KEAP1, ALK, ROS1, MET, BRAF, and 31 others — see `config/config.yaml`)
- **Expression**: RNA-seq (RSEM), same 40-gene panel, log2-transformed

**Why a curated panel, not the full transcriptome/exome:** an unfiltered pull returned a
157,142-row, 122MB mutation set and a 20,000-gene expression matrix — technically "more data,"
but impossible to defend gene-by-gene in an interview and mostly passenger noise. Filtering to 40
established LUAD driver genes (TCGA marker paper + COSMIC Cancer Gene Census) keeps every chart
and statistical test traceable to a specific, explainable biological question.

**How it was sourced/validated:** `src/validate.py` (Module 1) treats this exactly like an
analyst workflow where data engineering already handed off files — it runs schema checks, row/
column counts, and a missing-value summary against `data/raw/*.csv`, logged to
`data/acquisition_log.txt`.

## Architecture

```
OncoInsights/
├── config/config.yaml       # every path, threshold, and gene list — nothing hardcoded
├── data/
│   ├── raw/                 # clinical.csv, mutations.csv, expression_matrix.csv, samples.csv
│   ├── processed/           # clean_clinical.csv, clean_expression.csv, features.csv, oncoinsights.db
│   └── acquisition_log.txt  # Module 1 output
├── src/                     # one script per pipeline module (see below)
├── notebooks/                # 05_sql_analytics.ipynb — runs & explains every SQL query
├── sql/queries.sql          # 12 queries: aggregation, window functions, CTEs, a reusable view
├── dashboard/app.py         # Streamlit KPI dashboard
├── reports/                 # quality_report.html, statistical_results.md, feature_dictionary.md,
│                             # executive_summary.md + .pdf
├── figures/                 # exported EDA + statistical plots (PNG)
└── run.py                   # one command, runs the full pipeline end to end
```

## Pipeline (Modules 1–9)

| # | Module | Script | Output |
|---|---|---|---|
| 1 | Validate | `src/validate.py` | schema/row/null checks → `data/acquisition_log.txt` |
| 2 | Clean | `src/clean.py` | documented per-column missing-value strategy → `clean_clinical.csv`, `clean_expression.csv`, `quality_report.html` |
| 3 | EDA | `src/eda.py` | 6 figures (demographics, expression distributions, correlation heatmap, missing-value map, PCA, stage boxplots) + summary stats |
| 4 | Feature engineering | `src/features.py` | age buckets, expression quartiles, mutation count, composite `RISK_SCORE`, survival fields, `HIGH_RISK_FLAG` → `features.csv` + `feature_dictionary.md` |
| 5 | SQL analytics | `src/build_db.py` + `sql/queries.sql` | SQLite DB + 12 queries (aggregation, window functions, multi-step CTEs, a reusable `high_risk_cohort` view) + executed notebook |
| 6 | Statistics | `src/stats_analysis.py` | Welch's t-tests (FDR-corrected), chi-square, Pearson correlation, Kaplan-Meier + log-rank → `statistical_results.md` |
| 7 | Dashboard | `dashboard/app.py` | 5-tab Streamlit app with cross-cutting filters |
| 8 | Executive summary | `src/executive_summary.py` + `src/md_to_pdf.py` | `executive_summary.md` + `.pdf`, all figures computed live from the data, not hardcoded |
| 9 | Reproducibility | `run.py` | one command runs Modules 1–6 + 8 end to end with per-stage logging |

Each stage was built and reviewed independently before moving to the next, per a stage-gated
build process — Module 1+2 → Module 3+4 → Module 5+6 → Module 7 → Module 8+9.

## Dashboard

Run `streamlit run dashboard/app.py` and open `http://localhost:8501`. Five tabs — Overview,
Clinical & Demographic, Expression, Mutation, Survival — share one set of sidebar filters (age
range, gender, stage, mutation status by gene, expression gene selector). All charts are
interactive (Plotly). The Overview tab's KPIs and the Survival tab's log-rank p-value are computed
the exact same way as the standalone `stats_analysis.py` output — verified to match exactly
(p=0.0021 in both), so the dashboard and the offline analysis are one consistent source of truth,
not two diverging code paths.

Sample static figures (from `figures/`, generated by `src/eda.py` and `src/stats_analysis.py`):

- `figures/demographics.png` — age distribution, stage distribution, gender ratio
- `figures/pca_expression.png` — PCA of the 40-gene expression panel
- `figures/kaplan_meier_risk_group.png` — the risk-score-stratified survival curves

## Key Insights

1. **The engineered risk score meaningfully separates survival outcomes** — high-risk patients
   (top quartile of `RISK_SCORE`) show significantly worse overall survival (log-rank p=0.0021;
   median survival 38.5 vs 53.7 months). Not a foregone conclusion for a hand-weighted composite.
2. **TP53 is the dominant mutation (52% of patients)**, matching published TCGA-LUAD rates — a
   useful sanity check on the data pull itself.
3. **7 of 8 tested driver genes show a mutation-linked expression shift** (FDR-corrected Welch's
   t-test) — internal consistency between the mutation calls and expression data.
4. **TP53 mutation status is *not* associated with tumor stage** (chi-square p=0.55) — an early,
   stage-independent driver event rather than a late-stage marker.
5. **Mutation burden has a weak, unexpected *negative* correlation with age** (r=-0.165,
   p=0.0002) — flagged as counterintuitive rather than smoothed over.
6. **EGFR expression level alone is a weak prognostic signal** — mortality is roughly flat across
   EGFR expression quartiles, meaning it shouldn't be over-interpreted without mutation status and
   stage.

Full detail with plain-English interpretation for every test: `reports/statistical_results.md`
and `reports/executive_summary.pdf`.

## Tech Stack

Python 3.14 · pandas / numpy · scipy / statsmodels (FDR correction) · lifelines (Kaplan-Meier,
log-rank) · scikit-learn (PCA) · matplotlib / seaborn · SQLite · Streamlit + Plotly · reportlab
(PDF generation) · PyYAML (config)

## Installation

```bash
pip install -r requirements.txt
```

## Running the pipeline

```bash
python run.py                      # Modules 1-6 + 8, end to end, with logging
streamlit run dashboard/app.py     # Module 7, interactive dashboard
```

Everything is config-driven via `config/config.yaml` — file paths, the 40-gene panel, cleaning
thresholds, and the risk-score formula all live there, not hardcoded in `src/`.

## Results

- 503-patient clean cohort, 40-gene curated LUAD driver panel
- 12 SQL queries + a reusable `high_risk_cohort` view, all executed and explained in
  `notebooks/05_sql_analytics.ipynb`
- 4 statistical methods, each interpretable end to end (documented rationale for method choice)
- A composite risk score that is statistically validated against real survival outcomes, not just
  asserted
- A 5-tab interactive dashboard whose numbers are verified consistent with the offline analysis
- A one-command reproducible pipeline (`run.py`) and a PDF executive summary generated from live
  data, not hardcoded figures

## Future Work

- Expand the driver-gene panel or add a second curated panel (e.g. immune checkpoint genes) as a
  comparison arm
- Add Cox proportional-hazards regression as a second survival method once the univariate
  Kaplan-Meier/log-rank result is fully internalized
- Investigate the counterintuitive negative age/mutation-burden correlation with a larger cohort
- Improve clinical data completeness (11% of the pulled cohort had no usable clinical record at
  all) — a real bottleneck on statistical power identified in the executive summary
