# OncoInsights — Engineered Feature Dictionary

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
