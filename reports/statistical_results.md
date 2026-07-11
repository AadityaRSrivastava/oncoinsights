# OncoInsights — Statistical Analysis Results

## 1. Welch's t-test — gene expression by mutation status

**Why Welch's, not Student's:** mutant and wild-type groups have very different sample sizes and no reason to assume equal variance, so Welch's t-test (which doesn't assume equal variance) is the safer default.

| gene    |   n_mutant |   n_wildtype |   mean_expr_mutant |   mean_expr_wildtype |   t_stat |   p_value |   p_value_fdr_bh | significant_fdr_0.05   |
|:--------|-----------:|-------------:|-------------------:|---------------------:|---------:|----------:|-----------------:|:-----------------------|
| TP53    |        260 |          239 |            10.0301 |              10.4172 |  -5.1649 |    0      |           0      | True                   |
| KRAS    |        150 |          349 |            10.8854 |              10.3504 |   8.0339 |    0      |           0      | True                   |
| EGFR    |         65 |          434 |            10.9695 |               9.7355 |   5.7346 |    0      |           0      | True                   |
| STK11   |         72 |          427 |             9.1859 |               9.6009 |  -4.212  |    0.0001 |           0.0001 | True                   |
| KEAP1   |         92 |          407 |            10.6731 |              10.4038 |   3.0065 |    0.0033 |           0.0037 | True                   |
| SMARCA4 |         43 |          456 |            10.7135 |              11.6469 |  -4.2238 |    0.0001 |           0.0002 | True                   |
| ATM     |         41 |          458 |             9.6647 |               9.8445 |  -1.8486 |    0.0705 |           0.0705 | False                  |
| NF1     |         59 |          440 |            10.4026 |              10.8178 |  -3.5816 |    0.0007 |           0.0009 | True                   |

**Interpretation:** 
After Benjamini-Hochberg FDR correction across all 8 genes tested, TP53, KRAS, EGFR, STK11, KEAP1, SMARCA4, NF1 show a statistically significant difference in expression between mutated and wild-type patients (FDR-adjusted p < 0.05). This is a sanity check as much as a finding: a truncating/damaging mutation often does shift a gene's own expression (e.g. nonsense-mediated decay, altered transcriptional feedback), so significant genes here validate that the mutation calls are behaviorally consistent with the expression data.


## 2. Chi-square test — tumor stage vs TP53 mutation status

| STAGE_BINARY   |   Mutant |   Wild-type |
|:---------------|---------:|------------:|
| Early (I/II)   |      203 |         193 |
| Late (III/IV)  |       59 |          48 |


Chi2 = 0.364, dof = 1, p = 0.5463

**Interpretation:** TP53 mutation status is not significantly associated with early- vs late-stage disease at diagnosis (p >= 0.05). TP53 is the most frequently mutated gene in this cohort (52% of patients), so this test asks whether losing TP53 function tracks with more advanced disease at the time of diagnosis, or whether it's roughly evenly distributed across stages.


## 3. Pearson correlation — age vs panel mutation count

r = -0.165, p = 0.0002, n = 503


**Interpretation:** There is a weak negative correlation (r=-0.165) between patient age and driver-panel mutation count. This runs counter to the usual expectation that mutation burden rises with age, and would be worth flagging for follow-up.


## 4. Kaplan-Meier + log-rank test — high-risk vs low-risk survival

n(high-risk) = 126, n(low-risk) = 377

Median survival, high-risk group: 38.5 months

Median survival, low-risk group: 53.7 months

Log-rank test p-value: 0.002124


**Interpretation:** The engineered `RISK_SCORE` (built from mutation burden, age, and stage) does produce a statistically significant separation in overall survival between the top-quartile high-risk group and everyone else (log-rank p < 0.05). This validates the risk score as a meaningful stratification, not just an arbitrary composite.
