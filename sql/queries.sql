-- OncoInsights SQL Analytics
-- Database: data/processed/oncoinsights.db (SQLite)
-- Tables: patients (1 row/patient), mutations (1 row/mutation call),
--         expression (long format: patientId, sampleId, hugoGeneSymbol, log2_expression)


-- =====================================================================
-- Q1. Overview KPIs (aggregation)
-- Headline numbers for the dashboard's Overview tab.
-- =====================================================================
SELECT
    COUNT(*)                       AS patient_count,
    ROUND(AVG(AGE), 1)             AS avg_age,
    ROUND(AVG(RISK_SCORE), 3)      AS avg_risk_score,
    SUM(HIGH_RISK_FLAG)            AS high_risk_patients,
    ROUND(AVG(EVENT_OCCURRED) * 100, 1) AS pct_deceased
FROM patients;


-- =====================================================================
-- Q2. Patient count and average age by tumor stage (aggregation)
-- =====================================================================
SELECT
    STAGE_GROUP,
    COUNT(*)             AS patient_count,
    ROUND(AVG(AGE), 1)   AS avg_age,
    ROUND(AVG(RISK_SCORE), 3) AS avg_risk_score
FROM patients
GROUP BY STAGE_GROUP
ORDER BY CASE STAGE_GROUP WHEN 'I' THEN 1 WHEN 'II' THEN 2 WHEN 'III' THEN 3 WHEN 'IV' THEN 4 END;


-- =====================================================================
-- Q3. Average EGFR expression by tumor stage (aggregation, join)
-- =====================================================================
SELECT
    p.STAGE_GROUP,
    COUNT(*)                        AS n_samples,
    ROUND(AVG(e.log2_expression), 3) AS avg_egfr_log2_expr
FROM expression e
JOIN patients p ON p.patientId = e.patientId
WHERE e.hugoGeneSymbol = 'EGFR'
GROUP BY p.STAGE_GROUP
ORDER BY avg_egfr_log2_expr DESC;


-- =====================================================================
-- Q4. Average age by TP53 mutation status (aggregation, subquery)
-- =====================================================================
SELECT
    CASE WHEN p.patientId IN (SELECT DISTINCT patientId FROM mutations WHERE hugoGeneSymbol = 'TP53')
         THEN 'TP53 Mutant' ELSE 'TP53 Wild-type' END AS tp53_status,
    COUNT(*)            AS patient_count,
    ROUND(AVG(p.AGE), 1) AS avg_age
FROM patients p
GROUP BY tp53_status;


-- =====================================================================
-- Q5. Top 10 most frequently mutated genes in the panel (aggregation)
-- =====================================================================
SELECT
    hugoGeneSymbol,
    COUNT(DISTINCT patientId) AS patients_mutated,
    ROUND(100.0 * COUNT(DISTINCT patientId) / (SELECT COUNT(*) FROM patients), 1) AS pct_of_cohort
FROM mutations
GROUP BY hugoGeneSymbol
ORDER BY patients_mutated DESC
LIMIT 10;


-- =====================================================================
-- Q6. Rank patients by risk score within their stage group (window function)
-- Answers: "who are the highest-risk patients within each stage?"
-- =====================================================================
SELECT
    patientId,
    STAGE_GROUP,
    RISK_SCORE,
    RANK() OVER (PARTITION BY STAGE_GROUP ORDER BY RISK_SCORE DESC) AS risk_rank_in_stage
FROM patients
ORDER BY STAGE_GROUP, risk_rank_in_stage
LIMIT 20;


-- =====================================================================
-- Q7. Risk score quartile (NTILE) and cohort-wide risk percentile (window function)
-- =====================================================================
SELECT
    patientId,
    RISK_SCORE,
    NTILE(4) OVER (ORDER BY RISK_SCORE) AS risk_quartile,
    ROUND(PERCENT_RANK() OVER (ORDER BY RISK_SCORE) * 100, 1) AS risk_percentile
FROM patients
ORDER BY RISK_SCORE DESC
LIMIT 20;


-- =====================================================================
-- Q8. Average panel mutation count by stage (multi-step CTE)
-- Step 1: count mutations per patient. Step 2: join to stage and aggregate.
-- =====================================================================
WITH patient_mutation_counts AS (
    SELECT patientId, COUNT(*) AS mut_count
    FROM mutations
    GROUP BY patientId
),
patients_with_counts AS (
    SELECT
        p.patientId,
        p.STAGE_GROUP,
        COALESCE(pmc.mut_count, 0) AS mut_count
    FROM patients p
    LEFT JOIN patient_mutation_counts pmc ON pmc.patientId = p.patientId
)
SELECT
    STAGE_GROUP,
    COUNT(*)                      AS patient_count,
    ROUND(AVG(mut_count), 2)      AS avg_panel_mutation_count
FROM patients_with_counts
GROUP BY STAGE_GROUP
ORDER BY CASE STAGE_GROUP WHEN 'I' THEN 1 WHEN 'II' THEN 2 WHEN 'III' THEN 3 WHEN 'IV' THEN 4 END;


-- =====================================================================
-- Q9. Reusable view: high-risk cohort definition
-- =====================================================================
DROP VIEW IF EXISTS high_risk_cohort;
CREATE VIEW high_risk_cohort AS
SELECT
    patientId, sampleId, AGE, SEX, STAGE_GROUP, RISK_SCORE,
    PANEL_MUTATION_COUNT, SURVIVAL_MONTHS, EVENT_OCCURRED
FROM patients
WHERE HIGH_RISK_FLAG = 1;

-- Example use of the view: summarize the high-risk cohort.
SELECT
    COUNT(*)                          AS n_high_risk,
    ROUND(AVG(AGE), 1)                AS avg_age,
    ROUND(AVG(PANEL_MUTATION_COUNT),2) AS avg_mutation_count,
    ROUND(AVG(EVENT_OCCURRED) * 100,1) AS pct_deceased,
    ROUND(AVG(SURVIVAL_MONTHS), 1)     AS avg_survival_months
FROM high_risk_cohort;


-- =====================================================================
-- Q10. Survival outcomes by risk group (aggregation, business-relevant)
-- =====================================================================
SELECT
    CASE WHEN HIGH_RISK_FLAG = 1 THEN 'High risk' ELSE 'Low/medium risk' END AS risk_group,
    COUNT(*)                            AS patient_count,
    ROUND(AVG(SURVIVAL_MONTHS), 1)      AS avg_survival_months,
    ROUND(AVG(EVENT_OCCURRED) * 100, 1) AS pct_deceased
FROM patients
GROUP BY risk_group;


-- =====================================================================
-- Q11. Event rate by EGFR expression quartile (aggregation, join)
-- =====================================================================
SELECT
    EGFR_EXPR_QUARTILE,
    COUNT(*)                             AS patient_count,
    ROUND(AVG(EVENT_OCCURRED) * 100, 1)  AS pct_deceased,
    ROUND(AVG(SURVIVAL_MONTHS), 1)       AS avg_survival_months
FROM patients
WHERE EGFR_EXPR_QUARTILE IS NOT NULL
GROUP BY EGFR_EXPR_QUARTILE
ORDER BY EGFR_EXPR_QUARTILE;


-- =====================================================================
-- Q12. Gene mutation frequency by stage, ranked (multi-step CTE + window function)
-- Step 1: count mutated patients per gene per stage.
-- Step 2: rank genes within each stage by frequency.
-- =====================================================================
WITH gene_stage_counts AS (
    SELECT
        m.hugoGeneSymbol,
        p.STAGE_GROUP,
        COUNT(DISTINCT m.patientId) AS n_mutated
    FROM mutations m
    JOIN patients p ON p.patientId = m.patientId
    GROUP BY m.hugoGeneSymbol, p.STAGE_GROUP
),
ranked AS (
    SELECT
        STAGE_GROUP,
        hugoGeneSymbol,
        n_mutated,
        RANK() OVER (PARTITION BY STAGE_GROUP ORDER BY n_mutated DESC) AS rank_in_stage
    FROM gene_stage_counts
)
SELECT * FROM ranked
WHERE rank_in_stage <= 5
ORDER BY STAGE_GROUP, rank_in_stage;
