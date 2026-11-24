-- =====================================================================
-- U of R Session 3 | 15_ml_setup.sql
-- ML schema, compute pool, Git repository, synthetic training data, warehouse training proc.
-- The model is illustrative: features are synthetic, generated per real enrollment row.
-- =====================================================================
USE ROLE ACCOUNTADMIN;
USE WAREHOUSE UR_BATCH_WH;
USE DATABASE UOFR_XML_DEMO;
CREATE SCHEMA IF NOT EXISTS ML COMMENT = 'Machine learning demo: withdrawal risk.';
USE SCHEMA ML;

-- ---------- Compute pool for ML Jobs (CPU, one node, off when idle) ----------
CREATE COMPUTE POOL IF NOT EXISTS UR_ML_POOL
    MIN_NODES = 1 MAX_NODES = 1 INSTANCE_FAMILY = CPU_X64_S
    AUTO_RESUME = TRUE AUTO_SUSPEND_SECS = 300 INITIALLY_SUSPENDED = TRUE
    COMMENT = 'ML Jobs for the U of R demo. Suspends after 5 idle minutes.';

-- ---------- Git: the repository Snowflake reads the code from ----------
CREATE OR REPLACE GIT REPOSITORY UR_ML_REPO
    API_INTEGRATION = GITHUB_INTEGRATION
    ORIGIN = 'https://github.com/mhydara0624/uofr-ml-demo.git'
    COMMENT = 'U of R ML demo code (public repo).';

CREATE STAGE IF NOT EXISTS ML_PAYLOAD COMMENT = 'Payload stage for ML Jobs.';

-- ---------- Training data: one row per enrollment, synthetic signal ----------
-- u(n) = a deterministic pseudo-random number in [0,1) per enrollment row
CREATE OR REPLACE TABLE WITHDRAWAL_FEATURES
COMMENT = 'SYNTHETIC features and label for the withdrawal-risk demo, one row per Fall 2025 enrollment.' AS
WITH base AS (
    SELECT e.student_id, e.section_key, e.course_code, e.academic_period, e.gpa::FLOAT AS gpa,
           COUNT(*) OVER (PARTITION BY e.student_id, e.academic_period) AS course_load,
           (ABS(HASH(e.student_id, e.section_key, 'a')) % 1000000) / 1e6 AS u1,
           (ABS(HASH(e.student_id, e.section_key, 'b')) % 1000000) / 1e6 AS u2,
           (ABS(HASH(e.student_id, 'c')) % 1000000) / 1e6 AS u3,
           (ABS(HASH(e.student_id, 'd')) % 1000000) / 1e6 AS u4,
           (ABS(HASH(e.student_id, e.section_key, 'e')) % 1000000) / 1e6 AS u5,
           (ABS(HASH(e.student_id, e.section_key, 'f')) % 1000000) / 1e6 AS u6
    FROM UOFR_XML_DEMO.STUDENT.STUDENT_ENROLLMENT e
), f AS (
    SELECT student_id, section_key, course_code, academic_period, gpa, course_load,
           ROUND(LEAST(4.0, GREATEST(0.0, gpa - 0.6 + 1.2 * u1)), 1)            AS midterm_grade_points,
           ROUND(2 + 22 * u2 * (0.4 + gpa / 6.0))::INT                          AS lms_logins_last_14d,
           IFF(u3 < 0.10, TRUE, FALSE)                                          AS financial_hold,
           IFF(u4 < 0.22, TRUE, FALSE)                                          AS first_generation,
           FLOOR(u5 * 3.2)::INT                                                 AS advising_visits,
           u6
    FROM base
)
SELECT student_id, section_key, course_code, academic_period, gpa, course_load,
       midterm_grade_points, lms_logins_last_14d, financial_hold, first_generation, advising_visits,
       IFF(u6 < 1 / (1 + EXP(-( -3.4
                               + 1.3 * (2.0 - LEAST(midterm_grade_points, 2.0))
                               + 0.7 * (3.0 - gpa)
                               + 0.08 * (8 - LEAST(lms_logins_last_14d, 8))
                               + 1.2 * IFF(financial_hold, 1, 0)
                               + 0.35 * IFF(first_generation, 1, 0)
                               + 0.30 * (course_load - 1)
                               - 0.45 * advising_visits))), 1, 0)       AS withdrew
FROM f;

ALTER TABLE WITHDRAWAL_FEATURES MODIFY COLUMN STUDENT_ID SET TAG UOFR_XML_DEMO.GOVERNANCE.FERPA_CLASS = 'EDUCATION_RECORD';
ALTER TABLE WITHDRAWAL_FEATURES MODIFY COLUMN GPA        SET TAG UOFR_XML_DEMO.GOVERNANCE.FERPA_CLASS = 'EDUCATION_RECORD';

-- ---------- Warehouse training: train.py from Git, run as a stored procedure ----------
ALTER GIT REPOSITORY UR_ML_REPO FETCH;

CREATE OR REPLACE PROCEDURE TRAIN_WITHDRAWAL_MODEL(VERSION_NAME STRING)
RETURNS VARIANT
LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
PACKAGES = ('snowflake-snowpark-python', 'snowflake-ml-python', 'xgboost', 'scikit-learn', 'pandas')
IMPORTS = ('@UOFR_XML_DEMO.ML.UR_ML_REPO/branches/main/ml/train.py')
HANDLER = 'train.run'
EXECUTE AS CALLER
COMMENT = 'Runs ml/train.py from the Git repository on the current warehouse and registers the model.';

GRANT USAGE ON SCHEMA UOFR_XML_DEMO.ML TO ROLE UR_ANALYST;
GRANT USAGE ON SCHEMA UOFR_XML_DEMO.ML TO ROLE UR_REGISTRAR;
GRANT SELECT ON TABLE UOFR_XML_DEMO.ML.WITHDRAWAL_FEATURES TO ROLE UR_ANALYST;
GRANT SELECT ON TABLE UOFR_XML_DEMO.ML.WITHDRAWAL_FEATURES TO ROLE UR_REGISTRAR;
