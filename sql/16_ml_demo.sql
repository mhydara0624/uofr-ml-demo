-- =====================================================================
-- U of R Session 3 | 16_ml_demo.sql   (VS Code: Snowflake extension, run statement by statement)
-- Order: steps 1-2, then start ml/submit_job.py in the VS Code terminal (pool takes ~1-2 min),
--        steps 3-4 while it runs, then steps 5-7.
-- =====================================================================
USE ROLE ACCOUNTADMIN; USE WAREHOUSE UR_ML_WH; USE SCHEMA UOFR_XML_DEMO.ML;

-- 1. The data, read straight from Snowflake (synthetic features, real FERPA tags)
SELECT * FROM WITHDRAWAL_FEATURES LIMIT 10;

-- 2. The code lives in GitHub; Snowflake reads the repository as an object
ALTER GIT REPOSITORY UR_ML_REPO FETCH;
LS @UR_ML_REPO/branches/main/ml/;

--    >> In the VS Code terminal:  .venv/bin/python ml/submit_job.py
--       (same train.py, as an ML Job on UR_ML_POOL, image pinned with runtime_environment)

-- 3. Train on a warehouse: train.py from Git runs as a stored procedure on UR_ML_WH
CALL TRAIN_WITHDRAWAL_MODEL('V1_WAREHOUSE');

-- 4. Where the image is set: the runtime images Snowflake provides (pin one, or bring your own)
SELECT value:base_image_type::STRING AS base_image_type, value:python_version::STRING AS python,
       value:version::STRING AS version, value:image_path::STRING AS image_path
FROM TABLE(FLATTEN(INPUT => PARSE_JSON(SYSTEM$GET_LATEST_CONTAINER_RUNTIME_IMAGES())));
SHOW SERVICES IN COMPUTE POOL UR_ML_POOL;           -- the ML Job running on the pool

-- 5. Both versions in the Model Registry, with metrics and where they were trained
SHOW VERSIONS IN MODEL UR_WITHDRAWAL_RISK;
SELECT "name" AS version, "is_default_version" AS is_default,
       PARSE_JSON("metadata"):metrics:auc::FLOAT AS auc,
       PARSE_JSON("metadata"):metrics:trained_on::STRING AS trained_on
FROM TABLE(RESULT_SCAN(LAST_QUERY_ID()));
ALTER MODEL UR_WITHDRAWAL_RISK SET DEFAULT_VERSION = V2_COMPUTE_POOL;   -- promote

-- 6. Predict in SQL: the students most at risk of withdrawing
WITH mv AS MODEL UOFR_XML_DEMO.ML.UR_WITHDRAWAL_RISK
SELECT f.student_id, f.course_code, f.gpa, f.midterm_grade_points, f.financial_hold, f.advising_visits,
       ROUND(mv!PREDICT_PROBA(f.gpa, f.course_load, f.midterm_grade_points, f.lms_logins_last_14d,
             IFF(f.financial_hold, 1, 0), IFF(f.first_generation, 1, 0), f.advising_visits):output_feature_1::FLOAT, 3) AS withdrawal_risk
FROM WITHDRAWAL_FEATURES f
ORDER BY withdrawal_risk DESC LIMIT 10;

-- 7. FERPA still applies: the analyst cannot see IDs or grades, so cannot score them
USE ROLE UR_ANALYST; USE SECONDARY ROLES NONE;
SELECT student_id, gpa, midterm_grade_points, advising_visits FROM UOFR_XML_DEMO.ML.WITHDRAWAL_FEATURES LIMIT 5;
USE ROLE UR_REGISTRAR; USE SECONDARY ROLES NONE;
SELECT student_id, gpa, midterm_grade_points, advising_visits FROM UOFR_XML_DEMO.ML.WITHDRAWAL_FEATURES LIMIT 5;
USE ROLE ACCOUNTADMIN;
