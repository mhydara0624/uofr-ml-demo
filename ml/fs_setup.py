"""One-time setup: Feature Store for the withdrawal-risk demo.
  * Feature Store schema UOFR_XML_DEMO.ML_FEATURES
  * Entity ENROLLMENT (STUDENT_ID, SECTION_KEY)
  * Feature view WITHDRAWAL_SIGNALS V1 (managed, refreshed daily) over ML.WITHDRAWAL_FEATURES
  * Dataset WITHDRAWAL_TRAINING V1 = label spine + feature view  (training reads this -> lineage)
Run:  .venv/bin/python ml/fs_setup.py
"""
from snowflake.ml.feature_store import CreationMode, Entity, FeatureStore, FeatureView
from snowpark_session import create_snowpark_session

DB, FS_SCHEMA, WH = "UOFR_XML_DEMO", "ML_FEATURES", "UR_ML_WH"
SOURCE = f"{DB}.ML.WITHDRAWAL_FEATURES"

session = create_snowpark_session("THEMUSHYDARA")
session.use_role("ACCOUNTADMIN")
session.use_warehouse(WH)

fs = FeatureStore(session=session, database=DB, name=FS_SCHEMA, default_warehouse=WH,
                  creation_mode=CreationMode.CREATE_IF_NOT_EXIST)

entity = Entity(name="ENROLLMENT", join_keys=["STUDENT_ID", "SECTION_KEY"],
                desc="One student in one course section.")
if "ENROLLMENT" not in [r["NAME"] for r in fs.list_entities().collect()]:
    fs.register_entity(entity)

features_df = session.sql(f"""
    SELECT STUDENT_ID, SECTION_KEY,
           GPA::FLOAT                              AS GPA,
           COURSE_LOAD::NUMBER(10,0)               AS COURSE_LOAD,
           MIDTERM_GRADE_POINTS::FLOAT             AS MIDTERM_GRADE_POINTS,
           LMS_LOGINS_LAST_14D::NUMBER(10,0)       AS LMS_LOGINS_LAST_14D,
           IFF(FINANCIAL_HOLD, 1, 0)::NUMBER(10,0) AS FINANCIAL_HOLD,
           IFF(FIRST_GENERATION, 1, 0)::NUMBER(10,0) AS FIRST_GENERATION,
           ADVISING_VISITS::NUMBER(10,0)           AS ADVISING_VISITS
    FROM {SOURCE}""")

fv = FeatureView(name="WITHDRAWAL_SIGNALS", entities=[entity], feature_df=features_df,
                 refresh_freq="1 day",
                 desc="Academic and engagement signals per enrollment (synthetic demo features).")
fv = fv.attach_feature_desc({
    "GPA": "Cumulative GPA at the start of term",
    "COURSE_LOAD": "Number of sections the student is enrolled in this term",
    "MIDTERM_GRADE_POINTS": "Midterm grade in this section, 0-4 scale",
    "LMS_LOGINS_LAST_14D": "Learning-system logins in the last 14 days",
    "FINANCIAL_HOLD": "1 if the student has a financial hold",
    "FIRST_GENERATION": "1 if the student is first generation",
    "ADVISING_VISITS": "Advising visits this term",
})
fv = fs.register_feature_view(feature_view=fv, version="V1", overwrite=True)
print("feature view:", fv.name, fv.version, fv.status)

spine = session.table(SOURCE).select("STUDENT_ID", "SECTION_KEY", "WITHDREW")
try:
    fs.delete_dataset("WITHDRAWAL_TRAINING", "V1") if hasattr(fs, "delete_dataset") else None
except Exception:
    pass
try:
    session.sql(f"DROP DATASET IF EXISTS {DB}.{FS_SCHEMA}.WITHDRAWAL_TRAINING").collect()
except Exception:
    pass
ds = fs.generate_dataset(name="WITHDRAWAL_TRAINING", version="V1", spine_df=spine, features=[fv],
                         spine_label_cols=["WITHDREW"],
                         desc="Training data for UR_WITHDRAWAL_RISK: label + WITHDRAWAL_SIGNALS V1.")
print("dataset:", ds.fully_qualified_name, ds.selected_version.name, ds.read.to_snowpark_dataframe().count(), "rows")
print(ds.read.to_snowpark_dataframe().columns)
