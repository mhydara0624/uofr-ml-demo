"""Submit ml/train.py from the Git repository as a Snowflake ML Job on the UR_ML_POOL compute pool.
Run from VS Code:  .venv/bin/python ml/submit_job.py
"""
from snowflake.ml.jobs import submit_from_stage
from snowpark_session import create_snowpark_session

session = create_snowpark_session("THEMUSHYDARA")
session.use_role("ACCOUNTADMIN")
session.use_warehouse("UR_ML_WH")
session.use_database("UOFR_XML_DEMO")
session.use_schema("ML")
session.sql("ALTER GIT REPOSITORY UOFR_XML_DEMO.ML.UR_ML_REPO FETCH").collect()

job = submit_from_stage(
    "@UOFR_XML_DEMO.ML.UR_ML_REPO/branches/main/ml/",
    "UR_ML_POOL",
    entrypoint="train.py",
    args=["--version", "V2_COMPUTE_POOL"],
    stage_name="UOFR_XML_DEMO.ML.ML_PAYLOAD",
    runtime_environment="3.0.6-py312",    # Snowflake runtime image (CPU, Python 3.12), pinned
    # runtime_environment="cre@ur_approved",  # your own scanned image, once registered
    session=session,
)
print("Submitted job:", job.id)
print("Status:", job.status)
job.wait()
print("Final status:", job.status)
print(job.get_logs()[-3000:])
print("Result:", job.result())
