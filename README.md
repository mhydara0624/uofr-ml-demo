# U of R ML demo: withdrawal risk

Demo code for the University of Rochester Snowflake sessions. The model and its features are
**synthetic** and only show the mechanics.

| File | What it does |
| --- | --- |
| `ml/fs_setup.py` | One-time setup: Feature Store `UOFR_XML_DEMO.ML_FEATURES`, entity `ENROLLMENT`, feature view `WITHDRAWAL_SIGNALS` V1, training dataset `WITHDRAWAL_TRAINING` V1. |
| `ml/train.py` | Trains an XGBoost withdrawal-risk model from the `WITHDRAWAL_TRAINING` dataset, logs the run to experiment `UR_WITHDRAWAL_RISK`, and registers it in the Model Registry (lineage back to the source table). Runs on a warehouse (stored procedure) or a compute pool (ML Job). |
| `ml/submit_job.py` | Submits `ml/train.py` from this repo as a Snowflake ML Job on `UR_ML_POOL`. The `runtime_environment` line is where the container image is set. |
| `sql/15_ml_setup.sql` | Schema, compute pool, Git repository object, training data, warehouse training procedure. |
| `sql/16_ml_demo.sql` | The SQL steps used in the demo: train on the warehouse, compare versions, experiment, lineage, predict. |
| `notebooks/withdrawal_risk_notebook.ipynb` | The same model in a Snowflake Workspace notebook: Feature Store data, train, log to the experiment, feature importance, score in SQL. |

Snowflake reads this repository directly as `UOFR_XML_DEMO.ML.UR_ML_REPO`.
