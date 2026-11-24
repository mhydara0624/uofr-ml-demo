# U of R ML demo: withdrawal risk

Demo code for the University of Rochester Snowflake sessions. The model and its features are
**synthetic** and only show the mechanics.

| File | What it does |
| --- | --- |
| `ml/train.py` | Trains an XGBoost withdrawal-risk model on `UOFR_XML_DEMO.ML.WITHDRAWAL_FEATURES` and registers it in the Snowflake Model Registry. Runs on a warehouse (stored procedure) or a compute pool (ML Job). |
| `ml/submit_job.py` | Submits `ml/train.py` from this repo as a Snowflake ML Job on `UR_ML_POOL`. The `runtime_environment` line is where the container image is set. |
| `sql/15_ml_setup.sql` | Schema, compute pool, Git repository object, training data, warehouse training procedure. |
| `sql/16_ml_demo.sql` | The SQL steps used in the demo: train on the warehouse, compare versions, predict. |

Snowflake reads this repository directly as `UOFR_XML_DEMO.ML.UR_ML_REPO`.
