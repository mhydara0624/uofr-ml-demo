"""Withdrawal-risk model for the U of R demo.

The same file runs two ways:
  * on a warehouse, as the handler of stored procedure ML.TRAIN_WITHDRAWAL_MODEL   -> run(session, version_name)
  * on a compute pool, as a Snowflake ML Job (see submit_job.py)                  -> python train.py --version ...
Either way it reads the versioned Feature Store dataset UOFR_XML_DEMO.ML_FEATURES.WITHDRAWAL_TRAINING,
logs the run to the experiment UR_WITHDRAWAL_RISK, and registers a version of
UOFR_XML_DEMO.ML.UR_WITHDRAWAL_RISK in the Model Registry (lineage: table -> feature view -> dataset -> model).
"""
import argparse
import platform
from datetime import datetime

import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

DATABASE, SCHEMA = "UOFR_XML_DEMO", "ML"
DATASET, DATASET_VERSION = "UOFR_XML_DEMO.ML_FEATURES.WITHDRAWAL_TRAINING", "V1"
MODEL_NAME = "UR_WITHDRAWAL_RISK"
PARAMS = {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.08, "subsample": 0.9}
FEATURES = ["GPA", "COURSE_LOAD", "MIDTERM_GRADE_POINTS", "LMS_LOGINS_LAST_14D",
            "FINANCIAL_HOLD", "FIRST_GENERATION", "ADVISING_VISITS"]
LABEL = "WITHDREW"


def load(session):
    from snowflake.ml.dataset import load_dataset
    from snowflake.snowpark.functions import col
    from snowflake.snowpark.types import DecimalType, FloatType
    sdf = load_dataset(session, DATASET, DATASET_VERSION).read.to_snowpark_dataframe()
    if "GPA" not in [c.strip('"').upper() for c in sdf.columns]:  # warehouse runtime reads the dataset's parquet files as one object column
        typed = {"GPA": FloatType(), "MIDTERM_GRADE_POINTS": FloatType()}
        sdf = sdf.select([col("$1")[c].cast(typed.get(c, DecimalType(10, 0))).alias(c) for c in FEATURES + [LABEL]])
    df = sdf.select(FEATURES + [LABEL]).to_pandas()
    for c in ["COURSE_LOAD", "LMS_LOGINS_LAST_14D", "FINANCIAL_HOLD", "FIRST_GENERATION", "ADVISING_VISITS", LABEL]:
        df[c] = df[c].astype(int)
    for c in ["GPA", "MIDTERM_GRADE_POINTS"]:
        df[c] = df[c].astype(float)
    return sdf, df


def train(df):
    X, y = df[FEATURES], df[LABEL]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    model = XGBClassifier(**PARAMS, eval_metric="auc", random_state=42)
    model.fit(X_tr, y_tr)
    p = model.predict_proba(X_te)[:, 1]
    metrics = {"auc": round(float(roc_auc_score(y_te, p)), 4),
               "accuracy": round(float(accuracy_score(y_te, (p >= 0.5).astype(int))), 4),
               "rows_train": int(len(X_tr)), "rows_test": int(len(X_te)),
               "withdrawal_rate": round(float(y.mean()), 4)}
    return model, X_tr, metrics


def register(session, model, sample, metrics, version_name, compute):
    from snowflake.ml.registry import Registry
    reg = Registry(session=session, database_name=DATABASE, schema_name=SCHEMA)
    # re-runnable demo: replace this version if it exists (the default version cannot be deleted directly)
    if MODEL_NAME in reg.show_models()["name"].tolist():
        m = reg.get_model(MODEL_NAME)
        names = [v.version_name for v in m.versions()]
        if version_name in names:
            others = [n for n in names if n != version_name]
            if not others:
                reg.delete_model(MODEL_NAME)
            else:
                if m.default.version_name == version_name:
                    m.default = others[0]
                m.delete_version(version_name)
    mv = reg.log_model(
        model,
        model_name=MODEL_NAME,
        version_name=version_name,
        sample_input_data=sample,      # Snowpark DataFrame from the Dataset -> lineage
        conda_dependencies=["xgboost", "scikit-learn"],
        target_platforms=["WAREHOUSE"],
        metrics={**metrics, "trained_on": compute},
        comment=f"Withdrawal risk (synthetic demo data). Trained on {compute}.",
    )
    reg.get_model(MODEL_NAME).comment = "Probability a student withdraws from a course section. Demo on synthetic features."
    return mv.version_name


def track(session, run_name, metrics, compute):
    """Log params and metrics to the Snowflake experiment (AI & ML > Experiments)."""
    try:
        from snowflake.ml.experiment import ExperimentTracking
        exp = ExperimentTracking(session=session, database_name=DATABASE, schema_name=SCHEMA)
        exp.set_experiment(MODEL_NAME)
        with exp.start_run(run_name):
            exp.log_params({**PARAMS, "compute": compute, "dataset": f"{DATASET} {DATASET_VERSION}"})
            exp.log_metrics({k: float(v) for k, v in metrics.items()})
        return run_name
    except Exception as e:  # tracking must never fail a training run
        return f"not logged: {type(e).__name__}: {e}"[:200]


def main(session, version_name, compute):
    sdf, df = load(session)
    model, X_tr, metrics = train(df)
    run = track(session, f"{version_name}_{datetime.now():%Y%m%d_%H%M%S}", metrics, compute)
    v = register(session, model, sdf.select(FEATURES).limit(50), metrics, version_name, compute)
    out = {"model": f"{DATABASE}.{SCHEMA}.{MODEL_NAME}", "version": v, "trained_on": compute,
           "experiment_run": run, "python": platform.python_version(), **metrics}
    print(out)
    return out


def run(session, version_name):
    """Stored procedure handler (warehouse)."""
    wh = session.get_current_warehouse()
    return main(session, version_name, "warehouse " + (wh or "").strip('"'))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="V2_COMPUTE_POOL")
    ap.add_argument("--pool", default="UR_ML_POOL")
    args = ap.parse_args()
    from snowflake.snowpark import Session
    session = Session.builder.getOrCreate()
    session.use_database(DATABASE)
    session.use_schema(SCHEMA)
    __return__ = main(session, args.version, f"compute pool {args.pool}")
