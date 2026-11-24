"""Withdrawal-risk model for the U of R demo.

The same file runs two ways:
  * on a warehouse, as the handler of stored procedure ML.TRAIN_WITHDRAWAL_MODEL   -> run(session, version_name)
  * on a compute pool, as a Snowflake ML Job (see submit_job.py)                  -> python train.py --version ...
Either way it trains on UOFR_XML_DEMO.ML.WITHDRAWAL_FEATURES and registers a version of
UOFR_XML_DEMO.ML.UR_WITHDRAWAL_RISK in the Model Registry.
"""
import argparse
import os
import platform

import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

DATABASE, SCHEMA = "UOFR_XML_DEMO", "ML"
TABLE = f"{DATABASE}.{SCHEMA}.WITHDRAWAL_FEATURES"
MODEL_NAME = "UR_WITHDRAWAL_RISK"
FEATURES = ["GPA", "COURSE_LOAD", "MIDTERM_GRADE_POINTS", "LMS_LOGINS_LAST_14D",
            "FINANCIAL_HOLD", "FIRST_GENERATION", "ADVISING_VISITS"]
LABEL = "WITHDREW"


def load(session):
    df = session.table(TABLE).select(FEATURES + [LABEL]).to_pandas()
    for c in ["FINANCIAL_HOLD", "FIRST_GENERATION"]:
        df[c] = df[c].astype(int)
    for c in ["COURSE_LOAD", "LMS_LOGINS_LAST_14D", "ADVISING_VISITS", LABEL]:
        df[c] = df[c].astype(int)
    for c in ["GPA", "MIDTERM_GRADE_POINTS"]:
        df[c] = df[c].astype(float)
    return df


def train(df):
    X, y = df[FEATURES], df[LABEL]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    model = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.08, subsample=0.9,
                          eval_metric="auc", random_state=42)
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
    try:
        reg.get_model(MODEL_NAME).delete_version(version_name)   # re-runnable demo
    except Exception:
        pass
    mv = reg.log_model(
        model,
        model_name=MODEL_NAME,
        version_name=version_name,
        sample_input_data=sample.head(50),
        conda_dependencies=["xgboost", "scikit-learn"],
        target_platforms=["WAREHOUSE"],
        metrics={**metrics, "trained_on": compute},
        comment=f"Withdrawal risk (synthetic demo data). Trained on {compute}.",
    )
    reg.get_model(MODEL_NAME).comment = "Probability a student withdraws from a course section. Demo on synthetic features."
    return mv.version_name


def main(session, version_name, compute):
    df = load(session)
    model, X_tr, metrics = train(df)
    v = register(session, model, X_tr, metrics, version_name, compute)
    out = {"model": f"{DATABASE}.{SCHEMA}.{MODEL_NAME}", "version": v, "trained_on": compute,
           "python": platform.python_version(), **metrics}
    print(out)
    return out


def run(session, version_name):
    """Stored procedure handler (warehouse)."""
    wh = session.get_current_warehouse()
    return main(session, version_name, f"warehouse {wh}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="V2_COMPUTE_POOL")
    args = ap.parse_args()
    from snowflake.snowpark import Session
    session = Session.builder.getOrCreate()
    session.use_database(DATABASE)
    session.use_schema(SCHEMA)
    pool = os.environ.get("SNOWFLAKE_COMPUTE_POOL", "compute pool")
    __return__ = main(session, args.version, f"compute pool {pool}")
