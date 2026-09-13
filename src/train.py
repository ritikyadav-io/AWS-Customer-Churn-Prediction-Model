"""
train.py
--------
Trains the churn prediction model.

Usage (local, uses data/sample_customer_churn.csv by default):
    python src/train.py

Usage (AWS mode, reads cleaned data from S3 - e.g. after a Glue ETL job):
    python src/train.py --s3-bucket my-bucket --s3-key new-data/output.csv

Either way it produces:
    src/model/model.joblib   (used by the local Flask demo + tests)
    src/model/model.tar.gz   (used by SageMaker deployment)
"""

import argparse
import os
import tarfile

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                              recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

FEATURES = ["tenure", "MonthlyCharges", "Contract"]
TARGET = "Churn"

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOCAL_CSV = os.path.join(HERE, "..", "data", "sample_customer_churn.csv")
MODEL_DIR = os.path.join(HERE, "model")


def load_data(s3_bucket=None, s3_key=None, local_csv=DEFAULT_LOCAL_CSV):
    if s3_bucket and s3_key:
        import boto3
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=s3_bucket, Key=s3_key)
        df = pd.read_csv(obj["Body"])
        print(f"Loaded {df.shape[0]} rows from s3://{s3_bucket}/{s3_key}")
    else:
        df = pd.read_csv(local_csv)
        print(f"Loaded {df.shape[0]} rows from local file {local_csv}")
    return df


def build_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            ("contract_encoder", OneHotEncoder(handle_unknown="ignore"), ["Contract"]),
        ],
        remainder="passthrough",
    )
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)),
    ])


def train(df):
    df = df.dropna(subset=FEATURES + [TARGET])
    X = df[FEATURES]
    y = df[TARGET].map({"Yes": 1, "No": 0})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_pipeline()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    return model, metrics


def save_model(model):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib_path = os.path.join(MODEL_DIR, "model.joblib")
    joblib.dump(model, joblib_path)

    tar_path = os.path.join(MODEL_DIR, "model.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(joblib_path, arcname="model.joblib")

    print(f"Saved {joblib_path} and {tar_path}")
    return joblib_path, tar_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3-bucket", default=None)
    parser.add_argument("--s3-key", default=None)
    parser.add_argument("--upload-to-s3", action="store_true",
                         help="After training, upload model.tar.gz back to s3://<bucket>/model/")
    args = parser.parse_args()

    df = load_data(s3_bucket=args.s3_bucket, s3_key=args.s3_key)
    model, metrics = train(df)

    print("\n--- Model evaluation ---")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"Confusion matrix: {metrics['confusion_matrix']}")

    _, tar_path = save_model(model)

    if args.upload_to_s3 and args.s3_bucket:
        import boto3
        s3 = boto3.client("s3")
        s3.upload_file(tar_path, args.s3_bucket, "model/model.tar.gz")
        print(f"Uploaded to s3://{args.s3_bucket}/model/model.tar.gz")


if __name__ == "__main__":
    main()
