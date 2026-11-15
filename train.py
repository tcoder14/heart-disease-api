"""Train a Random Forest on the Heart Disease dataset and save it with joblib.

Usage:
    python train.py                     # expects data/heart.csv
    python train.py --data path/to.csv
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
TARGET = "target"  # 1 = heart disease present, 0 = absent


def main(data_path: str, out_path: str) -> None:
    df = pd.read_csv(data_path)
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise SystemExit(f"CSV is missing columns: {sorted(missing)}")

    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy: {acc:.3f}")
    print(classification_report(y_test, preds))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "test_accuracy": round(acc, 4)}, out_path)
    print(f"Saved model to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/heart.csv")
    parser.add_argument("--out", default="model/heart_model.joblib")
    args = parser.parse_args()
    main(args.data, args.out)
