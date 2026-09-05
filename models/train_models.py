from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.scoring import engineer_features

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    train = pd.read_csv(ROOT / "data" / "train.csv", parse_dates=["timestamp"])
    test = pd.read_csv(ROOT / "data" / "test.csv", parse_dates=["timestamp"])
    x_train = engineer_features(train)
    combined = pd.concat([train, test], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    all_features = engineer_features(combined)
    x_test = all_features[combined["timestamp"] > train["timestamp"].max()].reset_index(drop=True)
    y_train = (train["label"] == "cascade").astype(int)
    y_test = (test["label"] == "cascade").astype(int)
    classifier = XGBClassifier(n_estimators=160, max_depth=4, learning_rate=0.08, subsample=0.85, colsample_bytree=0.9, eval_metric="logloss", random_state=42)
    classifier.fit(x_train, y_train)
    forest = IsolationForest(n_estimators=180, contamination="auto", random_state=42)
    forest.fit(x_train)
    joblib.dump(classifier, ROOT / "models" / "xgb_model.pkl")
    joblib.dump(forest, ROOT / "models" / "isoforest_model.pkl")
    prediction = classifier.predict(x_test)
    print("Held-out test metrics")
    print(f"precision={precision_score(y_test, prediction, zero_division=0):.4f}")
    print(f"recall={recall_score(y_test, prediction, zero_division=0):.4f}")
    print(f"f1={f1_score(y_test, prediction, zero_division=0):.4f}")


if __name__ == "__main__":
    main()