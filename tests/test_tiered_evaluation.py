from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import evaluation
from app import scoring
from app.main import app
from app.routers import risk


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def evaluation_result():
    result = evaluation.evaluate()
    risk.METRICS_CACHE = result
    return result


def test_each_tier_is_present_in_train_and_test():
    train = pd.read_csv(ROOT / "data" / "train.csv")
    test = pd.read_csv(ROOT / "data" / "test.csv")
    metadata = pd.read_csv(ROOT / "data" / "tier_metadata.csv").set_index("transaction_id")
    for frame in (train, test):
        tiers = set(metadata.loc[frame.loc[frame.label == "cascade", "transaction_id"], "difficulty"])
        assert {"level_1", "level_2", "level_3"}.issubset(tiers)


def test_difficulty_metadata_is_not_a_model_feature():
    train = pd.read_csv(ROOT / "data" / "train.csv", parse_dates=["timestamp"])
    features = scoring.engineer_features(train)
    assert "difficulty" not in train.columns
    assert "difficulty" not in features.columns
    assert "_difficulty" not in features.columns


def test_ablation_configurations_are_distinct(evaluation_result):
    f1_values = {evaluation_result["ablation"][name]["overall"]["f1"] for name in ("individual_only", "individual_temporal", "full_cascade")}
    assert len(f1_values) > 1


def test_metrics_endpoint_returns_tiers_and_ablation(evaluation_result):
    with TestClient(app) as client:
        response = client.get("/api/risk/metrics")
    assert response.status_code == 200
    body = response.json()
    assert set(body["per_tier"]) == {"level_1", "level_2", "level_3"}
    assert set(body["ablation"]) == {"individual_only", "individual_temporal", "full_cascade"}
    assert set(body["overall"]) >= {"precision", "recall", "f1", "false_positive_rate"}


def test_tier_metrics_are_one_vs_rest(evaluation_result):
    test = pd.read_csv(ROOT / "data" / "test.csv")
    normal_count = int((test["label"] == "normal").sum())
    full = evaluation_result["ablation"]["full_cascade"]
    for tier, metrics in full["per_tier"].items():
        assert metrics["count"] == normal_count + {"level_1": 20, "level_2": 16, "level_3": 17}[tier]
        assert metrics["false_positive_rate"] == full["overall"]["false_positive_rate"]