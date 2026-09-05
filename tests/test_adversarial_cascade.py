from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.scoring import build_graph, cascade_score, load_models


def test_spaced_relational_cascade_reports_score(capsys):
    load_models()
    transactions = []
    start = datetime(2025, 3, 1, 12, 0)
    for index in range(6):
        transactions.append({
            "transaction_id": f"adversarial-{index}",
            "customer_id": f"adversarial-customer-{index}",
            "device_id": "shared-device-hard-case",
            "ip": "203.0.113.250",
            "payment_instrument": f"adversarial-card-{index}",
            "address": f"adversarial-address-{index}",
            "amount": 65.0 + index,
            "timestamp": start + timedelta(minutes=25 * index),
        })

    graph = build_graph(transactions)
    result = cascade_score(transactions[-1], graph, transactions[:-1])
    print({"cascade_score": result["cascade_score"], "action": result["action"], "temporal": result["temporal"], "relational": result["relational"]})
    assert 0 <= result["cascade_score"] <= 1
    assert result["relational"] > result["temporal"]
    captured = capsys.readouterr()
    assert "cascade_score" in captured.out


def test_print_causal_test_score_distribution():
    load_models()
    root = Path(__file__).resolve().parents[1]
    train = pd.read_csv(root / "data" / "train.csv", parse_dates=["timestamp"])
    test = pd.read_csv(root / "data" / "test.csv", parse_dates=["timestamp"])
    history = train.sort_values("timestamp").to_dict("records")
    graph = build_graph(history)
    scored = []
    for row in test.sort_values("timestamp").to_dict("records"):
        result = cascade_score(row, graph, history[-200:])
        scored.append({"label": row["label"], "cascade_score": result["cascade_score"], "action": result["action"]})
        history.append(row)
    distribution = pd.DataFrame(scored)
    print(distribution.groupby("label").cascade_score.agg(["min", "max", "mean", "median", "count"]).to_string())
    print(distribution.groupby(["label", "action"]).size().to_string())
    assert len(distribution) == len(test)
    assert distribution.cascade_score.between(0, 1).all()
