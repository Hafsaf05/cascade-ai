from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from app.scoring import build_graph, cascade_score, load_models, update_graph


ROOT = Path(__file__).resolve().parents[1]
CONFIGURATIONS = {
    "individual_only": {"include_temporal": False, "include_relational": False},
    "individual_temporal": {"include_temporal": True, "include_relational": False},
    "full_cascade": {"include_temporal": True, "include_relational": True},
}


def _metric_rows(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    actual = pd.Series([row["actual"] for row in rows], dtype=int)
    predicted = pd.Series([row["predicted"] for row in rows], dtype=int)
    tp = int(((predicted == 1) & (actual == 1)).sum())
    fp = int(((predicted == 1) & (actual == 0)).sum())
    fn = int(((predicted == 0) & (actual == 1)).sum())
    tn = int(((predicted == 0) & (actual == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0, "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0, "count": len(rows)}


def evaluate() -> dict[str, Any]:
    load_models()
    train = pd.read_csv(ROOT / "data" / "train.csv", parse_dates=["timestamp"])
    test = pd.read_csv(ROOT / "data" / "test.csv", parse_dates=["timestamp"]).sort_values("timestamp")
    metadata = pd.read_csv(ROOT / "data" / "tier_metadata.csv").set_index("transaction_id")
    history = train.sort_values("timestamp").to_dict("records")
    graph = build_graph(history)
    results = {name: [] for name in CONFIGURATIONS}
    for row in test.to_dict("records"):
        tier = metadata.loc[row["transaction_id"], "difficulty"] or None
        actual = int(row["label"] == "cascade")
        for name, options in CONFIGURATIONS.items():
            score = cascade_score(row, graph, history[-200:], **options)
            results[name].append({"actual": actual, "predicted": int(float(score["cascade_score"]) >= 0.5), "tier": tier})
        history.append(row)
        update_graph(graph, row)

    def grouped(rows: list[dict[str, Any]]) -> dict[str, Any]:
        output = {"overall": _metric_rows(rows), "per_tier": {}}
        for tier in ("level_1", "level_2", "level_3"):
            # Each tier is evaluated one-vs-rest: its positives plus every
            # held-out normal row, so precision and FPR share real negatives.
            tier_rows = [row for row in rows if row["tier"] == tier or row["actual"] == 0]
            if tier_rows:
                output["per_tier"][tier] = _metric_rows(tier_rows)
        return output

    ablation = {name: grouped(rows) for name, rows in results.items()}
    return {"test_transactions": len(test), "overall": ablation["full_cascade"]["overall"], "per_tier": ablation["full_cascade"]["per_tier"], "ablation": ablation}