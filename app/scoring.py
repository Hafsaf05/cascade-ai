from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


INDIVIDUAL_WEIGHT = 0.42
ANOMALY_WEIGHT = 0.18
TEMPORAL_WEIGHT = 0.22
RELATIONAL_WEIGHT = 0.18
ALLOW_THRESHOLD = 0.35
HOLD_THRESHOLD = 0.70
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
XGB_MODEL = None
ISOFOR_MODEL = None


def load_models() -> None:
    global XGB_MODEL, ISOFOR_MODEL
    XGB_MODEL = joblib.load(MODEL_DIR / "xgb_model.pkl")
    ISOFOR_MODEL = joblib.load(MODEL_DIR / "isoforest_model.pkl")


def _as_frame(transactions: list[dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    frame = transactions.copy() if isinstance(transactions, pd.DataFrame) else pd.DataFrame(transactions)
    if frame.empty:
        return frame
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame.sort_values("timestamp").reset_index(drop=True)


def engineer_features(transactions: list[dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    frame = _as_frame(transactions)
    features = []
    for index, row in frame.iterrows():
        prior = frame.iloc[:index]
        age = row["timestamp"] - prior["timestamp"] if not prior.empty else pd.Series(dtype="timedelta64[ns]")
        last_5 = prior[age <= pd.Timedelta(minutes=5)]
        last_30 = prior[age <= pd.Timedelta(minutes=30)]
        customer_5 = last_5[last_5["customer_id"] == row["customer_id"]]
        customer_30 = last_30[last_30["customer_id"] == row["customer_id"]]
        features.append({
            "amount": float(row["amount"]), "hour": row["timestamp"].hour, "day_of_week": row["timestamp"].dayofweek,
            "customer_txn_5m": len(customer_5), "customer_txn_30m": len(customer_30),
            "unique_device_30m": last_30["device_id"].nunique() if not last_30.empty else 0,
            "unique_ip_30m": last_30["ip"].nunique() if not last_30.empty else 0,
            "customer_amount_30m": float(customer_30["amount"].sum()) if not customer_30.empty else 0,
        })
    return pd.DataFrame(features).fillna(0)


def model_scores(transaction: dict[str, Any], recent_transactions: list[dict[str, Any]] | None = None) -> tuple[float, float]:
    feature_frame = engineer_features((recent_transactions or []) + [transaction])
    if XGB_MODEL is None or ISOFOR_MODEL is None:
        load_models()
    values = feature_frame.iloc[[-1]]
    fraud_probability = float(XGB_MODEL.predict_proba(values)[0][1])
    anomaly = float(np.clip((0.5 - ISOFOR_MODEL.decision_function(values)[0]) / 0.5, 0, 1))
    return fraud_probability, anomaly


def individual_risk(transaction: dict[str, Any], recent_transactions: list[dict[str, Any]] | None = None) -> float:
    probability, anomaly = model_scores(transaction, recent_transactions)
    return float(np.clip(probability * 0.70 + anomaly * 0.30, 0, 1))


def temporal_risk(transaction: dict[str, Any], recent_transactions: list[dict[str, Any]]) -> float:
    timestamp = pd.Timestamp(transaction["timestamp"])
    history = _as_frame(recent_transactions)
    if history.empty:
        return 0.0
    age = timestamp - history["timestamp"]
    recent_5 = history[age.between(pd.Timedelta(0), pd.Timedelta(minutes=5))]
    recent_30 = history[age.between(pd.Timedelta(0), pd.Timedelta(minutes=30))]
    velocity = min(len(recent_5) / 12, 1) * 0.5 + min(len(recent_30) / 30, 1) * 0.25
    accounts = min(recent_30["customer_id"].nunique() / 10, 1) * 0.15
    amount_velocity = min(float(recent_30["amount"].sum()) / 2500, 1) * 0.10
    return float(np.clip(velocity + accounts + amount_velocity, 0, 1))


def build_graph(transactions: list[dict[str, Any]] | pd.DataFrame) -> dict[str, Any]:
    graph = {key: defaultdict(set) for key in ("customer", "device", "ip", "address", "payment_instrument")}
    for row in _as_frame(transactions).to_dict("records"):
        customer = row["customer_id"]
        for key in ("device_id", "ip", "address", "payment_instrument"):
            graph["customer"][customer].add(row[key])
            graph["device" if key == "device_id" else key][row[key]].add(customer)
    return graph


def relational_risk(transaction: dict[str, Any], graph: dict[str, Any]) -> float:
    linked = set(graph.get("customer", {}).get(transaction["customer_id"], set()))
    counts = []
    for key, value_key in (("device", "device_id"), ("ip", "ip"), ("payment_instrument", "payment_instrument")):
        neighbors = graph.get(key, {}).get(transaction[value_key], set())
        linked.update(neighbors)
        counts.append(max(len(neighbors) - 1, 0))
    return float(np.clip(min(max(counts) / 8, 1) * 0.65 + min(max(len(linked) - 1, 0) / 12, 1) * 0.35, 0, 1))


def cascade_score(transaction: dict[str, Any], graph: dict[str, Any], recent_transactions: list[dict[str, Any]]) -> dict[str, float | str]:
    individual = individual_risk(transaction, recent_transactions)
    probability, anomaly = model_scores(transaction, recent_transactions)
    temporal = temporal_risk(transaction, recent_transactions)
    relational = relational_risk(transaction, graph)
    score = individual * INDIVIDUAL_WEIGHT + anomaly * ANOMALY_WEIGHT + temporal * TEMPORAL_WEIGHT + relational * RELATIONAL_WEIGHT
    action = "ALLOW" if score < ALLOW_THRESHOLD else "STEP_UP" if score <= HOLD_THRESHOLD else "HOLD"
    return {"individual": individual, "anomaly": anomaly, "temporal": temporal, "relational": relational, "cascade_score": float(np.clip(score, 0, 1)), "action": action, "fraud_probability": probability}