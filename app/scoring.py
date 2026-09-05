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
SHARED_DEVICE_WEIGHT = 0.35
SHARED_IP_WEIGHT = 0.25
SHARED_PAYMENT_WEIGHT = 0.20
CLUSTER_SIZE_WEIGHT = 0.20
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
    transaction_time = pd.Timestamp(transaction["timestamp"])
    history = [row for row in (recent_transactions or []) if pd.Timestamp(row["timestamp"]) <= transaction_time]
    history_frame = _as_frame(history)
    if history_frame.empty:
        history_frame = pd.DataFrame(columns=["timestamp", "customer_id", "device_id", "ip", "amount"])
    age = transaction_time - history_frame["timestamp"] if not history_frame.empty else pd.Series(dtype="timedelta64[ns]")
    last_5 = history_frame[age.between(pd.Timedelta(0), pd.Timedelta(minutes=5))] if not history_frame.empty else history_frame
    last_30 = history_frame[age.between(pd.Timedelta(0), pd.Timedelta(minutes=30))] if not history_frame.empty else history_frame
    customer_5 = last_5[last_5["customer_id"] == transaction["customer_id"]]
    customer_30 = last_30[last_30["customer_id"] == transaction["customer_id"]]
    feature_frame = pd.DataFrame([{
        "amount": float(transaction["amount"]), "hour": transaction_time.hour, "day_of_week": transaction_time.dayofweek,
        "customer_txn_5m": len(customer_5), "customer_txn_30m": len(customer_30),
        "unique_device_30m": last_30["device_id"].nunique() if not last_30.empty else 0,
        "unique_ip_30m": last_30["ip"].nunique() if not last_30.empty else 0,
        "customer_amount_30m": float(customer_30["amount"].sum()) if not customer_30.empty else 0,
    }])
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
    history = history[age >= pd.Timedelta(0)]
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
        update_graph(graph, row)
    return graph


def update_graph(graph: dict[str, Any], row: dict[str, Any]) -> None:
    customer = row["customer_id"]
    for key in ("device_id", "ip", "address", "payment_instrument"):
        graph["customer"][customer].add(row[key])
        graph["device" if key == "device_id" else key][row[key]].add(customer)


def relational_risk(transaction: dict[str, Any], graph: dict[str, Any]) -> float:
    linked = set(graph.get("customer", {}).get(transaction["customer_id"], set()))
    signals = {}
    for signal, key, value_key, cap in (("shared_device_count", "device", "device_id", 8), ("shared_ip_count", "ip", "ip", 8), ("shared_payment_instrument_count", "payment_instrument", "payment_instrument", 6)):
        neighbors = graph.get(key, {}).get(transaction[value_key], set())
        linked.update(neighbors)
        signals[signal] = float(np.clip(max(len(neighbors) - 1, 0) / cap, 0, 1))
    signals["cluster_size"] = float(np.clip(max(len(linked) - 1, 0) / 12, 0, 1))
    return float(np.clip(signals["shared_device_count"] * SHARED_DEVICE_WEIGHT + signals["shared_ip_count"] * SHARED_IP_WEIGHT + signals["shared_payment_instrument_count"] * SHARED_PAYMENT_WEIGHT + signals["cluster_size"] * CLUSTER_SIZE_WEIGHT, 0, 1))


def cascade_score(transaction: dict[str, Any], graph: dict[str, Any], recent_transactions: list[dict[str, Any]], include_temporal: bool = True, include_relational: bool = True) -> dict[str, float | str]:
    individual = individual_risk(transaction, recent_transactions)
    probability, anomaly = model_scores(transaction, recent_transactions)
    temporal = temporal_risk(transaction, recent_transactions)
    relational = relational_risk(transaction, graph)
    active = [(individual, INDIVIDUAL_WEIGHT), (anomaly, ANOMALY_WEIGHT)]
    if include_temporal:
        active.append((temporal, TEMPORAL_WEIGHT))
    if include_relational:
        active.append((relational, RELATIONAL_WEIGHT))
    score = sum(value * weight for value, weight in active) / sum(weight for _, weight in active)
    action = "ALLOW" if score < ALLOW_THRESHOLD else "STEP_UP" if score <= HOLD_THRESHOLD else "HOLD"
    return {"individual": individual, "anomaly": anomaly, "temporal": temporal, "relational": relational, "cascade_score": float(np.clip(score, 0, 1)), "action": action, "fraud_probability": probability}