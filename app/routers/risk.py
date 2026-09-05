from __future__ import annotations

from collections import defaultdict
from datetime import datetime

import pandas as pd
from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from app.database import AuditLog, SessionLocal, Transaction, as_dict
from app.llm_explain import explain_cascade
from app.schemas import ActionRequest, ScoreRequest, SimulationRequest
from app.scoring import build_graph, cascade_score, engineer_features, load_models
from app.simulator import simulate_cascade


router = APIRouter(prefix="/api/risk", tags=["risk"])


def _rows() -> list[dict]:
    with SessionLocal() as session:
        return [as_dict(row) for row in session.scalars(select(Transaction).order_by(Transaction.timestamp)).all()]


def _cascade_groups() -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for row in _rows():
        if row["label"] == "cascade":
            timestamp = pd.Timestamp(row["timestamp"])
            groups[f"cascade-{timestamp.strftime('%Y%m%d')}-{timestamp.hour:02d}"] .append(row)
    return dict(groups)


@router.post("/score")
def score_transaction(request: ScoreRequest):
    transaction = request.transaction.model_dump()
    recent = [row.model_dump() for row in request.recent_transactions]
    return cascade_score(transaction, build_graph(recent + [transaction]), recent)


@router.get("/cascade/{cascade_id}")
def get_cascade(cascade_id: str):
    cascade = _cascade_groups().get(cascade_id)
    if not cascade:
        raise HTTPException(status_code=404, detail="Cascade not found")
    graph = build_graph(cascade)
    scored = [dict(row, **cascade_score(row, graph, cascade[:index])) for index, row in enumerate(cascade)]
    breakdown = {key: sum(float(row[key]) for row in scored) / len(scored) for key in ("individual", "anomaly", "temporal", "relational", "cascade_score")}
    evidence = dict(breakdown, account_count=len({row["customer_id"] for row in cascade}), transaction_count=len(cascade), action=max((row["action"] for row in scored), key=("ALLOW", "STEP_UP", "HOLD").index))
    return {"cascade_id": cascade_id, "transactions": scored, "graph": {key: {entity: len(accounts) for entity, accounts in values.items()} for key, values in graph.items()}, "breakdown": breakdown, "explanation": explain_cascade(evidence)}


@router.post("/simulate")
def simulate(request: SimulationRequest):
    transactions = request.transactions or _cascade_groups().get(request.cascade_id)
    if not transactions:
        raise HTTPException(status_code=404, detail="Cascade not found")
    return simulate_cascade(transactions)


@router.post("/action")
def apply_action(request: ActionRequest):
    with SessionLocal.begin() as session:
        entry = AuditLog(cascade_id=request.cascade_id, action=request.action, actor=request.actor)
        session.add(entry)
    return {"status": "recorded", "cascade_id": request.cascade_id, "action": request.action, "actor": request.actor}


@router.get("/metrics")
def metrics():
    rows = _rows()
    frame = pd.DataFrame(rows).sort_values("timestamp")
    split = frame["timestamp"].quantile(0.8)
    test = frame[frame["timestamp"] > split].copy()
    from app.scoring import XGB_MODEL
    values = engineer_features(frame).iloc[-len(test):]
    predicted = (XGB_MODEL.predict_proba(values)[:, 1] >= 0.5).astype(int)
    actual = (test["label"] == "cascade").astype(int).to_numpy()
    tp = int(((predicted == 1) & (actual == 1)).sum())
    fp = int(((predicted == 1) & (actual == 0)).sum())
    fn = int(((predicted == 0) & (actual == 1)).sum())
    tn = int(((predicted == 0) & (actual == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0, "false_positive_rate": fp / (fp + tn) if fp + tn else 0, "test_transactions": len(test)}


@router.get("/audit")
def audit():
    with SessionLocal() as session:
        return [{"id": row.id, "timestamp": row.timestamp, "cascade_id": row.cascade_id, "action": row.action, "actor": row.actor} for row in session.scalars(select(AuditLog).order_by(AuditLog.timestamp.desc())).all()]


@router.get("/cascades")
def cascades():
    output = []
    for cascade_id, rows in _cascade_groups().items():
        graph = build_graph(rows)
        scores = [cascade_score(row, graph, rows[:index]) for index, row in enumerate(rows)]
        output.append({"cascade_id": cascade_id, "cascade_score": max(float(score["cascade_score"]) for score in scores), "transaction_count": len(rows), "account_count": len({row["customer_id"] for row in rows}), "start": min(row["timestamp"] for row in rows), "end": max(row["timestamp"] for row in rows)})
    return sorted(output, key=lambda row: row["cascade_score"], reverse=True)