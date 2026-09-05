from __future__ import annotations

from typing import Any

FALSE_POSITIVE_COST_MULTIPLIER = 0.35
STEP_UP_FRICTION_COST = 4.0
HOLD_FRICTION_COST = 18.0
STEP_UP_FRAUD_REDUCTION = 0.70
HOLD_FRAUD_REDUCTION = 0.98


def _transaction_cost(amount: float, probability: float, action: str) -> tuple[float, float]:
    legitimate = amount * (1 - probability)
    fraud = amount * probability
    if action == "ALLOW":
        return fraud, 0.0
    friction = STEP_UP_FRICTION_COST if action == "STEP_UP" else HOLD_FRICTION_COST
    reduction = STEP_UP_FRAUD_REDUCTION if action == "STEP_UP" else HOLD_FRAUD_REDUCTION
    false_positive = friction * legitimate / max(amount, 1)
    return fraud * (1 - reduction) + false_positive, false_positive


def simulate_cascade(transactions: list[dict[str, Any]]) -> dict[str, Any]:
    probabilities = [float(row.get("fraud_probability", row.get("cascade_score", 0.5))) for row in transactions]
    allow_loss = sum(row["amount"] * probability for row, probability in zip(transactions, probabilities))
    block_fp = sum(row["amount"] * (1 - probability) * FALSE_POSITIVE_COST_MULTIPLIER for row, probability in zip(transactions, probabilities))
    actions = []
    minimum_loss = 0.0
    minimum_fp = 0.0
    for row, probability in zip(transactions, probabilities):
        candidates = [(action, *_transaction_cost(row["amount"], probability, action)) for action in ("ALLOW", "STEP_UP", "HOLD")]
        action, cost, fp = min(candidates, key=lambda item: item[1])
        minimum_loss += cost
        minimum_fp += fp
        actions.append({"transaction_id": row["transaction_id"], "action": action, "fraud_probability": probability, "expected_cost": cost})
    strategies = {
        "allow_all": {"expected_loss": allow_loss, "false_positive_cost": 0.0, "actions": [{"transaction_id": row["transaction_id"], "action": "ALLOW"} for row in transactions]},
        "block_all": {"expected_loss": block_fp, "false_positive_cost": block_fp, "actions": [{"transaction_id": row["transaction_id"], "action": "HOLD"} for row in transactions]},
        "minimum_friction": {"expected_loss": minimum_loss, "false_positive_cost": minimum_fp, "actions": actions},
    }
    return {"strategies": strategies, "recommended": min(strategies, key=lambda name: strategies[name]["expected_loss"])}