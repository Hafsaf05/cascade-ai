from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class TransactionInput(BaseModel):
    transaction_id: str = "ad-hoc"
    customer_id: str
    device_id: str
    ip: str
    payment_instrument: str
    address: str
    amount: float = Field(gt=0)
    timestamp: datetime


class ScoreRequest(BaseModel):
    transaction: TransactionInput
    recent_transactions: list[TransactionInput] = Field(default_factory=list)


class SimulationRequest(BaseModel):
    cascade_id: str
    transactions: list[dict[str, Any]] | None = None


class ActionRequest(BaseModel):
    cascade_id: str
    action: Literal["ALLOW", "STEP_UP", "HOLD"]
    actor: str = "investigator"