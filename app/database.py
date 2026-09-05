from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import DateTime, Float, Integer, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

DATABASE_URL = f"sqlite:///{Path(__file__).resolve().parents[1] / 'sentinel.db'}"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Transaction(Base):
    __tablename__ = "transactions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    transaction_id: Mapped[str] = mapped_column(String, unique=True, index=True)
    customer_id: Mapped[str] = mapped_column(String, index=True)
    device_id: Mapped[str] = mapped_column(String)
    ip: Mapped[str] = mapped_column(String)
    payment_instrument: Mapped[str] = mapped_column(String)
    address: Mapped[str] = mapped_column(String)
    amount: Mapped[float] = mapped_column(Float)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
    label: Mapped[str] = mapped_column(String)


class AuditLog(Base):
    __tablename__ = "audit_log"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    cascade_id: Mapped[str] = mapped_column(String, index=True)
    action: Mapped[str] = mapped_column(String)
    actor: Mapped[str] = mapped_column(String)


def init_db() -> None:
    Base.metadata.create_all(engine)
    with SessionLocal.begin() as session:
        root = Path(__file__).resolve().parents[1]
        source_files = [root / "data" / "train.csv", root / "data" / "test.csv"]
        source_count = sum(len(pd.read_csv(file)) for file in source_files if file.exists())
        if session.scalar(select(Transaction.id).limit(1)) is not None and session.query(Transaction).count() == source_count:
            return
        session.query(Transaction).delete()
        for file in source_files:
            if file.exists():
                for row in pd.read_csv(file, parse_dates=["timestamp"]).to_dict("records"):
                    session.add(Transaction(**row))


def as_dict(transaction: Transaction) -> dict:
    return {column: getattr(transaction, column) for column in ("transaction_id", "customer_id", "device_id", "ip", "payment_instrument", "address", "amount", "timestamp", "label")}