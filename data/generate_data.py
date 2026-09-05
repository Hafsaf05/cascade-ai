from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import pandas as pd


SEED = 42
ROOT = Path(__file__).resolve().parents[1]


def build_dataset() -> pd.DataFrame:
    random.seed(SEED)
    np.random.seed(SEED)
    start = pd.Timestamp("2025-01-01 00:00:00")
    rows: list[dict] = []
    devices = [f"device-{index:04d}" for index in range(1, 1401)]
    ips = [f"198.51.100.{index}" for index in range(1, 241)]
    addresses = [f"address-{index:04d}" for index in range(1, 1601)]
    instruments = [f"card-{index:04d}" for index in range(1, 1801)]

    for index in range(3000):
        customer = random.randint(1, 1000)
        timestamp = start + pd.Timedelta(minutes=random.randint(0, 60 * 24 * 40))
        rows.append({
            "transaction_id": f"txn-{index + 1:05d}", "customer_id": f"customer-{customer:04d}",
            "device_id": random.choice(devices), "ip": random.choice(ips),
            "payment_instrument": random.choice(instruments), "address": random.choice(addresses),
            "amount": round(float(np.clip(np.random.lognormal(3.8, 0.65), 8, 850)), 2),
            "timestamp": timestamp, "label": "normal",
        })

    next_id = 3001
    for cascade_index in range(10):
        center = start + pd.Timedelta(days=3 + cascade_index * 4)
        shared_device = f"cascade-device-{cascade_index:02d}"
        shared_ip = f"203.0.113.{cascade_index + 10}"
        shared_instrument = f"cascade-card-{cascade_index:02d}"
        for member in range(random.randint(5, 10)):
            customer = f"cascade-customer-{cascade_index:02d}-{member:02d}"
            for _ in range(random.randint(3, 4)):
                timestamp = center + pd.Timedelta(minutes=random.randint(0, 14))
                rows.append({
                    "transaction_id": f"txn-{next_id:05d}", "customer_id": customer,
                    "device_id": shared_device if member != 0 else f"device-unique-{cascade_index}-{member}",
                    "ip": shared_ip if member % 3 else f"198.51.100.{cascade_index + 1}",
                    "payment_instrument": shared_instrument if member % 2 else f"card-unique-{cascade_index}-{member}",
                    "address": f"cascade-address-{cascade_index:02d}-{member:02d}",
                    "amount": round(float(np.random.uniform(120, 650)), 2),
                    "timestamp": timestamp, "label": "cascade",
                })
                next_id += 1
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def main() -> None:
    output_dir = ROOT / "data"
    frame = build_dataset()
    split_at = frame["timestamp"].quantile(0.8)
    train = frame[frame["timestamp"] <= split_at].copy()
    test = frame[frame["timestamp"] > split_at].copy()
    for output, subset in (("train.csv", train), ("test.csv", test)):
        subset.assign(timestamp=subset["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")).to_csv(output_dir / output, index=False)
    print(f"Generated {len(frame)} transactions: {len(train)} train, {len(test)} test")
    print(f"Cascade labels: {int((frame['label'] == 'cascade').sum())}")


if __name__ == "__main__":
    main()