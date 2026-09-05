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
            "device_id": random.choice(devices[:120]) if random.random() < 0.18 else random.choice(devices),
            "ip": random.choice(ips[:24]) if random.random() < 0.18 else random.choice(ips),
            "payment_instrument": random.choice(instruments), "address": random.choice(addresses),
            "amount": round(float(np.clip(np.random.lognormal(3.8, 0.65), 8, 850)), 2),
            "timestamp": timestamp, "label": "normal", "_difficulty": "", "_cascade_id": "",
        })

    next_id = 3001
    cascade_centers = [3, 6, 9, 12, 15, 18, 21, 24, 27, 33, 36, 39]
    for cascade_index, center_day in enumerate(cascade_centers):
        level = cascade_index % 3 + 1
        center = start + pd.Timedelta(days=center_day)
        shared_device = f"cascade-device-{cascade_index:02d}"
        shared_ip = f"203.0.113.{cascade_index + 10}"
        shared_instrument = f"cascade-card-{cascade_index:02d}"
        members = random.randint(5, 8)
        cascade_customers = [f"cascade-customer-{cascade_index:02d}-{member:02d}" for member in range(members)]
        for member, customer in enumerate(cascade_customers):
            for _ in range(random.randint(3, 4)):
                if level == 1:
                    timestamp = center + pd.Timedelta(minutes=random.randint(0, 14))
                elif level == 2:
                    timestamp = center + pd.Timedelta(minutes=random.randint(0, 150))
                else:
                    timestamp = center + pd.Timedelta(minutes=random.randint(0, 8 * 60))
                if level == 1:
                    device_id = shared_device
                    ip = shared_ip
                    payment_instrument = shared_instrument
                    amount = round(float(np.random.uniform(80, 420)), 2)
                elif level == 2:
                    device_id = shared_device if member % 2 == 0 else random.choice(devices)
                    ip = shared_ip
                    payment_instrument = shared_instrument if member == 0 else random.choice(instruments)
                    amount = round(float(np.clip(np.random.lognormal(3.8, 0.65), 8, 420)), 2)
                else:
                    device_id = random.choice(devices)
                    ip = shared_ip
                    payment_instrument = random.choice(instruments)
                    amount = round(float(np.clip(np.random.lognormal(3.8, 0.65), 8, 250)), 2)
                rows.append({
                    "transaction_id": f"txn-{next_id:05d}", "customer_id": customer,
                    "device_id": device_id, "ip": ip, "payment_instrument": payment_instrument,
                    "address": f"cascade-address-{cascade_index:02d}-{member:02d}",
                    "amount": amount, "timestamp": timestamp, "label": "cascade",
                    "_difficulty": f"level_{level}", "_cascade_id": f"cascade-{cascade_index:02d}",
                })
                next_id += 1
        if level == 3:
            for noise_index, customer in enumerate(cascade_customers[:3]):
                rows.append({
                    "transaction_id": f"txn-{next_id:05d}", "customer_id": customer,
                    "device_id": random.choice(devices), "ip": random.choice(ips),
                    "payment_instrument": random.choice(instruments), "address": random.choice(addresses),
                    "amount": round(float(np.clip(np.random.lognormal(3.8, 0.65), 8, 250)), 2),
                    "timestamp": center + pd.Timedelta(minutes=30 + noise_index * 70),
                    "label": "normal", "_difficulty": "", "_cascade_id": "",
                })
                next_id += 1
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def main() -> None:
    output_dir = ROOT / "data"
    frame = build_dataset()
    split_at = frame["timestamp"].quantile(0.8)
    train = frame[frame["timestamp"] <= split_at].copy()
    test = frame[frame["timestamp"] > split_at].copy()
    metadata = frame[["transaction_id", "_difficulty", "_cascade_id"]].rename(columns={"_difficulty": "difficulty", "_cascade_id": "cascade_id"})
    metadata.to_csv(output_dir / "tier_metadata.csv", index=False)
    model_columns = ["transaction_id", "customer_id", "device_id", "ip", "payment_instrument", "address", "amount", "timestamp", "label"]
    for output, subset in (("train.csv", train), ("test.csv", test)):
        subset[model_columns].assign(timestamp=subset["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")).to_csv(output_dir / output, index=False)
    print(f"Generated {len(frame)} transactions: {len(train)} train, {len(test)} test")
    print(f"Cascade labels: {int((frame['label'] == 'cascade').sum())}")


if __name__ == "__main__":
    main()