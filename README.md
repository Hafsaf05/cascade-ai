# Sentinel AI - Risk Cascade

Sentinel AI detects fraud cascades: individually plausible transactions that become risky when connected by time, customers, devices, IPs, addresses, or payment instruments. It combines XGBoost, IsolationForest, temporal velocity, and relational signals, then compares expected loss across defensive strategies.

## Setup

Python 3.11+ is recommended.

```powershell
pip install -r requirements.txt
python data/generate_data.py
python models/train_models.py
uvicorn app.main:app --reload
```

The API runs at `http://localhost:8000`. Swagger documentation is available at `/docs`.

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

The frontend uses `VITE_API_URL` and defaults to `http://localhost:8000` for local development. For deployment, create `frontend/.env` with the API origin:

```text
VITE_API_URL=https://your-api.example.com
```

## Current evaluation

The latest chronological training run produced these real held-out test metrics:

| Metric | Value |
| --- | ---: |
| Precision | 0.9672 |
| Recall | 1.0000 |
| F1 | 0.9833 |
| False-positive rate | 0.00334 |

These values are printed by `models/train_models.py` and recalculated by `GET /api/risk/metrics`; they are not hardcoded into the API.

## API

- `GET /api/risk/cascades` lists detected cascade windows for the dashboard.
- `POST /api/risk/score` scores one transaction and returns the signal breakdown and action tier.
- `GET /api/risk/cascade/{id}` returns transactions, graph neighborhood counts, and an investigator explanation.
- `POST /api/risk/simulate` compares Allow All, Block All, and Minimum-Friction expected loss.
- `POST /api/risk/action` records an investigator decision in SQLite `audit_log`.
- `GET /api/risk/metrics` runs the held-out test set and returns precision, recall, F1, and FPR.
- `GET /api/risk/audit` lists recorded decisions.

Ollama is optional. If `qwen2.5:7b` is available at the default local endpoint, the detail view uses it for the short explanation. If it is unavailable, a deterministic evidence-only explanation is returned and all scoring/simulation functionality continues to work.