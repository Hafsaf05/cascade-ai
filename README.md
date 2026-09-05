# Cascade

### Advanced Risk Engineering for Minimum-Friction Risk Detection

> **A fraud model sees transactions. An attack lives between them.**

Cascade is a defense-only payment-risk engineering prototype designed to detect **coordinated transaction patterns** that may be missed when transactions are evaluated independently.

Instead of asking only *“Is this transaction risky?”*, Cascade asks:

> **“Is this transaction part of a larger risk pattern?”**

It combines **individual, temporal, and relational signals** into an auditable cascade score, then uses **expected loss and False Positive Cost** to choose an appropriate level of intervention.

---

## Why Cascade?

Traditional transaction-level fraud detection can miss coordinated behavior.

A single transaction may look normal:

* reasonable amount
* normal-looking account
* no obvious anomaly
* no extreme velocity

But several transactions can become suspicious when viewed together.

For example:

```text
Account A ──┐
Account B ──┼── shared device ──┐
Account C ──┘                   │
                                ├── rapid transaction burst
Account D ───── shared IP ──────┘
```

Individually, these transactions may not cross a blocking threshold.

Together, they can represent a coordinated risk cascade.

Cascade is built to surface that context while avoiding the opposite problem: **over-intervening on legitimate customers.**

That trade-off is treated explicitly as **False Positive Cost**.

---

# Core Idea

Cascade evaluates payment risk across three dimensions:

### 1. Individual Risk

Evaluates whether a transaction is suspicious on its own.

* **XGBoost** probability
* **Isolation Forest** anomaly score

### 2. Temporal Risk

Evaluates transaction behavior over time.

* transaction velocity
* short-window bursts
* repeated activity within a limited time period

### 3. Relational Risk

Evaluates connections between entities.

* shared devices
* shared IP addresses
* shared payment instruments
* account relationships
* other reusable transaction entities

These signals are combined into a **cascade score**.

```text
Individual Risk
       │
       ├──────────────┐
Temporal Risk         │
       │              ├──→ Cascade Score → Risk Decision
Relational Risk       │
       │              │
       └──────────────┘
```

The result is an **Advanced Risk Engineering** approach rather than a single-model fraud classifier.

---

# From Detection to Decision

Detection is only half of fraud-risk management.

Once a transaction or cascade is considered risky, the system still has to decide:

> **How much intervention is justified?**

Blocking everything suspicious can reduce fraud but also creates unnecessary customer friction.

Cascade therefore compares:

* expected fraud loss
* intervention cost
* False Positive Cost
* different response strategies

The response layer supports three transaction-level actions:

| Action            | Purpose                                                     |
| ----------------- | ----------------------------------------------------------- |
| **ALLOW**         | Continue the transaction when intervention is not justified |
| **STEP-UP**       | Request additional verification                             |
| **HOLD / REVIEW** | Temporarily hold the transaction for investigation          |

The simulator also allows comparison of broader strategies such as:

* Allow All
* Block All
* Minimum-Friction

The objective is not:

> **“Block as much as possible.”**

It is:

> **“Reduce meaningful risk with the minimum necessary friction.”**

This is the core **Technical Product Management for Fraud Risk** decision in Cascade.

---

# AI Boundary

Cascade deliberately separates **risk decisioning** from **language-model generation**.

The LLM does **not** determine:

* the risk score
* whether a transaction is fraudulent
* the final risk threshold
* the intervention action

The risk decision is driven by the underlying models and deterministic risk signals.

An optional local **Qwen2.5:7B** model running through **Ollama** is used only to generate investigator-facing explanations of the evidence.

```text
Transaction
    │
    ▼
Risk Models + Risk Signals
    │
    ▼
Cascade Score
    │
    ├──→ Risk Decision
    │
    └──→ Evidence
              │
              ▼
        Optional Qwen2.5:7B
              │
              ▼
      Investigator Explanation
```

### Why this boundary matters

The language model explains the decision.

**It does not make the decision.**

This keeps the risk pipeline more predictable, auditable, and suitable for defense-oriented workflows.

If Ollama is unavailable, Cascade falls back to a deterministic explanation rather than failing the risk workflow.

---

# Evaluation

Cascade is evaluated using a **chronological held-out test set** rather than randomly mixing future and past transactions.

The evaluation pipeline:

1. Generate synthetic transaction data.
2. Plant coordinated risk cascades.
3. Add realistic noise and hard-negative behavior.
4. Split transactions chronologically.
5. Train only on the earlier portion.
6. Evaluate on later unseen transactions.
7. Compare progressively richer signal sets.

### Current held-out benchmark

| Configuration             | Precision | Recall |     F1 | False Positive Rate |
| ------------------------- | --------: | -----: | -----: | ------------------: |
| **Individual only**       |    24.50% | 92.45% | 38.74% |              25.17% |
| **Individual + Temporal** |    60.00% | 73.58% | 66.10% |           **4.33%** |
| **Full Cascade**          |    51.56% | 62.26% | 56.41% |               5.17% |

These results are from **synthetic held-out data**, not Razorpay production data.

### What the benchmark actually shows

The strongest measurable improvement in the current benchmark comes from **temporal context**.

Individual scoring alone produces a high false-positive rate:

> **25.17% FPR**

Adding temporal context reduces that to:

> **4.33% FPR**

The current full Cascade configuration has:

> **51.56% precision**
> **62.26% recall**
> **5.17% FPR**

The relational signal is retained as an investigative signal, but the current benchmark does **not** show that it improves aggregate classification performance.

That result is intentionally reported rather than hidden.

---

# Difficulty-Level Evaluation

The held-out benchmark also evaluates planted cascades by difficulty level using a **one-vs-rest** evaluation.

Each tier contains:

* the positive cascades belonging to that difficulty level
* **all held-out normal transactions as negatives**

This avoids artificially inflating per-tier performance by excluding normal negatives.

| Difficulty  | Precision | Recall |     F1 | False Positive Rate |
| ----------- | --------: | -----: | -----: | ------------------: |
| **Level 1** |    38.00% | 95.00% | 54.29% |               5.17% |
| **Level 2** |    18.42% | 43.75% | 25.93% |               5.17% |
| **Level 3** |    18.42% | 41.18% | 25.45% |               5.17% |

The degradation across harder tiers is important.

It shows that detection becomes substantially more difficult as coordinated behavior becomes less obvious.

That is a useful engineering result—not something to hide behind a single aggregate number.

---

# Understanding the Relational Signal

The current benchmark exposed an important limitation.

Hard negatives can share infrastructure in ways that resemble coordinated behavior.

In the latest audit:

* **Normal mean relational score:** 0.524149
* **Cascade mean relational score:** 0.378498
* Normal rows with relational score ≥ 0.5: **287**
* Cascade rows with relational score ≥ 0.5: **14**

At a 0.5 threshold:

| Configuration         | TP | FN |  FP |  TN |
| --------------------- | -: | -: | --: | --: |
| Individual only       | 49 |  4 | 151 | 449 |
| Individual + Temporal | 39 | 14 |  26 | 574 |
| Full Cascade          | 33 | 20 |  31 | 569 |

This means relational context can currently **increase false positives and suppress some true positives** when incorporated directly into the aggregate score.

Rather than retuning the benchmark solely to produce better README numbers, Cascade retains relational information as an **investigative signal** and documents the limitation.

Future work would focus on calibration and stronger relational features against these hard negatives.

---

# What Broke During Development

The first evaluation produced an apparently excellent result:

> **96.72% precision and 100% recall**

That looked impressive.

It was also a warning sign.

An audit showed that the synthetic cascades were too easy to distinguish from normal transactions. Their transaction amounts were substantially separated from the normal population, allowing the benchmark to reward shortcuts rather than genuine coordinated-risk reasoning.

The benchmark was therefore hardened with:

* overlapping transaction-amount ranges
* benign shared devices
* benign shared IP addresses
* smaller cascade groups
* realistic noise
* hard-negative examples
* stricter past-only feature construction
* intentional entity reuse across time

The metrics dropped.

That was the intended outcome.

A difficult benchmark that exposes weaknesses is more valuable than an easy benchmark that produces impressive numbers.

---

# Known Limitation

The current temporal signal is less effective against **slow or evasive cascades**.

For example, an adversarial sequence with transactions spaced approximately 25 minutes apart produced a relatively low temporal score:

```text
Cascade score: 0.3578
Temporal score: 0.0261
Relational score: 0.5250
Action: STEP_UP
```

This demonstrates that an attacker who spreads activity over a longer period can weaken velocity-based detection.

The current system therefore should not be interpreted as a complete fraud-defense solution.

It is a prototype exploring how **cross-transaction context, decision cost, and auditable intervention** can be combined.

---

# Product Flow

Cascade is designed around a short investigator workflow:

```text
Dashboard
    ↓
Suspicious Cascade
    ↓
Evidence + Relationship Graph
    ↓
Risk Explanation
    ↓
Response Simulator
    ↓
Apply Defensive Plan
    ↓
Audit Trail
```

The interface exposes:

* cascade-level risk
* supporting evidence
* temporal behavior
* relational connections
* recommended intervention
* response simulation
* applied actions
* audit history

The goal is to make the path from **detection → reasoning → action → audit** visible in one workflow.

---

# Architecture

```text
┌───────────────────────────────────────────────┐
│                  React UI                     │
│             TypeScript + Tailwind             │
└──────────────────────┬────────────────────────┘
                       │ REST API
                       ▼
┌───────────────────────────────────────────────┐
│                  FastAPI                      │
│                                               │
│  Risk Scoring                                │
│  Cascade Detection                           │
│  Response Simulation                         │
│  Defensive Actions                           │
│  Audit Trail                                 │
└──────────────┬────────────────────────────────┘
               │
       ┌───────┼───────────────┐
       ▼       ▼               ▼
   XGBoost  Isolation      Temporal /
              Forest       Relational Logic
       │       │               │
       └───────┴───────┬───────┘
                       ▼
                Cascade Score
                       │
                       ▼
              Decision / Simulator
                       │
                       ▼
                Audit Database

Optional:
                       │
                       ▼
                Ollama / Qwen2.5:7B
                       │
                       ▼
             Investigator Explanation
```

### Technology Stack

**Frontend**

* React
* TypeScript
* Tailwind CSS

**Backend**

* FastAPI
* SQLAlchemy
* SQLite

**Risk / ML**

* Python
* XGBoost
* Isolation Forest
* temporal feature engineering
* relational graph-style features

**AI Explanation**

* Ollama
* Qwen2.5:7B

---

# API

| Method | Endpoint                 | Purpose                               |
| ------ | ------------------------ | ------------------------------------- |
| `GET`  | `/api/risk/cascades`     | List detected cascades                |
| `POST` | `/api/risk/score`        | Score transaction risk                |
| `GET`  | `/api/risk/cascade/{id}` | Retrieve cascade details and evidence |
| `POST` | `/api/risk/simulate`     | Compare response strategies           |
| `POST` | `/api/risk/action`       | Apply a defensive action              |
| `GET`  | `/api/risk/metrics`      | Retrieve evaluation metrics           |
| `GET`  | `/api/risk/audit`        | Retrieve the audit trail              |

---

# Evaluation Integrity

The benchmark was designed to avoid obvious evaluation leakage.

### Chronological split

Training data comes before the held-out test period.

The test set begins after the maximum training timestamp.

### Disjoint transactions

Training and test transaction IDs are disjoint.

### Past-only features

Features are constructed without reading labels or future transactions.

### Intentional entity reuse

Customers, devices, IP addresses, payment instruments, and addresses can legitimately reappear across time.

This is intentional.

Cross-transaction relationships are the behavior Cascade is designed to detect.

### Hard negatives

Normal behavior can also contain:

* shared devices
* shared IPs
* repeated entities
* bursts
* unusual amounts

This prevents relational or temporal signals from becoming automatic proxies for fraud.

---

# Local Setup

## Prerequisites

* Node.js
* Python 3.10+
* npm
* Optional: Ollama for local AI explanations

## Clone

```bash
git clone <YOUR_PUBLIC_GITHUB_REPO_URL>
cd cascade-ai
```

## Backend

```bash
cd backend

python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### macOS / Linux

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the API:

```bash
uvicorn app.main:app --reload --port 8000
```

The backend will be available at:

```text
http://localhost:8000
```

---

## Frontend

From the frontend directory:

```bash
npm install
```

Create a local environment file:

```env
VITE_API_URL=http://localhost:8000
```

Then run:

```bash
npm run dev
```

The frontend will start using the configured API endpoint.

---

# Optional Local AI

Cascade can use **Qwen2.5:7B through Ollama** for investigator-facing explanations.

Install Ollama and pull the model:

```bash
ollama pull qwen2.5:7b
```

Then make sure Ollama is running locally.

The core risk pipeline does not depend on the LLM.

If the model is unavailable, Cascade uses its deterministic explanation fallback.

---

# Local End-to-End Verification

Once both services are running:

### 1. Open the dashboard

Verify that detected cascades are displayed.

### 2. Open a cascade

Inspect:

* cascade score
* individual risk
* temporal risk
* relational evidence
* linked entities

### 3. Generate the explanation

The optional local LLM should summarize the available evidence without changing the risk decision.

### 4. Run the response simulator

Compare:

* Allow All
* Block All
* Minimum-Friction

### 5. Apply the defensive plan

Verify that the selected transaction actions are recorded.

### 6. Open the audit trail

Confirm that the decision and applied action are traceable.

---

# Defense-Only Design

Cascade is designed strictly for defensive payment-risk use.

It does **not** provide:

* fraud execution
* payment abuse instructions
* evasion tooling
* credential attacks
* attack automation

Its purpose is to:

* detect suspicious coordinated behavior
* explain risk evidence
* compare defensive responses
* reduce unnecessary intervention
* maintain an auditable decision trail

---

# Production Considerations

Cascade is a **buildathon-ready, production-oriented prototype**, not production payment infrastructure.

A production deployment would additionally require:

* real payment-system integrations
* production-grade database infrastructure
* authentication and authorization
* secrets management
* encryption and key management
* observability and alerting
* model monitoring
* model/data drift detection
* scalable stream or event processing
* formal security review
* comprehensive load testing
* operational controls and rollback mechanisms
* validation against real, governed fraud data

The current evaluation intentionally uses synthetic data and should not be interpreted as production fraud performance.

---

# Why This Approach?

Cascade is built around three engineering principles:

### 1. Context over isolated predictions

Fraud does not always exist inside a single transaction.

Sometimes the meaningful signal exists in the relationships **between** transactions.

### 2. Decision quality over maximum blocking

A fraud system can reduce fraud while still creating excessive customer friction.

Risk management therefore needs to account for **False Positive Cost**, not only detection rate.

### 3. Honest evaluation over impressive metrics

A model that performs well on an easy benchmark is not necessarily a useful model.

Cascade deliberately hardened its synthetic benchmark, reported the resulting performance drop, and documented where the current approach still fails.

---

# Project Philosophy

> **Detect the attack between transactions.**
>
> **Account for False Positive Cost.**
>
> **Intervene with the minimum necessary friction.**
>
> **Keep the decision explainable and auditable.**

Cascade is ultimately an **Advanced Risk Engineering** experiment in turning transaction-level fraud signals into coordinated-risk decisions and actionable, auditable responses.

---

## License

This project is provided for buildathon, demonstration, and educational purposes.
