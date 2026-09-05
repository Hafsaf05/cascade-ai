from __future__ import annotations

import json
import urllib.request


def _fallback(evidence: dict) -> dict:
    score = float(evidence.get("cascade_score", 0))
    highest = max(("individual", evidence.get("individual", 0)), ("anomaly", evidence.get("anomaly", 0)), ("temporal", evidence.get("temporal", 0)), ("relational", evidence.get("relational", 0)), key=lambda item: item[1])[0]
    return {"summary": f"{evidence.get('account_count', 0)} accounts generated {evidence.get('transaction_count', 0)} transactions; cascade score {score:.2f}, driven primarily by {highest} risk.", "recommended_action": evidence.get("action", "STEP_UP"), "source": "deterministic-fallback"}


def explain_cascade(evidence: dict) -> dict:
    prompt = "Return JSON with summary and recommended_action. Use only numeric facts present in this evidence; do not invent numbers. Evidence: " + json.dumps(evidence, default=str)
    try:
        request = urllib.request.Request("http://localhost:11434/api/generate", data=json.dumps({"model": "qwen2.5:7b", "prompt": prompt, "stream": False}).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=0.5) as response:
            result = json.loads(response.read())
        parsed = json.loads(result["response"])
        return {"summary": parsed["summary"], "recommended_action": parsed["recommended_action"], "source": "ollama"}
    except Exception:
        return _fallback(evidence)