from __future__ import annotations

import re
import requests

PUREGLOBAL_CLINICAL_DATABASE_URL = "https://www.pureglobal.ai/clinical-trials/database"


def fetch_pureglobal_snapshot(timeout: int = 20) -> dict:
    """
    Scaffold for PureGlobal ingestion.
    If the endpoint is a rendered web page instead of JSON, store raw text and
    parse it with custom logic in your own environment and under site terms.
    """
    response = requests.get(PUREGLOBAL_CLINICAL_DATABASE_URL, timeout=timeout)
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "")
    if "application/json" in content_type:
        payload = response.json()
        return {"type": "json", "payload": payload, "metrics": {"record_count": len(str(payload))}}

    html = response.text[:50000]
    return {"type": "html", "payload": html, "metrics": extract_pureglobal_metrics(html)}


def extract_pureglobal_metrics(payload: str) -> dict:
    text = payload.lower()
    return {
        "phase_1_mentions": len(re.findall(r"\bphase\s*1\b", text)),
        "phase_2_mentions": len(re.findall(r"\bphase\s*2\b", text)),
        "phase_3_mentions": len(re.findall(r"\bphase\s*3\b", text)),
        "recruiting_mentions": len(re.findall(r"\brecruiting\b", text)),
        "completed_mentions": len(re.findall(r"\bcompleted\b", text)),
    }
