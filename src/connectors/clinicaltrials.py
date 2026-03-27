from __future__ import annotations

from src.connectors.http import get_json


CTGOV_STUDIES_API = "https://clinicaltrials.gov/api/v2/studies"


def fetch_clinical_trials(company: str, page_size: int = 5) -> list[dict]:
    params = {
        "query.term": company,
        "pageSize": page_size,
        "format": "json",
    }
    data = get_json(CTGOV_STUDIES_API, params=params)
    return data.get("studies", [])
