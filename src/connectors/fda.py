from __future__ import annotations

from src.connectors.http import get_json


OPEN_FDA_DRUG_EVENT = "https://api.fda.gov/drug/event.json"


def fetch_fda_drug_events(company: str, limit: int = 5) -> list[dict]:
    query = f'patient.drug.medicinalproduct:"{company}"'
    data = get_json(OPEN_FDA_DRUG_EVENT, params={"search": query, "limit": limit})
    return data.get("results", [])
