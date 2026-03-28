from __future__ import annotations

from typing import Dict, List

from src.connectors.http import get_json


FEDERAL_REGISTER_URL = "https://www.federalregister.gov/api/v1/documents.json"


def fetch_fda_adcom_calendar(limit: int = 25) -> List[Dict]:
    """
    Pull FDA advisory committee notices from Federal Register API.
    """
    params = {
        "conditions[agencies][]": "food-and-drug-administration",
        "conditions[term]": "advisory committee",
        "order": "newest",
        "per_page": str(limit),
    }
    data = get_json(FEDERAL_REGISTER_URL, params=params, timeout=20)
    results = data.get("results", []) if isinstance(data, dict) else []
    out: List[Dict] = []
    for item in results:
        out.append(
            {
                "type": "FDA_AdCom_Notice",
                "title": item.get("title", ""),
                "url": item.get("html_url", ""),
                "target_date": item.get("publication_date", ""),
                "summary": item.get("abstract", ""),
            }
        )
    return out


def fetch_fda_pdufa_mentions(limit: int = 25) -> List[Dict]:
    """
    Pull recent FDA-related Federal Register notices mentioning PDUFA.
    """
    params = {
        "conditions[agencies][]": "food-and-drug-administration",
        "conditions[term]": "PDUFA",
        "order": "newest",
        "per_page": str(limit),
    }
    data = get_json(FEDERAL_REGISTER_URL, params=params, timeout=20)
    results = data.get("results", []) if isinstance(data, dict) else []
    out: List[Dict] = []
    for item in results:
        out.append(
            {
                "type": "PDUFA_Mention",
                "title": item.get("title", ""),
                "url": item.get("html_url", ""),
                "target_date": item.get("publication_date", ""),
                "summary": item.get("abstract", ""),
            }
        )
    return out
