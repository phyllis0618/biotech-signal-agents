from __future__ import annotations

from typing import Dict, List

from src.connectors.http import get_json


NASDAQ_SCREENER_URL = "https://api.nasdaq.com/api/screener/stocks"


def fetch_us_biotech_from_nasdaq() -> List[Dict]:
    """
    Pull stock universe from Nasdaq screener and filter US biotechnology names.
    """
    params = {
        "tableonly": "true",
        "limit": "10000",
        "offset": "0",
        "download": "true",
    }
    data = get_json(
        NASDAQ_SCREENER_URL,
        params=params,
        timeout=20,
        extra_headers={
            "Accept": "application/json, text/plain, */*",
            "Origin": "https://www.nasdaq.com",
            "Referer": "https://www.nasdaq.com/",
        },
    )
    rows = data.get("data", {}).get("rows", []) if isinstance(data, dict) else []
    out: List[Dict] = []
    for row in rows:
        ticker = str(row.get("symbol", "")).strip().upper()
        name = str(row.get("name", "")).strip()
        country = str(row.get("country", "")).strip().lower()
        sector = str(row.get("sector", "")).strip().lower()
        industry = str(row.get("industry", "")).strip().lower()
        if not ticker or not name:
            continue
        if "united states" not in country and country != "usa":
            continue
        if "biotech" in industry or "biotechnology" in industry:
            out.append(
                {
                    "ticker": ticker,
                    "company": name,
                    "sector": row.get("sector", ""),
                    "industry": row.get("industry", ""),
                }
            )
            continue
        # Include Healthcare/Biotechnology style labels if vendor format varies.
        if "biotech" in sector:
            out.append(
                {
                    "ticker": ticker,
                    "company": name,
                    "sector": row.get("sector", ""),
                    "industry": row.get("industry", ""),
                }
            )
    return out
