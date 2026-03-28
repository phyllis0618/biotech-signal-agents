from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List


def load_biotech_universe(csv_path: str) -> List[Dict]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Universe file not found: {csv_path}")

    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = (row.get("ticker") or "").strip().upper()
            company = (row.get("company") or "").strip()
            if not ticker or not company:
                continue
            rows.append(
                {
                    "ticker": ticker,
                    "company": company,
                    "cash_runway_months": int(row.get("cash_runway_months", 18) or 18),
                    "single_asset_exposure": str(
                        row.get("single_asset_exposure", "false")
                    ).strip().lower()
                    in {"1", "true", "yes", "y"},
                }
            )
    return rows
