from __future__ import annotations

from typing import Any, Dict, List

from src.connectors.clinicaltrials import fetch_clinical_trials
from src.connectors.fda import fetch_fda_drug_events
from src.connectors.fda_calendar import fetch_fda_adcom_calendar, fetch_fda_pdufa_mentions


class DataExtractor:
    """
    DataExtractor Agent: ClinicalTrials.gov + FDA approval-path signals.
    """

    def __init__(self, ct_page_size: int = 8, fda_limit: int = 5) -> None:
        self.ct_page_size = ct_page_size
        self.fda_limit = fda_limit

    def run(self, ticker: str, company: str) -> Dict[str, Any]:
        trials: List[dict] = []
        fda_events: List[dict] = []
        fda_calendar: List[dict] = []
        errors: List[str] = []

        try:
            trials = fetch_clinical_trials(company=company, page_size=self.ct_page_size)
        except Exception as e:
            errors.append(f"clinicaltrials: {e}")

        try:
            fda_events = fetch_fda_drug_events(company=company, limit=self.fda_limit)
        except Exception as e:
            errors.append(f"openfda: {e}")

        try:
            fda_calendar = fetch_fda_adcom_calendar(limit=10) + fetch_fda_pdufa_mentions(limit=10)
        except Exception as e:
            errors.append(f"fda_calendar: {e}")

        phase23 = 0
        active = 0
        for t in trials:
            st = (
                t.get("protocolSection", {})
                .get("statusModule", {})
                .get("overallStatus", "")
                .lower()
            )
            if "recruit" in st or "active" in st:
                active += 1
            phases = t.get("protocolSection", {}).get("designModule", {}).get("phases", [])
            pt = " ".join(phases).lower() if isinstance(phases, list) else str(phases).lower()
            if "phase 2" in pt or "phase 3" in pt:
                phase23 += 1

        return {
            "ticker": ticker,
            "company": company,
            "trial_count": len(trials),
            "phase2_or_3_active": phase23,
            "active_recruiting_trials": active,
            "fda_adverse_event_sample_size": len(fda_events),
            "fda_calendar_headlines": [x.get("title", "")[:120] for x in fda_calendar[:5]],
            "raw_trials": trials[:3],
            "errors": errors,
        }
