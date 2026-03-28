from __future__ import annotations

from typing import Dict, List


def build_pipeline_table_rows(
    ctgov_studies: List[Dict],
    ticker: str,
    company: str,
    max_rows: int = 20,
) -> List[Dict]:
    rows: List[Dict] = []
    for study in ctgov_studies[:max_rows]:
        protocol = study.get("protocolSection", {})
        ident = protocol.get("identificationModule", {})
        status_mod = protocol.get("statusModule", {})
        design_mod = protocol.get("designModule", {})
        cond_mod = protocol.get("conditionsModule", {})

        phases = design_mod.get("phases", [])
        phase_text = " / ".join(phases) if isinstance(phases, list) else str(phases or "")
        conditions = cond_mod.get("conditions", [])
        cond_text = "; ".join(conditions[:2]) if isinstance(conditions, list) else str(conditions or "")

        rows.append(
            {
                "ticker": ticker,
                "company": company,
                "nct_id": ident.get("nctId", ""),
                "brief_title": ident.get("briefTitle", ""),
                "phase": phase_text or "N/A",
                "status": status_mod.get("overallStatus", "N/A"),
                "last_update": status_mod.get("lastUpdatePostDateStruct", {}).get("date", ""),
                "completion_date": status_mod.get("completionDateStruct", {}).get("date", ""),
                "conditions": cond_text,
            }
        )
    return rows


def summarize_pipeline_strength(ctgov_studies: List[Dict]) -> Dict:
    total = len(ctgov_studies)
    phase23 = 0
    completed = 0
    for study in ctgov_studies:
        protocol = study.get("protocolSection", {})
        status = str(protocol.get("statusModule", {}).get("overallStatus", "")).lower()
        if "complete" in status:
            completed += 1
        phases = protocol.get("designModule", {}).get("phases", [])
        phase_text = " ".join(phases).lower() if isinstance(phases, list) else str(phases).lower()
        if "phase 2" in phase_text or "phase 3" in phase_text:
            phase23 += 1
    return {
        "total_trials": total,
        "phase23_trials": phase23,
        "completed_trials": completed,
    }
