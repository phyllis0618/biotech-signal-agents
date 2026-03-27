from __future__ import annotations

from src.connectors.clinicaltrials import fetch_clinical_trials
from src.connectors.fda import fetch_fda_drug_events
from src.connectors.pureglobal import fetch_pureglobal_snapshot
from src.models.messages import AgentMessage, Evidence


def run_ingestion_agent(ticker: str, company: str) -> tuple[AgentMessage, dict]:
    fda_events = []
    ctgov_studies = []
    pureglobal_data = {"type": "none", "payload": ""}

    # Keep pipeline alive when one source is temporarily unavailable.
    try:
        fda_events = fetch_fda_drug_events(company=company, limit=5)
    except Exception:
        fda_events = []
    try:
        ctgov_studies = fetch_clinical_trials(company=company, page_size=5)
    except Exception:
        ctgov_studies = []
    try:
        pureglobal_data = fetch_pureglobal_snapshot()
    except Exception:
        pureglobal_data = {"type": "none", "payload": ""}

    evidence: list[Evidence] = []
    for item in fda_events[:2]:
        evidence.append(
            Evidence(
                source="openFDA",
                title="FDA drug event",
                url="https://api.fda.gov/drug/event.json",
                snippet=str(item)[:300],
            )
        )
    for study in ctgov_studies[:2]:
        evidence.append(
            Evidence(
                source="ClinicalTrials.gov",
                title="Clinical trial study record",
                url="https://clinicaltrials.gov/",
                snippet=str(study)[:300],
            )
        )

    if pureglobal_data.get("payload"):
        evidence.append(
            Evidence(
                source="PureGlobal",
                title="PureGlobal clinical trials snapshot",
                url="https://www.pureglobal.ai/clinical-trials/database",
                snippet=str(pureglobal_data["payload"])[:300],
            )
        )

    msg = AgentMessage(
        agent="ingestion",
        ticker=ticker,
        company=company,
        summary=f"Ingested {len(fda_events)} FDA events, {len(ctgov_studies)} CTGov studies, and PureGlobal snapshot.",
        confidence=70 if evidence else 30,
        signal_hint="neutral",
        evidence=evidence,
        tags=["data_ingestion"],
    )

    raw_data = {
        "fda_events": fda_events,
        "ctgov_studies": ctgov_studies,
        "pureglobal_data": pureglobal_data,
        "pureglobal_metrics": pureglobal_data.get("metrics", {}),
    }
    return msg, raw_data
