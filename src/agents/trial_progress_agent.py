from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.models.messages import AgentMessage, Evidence


def run_trial_progress_agent(ticker: str, company: str, raw_data: dict) -> AgentMessage:
    studies = raw_data.get("ctgov_studies", [])
    completed = 0
    recruiting = 0
    phase_2_or_3 = 0
    recent_updates = 0
    window_start = datetime.now(timezone.utc) - timedelta(days=180)

    for study in studies:
        protocol = study.get("protocolSection", {})
        status_module = protocol.get("statusModule", {})
        design_module = protocol.get("designModule", {})
        status = str(status_module.get("overallStatus", "")).lower()

        if "complete" in status:
            completed += 1
        if "recruit" in status:
            recruiting += 1

        phases = design_module.get("phases", [])
        phase_text = " ".join(phases).lower() if isinstance(phases, list) else str(phases).lower()
        if "phase 2" in phase_text or "phase 3" in phase_text:
            phase_2_or_3 += 1

        completion_date = status_module.get("completionDateStruct", {}).get("date", "")
        if isinstance(completion_date, str) and completion_date:
            parsed = _parse_yyyy_mm(completion_date)
            if parsed and parsed >= window_start:
                recent_updates += 1

    hint = "neutral"
    confidence = 55
    if completed > recruiting and phase_2_or_3 > 0:
        hint = "bullish"
        confidence = 70
    elif recruiting > completed and recruiting > 2:
        hint = "neutral"
        confidence = 50

    if recent_updates >= 2:
        confidence = min(85, confidence + 5)

    return AgentMessage(
        agent="trial_progress",
        ticker=ticker,
        company=company,
        summary=(
            f"Trial mix: completed={completed}, recruiting={recruiting}, "
            f"phase2/3={phase_2_or_3}, recent_180d_updates={recent_updates}, total={len(studies)}."
        ),
        confidence=confidence,
        signal_hint=hint,
        evidence=[
            Evidence(
                source="ClinicalTrials.gov",
                title="Trial status summary",
                url="https://clinicaltrials.gov/",
                snippet=(
                    f"completed={completed}, recruiting={recruiting}, "
                    f"phase2/3={phase_2_or_3}, recent_180d={recent_updates}"
                ),
            )
        ],
        tags=["trial_progress"],
    )


def _parse_yyyy_mm(value: str):
    value = value.strip()
    for fmt in ("%Y-%m-%d", "%Y-%m"):
        try:
            parsed = datetime.strptime(value, fmt)
            return parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None
