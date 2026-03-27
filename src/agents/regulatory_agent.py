from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.models.messages import AgentMessage, Evidence


def run_regulatory_agent(ticker: str, company: str, raw_data: dict) -> AgentMessage:
    fda_events = raw_data.get("fda_events", [])
    serious_cases = 0
    recent_serious_cases = 0
    window_start = datetime.now(timezone.utc) - timedelta(days=180)

    for item in fda_events:
        reactions = item.get("patient", {}).get("reaction", [])
        event_date = _parse_yyyymmdd(str(item.get("receiptdate", "")))
        for reaction in reactions:
            outcome = str(reaction.get("reactionmeddrapt", "")).lower()
            if any(key in outcome for key in ["death", "hospital", "life"]):
                serious_cases += 1
                if event_date and event_date >= window_start:
                    recent_serious_cases += 1

    hint = "neutral"
    confidence = 52
    if recent_serious_cases >= 2 or serious_cases >= 4:
        hint = "bearish"
        confidence = 72

    return AgentMessage(
        agent="regulatory",
        ticker=ticker,
        company=company,
        summary=(
            f"Regulatory safety scan found serious={serious_cases}, "
            f"recent_180d_serious={recent_serious_cases} in sampled FDA events."
        ),
        confidence=confidence,
        signal_hint=hint,
        evidence=[
            Evidence(
                source="openFDA",
                title="Safety event reaction scan",
                url="https://api.fda.gov/drug/event.json",
                snippet=f"serious_mentions={serious_cases}, recent_180d={recent_serious_cases}",
            )
        ],
        tags=["regulatory", "safety"],
    )


def _parse_yyyymmdd(value: str):
    value = value.strip()
    if len(value) != 8 or not value.isdigit():
        return None
    try:
        parsed = datetime.strptime(value, "%Y%m%d")
        return parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None
