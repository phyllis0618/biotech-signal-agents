from __future__ import annotations

from src.models.messages import AgentMessage, FinalReport
from src.prompts.coordinator_prompt import COORDINATOR_SYSTEM_PROMPT


def _to_final_signal(hint: str) -> str:
    if hint == "bullish":
        return "long"
    if hint == "bearish":
        return "short"
    return "no_trade"


def _pick_horizon(messages: list[AgentMessage]) -> str:
    text = " ".join(m.summary.lower() for m in messages)
    if "recent_180d" in text or "phase2/3" in text:
        return "1w"
    if "no strong" in text:
        return "1m"
    return "1w"


def run_coordinator_agent(messages: list[AgentMessage]) -> FinalReport:
    if not messages:
        raise ValueError("messages cannot be empty")

    ticker = messages[0].ticker
    company = messages[0].company

    bullish = sum(1 for m in messages if m.signal_hint == "bullish")
    bearish = sum(1 for m in messages if m.signal_hint == "bearish")
    neutral = sum(1 for m in messages if m.signal_hint == "neutral")

    if bullish > bearish and bullish >= 2:
        final_hint = "bullish"
    elif bearish > bullish and bearish >= 2:
        final_hint = "bearish"
    else:
        final_hint = "neutral"

    base_confidence = int(sum(m.confidence for m in messages) / len(messages))
    disagreement_penalty = 10 if bullish > 0 and bearish > 0 else 0
    confidence = max(20, min(90, base_confidence - disagreement_penalty))

    evidence = []
    for m in messages:
        evidence.extend(m.evidence[:1])

    key_points = [
        f"Coordinator prompt policy loaded ({len(COORDINATOR_SYSTEM_PROMPT)} chars).",
        f"Agent votes: bullish={bullish}, bearish={bearish}, neutral={neutral}.",
        "Confidence is penalized when agent views conflict.",
    ]

    risk_flags = []
    if disagreement_penalty:
        risk_flags.append("Agent disagreement on direction.")
    if confidence < 45:
        risk_flags.append("Low confidence signal quality.")
    if final_hint == "neutral":
        risk_flags.append("No strong directional consensus.")

    return FinalReport(
        ticker=ticker,
        company=company,
        final_signal=_to_final_signal(final_hint),  # type: ignore[arg-type]
        confidence=confidence,
        horizon=_pick_horizon(messages),  # type: ignore[arg-type]
        key_points=key_points,
        risk_flags=risk_flags,
        evidence=evidence,
    )
