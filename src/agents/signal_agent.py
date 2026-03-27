from __future__ import annotations

from src.models.messages import AgentMessage, Evidence


def run_signal_agent(
    ticker: str,
    company: str,
    trial_msg: AgentMessage,
    reg_msg: AgentMessage,
    market_msg: AgentMessage,
) -> AgentMessage:
    hints = [trial_msg.signal_hint, reg_msg.signal_hint, market_msg.signal_hint]
    weights = [trial_msg.confidence, reg_msg.confidence, market_msg.confidence]
    bullish = hints.count("bullish")
    bearish = hints.count("bearish")
    weighted_score = 0
    for hint, weight in zip(hints, weights):
        if hint == "bullish":
            weighted_score += weight
        elif hint == "bearish":
            weighted_score -= weight

    if weighted_score > 10:
        hint = "bullish"
    elif weighted_score < -10:
        hint = "bearish"
    else:
        hint = "neutral"

    confidence = 50 + min(30, abs(weighted_score) // 6)

    return AgentMessage(
        agent="signal",
        ticker=ticker,
        company=company,
        summary=f"Signal vote bullish={bullish}, bearish={bearish}, weighted_score={weighted_score}.",
        confidence=min(int(confidence), 88),
        signal_hint=hint,
        evidence=[
            Evidence(
                source="Internal",
                title="Signal vote",
                url="internal://signal-vote",
                snippet=f"hints={hints}, weights={weights}, weighted_score={weighted_score}",
            )
        ],
        tags=["signal"],
    )
