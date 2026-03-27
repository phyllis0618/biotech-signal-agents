from __future__ import annotations

from src.models.messages import AgentMessage, Evidence


def run_market_impact_agent(
    ticker: str,
    company: str,
    trial_msg: AgentMessage,
    reg_msg: AgentMessage,
    raw_data: dict,
    risk_profile: dict,
) -> AgentMessage:
    score = 0
    if trial_msg.signal_hint == "bullish":
        score += 1
    if reg_msg.signal_hint == "bearish":
        score -= 1

    runway_months = int(risk_profile.get("cash_runway_months", 18))
    single_asset = bool(risk_profile.get("single_asset_exposure", False))
    pureglobal_metrics = raw_data.get("pureglobal_metrics", {})
    phase3_mentions = int(pureglobal_metrics.get("phase_3_mentions", 0))

    if runway_months < 12:
        score -= 1
    if single_asset:
        score -= 1
    if phase3_mentions > 3:
        score += 1

    hint = "neutral"
    if score > 0:
        hint = "bullish"
    elif score < 0:
        hint = "bearish"

    return AgentMessage(
        agent="market_impact",
        ticker=ticker,
        company=company,
        summary=(
            "Estimated market impact from trial/regulatory and company risk "
            f"(runway_months={runway_months}, single_asset={single_asset}, phase3_mentions={phase3_mentions})."
        ),
        confidence=60,
        signal_hint=hint,
        evidence=[
            Evidence(
                source="Internal",
                title="Impact scoring",
                url="internal://market-impact",
                snippet=(
                    f"score={score}, trial={trial_msg.signal_hint}, reg={reg_msg.signal_hint}, "
                    f"runway_months={runway_months}, single_asset={single_asset}, phase3={phase3_mentions}"
                ),
            )
        ],
        tags=["market_impact"],
    )
