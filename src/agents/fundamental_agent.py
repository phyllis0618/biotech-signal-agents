from __future__ import annotations

import json
from typing import Dict, Tuple

from src.connectors.sec_edgar import build_fundamental_context, fetch_latest_8k_plain_text
from src.fundamentals.llm_extract import extract_json_with_llm
from src.models.fundamental_schemas import (
    CashRunwayJson,
    ClinicalResultsJson,
    DealFlowJson,
    MarketTAMJson,
    RegulatoryCatalystJson,
)
from src.models.messages import AgentMessage, Evidence
from src.prompts.fundamental_prompts import (
    PROMPT_CASH_RUNWAY,
    PROMPT_CLINICAL_RESULTS,
    PROMPT_DEAL_FLOW,
    PROMPT_MARKET_TAM,
    PROMPT_REGULATORY_CATALYST,
)


def _aggregate_fundamental_signal(
    clinical: ClinicalResultsJson,
    regulatory: RegulatoryCatalystJson,
    cash: CashRunwayJson,
    deal: DealFlowJson,
) -> Tuple[str, int]:
    score = 0
    cs = (clinical.status or "").strip().lower()
    if cs == "positive" and clinical.primary_endpoint_met:
        score += 35
    elif cs == "negative":
        score -= 35
    elif cs == "mixed":
        score -= 5

    et = (regulatory.event_type or "").upper()
    if et == "CRL":
        score -= 40
    elif et == "PDUFA" and regulatory.milestone_date:
        score += 5
    if "favor" in (regulatory.adcom_sentiment or "").lower():
        score += 15

    dr = (cash.dilution_risk or "").strip().lower()
    if dr == "high":
        score -= 25
    elif dr == "low":
        score += 5
    rq = cash.runway_quarters
    if rq is not None and rq < 3:
        score -= 25

    dt = (deal.type or "").strip()
    if dt == "Acquisition":
        score += 20
    elif dt == "Licensing" and (deal.total_value or 0) > 0:
        score += 10

    if score > 15:
        return "bullish", min(88, 55 + abs(score) // 3)
    if score < -15:
        return "bearish", min(88, 55 + abs(score) // 3)
    return "neutral", 52


def run_fundamental_agent(ticker: str, company: str, raw_data: Dict) -> AgentMessage:
    ctgov = raw_data.get("ctgov_studies", [])
    fda_cal = raw_data.get("fda_calendar_events", [])
    filing_text = fetch_latest_8k_plain_text(ticker)
    user_text = build_fundamental_context(ticker, company, ctgov, fda_cal, filing_text)

    clinical = extract_json_with_llm(PROMPT_CLINICAL_RESULTS, user_text, ClinicalResultsJson)
    regulatory = extract_json_with_llm(PROMPT_REGULATORY_CATALYST, user_text, RegulatoryCatalystJson)
    cash = extract_json_with_llm(PROMPT_CASH_RUNWAY, user_text, CashRunwayJson)
    deal = extract_json_with_llm(PROMPT_DEAL_FLOW, user_text, DealFlowJson)
    tam = extract_json_with_llm(PROMPT_MARKET_TAM, user_text, MarketTAMJson)

    hint, confidence = _aggregate_fundamental_signal(clinical, regulatory, cash, deal)

    payload = {
        "clinical": clinical.model_dump(),
        "regulatory": regulatory.model_dump(),
        "cash_runway": cash.model_dump(),
        "deal_flow": deal.model_dump(),
        "market_tam": tam.model_dump(),
        "filing_chars": len(filing_text),
    }

    summary = (
        f"Fundamentals: clinical={clinical.status}, reg={regulatory.event_type}, "
        f"dilution={cash.dilution_risk}, deal={deal.type}, tam_est={tam.estimated_tam}."
    )

    evidence = [
        Evidence(
            source="Fundamental LLM",
            title="Structured fundamentals JSON",
            url="internal://fundamentals",
            snippet=json.dumps(payload, ensure_ascii=True)[:900],
        )
    ]
    if filing_text:
        evidence.append(
            Evidence(
                source="SEC EDGAR",
                title="Latest 8-K excerpt used as context",
                url=f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=8-K",
                snippet=f"chars={len(filing_text)}",
            )
        )

    return AgentMessage(
        agent="fundamental",
        ticker=ticker,
        company=company,
        summary=summary,
        confidence=confidence,
        signal_hint=hint,  # type: ignore[arg-type]
        evidence=evidence,
        tags=["fundamental", "clinical", "regulatory", "cash", "deals", "tam"],
    )
