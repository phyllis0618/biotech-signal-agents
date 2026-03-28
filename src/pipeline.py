from __future__ import annotations

from typing import Dict, List, Tuple

from src.agents.coordinator_agent import run_coordinator_agent
from src.agents.fundamental_agent import run_fundamental_agent
from src.agents.ingestion_agent import run_ingestion_agent
from src.agents.market_impact_agent import run_market_impact_agent
from src.agents.regulatory_agent import run_regulatory_agent
from src.agents.signal_agent import run_signal_agent
from src.agents.trial_progress_agent import run_trial_progress_agent
from src.models.messages import AgentMessage, FinalReport


def run_pipeline(
    ticker: str,
    company: str,
    cash_runway_months: int = 18,
    single_asset_exposure: bool = False,
) -> Tuple[List[AgentMessage], FinalReport, Dict]:
    ingestion_msg, raw_data = run_ingestion_agent(ticker, company)
    raw_data["risk_profile"] = {
        "cash_runway_months": cash_runway_months,
        "single_asset_exposure": single_asset_exposure,
    }
    fundamental_msg = run_fundamental_agent(ticker, company, raw_data)
    trial_msg = run_trial_progress_agent(ticker, company, raw_data)
    reg_msg = run_regulatory_agent(ticker, company, raw_data)
    market_msg = run_market_impact_agent(
        ticker,
        company,
        trial_msg,
        reg_msg,
        raw_data,
        {
            "cash_runway_months": cash_runway_months,
            "single_asset_exposure": single_asset_exposure,
        },
    )
    signal_msg = run_signal_agent(
        ticker, company, trial_msg, reg_msg, market_msg, fundamental_msg
    )
    agent_messages = [
        ingestion_msg,
        fundamental_msg,
        trial_msg,
        reg_msg,
        market_msg,
        signal_msg,
    ]
    report = run_coordinator_agent(agent_messages)
    return agent_messages, report, raw_data
