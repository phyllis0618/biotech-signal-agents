from __future__ import annotations

"""
Fallback when LangGraph is not installed: same semantics, sequential execution.
"""

from typing import Any, Dict

from src.agentic_research.critic_agent import CriticAgent
from src.agentic_research.data_extractor import DataExtractor
from src.agentic_research.financial_analyst import FinancialAnalyst
from src.agentic_research.state import ResearchState


def run_research_fsm(ticker: str, company: str) -> ResearchState:
    state: ResearchState = {"ticker": ticker, "company": company, "errors": []}

    plan = (
        f"1) Pull CTGov + FDA path for {company} ({ticker}). "
        f"2) Parse SEC liquidity. "
        f"3) Critic reflects on bullish bias. "
        f"4) Emit alpha with risk-adjusted view."
    )
    state["plan"] = plan

    de = DataExtractor()
    fa = FinancialAnalyst()
    state["extraction"] = de.run(ticker, company)
    state["financials"] = fa.run(ticker, company)
    state["research_bundle"] = {
        "extraction": state["extraction"],
        "financials": state["financials"],
    }

    critic = CriticAgent()
    state["reflection"] = critic.run(state["research_bundle"])

    ext = state["extraction"]
    fin = state["financials"]
    ref = state["reflection"]

    phase_boost = min(0.3, 0.05 * ext.get("phase2_or_3_active", 0))
    liq_pen = 0.0
    rw = fin.get("runway_months_est")
    if rw is not None and rw < 12:
        liq_pen = 0.25
    critic_pen = float(ref.get("bull_case_undercut_score", 0.0))

    raw_alpha = 0.5 + phase_boost - liq_pen - critic_pen
    signal_strength = max(-1.0, min(1.0, (raw_alpha - 0.5) * 2.0))

    state["final_alpha"] = {
        "signal_strength": round(signal_strength, 4),
        "rationale": {
            "phase2_3_active": ext.get("phase2_or_3_active"),
            "runway_months_est": rw,
            "critic_penalty": critic_pen,
        },
        "disclaimer": "Research-only; not investment advice.",
    }
    return state
