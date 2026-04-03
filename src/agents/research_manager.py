from __future__ import annotations

"""
Research Manager: orchestrates DataExtractor, FinancialAnalyst, CriticAgent
via LangGraph (preferred) or a simple state machine fallback.

Flow: Input -> Plan -> Research -> Reflect -> Final Alpha Signal
"""

import json
from typing import Any, Dict

from src.agentic_research.state import ResearchState


def run_agentic_research(ticker: str, company: str) -> ResearchState:
    try:
        from src.agentic_research.research_graph import run_research_workflow

        return run_research_workflow(ticker, company)
    except Exception:
        from src.agentic_research.research_fsm import run_research_fsm

        return run_research_fsm(ticker, company)


def research_report_json(state: ResearchState) -> str:
    return json.dumps(
        {
            "ticker": state.get("ticker"),
            "company": state.get("company"),
            "plan": state.get("plan"),
            "extraction_summary": _summarize_ext(state.get("extraction", {})),
            "financials": state.get("financials"),
            "reflection": state.get("reflection"),
            "final_alpha": state.get("final_alpha"),
        },
        indent=2,
        ensure_ascii=True,
    )


def _summarize_ext(ext: Dict[str, Any]) -> Dict[str, Any]:
    if not ext:
        return {}
    return {
        "trial_count": ext.get("trial_count"),
        "phase2_or_3_active": ext.get("phase2_or_3_active"),
        "fda_calendar_headlines": ext.get("fda_calendar_headlines", [])[:3],
        "errors": ext.get("errors", []),
    }
