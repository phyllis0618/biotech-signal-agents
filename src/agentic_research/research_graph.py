from __future__ import annotations

from typing import Any, Dict

from src.agentic_research.critic_agent import CriticAgent
from src.agentic_research.data_extractor import DataExtractor
from src.agentic_research.financial_analyst import FinancialAnalyst
from src.agentic_research.state import ResearchState


def _plan(state: ResearchState) -> Dict[str, Any]:
    t = state.get("ticker", "")
    c = state.get("company", "")
    plan = (
        f"1) Pull CTGov + FDA path for {c} ({t}). "
        f"2) Parse SEC liquidity. "
        f"3) Critic reflects on bullish bias. "
        f"4) Emit alpha with risk-adjusted view."
    )
    return {"plan": plan}


def _research(state: ResearchState) -> Dict[str, Any]:
    ticker = state["ticker"]
    company = state["company"]
    de = DataExtractor()
    fa = FinancialAnalyst()
    extraction = de.run(ticker, company)
    financials = fa.run(ticker, company)
    bundle = {"extraction": extraction, "financials": financials}
    return {"extraction": extraction, "financials": financials, "research_bundle": bundle}


def _reflect(state: ResearchState) -> Dict[str, Any]:
    bundle = state.get("research_bundle") or {}
    critic = CriticAgent()
    reflection = critic.run(bundle)
    return {"reflection": reflection}


def _finalize(state: ResearchState) -> Dict[str, Any]:
    ext = state.get("extraction", {})
    fin = state.get("financials", {})
    ref = state.get("reflection", {})

    phase_boost = min(0.3, 0.05 * ext.get("phase2_or_3_active", 0))
    liq_pen = 0.0
    rw = fin.get("runway_months_est")
    if rw is not None and rw < 12:
        liq_pen = 0.25
    critic_pen = float(ref.get("bull_case_undercut_score", 0.0))

    raw_alpha = 0.5 + phase_boost - liq_pen - critic_pen
    signal_strength = max(-1.0, min(1.0, (raw_alpha - 0.5) * 2.0))

    final_alpha = {
        "signal_strength": round(signal_strength, 4),
        "rationale": {
            "phase2_3_active": ext.get("phase2_or_3_active"),
            "runway_months_est": rw,
            "critic_penalty": critic_pen,
        },
        "disclaimer": "Research-only; not investment advice.",
    }
    return {"final_alpha": final_alpha}


def build_research_graph():
    """LangGraph: Plan -> Research -> Reflect -> Final Alpha."""
    from langgraph.graph import StateGraph, END

    g = StateGraph(ResearchState)
    g.add_node("plan", _plan)
    g.add_node("research", _research)
    g.add_node("reflect", _reflect)
    g.add_node("finalize", _finalize)

    g.set_entry_point("plan")
    g.add_edge("plan", "research")
    g.add_edge("research", "reflect")
    g.add_edge("reflect", "finalize")
    g.add_edge("finalize", END)

    return g.compile()


def run_research_workflow(ticker: str, company: str) -> ResearchState:
    graph = build_research_graph()
    out = graph.invoke({"ticker": ticker, "company": company, "errors": []})
    return out  # type: ignore[return-value]
