from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List

from src.agentic_research.critic_agent import CriticAgent
from src.agentic_research.financial_analyst import FinancialAnalyst
from src.memory.context_store import BiotechContextMemory

from src.agentic_pm.state import ReasoningState, TimelineStep
from src.agentic_pm.tools import (
    tool_clinical_trials_fda_bundle,
    tool_price_backtest_proxy,
    tool_sec_material_events,
)


class ReasoningManager:
    """
    Graph-style agentic loop (Plan → Tool Use → Memory → Reflect → Decision).
    Intended for PM-grade audit trails and HITL reasoning traces.
    """

    def __init__(self, memory: BiotechContextMemory | None = None) -> None:
        self._memory = memory or BiotechContextMemory()

    def run(self, ticker: str, company: str) -> ReasoningState:
        timeline: List[TimelineStep] = []
        errors: List[str] = []

        plan_tasks = [
            f"FDA / regulatory path: verify PDUFA / AdCom catalysts for {company}",
            "Phase 2/3 evidence: active registrational trials vs early-stage noise",
            f"Cash runway & financing: SEC 8-K + filings vs trial burn",
            "Price risk: realized vol and drawdown vs proposed alpha",
        ]
        timeline.append(
            {
                "phase": "planning",
                "title": "Multi-step validation plan",
                "detail": "\n".join(f"• {t}" for t in plan_tasks),
                "payload": {"tasks": plan_tasks},
            }
        )

        tool_results: Dict[str, Any] = {}
        try:
            tool_results["sec"] = tool_sec_material_events(ticker)
        except Exception as e:
            errors.append(f"sec_tool:{e}")
            tool_results["sec"] = {"error": str(e)}
        try:
            tool_results["ct_fda"] = tool_clinical_trials_fda_bundle(ticker, company)
        except Exception as e:
            errors.append(f"ct_fda:{e}")
            tool_results["ct_fda"] = {"error": str(e)}
        try:
            tool_results["price"] = tool_price_backtest_proxy(ticker)
        except Exception as e:
            errors.append(f"price:{e}")
            tool_results["price"] = {"error": str(e)}

        timeline.append(
            {
                "phase": "tool_use",
                "title": "Tool calls (SEC, CT.gov/FDA, price backtest)",
                "detail": json.dumps({k: _short(v) for k, v in tool_results.items()}, ensure_ascii=True)[:2000],
                "payload": {"keys": list(tool_results.keys())},
            }
        )

        fa = FinancialAnalyst()
        financials = fa.run(ticker, company)
        ext = (tool_results.get("ct_fda") or {}).get("bundle") or {}
        if not ext and "error" not in str(tool_results.get("ct_fda")):
            ext = {}

        research_bundle = {
            "extraction": ext,
            "financials": financials,
            "sec_excerpt_chars": (tool_results.get("sec") or {}).get("chars", 0),
            "price_risk": tool_results.get("price"),
        }

        mem_query = f"{company} {ticker} biotech clinical catalyst financing"
        memory_hits = self._memory.query_similar(mem_query, k=3)
        timeline.append(
            {
                "phase": "memory",
                "title": "Historical context (vector / keyword store)",
                "detail": f"Retrieved {len(memory_hits)} similar contexts for contrarian screening.",
                "payload": {"hits": memory_hits},
            }
        )

        critic = CriticAgent()
        reflection = critic.run(research_bundle)
        counter_thesis = critic.counter_thesis(reflection, research_bundle, memory_hits)
        timeline.append(
            {
                "phase": "reflection",
                "title": "CriticAgent — counter-thesis",
                "detail": counter_thesis[:1200],
                "payload": {"reflection": reflection},
            }
        )

        phase_boost = min(0.3, 0.05 * int(ext.get("phase2_or_3_active") or 0))
        liq_pen = 0.0
        rw = financials.get("runway_months_est")
        if rw is not None and rw < 12:
            liq_pen = 0.25
        critic_pen = float(reflection.get("bull_case_undercut_score", 0.0))

        mem_pen = min(0.15, 0.05 * len(memory_hits)) if memory_hits else 0.0
        price = tool_results.get("price") or {}
        vol_pen = 0.0
        if isinstance(price, dict) and price.get("realized_vol_daily"):
            vol_pen = min(0.2, float(price["realized_vol_daily"]) * 0.25)

        raw_alpha = 0.5 + phase_boost - liq_pen - critic_pen - mem_pen - vol_pen
        signal_strength = max(-1.0, min(1.0, (raw_alpha - 0.5) * 2.0))

        final_alpha = {
            "signal_strength": round(signal_strength, 4),
            "rationale": {
                "phase2_3_active": ext.get("phase2_or_3_active"),
                "runway_months_est": rw,
                "critic_penalty": critic_pen,
                "memory_penalty": mem_pen,
                "vol_penalty": vol_pen,
            },
            "disclaimer": "Research-only; not investment advice.",
        }

        timeline.append(
            {
                "phase": "decision",
                "title": "Alpha synthesis (pre-execution)",
                "detail": json.dumps(final_alpha, ensure_ascii=True)[:1500],
                "payload": {"final_alpha": final_alpha},
            }
        )

        doc_id = f"{ticker}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        try:
            self._memory.upsert_context(
                doc_id,
                f"{company} | strengths:{final_alpha['signal_strength']} | {counter_thesis[:400]}",
                metadata={"ticker": ticker, "signal": str(final_alpha["signal_strength"])},
            )
        except Exception as e:
            errors.append(f"memory_upsert:{e}")

        state: ReasoningState = {
            "ticker": ticker,
            "company": company,
            "plan_tasks": plan_tasks,
            "tool_results": tool_results,
            "memory_hits": memory_hits,
            "research_bundle": research_bundle,
            "reflection": reflection,
            "counter_thesis": counter_thesis,
            "final_alpha": final_alpha,
            "timeline": timeline,
            "errors": errors,
        }
        return state


def _short(obj: Any, limit: int = 400) -> Any:
    if isinstance(obj, dict):
        return {k: _short(v, 200) for k, v in list(obj.items())[:12]}
    if isinstance(obj, list):
        return [_short(x, 120) for x in obj[:5]]
    if isinstance(obj, str) and len(obj) > limit:
        return obj[:limit] + "…"
    return obj


def reasoning_timeline_json(state: ReasoningState) -> str:
    return json.dumps(state.get("timeline") or [], indent=2, ensure_ascii=True)
