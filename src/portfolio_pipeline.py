from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.agentic_pm.reasoning_manager import ReasoningManager
from src.agentic_pm.state import ReasoningState
from src.approval.queue import ApprovalQueue, PendingApproval
from src.connectors.market_data import fetch_yahoo_daily_bars
from src.models.messages import AgentMessage, FinalReport
from src.pipeline import run_pipeline
from src.rl.load_config import load_rl_hyperparams
from src.rl.pm_model import compute_pm_weights_preview
from src.risk.metrics import daily_returns_from_prices, max_drawdown_from_equity, unrealized_pnl_pct
from src.risk.trade_executor import TradeExecutor


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def enrich_report_with_pm(
    report: FinalReport,
    rs: ReasoningState,
    messages: List[AgentMessage],
) -> FinalReport:
    lines = [dict(t) for t in (rs.get("timeline") or [])]
    pm = compute_pm_weights_preview(report, messages)

    bars = fetch_yahoo_daily_bars(report.ticker, range_="60d", max_rows=65)
    prices = [float(b["close"]) for b in bars if b.get("close") is not None]
    rets = daily_returns_from_prices(prices)
    executor = TradeExecutor()
    check = executor.evaluate(rets, has_open_risk=report.final_signal != "no_trade")

    mdd = max_drawdown_from_equity(prices) if prices else 0.0
    u_pnl = unrealized_pnl_pct(prices[0], prices[-1], "long") if len(prices) >= 2 else 0.0

    risk_line = (
        f"{check.decision.value} | VaR95≈{check.daily_var_95:.4f} | "
        f"trail_dd≈{mdd:.2%} | uPnL≈{u_pnl:.2f}% (60d)"
    )

    return report.model_copy(
        update={
            "reasoning_trace": lines,
            "counter_thesis": rs.get("counter_thesis", ""),
            "pm_weights_preview": pm,
            "risk_status": risk_line,
        }
    )


def _json_safe(obj: Any) -> Any:
    return json.loads(json.dumps(obj, default=str))


def write_pm_dashboard_state(
    report: FinalReport,
    rs: ReasoningState,
    *,
    ppo_episode_returns_tail: Optional[List[float]] = None,
    batch_runs: Optional[List[Dict[str, Any]]] = None,
    universe_snapshot: Optional[List[Dict[str, Any]]] = None,
    catalyst_calendar: Optional[List[Dict[str, Any]]] = None,
    institutional_scorecard: Optional[List[Dict[str, Any]]] = None,
) -> Path:
    """JSON for Next.js PM dashboard (`web/app/api/state`)."""
    out = _project_root() / "outputs" / "pm_dashboard_state.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    rl_cfg = load_rl_hyperparams()
    payload: Dict[str, Any] = {
        "system_tag": report.system_tag,
        "ticker": report.ticker,
        "company": report.company,
        "confidence": report.confidence,
        "final_signal": report.final_signal,
        "counter_thesis": report.counter_thesis,
        "pm_weights_preview": report.pm_weights_preview,
        "risk_status": report.risk_status,
        "reasoning_timeline": rs.get("timeline") or [],
        "rl_hyperparams": {
            "learning_rate": rl_cfg.learning_rate,
            "entropy_coef": rl_cfg.entropy_coef,
            "gamma": rl_cfg.gamma,
            "reward_clip": rl_cfg.reward_clip,
        },
        "rl_training": {
            "status": "idle",
            "last_episode_returns_tail": ppo_episode_returns_tail or [],
            "note": "Run `python -m scripts.train_ppo_demo` to refresh PPO metrics.",
        },
        "hitl": {
            "trade_id": report.trade_id,
            "execution_status": report.execution_status,
            "approval_required": True,
        },
    }
    if batch_runs is not None:
        payload["batch_runs"] = batch_runs
        payload["batch_note"] = "Multi-ticker demo; detail timeline is from the primary ticker row above."
    if universe_snapshot is not None:
        payload["universe"] = _json_safe(universe_snapshot)
    if catalyst_calendar is not None:
        payload["catalyst_calendar"] = _json_safe(catalyst_calendar)
    if institutional_scorecard is not None:
        payload["institutional_scorecard"] = _json_safe(institutional_scorecard)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return out


def run_portfolio_pm_cycle(
    ticker: str,
    company: str,
    cash_runway_months: int = 18,
    single_asset_exposure: bool = False,
    *,
    write_dashboard: bool = True,
) -> Tuple[List[AgentMessage], FinalReport, Dict[str, Any], ReasoningState]:
    """
    Full Agentic PM path: ReasoningManager (plan/tools/memory/critic) → main signal pipeline
    → enrich report → approval queue → dashboard JSON.
    """
    rm = ReasoningManager()
    rs = rm.run(ticker, company)
    messages, report, raw = run_pipeline(
        ticker,
        company,
        cash_runway_months=cash_runway_months,
        single_asset_exposure=single_asset_exposure,
    )
    report = enrich_report_with_pm(report, rs, messages)

    risk_snap = {
        "risk_status": report.risk_status,
        "counter_thesis_excerpt": (report.counter_thesis or "")[:280],
    }
    q = ApprovalQueue()
    q.enqueue(
        PendingApproval(
            trade_id=report.trade_id,
            ticker=report.ticker,
            company=report.company,
            final_signal=report.final_signal,
            confidence=report.confidence,
            counter_thesis=report.counter_thesis,
            reasoning_trace=report.reasoning_trace,
            risk_snapshot=risk_snap,
        )
    )
    if write_dashboard:
        write_pm_dashboard_state(report, rs)
    return messages, report, raw, rs
