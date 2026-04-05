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


_DASHBOARD_SLICE_KEYS = (
    "ticker",
    "company",
    "confidence",
    "final_signal",
    "counter_thesis",
    "pm_weights_preview",
    "risk_status",
    "reasoning_timeline",
    "rl_deployed",
    "hitl",
)


def dashboard_slice_for_report(report: FinalReport, rs: ReasoningState) -> Dict[str, Any]:
    """Per-ticker snapshot for `by_ticker` in pm_dashboard_state.json (Next.js merges by workspace ticker)."""
    return {
        "ticker": report.ticker,
        "company": report.company,
        "confidence": report.confidence,
        "final_signal": report.final_signal,
        "counter_thesis": report.counter_thesis,
        "pm_weights_preview": report.pm_weights_preview,
        "risk_status": report.risk_status,
        "reasoning_timeline": rs.get("timeline") or [],
        "rl_deployed": {
            "state_key": getattr(report, "rl_state", "") or "",
            "policy_action": getattr(report, "rl_action", "") or "",
            "final_signal_after_pipeline": report.final_signal,
            "source_table": "outputs/rl_qtable.json",
            "forward_path": "run_pipeline → run_rl_policy_agent → trader_review",
        },
        "hitl": {
            "trade_id": report.trade_id,
            "execution_status": report.execution_status,
            "approval_required": True,
        },
    }


def _apply_dashboard_slice(payload: Dict[str, Any], sl: Dict[str, Any]) -> None:
    for k in _DASHBOARD_SLICE_KEYS:
        if k in sl:
            payload[k] = sl[k]


def _load_tabular_rl_training_snapshot() -> Dict[str, Any]:
    """
    Surfaces overnight TD Q backtest (`outputs/overnight_train_report.json`) for the dashboard.
    Forward execution uses the same `outputs/rl_qtable.json` via `run_rl_policy_agent`.
    """
    root = _project_root()
    report_path = root / "outputs" / "overnight_train_report.json"
    q_path = root / "outputs" / "rl_qtable.json"
    snap: Dict[str, Any] = {
        "flow_summary": (
            "100d Yahoo: TD Q on first 70 return days → outputs/rl_qtable.json (or rl_qtable_<TICKER>.json with "
            "--multi-demo). overnight_train_report.json may include by_ticker for the same three symbols as the dashboard. "
            "PnL OOS = last 30 real days; live: coordinator → rl_policy_agent → trader_review → execution."
        ),
        "train_script": "scripts/train_tabular_q_overnight.py",
        "report_file": "outputs/overnight_train_report.json",
        "q_table_file": "outputs/rl_qtable.json",
        "report_available": report_path.exists(),
        "q_table_available": q_path.exists(),
    }
    if report_path.exists():
        try:
            snap["overnight_report"] = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception as e:
            snap["overnight_report"] = None
            snap["report_load_error"] = str(e)
    if q_path.exists():
        try:
            q = json.loads(q_path.read_text(encoding="utf-8"))
            snap["q_state_count"] = len(q) if isinstance(q, dict) else 0
        except Exception:
            snap["q_state_count"] = None
    return snap


def write_pm_dashboard_state(
    report: FinalReport,
    rs: ReasoningState,
    *,
    ppo_episode_returns_tail: Optional[List[float]] = None,
    batch_runs: Optional[List[Dict[str, Any]]] = None,
    universe_snapshot: Optional[List[Dict[str, Any]]] = None,
    catalyst_calendar: Optional[List[Dict[str, Any]]] = None,
    institutional_scorecard: Optional[List[Dict[str, Any]]] = None,
    by_ticker: Optional[Dict[str, Dict[str, Any]]] = None,
    default_ticker: Optional[str] = None,
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
            "note": (
                "PPO (Gym, config/rl_config.json) is separate from tabular Q. "
                "Deployed tabular policy: train with scripts/train_tabular_q_overnight.py → rl_qtable.json; "
                "see tabular_rl_training on this dashboard."
            ),
        },
        "tabular_rl_training": _load_tabular_rl_training_snapshot(),
        "rl_deployed": {
            "state_key": getattr(report, "rl_state", "") or "",
            "policy_action": getattr(report, "rl_action", "") or "",
            "final_signal_after_pipeline": report.final_signal,
            "source_table": "outputs/rl_qtable.json",
            "forward_path": "run_pipeline → run_rl_policy_agent → trader_review",
        },
        "hitl": {
            "trade_id": report.trade_id,
            "execution_status": report.execution_status,
            "approval_required": True,
        },
    }
    if by_ticker:
        payload["by_ticker"] = by_ticker
        dt = default_ticker
        if not dt and by_ticker:
            dt = next(iter(by_ticker.keys()))
        if dt and dt in by_ticker:
            payload["default_ticker"] = dt
            _apply_dashboard_slice(payload, by_ticker[dt])
        payload["batch_note"] = (
            "Multi-ticker demo: each symbol has its own Agentic timeline in `by_ticker`; "
            "top-level fields mirror `default_ticker` (first successful in demo order when set)."
        )
    if batch_runs is not None:
        payload["batch_runs"] = batch_runs
        if not by_ticker:
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
