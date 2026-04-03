from __future__ import annotations

import json
from typing import Any, Dict, List

from src.models.messages import AgentMessage, FinalReport


def build_factor_trace(
    messages: List[AgentMessage],
    report: FinalReport,
) -> Dict[str, Any]:
    """
    Human-readable map of how factors/agents interact before execution is gated.
    """
    agent_rows = []
    for m in messages:
        agent_rows.append(
            {
                "step": m.agent,
                "signal_hint": m.signal_hint,
                "confidence": m.confidence,
                "tags": ",".join(m.tags) if m.tags else "",
                "summary_short": (m.summary[:160] + "…") if len(m.summary) > 160 else m.summary,
            }
        )

    edges: List[Dict[str, str]] = [
        {"from": "ingestion", "to": "fundamental / trial / regulatory", "data": "raw_data (FDA, CTGov, PureGlobal, SEC)"},
        {"from": "trial_progress", "to": "market_impact", "data": "trial mix vs regulatory balance"},
        {"from": "regulatory", "to": "market_impact", "data": "safety / FAERS scan"},
        {"from": "fundamental", "to": "signal", "data": "LLM JSON (clinical/reg/cash/deal/TAM); weight ×1.15"},
        {"from": "trial + reg + market", "to": "signal", "data": "weighted vote → signal_hint"},
        {"from": "all upstream", "to": "coordinator", "data": "bull/bear/neutral vote count + confidence blend"},
        {"from": "coordinator", "to": "rl_policy", "data": f"draft: {report.coordinator_signal} @ {report.coordinator_confidence}"},
        {"from": "rl_policy", "to": "trader_review", "data": f"tabular Q state={report.rl_state} → action={report.rl_action}"},
        {"from": "trader_review", "to": "OMS / execution", "data": "BLOCKED until approved (execution_status)"},
    ]

    example_cli = (
        f'python -m scripts.apply_trader_feedback \\\n'
        f'  --trade-id "{report.trade_id}" \\\n'
        f'  --ticker {report.ticker} \\\n'
        f'  --coordinator-signal {report.coordinator_signal or "no_trade"} \\\n'
        f'  --coordinator-confidence {report.coordinator_confidence or 0} \\\n'
        f'  --rl-action {report.rl_action or report.final_signal} \\\n'
        f'  --decision approved \\\n'
        f'  --guidance "Desk: size cap 2% NAV; watch liquidity"'
    )

    human_block = {
        "execution_status": report.execution_status,
        "trade_id": report.trade_id,
        "example_cli": example_cli,
        "how_to_intervene": [
            "1) Copy trade_id from report or table below.",
            "2) Run: python -m scripts.apply_trader_feedback --trade-id <ID> --ticker TICKER "
            "--coordinator-signal <coord> --coordinator-confidence <n> --rl-action <rl> "
            "--decision approved|rejected|deferred --guidance \"your desk notes\"",
            "3) Feedback updates outputs/rl_qtable.json (tabular RL) for future runs.",
        ],
        "no_auto_execution": True,
    }

    rl_block = {
        "layer_1_tabular_q": {
            "role": "After coordinator: discretized state → pick long/short/flat; learns from trader decisions.",
            "persistence": "outputs/rl_qtable.json",
            "code": "src/agents/rl_policy_agent.py, src/trading/q_learning.py",
        },
        "layer_2_ppo_sizing": {
            "role": "Separate Gym env: position sizing (Hold / +5% / +10% / Sell all) on [signal, vol, runway, FDA days].",
            "training": "python -m scripts.train_ppo_demo",
            "bridge": "python -m scripts.run_joint_backtest (research → env)",
            "code": "src/rl/trading_env.py, src/rl/ppo_agent.py",
        },
        "note": "Layer 1 shapes direction; Layer 2 scales exposure. Trader gate is after Layer 1 in the main pipeline.",
    }

    return {
        "factor_agent_table": agent_rows,
        "interaction_edges": edges,
        "final_report_snapshot": {
            "coordinator_signal": report.coordinator_signal,
            "coordinator_confidence": report.coordinator_confidence,
            "after_rl_signal": report.final_signal,
            "after_rl_confidence": report.confidence,
            "rl_state": report.rl_state,
            "rl_q_preview": report.rl_q_preview[:300] if report.rl_q_preview else "",
        },
        "human_intervention": human_block,
        "reinforcement_learning": rl_block,
    }


def factor_trace_json(messages: List[AgentMessage], report: FinalReport) -> str:
    return json.dumps(build_factor_trace(messages, report), indent=2, ensure_ascii=True)
