from __future__ import annotations

import json
from typing import Tuple

from src.models.messages import AgentMessage, Evidence, FinalReport
from src.trading.feedback_store import new_trade_id


def run_trader_review_agent(
    rl_report: FinalReport,
    rl_policy_msg: AgentMessage,
) -> Tuple[AgentMessage, FinalReport]:
    """
    Human-in-the-loop gate: marks execution as pending trader review.
    Actual approve/reject happens via scripts/apply_trader_feedback.py or dashboard.
    """
    tid = new_trade_id()
    report = rl_report.model_copy(
        update={
            "trade_id": tid,
            "execution_status": "pending_trader_review",
            "trader_guidance": None,
            "key_points": rl_report.key_points
            + [
                f"Trade id {tid} awaits trader review before any execution.",
                "RL action is advisory; trader may override with guidance in feedback tool.",
            ],
        }
    )

    msg = AgentMessage(
        agent="trader_review",
        ticker=report.ticker,
        company=report.company,
        summary=(
            f"Trader gate: status=pending_trader_review, trade_id={tid}, "
            f"proposed={report.final_signal} @ {report.confidence}."
        ),
        confidence=report.confidence,
        signal_hint="neutral",
        evidence=[
            Evidence(
                source="TraderReview",
                title="Execution blocked until human approval",
                url="internal://trader-review",
                snippet=json.dumps(
                    {
                        "trade_id": tid,
                        "proposed_signal": report.final_signal,
                        "rl_state": report.rl_state,
                        "rl_action": report.rl_action,
                    },
                    ensure_ascii=True,
                )[:800],
            )
        ],
        tags=["human_in_the_loop", "execution_gate"],
    )
    return msg, report
