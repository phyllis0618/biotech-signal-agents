from __future__ import annotations

import json
from typing import List, Tuple

from src.models.messages import AgentMessage, Evidence, FinalReport
from src.trading.q_learning import q_values_for_state, select_action, state_key


def run_rl_policy_agent(
    upstream_messages: List[AgentMessage],
    coordinator_report: FinalReport,
) -> Tuple[AgentMessage, FinalReport]:
    """
    Tabular Q-policy over (signal bucket) states. Refines coordinator output.
    Learns from trader feedback via outputs/rl_qtable.json updates.
    """
    pre_sig = coordinator_report.final_signal
    pre_conf = coordinator_report.confidence
    st = state_key(pre_sig, pre_conf)
    action = select_action(st)
    q_prev = q_values_for_state(st)

    conf = pre_conf
    if action == pre_sig:
        conf = min(95, conf + 3)
    else:
        conf = max(25, conf - 8)

    report = coordinator_report.model_copy(
        update={
            "coordinator_signal": pre_sig,
            "coordinator_confidence": pre_conf,
            "final_signal": action,  # type: ignore[arg-type]
            "confidence": conf,
            "rl_state": st,
            "rl_action": action,
            "rl_q_preview": json.dumps(q_prev, ensure_ascii=True)[:500],
        }
    )

    msg = AgentMessage(
        agent="rl_policy",
        ticker=report.ticker,
        company=report.company,
        summary=(
            f"RL policy: state={st}, selected_action={action}, "
            f"Q_preview={q_prev}. Coordinator was {pre_sig}@{pre_conf}."
        ),
        confidence=conf,
        signal_hint="bullish"
        if action == "long"
        else ("bearish" if action == "short" else "neutral"),
        evidence=[
            Evidence(
                source="RL",
                title="Tabular Q selection",
                url="internal://rl-policy",
                snippet=f"state={st}, action={action}, q={q_prev}",
            )
        ],
        tags=["reinforcement_learning", "tabular_q"],
    )
    return msg, report
