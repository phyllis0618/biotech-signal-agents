from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.trading.q_learning import apply_trader_reward, state_key

LOG_PATH = Path("outputs/trader_feedback.jsonl")


def append_feedback(
    trade_id: str,
    ticker: str,
    coordinator_signal: str,
    coordinator_confidence: int,
    rl_action_taken: str,
    decision: str,
    trader_guidance: Optional[str] = None,
    update_rl: bool = True,
) -> Dict[str, Any]:
    """RL state matches state_key(coordinator output before RL refinement)."""
    st = state_key(coordinator_signal, coordinator_confidence)
    row = {
        "trade_id": trade_id,
        "ticker": ticker,
        "state": st,
        "rl_action_taken": rl_action_taken,
        "decision": decision,
        "trader_guidance": trader_guidance or "",
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")

    if update_rl and decision in ("approved", "rejected", "deferred"):
        apply_trader_reward(st, rl_action_taken, decision)
    return row


def new_trade_id() -> str:
    return str(uuid.uuid4())
