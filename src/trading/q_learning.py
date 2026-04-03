from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

Q_PATH = Path(os.getenv("RL_QTABLE_PATH", "outputs/rl_qtable.json"))


def _default_q() -> Dict[str, Dict[str, float]]:
    return {}


def load_q_table() -> Dict[str, Dict[str, float]]:
    if not Q_PATH.exists():
        return _default_q()
    try:
        data = json.loads(Q_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else _default_q()
    except Exception:
        return _default_q()


def save_q_table(table: Dict[str, Dict[str, float]]) -> None:
    Q_PATH.parent.mkdir(parents=True, exist_ok=True)
    Q_PATH.write_text(json.dumps(table, indent=2), encoding="utf-8")


def state_key(signal: str, confidence: int) -> str:
    bucket = max(0, min(9, confidence // 10))
    return f"{signal}|{bucket}"


def available_actions() -> List[str]:
    return ["long", "short", "no_trade"]


def select_action(state: str, epsilon: float = 0.12) -> str:
    q = load_q_table()
    actions = available_actions()
    row = q.get(state, {})
    if not row or random.random() < epsilon:
        return random.choice(actions)
    best_a = max(actions, key=lambda a: row.get(a, 0.0))
    return best_a


def q_values_for_state(state: str) -> Dict[str, float]:
    q = load_q_table()
    row = q.get(state, {})
    return {a: float(row.get(a, 0.0)) for a in available_actions()}


def update_q(
    state: str,
    action: str,
    reward: float,
    learning_rate: float = 0.35,
    discount: float = 0.85,
) -> None:
    """Tabular Q update for (state, action). Next-state bootstrap omitted (bandit-style)."""
    q = load_q_table()
    if state not in q:
        q[state] = {a: 0.0 for a in available_actions()}
    old = q[state].get(action, 0.0)
    target = reward
    q[state][action] = old + learning_rate * (target - old)
    save_q_table(q)


def apply_trader_reward(
    state: str,
    action_taken: str,
    decision: str,
    guidance_weight: float = 0.5,
) -> None:
    """
    Map trader decision to reward for RL update.
    decision: approved | rejected | deferred
    """
    if decision == "approved":
        reward = 1.0
    elif decision == "rejected":
        reward = -1.0
    else:
        reward = guidance_weight * 0.0
    update_q(state, action_taken, reward)
