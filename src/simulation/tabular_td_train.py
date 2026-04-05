"""
TD tabular Q training on momentum-derived states (same as train_tabular_q_overnight).
Used for a fixed train slice (e.g. first 70 of 100 return days).
"""

from __future__ import annotations

import math
import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.trading.q_learning import available_actions, state_key


def momentum_coordinator_style(window: float, mom_threshold: float = 0.002) -> Tuple[str, int]:
    if window > mom_threshold:
        return "long", min(90, 55 + int(abs(window) * 500))
    if window < -mom_threshold:
        return "short", min(90, 55 + int(abs(window) * 500))
    return "no_trade", 48


def position_weight(action: str) -> float:
    if action == "long":
        return 1.0
    if action == "short":
        return -1.0
    return 0.0


def session_for_return_index(
    rets: List[float],
    dates: List[str],
    lookback: int,
    mom_threshold: float,
    i: int,
) -> Dict[str, Any]:
    """One training row: return index i >= lookback."""
    window = sum(rets[i - lookback : i])
    cs, cc = momentum_coordinator_style(window, mom_threshold)
    return {
        "date": dates[i + 1],
        "daily_return": float(rets[i]),
        "coordinator_signal": cs,
        "coordinator_confidence": int(cc),
        "momentum_20d": float(window),
    }


def build_train_sessions(
    rets: List[float],
    dates: List[str],
    *,
    lookback: int,
    mom_threshold: float,
    train_days: int,
) -> List[Dict[str, Any]]:
    """Sessions for TD: return indices lookback .. train_days-1."""
    out: List[Dict[str, Any]] = []
    for i in range(lookback, train_days):
        out.append(session_for_return_index(rets, dates, lookback, mom_threshold, i))
    return out


def _ensure_state_row(q: Dict[str, Dict[str, float]], s: str) -> None:
    if s not in q:
        q[s] = {a: 0.0 for a in available_actions()}


def max_q(q: Dict[str, Dict[str, float]], s: str) -> float:
    _ensure_state_row(q, s)
    row = q[s]
    return max(float(row.get(a, 0.0)) for a in available_actions())


def select_epsilon_greedy(
    q: Dict[str, Dict[str, float]],
    s: str,
    epsilon: float,
    rng: random.Random,
) -> str:
    actions = available_actions()
    _ensure_state_row(q, s)
    if rng.random() < epsilon:
        return rng.choice(actions)
    row = q[s]
    return max(actions, key=lambda a: float(row.get(a, 0.0)))


def td_update(
    q: Dict[str, Dict[str, float]],
    s: str,
    a: str,
    r: float,
    s_next: Optional[str],
    *,
    lr: float,
    gamma: float,
    terminal: bool,
) -> None:
    _ensure_state_row(q, s)
    bootstrap = 0.0 if terminal else gamma * max_q(q, s_next or "")
    target = r + bootstrap
    old = float(q[s].get(a, 0.0))
    q[s][a] = old + lr * (target - old)


def train_td_q_on_sessions(
    sessions: List[Dict[str, Any]],
    *,
    episodes: int,
    lr: float,
    gamma: float,
    epsilon_start: float,
    epsilon_min: float,
    epsilon_decay: float,
    reward_scale: float,
    seed: int,
    q_init: Optional[Dict[str, Dict[str, float]]] = None,
    eval_every: int = 0,
    eval_callback: Optional[Callable[[int, Dict[str, Dict[str, float]]], None]] = None,
    max_seconds: float = 0.0,
) -> Tuple[Dict[str, Dict[str, float]], int]:
    """In-place TD Q-learning over the given session list (one walk per episode)."""
    rng = random.Random(seed)
    q: Dict[str, Dict[str, float]] = dict(q_init) if q_init else {}
    n = len(sessions)
    if n == 0:
        return q, 0
    eps = epsilon_start
    t_wall = time.perf_counter()
    completed = 0
    for ep in range(episodes):
        if max_seconds > 0 and (time.perf_counter() - t_wall) >= max_seconds:
            break
        for t in range(n):
            row = sessions[t]
            sig = str(row["coordinator_signal"])
            conf = int(row["coordinator_confidence"])
            s = state_key(sig, conf)
            dr = float(row["daily_return"])
            a = select_epsilon_greedy(q, s, eps, rng)
            r = reward_scale * position_weight(a) * dr
            terminal = t == n - 1
            if terminal:
                td_update(q, s, a, r, None, lr=lr, gamma=gamma, terminal=True)
            else:
                nxt = sessions[t + 1]
                s_next = state_key(str(nxt["coordinator_signal"]), int(nxt["coordinator_confidence"]))
                td_update(q, s, a, r, s_next, lr=lr, gamma=gamma, terminal=False)
        eps = max(epsilon_min, eps * epsilon_decay)
        completed = ep + 1
        if eval_every > 0 and eval_callback and (ep + 1) % eval_every == 0:
            eval_callback(ep + 1, q)
    return q, completed


def greedy_equity_on_sessions(
    q: Dict[str, Dict[str, float]],
    sessions: List[Dict[str, Any]],
) -> Tuple[float, float, float]:
    """Greedy policy equity and Sharpe-like on session list."""
    equity = 1.0
    strat_rets: List[float] = []
    actions = available_actions()
    for row in sessions:
        sig = str(row["coordinator_signal"])
        conf = int(row["coordinator_confidence"])
        s = state_key(sig, conf)
        dr = float(row["daily_return"])
        _ensure_state_row(q, s)
        rowq = q[s]
        best_a = max(actions, key=lambda a: float(rowq.get(a, 0.0)))
        r = position_weight(best_a) * dr
        strat_rets.append(r)
        equity *= 1.0 + r

    if len(strat_rets) < 2:
        return equity, 0.0, (equity - 1.0) * 100.0
    m = sum(strat_rets) / len(strat_rets)
    var = sum((x - m) ** 2 for x in strat_rets) / (len(strat_rets) - 1)
    vol = math.sqrt(var) if var > 1e-18 else 1e-8
    sharpe = (m / vol) * math.sqrt(252.0)
    total_pct = (equity - 1.0) * 100.0
    return equity, sharpe, total_pct
