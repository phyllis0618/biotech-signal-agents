"""
100-day (or configurable) PnL simulation aligned with tabular RL training.

In-sample: same momentum→coordinator-style state as `train_tabular_q_overnight.py` →
greedy action from `outputs/rl_qtable.json` (fallback to raw momentum rule if state missing).

Optional Monte Carlo helpers (legacy): `simulate_forward_pnl`, `simulate_forward_tabular_q`.

**Primary product path** (see `simulate_strategy_pnl.py`): train TD Q on the first N return days,
then **real** OOS forward on the remaining days — no simulated future prices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.trading.q_learning import available_actions, load_q_table, state_key


@dataclass
class BacktestResult:
    dates: List[str]
    closes: List[float]
    daily_returns: List[float]
    positions: List[float]
    equity: List[float]
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_like: float


@dataclass
class ForwardSimResult:
    labels: List[str]
    equity: List[float]
    cumulative_pnl_pct: float
    final_equity: float
    extended_returns: Optional[List[float]] = None
    mode: str = "real_oos"  # "real_oos" | "monte_carlo"


def _max_drawdown_pct(equity: List[float]) -> float:
    peak = equity[0]
    mdd = 0.0
    for e in equity:
        peak = max(peak, e)
        if peak > 1e-12:
            mdd = max(mdd, (peak - e) / peak)
    return float(mdd * 100.0)


def momentum_coordinator_style(window: float, mom_threshold: float = 0.002) -> Tuple[str, int]:
    """Same proxy as build_interactive_sim / train_tabular_q_overnight."""
    if window > mom_threshold:
        return "long", min(90, 55 + int(abs(window) * 500))
    if window < -mom_threshold:
        return "short", min(90, 55 + int(abs(window) * 500))
    return "no_trade", 48


def _momentum_position_rule(window: float, mom_threshold: float) -> float:
    if window > mom_threshold:
        return 1.0
    if window < -mom_threshold:
        return -1.0
    return 0.0


def _action_to_position(action: str) -> float:
    if action == "long":
        return 1.0
    if action == "short":
        return -1.0
    return 0.0


def position_from_tabular_q(
    q: Dict[str, Dict[str, float]],
    window: float,
    mom_threshold: float,
) -> float:
    """Greedy tabular action → {-1,0,1}; fallback to momentum rule if state absent."""
    sig, conf = momentum_coordinator_style(window, mom_threshold)
    st = state_key(sig, conf)
    row = q.get(st)
    if not row:
        return _momentum_position_rule(window, mom_threshold)
    actions = available_actions()
    best_a = max(actions, key=lambda a: float(row.get(a, 0.0)))
    return _action_to_position(best_a)


def run_momentum_backtest(
    dates: List[str],
    closes: List[float],
    *,
    lookback: int = 20,
    mom_threshold: float = 0.002,
    history_days: int = 100,
) -> BacktestResult:
    """
    Use the last `history_days` trading days from dates/closes.
    Position on day t uses sum of previous `lookback` daily returns (no lookahead).
    """
    if len(closes) < history_days:
        raise ValueError(
            f"Not enough bars: have {len(closes)}, need at least {history_days} daily closes"
        )

    dates = dates[-history_days:]
    closes = closes[-history_days:]
    rets: List[float] = []
    for i in range(1, len(closes)):
        rets.append((closes[i] / closes[i - 1]) - 1.0)

    n = len(rets)
    pos: List[float] = [0.0] * n
    for i in range(lookback, n):
        window = sum(rets[i - lookback : i])
        if window > mom_threshold:
            pos[i] = 1.0
        elif window < -mom_threshold:
            pos[i] = -1.0
        else:
            pos[i] = 0.0

    equity: List[float] = [1.0]
    for i in range(n):
        equity.append(equity[-1] * (1.0 + pos[i] * rets[i]))

    tr = (equity[-1] - 1.0) * 100.0
    mdd = _max_drawdown_pct(equity)
    excess = np.array(rets) - np.mean(rets)
    vol = float(np.std(rets)) or 1e-8
    sharpe = float(np.mean(rets) / vol * math.sqrt(252)) if vol > 1e-12 else 0.0

    return BacktestResult(
        dates=dates,
        closes=closes,
        daily_returns=rets,
        positions=pos,
        equity=equity[1:],
        total_return_pct=float(tr),
        max_drawdown_pct=float(mdd),
        sharpe_like=float(sharpe),
    )


def run_tabular_q_backtest(
    dates: List[str],
    closes: List[float],
    *,
    lookback: int = 20,
    mom_threshold: float = 0.002,
    history_days: int = 100,
) -> BacktestResult:
    """
    Same windowing as momentum backtest, but positions come from greedy `outputs/rl_qtable.json`
    (same states as overnight training). Falls back to momentum rule if Q row missing.
    """
    if len(closes) < history_days:
        raise ValueError(
            f"Not enough bars: have {len(closes)}, need at least {history_days} daily closes"
        )

    dates = dates[-history_days:]
    closes = closes[-history_days:]
    rets: List[float] = []
    for i in range(1, len(closes)):
        rets.append((closes[i] / closes[i - 1]) - 1.0)

    q = load_q_table()
    n = len(rets)
    pos: List[float] = [0.0] * n
    for i in range(lookback, n):
        window = sum(rets[i - lookback : i])
        pos[i] = position_from_tabular_q(q, window, mom_threshold)

    equity: List[float] = [1.0]
    for i in range(n):
        equity.append(equity[-1] * (1.0 + pos[i] * rets[i]))

    tr = (equity[-1] - 1.0) * 100.0
    mdd = _max_drawdown_pct(equity)
    vol = float(np.std(rets)) or 1e-8
    sharpe = float(np.mean(rets) / vol * math.sqrt(252)) if vol > 1e-12 else 0.0

    return BacktestResult(
        dates=dates,
        closes=closes,
        daily_returns=rets,
        positions=pos,
        equity=equity[1:],
        total_return_pct=float(tr),
        max_drawdown_pct=float(mdd),
        sharpe_like=float(sharpe),
    )


def simulate_forward_pnl(
    *,
    last_equity: float,
    last_position: float,
    historical_returns: List[float],
    forward_days: int = 30,
    seed: Optional[int] = None,
) -> ForwardSimResult:
    """
    Monte Carlo forward path: daily return ~ N(0, sigma) with sigma = std of recent returns;
    optional bootstrap alternative mixed in for fat tails.
    """
    rng = np.random.default_rng(seed)
    rets_hist = np.array(historical_returns[-60:], dtype=np.float64)
    sigma = float(np.std(rets_hist)) if len(rets_hist) > 1 else 0.02
    sigma = max(sigma, 0.005)

    eq = float(last_equity)
    pos = float(last_position)
    curve = [eq]
    labels: List[str] = []

    for t in range(forward_days):
        # blend Gaussian noise with random historical day (bootstrap)
        z = rng.normal(0.0, sigma)
        if len(rets_hist) > 0:
            boot = float(rng.choice(rets_hist))
            r = 0.6 * z + 0.4 * boot
        else:
            r = z
        pos = float(np.clip(pos * 0.98 + 0.02 * np.sign(r), -1.0, 1.0))
        eq *= 1.0 + pos * r
        curve.append(eq)
        labels.append(f"sim+{t + 1}")

    pnl_pct = (curve[-1] / curve[0] - 1.0) * 100.0 if curve[0] > 1e-12 else 0.0
    return ForwardSimResult(
        labels=labels,
        equity=curve,
        cumulative_pnl_pct=float(pnl_pct),
        final_equity=float(curve[-1]),
        extended_returns=None,
        mode="monte_carlo",
    )


def run_tabular_q_train_backtest(
    rets: List[float],
    dates: List[str],
    *,
    lookback: int,
    mom_threshold: float,
    train_days: int,
    q: Dict[str, Dict[str, float]],
) -> BacktestResult:
    """
    In-sample equity on return indices 0 .. train_days-1 using greedy Q (already trained on train slice).
    """
    n = train_days
    if len(rets) < n:
        raise ValueError(f"need at least {n} returns, got {len(rets)}")
    pos: List[float] = [0.0] * n
    for i in range(lookback, n):
        window = sum(rets[i - lookback : i])
        pos[i] = position_from_tabular_q(q, window, mom_threshold)

    equity: List[float] = [1.0]
    for i in range(n):
        equity.append(equity[-1] * (1.0 + pos[i] * rets[i]))

    tr = (equity[-1] - 1.0) * 100.0
    mdd = _max_drawdown_pct(equity)
    vol = float(np.std(np.array(rets[:n]))) or 1e-8
    sharpe = float(np.mean(rets[:n]) / vol * math.sqrt(252)) if vol > 1e-12 else 0.0

    return BacktestResult(
        dates=dates[: n + 1],
        closes=[],
        daily_returns=rets[:n],
        positions=pos,
        equity=equity[1:],
        total_return_pct=float(tr),
        max_drawdown_pct=float(mdd),
        sharpe_like=float(sharpe),
    )


def run_real_oos_forward(
    rets: List[float],
    dates: List[str],
    q: Dict[str, Dict[str, float]],
    *,
    lookback: int,
    mom_threshold: float,
    train_days: int,
    total_days: int,
    entry_equity: float,
) -> ForwardSimResult:
    """
    Real out-of-sample: return indices train_days .. total_days-1 (actual Yahoo history).
    `entry_equity` is portfolio value at end of train (curve starts here, then 30 real steps).
    """
    if total_days > len(rets):
        raise ValueError("total_days exceeds available returns")
    eq = float(entry_equity)
    curve = [eq]
    labels: List[str] = []
    oos_rets: List[float] = []
    for i in range(train_days, total_days):
        window = sum(rets[i - lookback : i])
        pos = position_from_tabular_q(q, window, mom_threshold)
        r = rets[i]
        oos_rets.append(r)
        eq *= 1.0 + pos * r
        curve.append(eq)
        labels.append(str(dates[i + 1]))

    pnl_pct = (curve[-1] / curve[0] - 1.0) * 100.0 if curve[0] > 1e-12 else 0.0
    return ForwardSimResult(
        labels=labels,
        equity=curve,
        cumulative_pnl_pct=float(pnl_pct),
        final_equity=float(curve[-1]),
        extended_returns=oos_rets,
        mode="real_oos",
    )


def simulate_forward_tabular_q(
    *,
    last_equity: float,
    historical_returns: List[float],
    forward_days: int = 30,
    lookback: int = 20,
    mom_threshold: float = 0.002,
    seed: Optional[int] = None,
) -> ForwardSimResult:
    """
    Monte Carlo forward path with the same synthetic shocks as `simulate_forward_pnl`, but
    position each day = greedy tabular Q on rolling momentum state (matches overnight training).
    `extended_returns` is history + simulated days (for `--loop` continuation).
    """
    rng = np.random.default_rng(seed)
    rets_hist = np.array(historical_returns[-60:], dtype=np.float64)
    sigma = float(np.std(rets_hist)) if len(rets_hist) > 1 else 0.02
    sigma = max(sigma, 0.005)
    q = load_q_table()

    rets = list(historical_returns)
    eq = float(last_equity)
    curve = [eq]
    labels: List[str] = []

    for t in range(forward_days):
        L = len(rets)
        if L < lookback:
            break
        window = sum(rets[L - lookback : L])
        pos = position_from_tabular_q(q, window, mom_threshold)
        z = rng.normal(0.0, sigma)
        if len(rets_hist) > 0:
            boot = float(rng.choice(rets_hist))
            r = 0.6 * z + 0.4 * boot
        else:
            r = z
        eq *= 1.0 + pos * r
        rets.append(r)
        curve.append(eq)
        labels.append(f"sim+{t + 1}")

    pnl_pct = (curve[-1] / curve[0] - 1.0) * 100.0 if curve[0] > 1e-12 else 0.0
    return ForwardSimResult(
        labels=labels,
        equity=curve,
        cumulative_pnl_pct=float(pnl_pct),
        final_equity=float(curve[-1]),
        extended_returns=rets,
        mode="monte_carlo",
    )


def one_forward_step(
    current_equity: float,
    current_position: float,
    historical_returns: List[float],
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    """Single simulated day: return (new_equity, new_position, daily_return)."""
    rets_hist = np.array(historical_returns[-60:], dtype=np.float64)
    sigma = float(np.std(rets_hist)) if len(rets_hist) > 1 else 0.02
    sigma = max(sigma, 0.005)
    z = rng.normal(0.0, sigma)
    if len(rets_hist) > 0:
        boot = float(rng.choice(rets_hist))
        r = 0.6 * z + 0.4 * boot
    else:
        r = z
    new_pos = float(np.clip(current_position * 0.98 + 0.02 * np.sign(r), -1.0, 1.0))
    new_eq = current_equity * (1.0 + new_pos * r)
    return new_eq, new_pos, float(r)


def one_forward_step_tabular_q(
    current_equity: float,
    rets_mutable: List[float],
    rng: np.random.Generator,
    lookback: int,
    mom_threshold: float,
) -> Tuple[float, float, float]:
    """
    One synthetic day: draw return, append to `rets_mutable`, position from tabular Q.
    Returns (new_equity, position_used, daily_return).
    """
    rets_hist = np.array(rets_mutable[-60:], dtype=np.float64)
    sigma = float(np.std(rets_hist)) if len(rets_hist) > 1 else 0.02
    sigma = max(sigma, 0.005)
    z = rng.normal(0.0, sigma)
    if len(rets_hist) > 0:
        boot = float(rng.choice(rets_hist))
        r = 0.6 * z + 0.4 * boot
    else:
        r = z

    L = len(rets_mutable)
    if L < lookback:
        window = sum(rets_mutable) if rets_mutable else 0.0
    else:
        window = sum(rets_mutable[L - lookback : L])
    pos = position_from_tabular_q(load_q_table(), window, mom_threshold)
    new_eq = current_equity * (1.0 + pos * r)
    rets_mutable.append(float(r))
    return new_eq, pos, float(r)
