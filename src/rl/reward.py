from __future__ import annotations

from typing import Tuple


def pnl_drawdown_reward(
    pnl_step: float,
    max_drawdown: float,
    lambda_dd: float = 0.5,
) -> float:
    """
    Reward = PnL - λ * Max_Drawdown (Jordan-style risk-adjusted objective).
    """
    return float(pnl_step - lambda_dd * max_drawdown)


def sharpe_vol_penalty_reward(
    returns_window: list,
    event_volatility_spike: float,
    mu_risk_free: float = 0.0,
    vol_penalty: float = 0.15,
) -> float:
    """
    Proxy for Sharpe with extra penalty when volatility spikes during binary events.
    """
    import math

    if not returns_window:
        return 0.0
    mean_r = sum(returns_window) / len(returns_window)
    var = sum((x - mean_r) ** 2 for x in returns_window) / max(len(returns_window), 1)
    std = math.sqrt(var) + 1e-8
    sharpe_like = (mean_r - mu_risk_free) / std
    return float(sharpe_like - vol_penalty * event_volatility_spike)
