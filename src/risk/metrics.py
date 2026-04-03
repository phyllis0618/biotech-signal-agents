from __future__ import annotations

from typing import List, Sequence


def historical_var_95(returns: Sequence[float]) -> float:
    """Daily historical VaR at ~95% (5th percentile of returns)."""
    r = [float(x) for x in returns if x == x]
    if len(r) < 5:
        return 0.0
    r_sorted = sorted(r)
    idx = max(0, int(0.05 * (len(r_sorted) - 1)))
    return float(r_sorted[idx])


def max_drawdown_from_equity(equity_curve: Sequence[float]) -> float:
    if not equity_curve:
        return 0.0
    peak = float(equity_curve[0])
    mdd = 0.0
    for e in equity_curve:
        e = float(e)
        peak = max(peak, e)
        dd = (peak - e) / peak if peak > 1e-12 else 0.0
        mdd = max(mdd, dd)
    return float(mdd)


def unrealized_pnl_pct(entry_price: float, mark_price: float, side: str) -> float:
    if entry_price <= 0 or mark_price <= 0:
        return 0.0
    if side == "short":
        return (entry_price - mark_price) / entry_price * 100.0
    return (mark_price - entry_price) / entry_price * 100.0


def daily_returns_from_prices(prices: Sequence[float]) -> List[float]:
    out: List[float] = []
    for i in range(1, len(prices)):
        a, b = float(prices[i - 1]), float(prices[i])
        if a > 0:
            out.append((b - a) / a)
    return out
