from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from src.risk.metrics import historical_var_95


class ExecutionDecision(str, Enum):
    ALLOW = "allow"
    HALT = "halt"
    HEDGE = "auto_hedge"


@dataclass
class RiskLimits:
    max_daily_var_pct: float = 2.5
    """If |VaR| exceeds this (in % of daily return distribution), trigger risk action."""


@dataclass
class RiskCheckResult:
    decision: ExecutionDecision
    daily_var_95: float
    reason: str
    hedge_suggestion: Optional[str] = None


class TradeExecutor:
    """
    Enforces PM risk parameters. If estimated daily VaR is beyond limit, halt or suggest hedge.
    VaR here is a simple historical 5th percentile of recent daily returns (negative = loss tail).
    """

    def __init__(self, limits: Optional[RiskLimits] = None) -> None:
        self.limits = limits or RiskLimits()

    def evaluate(
        self,
        daily_returns: list[float],
        *,
        has_open_risk: bool = True,
    ) -> RiskCheckResult:
        var_est = historical_var_95(daily_returns)
        lim = self.limits.max_daily_var_pct / 100.0

        if not has_open_risk:
            return RiskCheckResult(
                decision=ExecutionDecision.ALLOW,
                daily_var_95=var_est,
                reason="No open risk — monitoring only.",
            )

        if var_est < -lim:
            return RiskCheckResult(
                decision=ExecutionDecision.HALT,
                daily_var_95=var_est,
                reason=f"Daily VaR tail {var_est:.4f} exceeds -{lim:.4f} — halt new risk.",
                hedge_suggestion="Reduce gross exposure or pair vs XBI/sector ETF.",
            )

        if var_est < -0.7 * lim:
            return RiskCheckResult(
                decision=ExecutionDecision.HEDGE,
                daily_var_95=var_est,
                reason=f"VaR approaching limit ({var_est:.4f}).",
                hedge_suggestion="Consider partial hedge (e.g. -0.3 beta to XBI).",
            )

        return RiskCheckResult(
            decision=ExecutionDecision.ALLOW,
            daily_var_95=var_est,
            reason="Within VaR budget.",
        )

    def to_dict(self, result: RiskCheckResult) -> Dict[str, Any]:
        return {
            "decision": result.decision.value,
            "daily_var_95": result.daily_var_95,
            "reason": result.reason,
            "hedge_suggestion": result.hedge_suggestion,
        }
