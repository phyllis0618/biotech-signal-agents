from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


AgentName = Literal[
    "ingestion",
    "fundamental",
    "trial_progress",
    "regulatory",
    "market_impact",
    "signal",
    "coordinator",
    "rl_policy",
    "trader_review",
]


class Evidence(BaseModel):
    source: str
    title: str
    url: str
    snippet: str = ""
    published_at: Optional[str] = None


class AgentMessage(BaseModel):
    agent: AgentName
    ticker: str
    company: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    summary: str
    confidence: int = Field(ge=0, le=100, default=50)
    signal_hint: Literal["bullish", "bearish", "neutral"] = "neutral"
    evidence: List[Evidence] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)


ExecutionStatus = Literal["pending_trader_review", "approved", "rejected", "deferred"]


class FinalReport(BaseModel):
    ticker: str
    company: str
    final_signal: Literal["long", "short", "no_trade"]
    confidence: int = Field(ge=0, le=100)
    horizon: Literal["1d", "1w", "1m"]
    key_points: List[str]
    risk_flags: List[str]
    evidence: List[Evidence]
    # Snapshot before RL (for audit / trader UI)
    coordinator_signal: Optional[str] = None
    coordinator_confidence: Optional[int] = None
    # Reinforcement-learning policy layer (tabular Q; learns from trader feedback)
    rl_state: str = ""
    rl_action: str = ""
    rl_q_preview: str = ""
    # Human gate — no automated execution without trader review
    trade_id: str = ""
    execution_status: ExecutionStatus = "pending_trader_review"
    trader_guidance: Optional[str] = None
    # Agentic PM / WorldQuant-style audit fields
    reasoning_trace: List[Dict[str, Any]] = Field(default_factory=list)
    counter_thesis: str = ""
    pm_weights_preview: str = ""
    risk_status: str = ""
    system_tag: str = "AUTONOMOUS_COGNITIVE_SYSTEM"
