from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


AgentName = Literal[
    "ingestion",
    "fundamental",
    "trial_progress",
    "regulatory",
    "market_impact",
    "signal",
    "coordinator",
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


class FinalReport(BaseModel):
    ticker: str
    company: str
    final_signal: Literal["long", "short", "no_trade"]
    confidence: int = Field(ge=0, le=100)
    horizon: Literal["1d", "1w", "1m"]
    key_points: List[str]
    risk_flags: List[str]
    evidence: List[Evidence]
