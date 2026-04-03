from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class ResearchState(TypedDict, total=False):
    """Shared graph state for LangGraph workflow."""

    ticker: str
    company: str
    plan: str
    extraction: Dict[str, Any]
    financials: Dict[str, Any]
    research_bundle: Dict[str, Any]
    reflection: Dict[str, Any]
    final_alpha: Dict[str, Any]
    errors: List[str]
