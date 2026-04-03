from __future__ import annotations

from typing import Any, Dict, List, Literal, TypedDict


class TimelineStep(TypedDict, total=False):
    phase: Literal["planning", "tool_use", "memory", "reflection", "decision"]
    title: str
    detail: str
    payload: Dict[str, Any]


class ReasoningState(TypedDict, total=False):
    ticker: str
    company: str
    plan_tasks: List[str]
    tool_results: Dict[str, Any]
    memory_hits: List[Dict[str, Any]]
    research_bundle: Dict[str, Any]
    reflection: Dict[str, Any]
    counter_thesis: str
    final_alpha: Dict[str, Any]
    timeline: List[TimelineStep]
    errors: List[str]
