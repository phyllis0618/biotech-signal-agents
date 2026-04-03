from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _outputs() -> Path:
    root = Path(__file__).resolve().parents[2]
    out = root / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out


@dataclass
class PendingApproval:
    trade_id: str
    ticker: str
    company: str
    final_signal: str
    confidence: int
    counter_thesis: str
    reasoning_trace: List[Dict[str, Any]]
    risk_snapshot: Dict[str, Any]
    status: str = "pending_pm_review"
    created_at: str = ""


class ApprovalQueue:
    """
    Mandatory HITL queue for high-stakes / all proposed trades.
    Persists to JSONL for audit and Next.js dashboard consumption.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or (_outputs() / "approval_queue.jsonl")

    def enqueue(self, item: PendingApproval) -> None:
        if not item.created_at:
            item.created_at = datetime.now(timezone.utc).isoformat()
        row = asdict(item)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    def recent(self, n: int = 20) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").strip().splitlines()
        out: List[Dict[str, Any]] = []
        for line in lines[-n:]:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out
