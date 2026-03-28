from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


HISTORY_FILE = Path("outputs/signal_history.jsonl")


def append_signal_history(record: Dict) -> None:
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def read_signal_history(limit: int = 2000) -> List[Dict]:
    if not HISTORY_FILE.exists():
        return []
    lines = HISTORY_FILE.read_text(encoding="utf-8").splitlines()
    rows = []
    for line in lines[-limit:]:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows
