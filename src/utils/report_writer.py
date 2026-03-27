from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.models.messages import FinalReport


def write_report(report: FinalReport, output_dir: str = "outputs") -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    path = out / f"{report.ticker}-{stamp}.json"
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return path


def final_report_to_pretty_json(report: FinalReport) -> str:
    return json.dumps(report.model_dump(), indent=2)
