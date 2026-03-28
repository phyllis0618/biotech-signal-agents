from __future__ import annotations

from typing import Dict


def compute_institutional_scorecard(
    pipeline_summary: Dict,
    report_confidence: int,
    signal: str,
    cash_runway_months: int,
    single_asset_exposure: bool,
) -> Dict:
    phase23 = int(pipeline_summary.get("phase23_trials", 0))
    completed = int(pipeline_summary.get("completed_trials", 0))
    total = int(pipeline_summary.get("total_trials", 0))

    pipeline_depth = min(100, phase23 * 12 + completed * 8 + min(total, 10) * 3)
    pipeline_execution = min(100, completed * 10 + max(0, total - completed) * 2)
    financing_quality = 70
    if cash_runway_months < 12:
        financing_quality -= 25
    elif cash_runway_months < 18:
        financing_quality -= 10
    if single_asset_exposure:
        financing_quality -= 20
    financing_quality = max(0, financing_quality)

    signal_alignment = report_confidence
    if signal == "long":
        signal_alignment = min(100, signal_alignment + 5)
    elif signal == "short":
        signal_alignment = max(0, signal_alignment - 5)

    composite = int(
        0.35 * pipeline_depth
        + 0.25 * pipeline_execution
        + 0.20 * financing_quality
        + 0.20 * signal_alignment
    )

    return {
        "pipeline_depth": pipeline_depth,
        "pipeline_execution": pipeline_execution,
        "financing_quality": financing_quality,
        "signal_alignment": signal_alignment,
        "composite_score": composite,
    }
