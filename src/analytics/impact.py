from __future__ import annotations

from typing import Dict, List

import pandas as pd


def estimate_signal_price_impact(history_rows: List[Dict]) -> Dict[str, float]:
    """
    Estimate expected price reaction by signal from historical observations.
    Expects rows to include final_signal and change_pct.
    """
    if not history_rows:
        return {"long": 0.0, "short": 0.0, "no_trade": 0.0}
    df = pd.DataFrame(history_rows)
    if "final_signal" not in df.columns or "change_pct" not in df.columns:
        return {"long": 0.0, "short": 0.0, "no_trade": 0.0}
    df = df.dropna(subset=["final_signal", "change_pct"]).copy()
    if df.empty:
        return {"long": 0.0, "short": 0.0, "no_trade": 0.0}
    df["change_pct"] = pd.to_numeric(df["change_pct"], errors="coerce")
    out: Dict[str, float] = {}
    for signal in ["long", "short", "no_trade"]:
        sub = df[df["final_signal"] == signal]
        out[signal] = round(float(sub["change_pct"].mean()), 4) if not sub.empty else 0.0
    return out
