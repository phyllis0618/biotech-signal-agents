from __future__ import annotations

from typing import Dict, List

import pandas as pd


def compute_event_window_attribution(history_rows: List[Dict]) -> pd.DataFrame:
    """
    Attribution proxy based on observed same-cycle change_pct.
    """
    if not history_rows:
        return pd.DataFrame()
    df = pd.DataFrame(history_rows)
    needed = {"final_signal", "change_pct", "confidence"}
    if not needed.issubset(df.columns):
        return pd.DataFrame()

    df["change_pct"] = pd.to_numeric(df["change_pct"], errors="coerce")
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df = df.dropna(subset=["change_pct", "confidence", "final_signal"])
    if df.empty:
        return pd.DataFrame()

    def _hit(row: pd.Series) -> int:
        if row["final_signal"] == "long" and row["change_pct"] > 0:
            return 1
        if row["final_signal"] == "short" and row["change_pct"] < 0:
            return 1
        if row["final_signal"] == "no_trade":
            return 1 if abs(row["change_pct"]) < 1.5 else 0
        return 0

    df["hit"] = df.apply(_hit, axis=1)
    df["conf_bucket"] = pd.cut(
        df["confidence"],
        bins=[0, 40, 55, 70, 100],
        labels=["low", "med_low", "med_high", "high"],
        include_lowest=True,
    )

    grp = (
        df.groupby(["final_signal", "conf_bucket"], dropna=False)
        .agg(
            obs=("change_pct", "count"),
            avg_change_pct=("change_pct", "mean"),
            hit_rate=("hit", "mean"),
        )
        .reset_index()
    )
    grp["avg_change_pct"] = grp["avg_change_pct"].round(4)
    grp["hit_rate"] = (grp["hit_rate"] * 100).round(2)
    return grp
