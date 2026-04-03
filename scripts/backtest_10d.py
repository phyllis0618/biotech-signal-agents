#!/usr/bin/env python3
"""
10-trading-day backtest using Yahoo daily closes + one-shot pipeline signal.

Strategy (simple): apply constant directional exposure from `final_signal` to each
daily return over the last N days (default 10). Compare vs buy-and-hold.

Note: Uses current pipeline snapshot as a stand-in for historical point-in-time signals;
for production, store daily signals or use walk-forward logic.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.connectors.market_data import fetch_yahoo_daily_bars
from src.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="10-day daily backtest (Yahoo + pipeline signal)")
    p.add_argument("--ticker", required=True)
    p.add_argument("--company", required=True)
    p.add_argument("--days", type=int, default=10, help="Trading days of history (default 10)")
    p.add_argument("--cash-runway-months", type=int, default=18)
    p.add_argument("--single-asset-exposure", action="store_true")
    p.add_argument(
        "--output-csv",
        default="",
        help="Optional path to save daily rows (default: outputs/backtest_10d_{ticker}.csv)",
    )
    return p.parse_args()


def _position_from_signal(sig: str) -> float:
    return {"long": 1.0, "short": -1.0, "no_trade": 0.0}.get(sig, 0.0)


def main() -> None:
    args = parse_args()
    bars = fetch_yahoo_daily_bars(args.ticker, range_="3mo", max_rows=args.days + 1)
    if len(bars) < 2:
        print("Not enough daily bars; check ticker or network.")
        return

    _, report, _ = run_pipeline(
        args.ticker,
        args.company,
        cash_runway_months=args.cash_runway_months,
        single_asset_exposure=args.single_asset_exposure,
    )
    sig = report.final_signal
    weight = _position_from_signal(sig)

    daily_rows = []
    strat_cum = 1.0
    bh_cum = 1.0

    for i in range(1, len(bars)):
        p0, p1 = bars[i - 1]["close"], bars[i]["close"]
        r = (p1 / p0) - 1.0 if p0 else 0.0
        strat_cum *= 1.0 + weight * r
        bh_cum *= 1.0 + r
        daily_rows.append(
            {
                "date": bars[i]["date"],
                "close": p1,
                "daily_return": round(r, 6),
                "signal": sig,
                "position_weight": weight,
                "strategy_gross": round(strat_cum, 6),
                "buy_hold_gross": round(bh_cum, 6),
            }
        )

    out = {
        "ticker": args.ticker,
        "company": args.company,
        "days_used": len(daily_rows),
        "pipeline_signal": sig,
        "confidence": report.confidence,
        "strategy_total_return_pct": round((strat_cum - 1.0) * 100, 4),
        "buy_hold_total_return_pct": round((bh_cum - 1.0) * 100, 4),
        "trade_id": report.trade_id,
        "execution_status": report.execution_status,
    }

    print(json.dumps(out, indent=2, ensure_ascii=True))

    csv_path = args.output_csv or f"outputs/backtest_10d_{args.ticker}.csv"
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    if daily_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(daily_rows[0].keys()))
            writer.writeheader()
            writer.writerows(daily_rows)
        print(f"\nSaved daily series: {csv_path}")


if __name__ == "__main__":
    main()
