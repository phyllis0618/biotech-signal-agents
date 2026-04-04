#!/usr/bin/env python3
"""
Simulated multi-ticker backtest (no required CLI args).

- **Default**: reads `data/demo_biotech_frontend.csv`, runs the live pipeline per row,
  applies the *current* `final_signal` as a fixed long/short/flat weight to each daily
  return over the last N trading days (same idea as `backtest_10d.py`), vs buy-and-hold.

- **--fast**: skips the pipeline (no LLM / APIs beyond Yahoo prices). Uses a simple
  momentum rule: 5-day trailing return sign → long / short / flat — useful for dry runs.

Caveat: this is a **snapshot** signal applied to recent history, not a true walk-forward test.

Usage (from repo root):

  python3 scripts/sim_backtest_demo.py
  python3 scripts/sim_backtest_demo.py --days 15 --universe-csv data/demo_biotech_frontend.csv
  python3 scripts/sim_backtest_demo.py --fast
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.connectors.market_data import fetch_yahoo_daily_bars
from src.pipeline import run_pipeline
from src.universe import load_biotech_universe


def _position_from_signal(sig: str) -> float:
    return {"long": 1.0, "short": -1.0, "no_trade": 0.0}.get(sig, 0.0)


def _momentum_signal(bars: List[Dict[str, Any]]) -> str:
    """Toy rule: last 5 closes → sign of cumulative return."""
    if len(bars) < 6:
        return "no_trade"
    closes = [float(b["close"]) for b in bars[-6:] if b.get("close") is not None]
    if len(closes) < 6:
        return "no_trade"
    r = (closes[-1] / closes[0]) - 1.0
    if r > 0.01:
        return "long"
    if r < -0.01:
        return "short"
    return "no_trade"


def _run_equity_curve(
    bars: List[Dict[str, Any]], sig: str
) -> Tuple[float, float, List[Dict[str, Any]]]:
    weight = _position_from_signal(sig)
    strat_cum = 1.0
    bh_cum = 1.0
    daily_rows: List[Dict[str, Any]] = []
    for i in range(1, len(bars)):
        p0, p1 = bars[i - 1]["close"], bars[i]["close"]
        r = (float(p1) / float(p0)) - 1.0 if p0 else 0.0
        strat_cum *= 1.0 + weight * r
        bh_cum *= 1.0 + r
        daily_rows.append(
            {
                "date": bars[i].get("date", ""),
                "daily_return": round(r, 6),
                "signal": sig,
                "position_weight": weight,
                "strategy_gross": round(strat_cum, 6),
                "buy_hold_gross": round(bh_cum, 6),
            }
        )
    return strat_cum, bh_cum, daily_rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simulated multi-ticker backtest (defaults: no args)")
    p.add_argument(
        "--universe-csv",
        default=str(_ROOT / "data" / "demo_biotech_frontend.csv"),
        help="Universe CSV (ticker, company, cash_runway_months, single_asset_exposure)",
    )
    p.add_argument("--days", type=int, default=10, help="Trading days of history")
    p.add_argument(
        "--fast",
        action="store_true",
        help="Skip pipeline; use price-only momentum signal (Yahoo only)",
    )
    p.add_argument(
        "--output-json",
        default=str(_ROOT / "outputs" / "sim_backtest_demo.json"),
        help="Summary JSON path",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        rows = load_biotech_universe(args.universe_csv)
    except FileNotFoundError:
        print(f"Universe not found: {args.universe_csv}", file=sys.stderr)
        sys.exit(1)

    results: List[Dict[str, Any]] = []
    for item in rows:
        ticker = item["ticker"]
        company = item["company"]
        bars = fetch_yahoo_daily_bars(ticker, range_="3mo", max_rows=args.days + 5)
        if len(bars) < 3:
            results.append(
                {
                    "ticker": ticker,
                    "error": "not_enough_bars",
                    "n_bars": len(bars),
                }
            )
            continue

        if args.fast:
            sig = _momentum_signal(bars)
            conf = None
            trade_id = None
            mode = "fast_momentum"
        else:
            try:
                _, report, _ = run_pipeline(
                    ticker,
                    company,
                    cash_runway_months=item["cash_runway_months"],
                    single_asset_exposure=item["single_asset_exposure"],
                )
                sig = report.final_signal
                conf = report.confidence
                trade_id = report.trade_id
                mode = "pipeline"
            except Exception as e:
                results.append({"ticker": ticker, "error": str(e)})
                continue

        strat_cum, bh_cum, daily_rows = _run_equity_curve(bars, sig)
        results.append(
            {
                "ticker": ticker,
                "company": company,
                "mode": mode,
                "pipeline_signal": sig,
                "confidence": conf,
                "trade_id": trade_id,
                "days_used": len(daily_rows),
                "strategy_total_return_pct": round((strat_cum - 1.0) * 100, 4),
                "buy_hold_total_return_pct": round((bh_cum - 1.0) * 100, 4),
                "excess_vs_bh_pct": round(
                    (strat_cum - bh_cum) * 100, 4
                ),
            }
        )

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "days": args.days,
        "fast": args.fast,
        "universe_csv": args.universe_csv,
        "rows": results,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    # Console table
    print("Simulated backtest summary (see caveat in script docstring)\n")
    print(f"{'Ticker':<8} {'Mode':<14} {'Signal':<10} {'Strat%':>10} {'B&H%':>10} {'Excess%':>10}")
    print("-" * 64)
    for r in results:
        if "error" in r:
            print(f"{r.get('ticker','?'):<8} {'ERROR':<14} {'—':<10} {'—':>10} {'—':>10} {r.get('error','')[:20]}")
            continue
        print(
            f"{r['ticker']:<8} {r['mode']:<14} {r['pipeline_signal']:<10} "
            f"{r['strategy_total_return_pct']:>10.4f} {r['buy_hold_total_return_pct']:>10.4f} "
            f"{r['excess_vs_bh_pct']:>10.4f}"
        )
    print(f"\nWrote: {out_path}")

    csv_path = out_path.with_suffix(".csv")
    ok_rows = [r for r in results if "error" not in r]
    if ok_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(ok_rows[0].keys()))
            w.writeheader()
            w.writerows(ok_rows)
        print(f"Wrote: {csv_path}")


if __name__ == "__main__":
    main()
