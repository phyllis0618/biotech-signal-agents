#!/usr/bin/env python3
"""
Build outputs/interactive_sim.json for the Next.js walk-forward trader drill.

Uses the **same Yahoo window as PnL sim**: last `total_days` daily returns (default 100),
then keeps only the **last `oos_days` sessions** (default 30) — the forward-test / OOS slice
(aligned with 70 train + 30 OOS in simulate_strategy_pnl.py).

Run from repo root:
  python3 scripts/build_interactive_sim.py
  python3 scripts/build_interactive_sim.py --ticker GILD --total-days 100 --oos-days 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.connectors.market_data import fetch_yahoo_daily_bars


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", default="AMGN")
    p.add_argument("--company", default="Amgen Inc.")
    p.add_argument(
        "--total-days",
        type=int,
        default=100,
        help="Rolling Yahoo return window (must match PnL sim; default 100)",
    )
    p.add_argument(
        "--oos-days",
        type=int,
        default=30,
        help="Drill length = OOS forward slice (last N sessions; default 30)",
    )
    p.add_argument(
        "--days",
        type=int,
        default=None,
        help="Deprecated: use --oos-days instead",
    )
    p.add_argument("--lookback", type=int, default=20)
    p.add_argument("--with-pipeline", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    total_days = args.total_days
    oos_days = args.oos_days if args.days is None else args.days
    if args.days is not None:
        print("Warning: --days is deprecated; use --oos-days", file=sys.stderr)

    bars = fetch_yahoo_daily_bars(
        args.ticker, range_="2y", max_rows=total_days + args.lookback + 5
    )
    if len(bars) < total_days + 1:
        print("Not enough bars", file=sys.stderr)
        sys.exit(1)

    chunk = bars[-(total_days + 1) :]
    dates = [str(b["date"]) for b in chunk]
    closes = [float(b["close"]) for b in chunk]
    rets: list[float] = []
    for i in range(1, len(closes)):
        rets.append((closes[i] / closes[i - 1]) - 1.0)

    if len(rets) != total_days:
        print(f"internal: expected {total_days} returns, got {len(rets)}", file=sys.stderr)
        sys.exit(1)

    days_out: list[dict] = []
    thr = 0.002
    for i in range(args.lookback, total_days):
        window = sum(rets[i - args.lookback : i])
        if window > thr:
            cs, cc = "long", min(90, 55 + int(abs(window) * 500))
        elif window < -thr:
            cs, cc = "short", min(90, 55 + int(abs(window) * 500))
        else:
            cs, cc = "no_trade", 48
        days_out.append(
            {
                "date": dates[i + 1],
                "daily_return": round(rets[i], 6),
                "coordinator_signal": cs,
                "coordinator_confidence": cc,
                "momentum_20d": round(window, 6),
            }
        )

    n_sess = len(days_out)
    if oos_days > n_sess:
        print(
            f"oos_days ({oos_days}) > available sessions ({n_sess}); reduce oos or increase total_days",
            file=sys.stderr,
        )
        sys.exit(1)
    days_out = days_out[-oos_days:]

    train_days = total_days - oos_days
    agent_context = (
        f"{args.ticker} / {args.company}: drill = last {oos_days} trading days of a {total_days}-day Yahoo "
        f"window (same OOS segment as PnL: {train_days}d train region + {oos_days}d forward test). "
        "Momentum→coordinator-style state; RL tabular Q updates only from approve/reject/defer."
    )
    if args.with_pipeline:
        try:
            from src.pipeline import run_pipeline

            _m, rep, _r = run_pipeline(
                args.ticker,
                args.company,
                cash_runway_months=18,
                single_asset_exposure=False,
            )
            agent_context = (
                f"Latest pipeline snapshot ({rep.final_signal} @ {rep.confidence}): "
                + "; ".join(rep.key_points[:4])
                + f" — drill still uses last {oos_days}d OOS slice ({total_days}d window)."
            )
        except Exception as e:
            agent_context += f" (pipeline optional failed: {e})"

    payload = {
        "version": 1,
        "ticker": args.ticker,
        "company": args.company,
        "split": {
            "total_return_days": total_days,
            "oos_drill_days": oos_days,
            "train_days": train_days,
        },
        "agent_context": agent_context,
        "days": days_out,
        "sim_q": {},
        "current_index": 0,
        "equity": 1.0,
        "event_log": [],
        "settings": {
            "epsilon": 0.12,
            "tabular_q_lr": 0.35,
            "gamma": 0.85,
        },
        "status": "ready",
    }

    out = _ROOT / "outputs" / "interactive_sim.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"Wrote {out} ({len(days_out)} OOS sessions, {total_days}-day window)")


if __name__ == "__main__":
    main()
