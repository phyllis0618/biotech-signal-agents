#!/usr/bin/env python3
"""
Rolling Yahoo window: last `total_days` daily returns.

1) First `train_days` returns: TD tabular Q training + in-sample backtest (writes outputs/rl_qtable.json).
2) Remaining `oos_days` returns: **real** forward (no Monte Carlo) — actual historical OOS.

`--loop`: poll Yahoo; when a **new** calendar bar appears vs `last_bar_date` in the output file,
rebuild the 100-day window and append the new end-of-day equity (still no simulated prices).

Examples:
  python3 scripts/simulate_strategy_pnl.py
  python3 scripts/simulate_strategy_pnl.py --ticker AMGN --total-days 100 --train-days 70 --oos-days 30
  python3 scripts/simulate_strategy_pnl.py --loop --interval 60
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.connectors.market_data import fetch_yahoo_daily_bars
from src.simulation.pnl_simulator import run_real_oos_forward, run_tabular_q_train_backtest
from src.simulation.tabular_td_train import build_train_sessions, train_td_q_on_sessions
from src.trading.q_learning import save_q_table

# Same names/order as scripts/run_demo_for_frontend.py — one full PnL run per symbol.
DEMO_TICKERS = [
    ("AMGN", "Amgen Inc."),
    ("GILD", "Gilead Sciences Inc."),
    ("XBI", "SPDR S&P Biotech ETF"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="70d train + 30d real OOS tabular Q PnL")
    p.add_argument("--ticker", default="AMGN")
    p.add_argument("--company", default="Amgen Inc.")
    p.add_argument("--total-days", type=int, default=100, help="Number of daily returns in the window")
    p.add_argument("--train-days", type=int, default=70, help="Train + in-sample BT on first N returns")
    p.add_argument("--oos-days", type=int, default=30, help="Real OOS on last N returns (must sum with train = total)")
    p.add_argument("--lookback", type=int, default=20)
    p.add_argument("--mom-threshold", type=float, default=0.002)
    p.add_argument("--train-episodes", type=int, default=25_000)
    p.add_argument("--lr", type=float, default=0.25)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--epsilon-start", type=float, default=0.35)
    p.add_argument("--epsilon-min", type=float, default=0.05)
    p.add_argument("--epsilon-decay", type=float, default=0.9995)
    p.add_argument("--reward-scale", type=float, default=100.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--loop",
        action="store_true",
        help="Poll Yahoo; on new bar date, rebuild window and append live equity (real data only)",
    )
    p.add_argument("--interval", type=float, default=60.0, help="Seconds between Yahoo polls in --loop")
    p.add_argument(
        "--output",
        default=str(_ROOT / "outputs" / "pnl_simulation_state.json"),
        help="JSON path for Next /api/pnl",
    )
    p.add_argument(
        "--multi-demo",
        action="store_true",
        help=(
            "Run the same train+BT+OOS logic for each demo ticker (AMGN, GILD, XBI); "
            "write `by_ticker` + top-level copy of default (first successful). "
            "Conflicts with --loop. `outputs/rl_qtable.json` is saved from the last symbol only."
        ),
    )
    return p.parse_args()


def _write_state(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _load_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def slice_last_returns(
    bars: List[Dict[str, Any]], total_days: int
) -> Tuple[List[float], List[str]]:
    """Last `total_days` daily returns and aligned date series (len dates == total_days + 1)."""
    if len(bars) < total_days + 1:
        raise ValueError(f"need at least {total_days + 1} daily bars, got {len(bars)}")
    chunk = bars[-(total_days + 1) :]
    dates = [str(b["date"]) for b in chunk]
    closes = [float(b["close"]) for b in chunk]
    rets: List[float] = []
    for i in range(1, len(closes)):
        rets.append((closes[i] / closes[i - 1]) - 1.0)
    assert len(rets) == total_days
    return rets, dates


def build_payload(
    *,
    ticker: str,
    company: str,
    bt,
    fwd,
    lookback: int,
    total_days: int,
    train_days: int,
    oos_days: int,
    train_episodes: int,
    last_bar_date: str,
) -> Dict[str, Any]:
    train_from = bt.dates[1] if len(bt.dates) > 1 else ""
    train_to = bt.dates[-1] if bt.dates else ""
    oos_from = fwd.labels[0] if fwd.labels else ""
    oos_to = fwd.labels[-1] if fwd.labels else ""

    return {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "last_bar_date": last_bar_date,
        "ticker": ticker,
        "company": company,
        "split": {
            "total_return_days": total_days,
            "train_days": train_days,
            "oos_days": oos_days,
            "train_date_range": [train_from, train_to],
            "oos_date_range": [oos_from, oos_to],
            "forward_mode": "real_historical_oos",
        },
        "strategy": {
            "name": "tabular_q_train_oos_split",
            "q_table": "outputs/rl_qtable.json",
            "lookback": lookback,
            "train_episodes": train_episodes,
            "backtest_total_return_pct": round(bt.total_return_pct, 4),
            "backtest_max_drawdown_pct": round(bt.max_drawdown_pct, 4),
            "backtest_sharpe_like": round(bt.sharpe_like, 4),
            "note": "BT = first train_days (in-sample). Forward = real Yahoo returns on last oos_days.",
        },
        "backtest_curve": {
            "dates": bt.dates[1:],
            "equity": [round(x, 6) for x in bt.equity],
            "positions": bt.positions,
            "daily_returns": [round(x, 6) for x in bt.daily_returns],
        },
        "forward_sim": {
            "mode": fwd.mode,
            "labels": fwd.labels,
            "equity": [round(x, 6) for x in fwd.equity],
            "cumulative_pnl_pct": round(fwd.cumulative_pnl_pct, 4),
            "final_equity": round(fwd.final_equity, 6),
            "oos_daily_returns": [round(x, 6) for x in (fwd.extended_returns or [])],
        },
        "live_sim": {
            "running": False,
            "step": 0,
            "equity": [round(fwd.equity[-1], 6)],
            "times": [datetime.now(timezone.utc).isoformat()],
            "note": "Without --loop, single point = end of real OOS. With --loop, appends when Yahoo has a new bar.",
        },
    }


def _validate_window_args(args: argparse.Namespace) -> None:
    if args.train_days + args.oos_days != args.total_days:
        print("train_days + oos_days must equal total_days", file=sys.stderr)
        sys.exit(1)
    if args.train_days <= args.lookback:
        print("train_days must be > lookback (need at least one train session)", file=sys.stderr)
        sys.exit(1)


def compute_pnl_payload(
    args: argparse.Namespace,
    ticker: str,
    company: str,
    *,
    save_q: bool,
    seed_offset: int = 0,
) -> Dict[str, Any]:
    """Train tabular Q on this symbol’s Yahoo window, BT, real OOS; optional `save_q` writes rl_qtable.json."""
    bars = fetch_yahoo_daily_bars(
        ticker, range_="2y", max_rows=max(130, args.total_days + args.lookback + 5)
    )
    rets, dates = slice_last_returns(bars, args.total_days)
    last_bar_date = str(bars[-1]["date"])

    train_sessions = build_train_sessions(
        rets,
        dates,
        lookback=args.lookback,
        mom_threshold=args.mom_threshold,
        train_days=args.train_days,
    )
    q, _n_ep = train_td_q_on_sessions(
        train_sessions,
        episodes=args.train_episodes,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        reward_scale=args.reward_scale,
        seed=args.seed + seed_offset,
    )
    if save_q:
        save_q_table(q)

    bt = run_tabular_q_train_backtest(
        rets,
        dates,
        lookback=args.lookback,
        mom_threshold=args.mom_threshold,
        train_days=args.train_days,
        q=q,
    )

    fwd = run_real_oos_forward(
        rets,
        dates,
        q,
        lookback=args.lookback,
        mom_threshold=args.mom_threshold,
        train_days=args.train_days,
        total_days=args.total_days,
        entry_equity=bt.equity[-1],
    )

    return build_payload(
        ticker=ticker,
        company=company,
        bt=bt,
        fwd=fwd,
        lookback=args.lookback,
        total_days=args.total_days,
        train_days=args.train_days,
        oos_days=args.oos_days,
        train_episodes=args.train_episodes,
        last_bar_date=last_bar_date,
    )


def run_once(args: argparse.Namespace, out_path: Path) -> Dict[str, Any]:
    _validate_window_args(args)
    payload = compute_pnl_payload(args, args.ticker, args.company, save_q=True)
    _write_state(out_path, payload)
    return payload


def run_multi_demo(args: argparse.Namespace, out_path: Path) -> Dict[str, Any]:
    _validate_window_args(args)
    by_ticker: Dict[str, Any] = {}
    for i, (t, c) in enumerate(DEMO_TICKERS):
        last = i == len(DEMO_TICKERS) - 1
        try:
            by_ticker[t] = compute_pnl_payload(args, t, c, save_q=last, seed_offset=i * 1000)
        except Exception as e:
            by_ticker[t] = {"error": str(e), "ticker": t, "company": c}
    default_t: str | None = None
    for t, _ in DEMO_TICKERS:
        if t in by_ticker and "error" not in by_ticker[t]:
            default_t = t
            break
    merged: Dict[str, Any] = {}
    if default_t:
        merged = dict(by_ticker[default_t])
    payload: Dict[str, Any] = {"by_ticker": by_ticker, "default_ticker": default_t, **merged}
    _write_state(out_path, payload)
    print(
        "[multi-demo] outputs/rl_qtable.json saved from last symbol in DEMO_TICKERS "
        f"({DEMO_TICKERS[-1][0]}) if that run succeeded.",
        flush=True,
    )
    return payload


def main() -> None:
    args = parse_args()
    out_path = Path(args.output)

    if args.loop and args.multi_demo:
        print("--multi-demo cannot be used with --loop", file=sys.stderr)
        sys.exit(1)

    if args.multi_demo:
        payload = run_multi_demo(args, out_path)
        print(json.dumps(payload.get("strategy"), indent=2))
        print(json.dumps(payload.get("split"), indent=2))
        print(f"\nWrote {out_path} (by_ticker keys: {list((payload.get('by_ticker') or {}).keys())})")
        return

    if not args.loop:
        payload = run_once(args, out_path)
        print(json.dumps(payload["strategy"], indent=2))
        print(json.dumps(payload["split"], indent=2))
        print(f"\nWrote {out_path}")
        return

    prev = _load_state(out_path)
    last_seen = prev.get("last_bar_date")
    print(f"Loop mode: poll every {args.interval}s; last_bar_date={last_seen!r} → rebuild on new Yahoo bar.")

    try:
        while True:
            time.sleep(args.interval)
            bars = fetch_yahoo_daily_bars(
                args.ticker, range_="2y", max_rows=max(130, args.total_days + args.lookback + 5)
            )
            if len(bars) < args.total_days + 1:
                continue
            latest = str(bars[-1]["date"])
            if latest == last_seen:
                continue

            before = _load_state(out_path)
            payload = run_once(args, out_path)
            last_seen = payload["last_bar_date"]

            eq_list = list((before.get("live_sim") or {}).get("equity") or [])
            times_list = list((before.get("live_sim") or {}).get("times") or [])
            pt = payload["forward_sim"]["equity"][-1]
            eq_list.append(round(pt, 6))
            times_list.append(datetime.now(timezone.utc).isoformat())
            if len(eq_list) > 500:
                eq_list = eq_list[-500:]
                times_list = times_list[-500:]

            payload["live_sim"] = {
                "running": True,
                "step": len(eq_list),
                "equity": eq_list,
                "times": times_list,
                "note": "Appends one point each time Yahoo reports a new daily bar (rolling window).",
            }
            payload["updated_at"] = datetime.now(timezone.utc).isoformat()
            _write_state(out_path, payload)
            print(f"[loop] new bar {last_seen} → equity tail {eq_list[-3:]}", flush=True)
    except KeyboardInterrupt:
        st = _load_state(out_path)
        st["live_sim"] = {**(st.get("live_sim") or {}), "running": False}
        st["updated_at"] = datetime.now(timezone.utc).isoformat()
        _write_state(out_path, st)
        print("\nStopped; state saved.")


if __name__ == "__main__":
    main()
