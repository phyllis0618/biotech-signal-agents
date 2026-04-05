#!/usr/bin/env python3
"""
TD tabular Q on the **first train_days** of a total_days Yahoo window (same states as PnL sim).
Default: total 100 return days, train on 70 (OOS is handled by simulate_strategy_pnl.py).

Single run: writes outputs/rl_qtable.json and outputs/overnight_train_report.json

Multi (same universe as dashboard / PnL multi-demo):
  python3 scripts/train_tabular_q_overnight.py --multi-demo
  → outputs/rl_qtable_AMGN.json, rl_qtable_GILD.json, rl_qtable_XBI.json
  → overnight_train_report.json with by_ticker + default_ticker; copies default Q → rl_qtable.json
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.connectors.market_data import fetch_yahoo_daily_bars
from src.simulation.tabular_td_train import (
    build_train_sessions,
    greedy_equity_on_sessions,
    train_td_q_on_sessions,
)
from src.trading.q_learning import save_q_table

# Same order as run_demo_for_frontend.py / simulate_strategy_pnl --multi-demo
DEMO_TICKERS = [
    ("AMGN", "Amgen Inc."),
    ("GILD", "Gilead Sciences Inc."),
    ("XBI", "SPDR S&P Biotech ETF"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TD Q on train slice of Yahoo window")
    p.add_argument("--ticker", default="AMGN")
    p.add_argument("--company", default="Amgen Inc.")
    p.add_argument("--total-days", type=int, default=100, help="Return days in rolling window")
    p.add_argument("--train-days", type=int, default=70, help="Train only on first N return days")
    p.add_argument("--lookback", type=int, default=20)
    p.add_argument("--mom-threshold", type=float, default=0.002)
    p.add_argument("--episodes", type=int, default=500_000)
    p.add_argument("--max-seconds", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=0.25)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--epsilon-start", type=float, default=0.35)
    p.add_argument("--epsilon-min", type=float, default=0.05)
    p.add_argument("--epsilon-decay", type=float, default=0.9995)
    p.add_argument("--reward-scale", type=float, default=100.0)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warm-start", action="store_true")
    p.add_argument("--q-out", type=str, default=str(_ROOT / "outputs" / "rl_qtable.json"))
    p.add_argument("--report-out", type=str, default=str(_ROOT / "outputs" / "overnight_train_report.json"))
    p.add_argument(
        "--multi-demo",
        action="store_true",
        help=(
            "Train TD Q for AMGN, GILD, XBI with the same hyperparameters; "
            "write outputs/rl_qtable_<TICKER>.json each, merged overnight_train_report.json (by_ticker), "
            "and copy default ticker's Q to outputs/rl_qtable.json."
        ),
    )
    return p.parse_args()


def slice_last_returns(bars: List[Dict[str, Any]], total_days: int) -> Tuple[List[float], List[str]]:
    if len(bars) < total_days + 1:
        raise SystemExit(f"need at least {total_days + 1} daily bars, got {len(bars)}")
    chunk = bars[-(total_days + 1) :]
    dates = [str(b["date"]) for b in chunk]
    closes = [float(b["close"]) for b in chunk]
    rets: List[float] = []
    for i in range(1, len(closes)):
        rets.append((closes[i] / closes[i - 1]) - 1.0)
    return rets, dates


def load_q_file(path: Path) -> Dict[str, Dict[str, float]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=True), encoding="utf-8")


@dataclass
class BestSnapshot:
    episode: int
    equity: float
    sharpe_like: float
    total_return_pct: float


def train_one_symbol(
    args: argparse.Namespace,
    ticker: str,
    company: str,
    q_out: Path,
    seed_offset: int,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
    """Train one symbol; write best checkpoints to `q_out`; return final Q and report dict."""
    t0 = time.perf_counter()
    bars = fetch_yahoo_daily_bars(
        ticker, range_="2y", max_rows=max(130, args.total_days + args.lookback + 5)
    )
    rets, dates = slice_last_returns(bars, args.total_days)
    train_sessions = build_train_sessions(
        rets,
        dates,
        lookback=args.lookback,
        mom_threshold=args.mom_threshold,
        train_days=args.train_days,
    )
    n = len(train_sessions)
    print(
        f"[train] {ticker}: {n} train sessions (total_days={args.total_days}, train_days={args.train_days}), "
        f"lookback={args.lookback}, thr={args.mom_threshold}",
        flush=True,
    )

    q_init = load_q_file(q_out) if args.warm_start else None
    if args.warm_start:
        print(f"[train] warm-start from {q_out} ({len(q_init or {})} states)", flush=True)

    best: Optional[BestSnapshot] = None

    def on_eval(ep: int, q_table: Dict[str, Dict[str, float]]) -> None:
        nonlocal best
        eq, sh, trp = greedy_equity_on_sessions(q_table, train_sessions)
        elapsed = time.perf_counter() - t0
        print(
            f"[train] ep={ep} greedy_equity={eq:.6f} ret%={trp:.4f} sharpe≈{sh:.4f} t={elapsed:.1f}s states={len(q_table)}",
            flush=True,
        )
        if best is None or eq > best.equity or (math.isclose(eq, best.equity, rel_tol=1e-9) and sh > best.sharpe_like):
            best = BestSnapshot(episode=ep, equity=eq, sharpe_like=sh, total_return_pct=trp)
            write_json(q_out, q_table)
            print(f"  -> new best saved to {q_out}", flush=True)

    q, episode_done = train_td_q_on_sessions(
        train_sessions,
        episodes=args.episodes,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        reward_scale=args.reward_scale,
        seed=args.seed + seed_offset,
        q_init=q_init,
        eval_every=args.eval_every,
        eval_callback=on_eval if args.eval_every > 0 else None,
        max_seconds=args.max_seconds,
    )
    if args.max_seconds > 0:
        print(f"[train] completed {episode_done} episodes (max_seconds may have cut short)", flush=True)

    eq_f, sh_f, trp_f = greedy_equity_on_sessions(q, train_sessions)
    if best is None:
        best = BestSnapshot(episode=episode_done, equity=eq_f, sharpe_like=sh_f, total_return_pct=trp_f)
    elapsed = time.perf_counter() - t0

    write_json(q_out, q)

    report = {
        "ticker": ticker,
        "company": company,
        "total_days": args.total_days,
        "train_days": args.train_days,
        "train_sessions": n,
        "lookback": args.lookback,
        "mom_threshold": args.mom_threshold,
        "episodes_trained": int(episode_done),
        "seconds_wall": round(elapsed, 3),
        "hyperparams": {
            "lr": args.lr,
            "gamma": args.gamma,
            "epsilon_start": args.epsilon_start,
            "epsilon_min": args.epsilon_min,
            "epsilon_decay": args.epsilon_decay,
            "reward_scale": args.reward_scale,
        },
        "best_greedy_eval": asdict(best) if best else None,
        "final_greedy_eval": {"equity": eq_f, "sharpe_like": sh_f, "total_return_pct": trp_f},
        "q_states": len(q),
        "q_out": str(q_out.resolve()),
        "note": "Train slice only; OOS real forward is simulate_strategy_pnl.py.",
    }
    return q, report


def run_multi_demo(args: argparse.Namespace) -> None:
    out_dir = _ROOT / "outputs"
    by_ticker: Dict[str, Any] = {}
    for i, (t, c) in enumerate(DEMO_TICKERS):
        q_path = out_dir / f"rl_qtable_{t}.json"
        try:
            _q, rep = train_one_symbol(args, t, c, q_path, seed_offset=i * 1000)
            by_ticker[t] = rep
        except Exception as e:
            by_ticker[t] = {"error": str(e), "ticker": t, "company": c}
            print(f"[multi-demo] ERROR {t}: {e}", file=sys.stderr, flush=True)

    default_t: str | None = None
    for t, _ in DEMO_TICKERS:
        if t in by_ticker and "error" not in by_ticker[t]:
            default_t = t
            break

    merged: Dict[str, Any] = {}
    if default_t and default_t in by_ticker and "error" not in by_ticker[default_t]:
        merged = dict(by_ticker[default_t])

    payload: Dict[str, Any] = {
        "by_ticker": by_ticker,
        "default_ticker": default_t,
        **merged,
    }
    report_path = Path(args.report_out)
    write_json(report_path, payload)
    print(json.dumps(payload.get("best_greedy_eval"), indent=2), flush=True)

    if default_t:
        src = out_dir / f"rl_qtable_{default_t}.json"
        dst = out_dir / "rl_qtable.json"
        if src.exists():
            shutil.copyfile(src, dst)
            print(f"[multi-demo] Copied {src.name} → {dst.name} (workspace default {default_t})", flush=True)

    print(
        f"[multi-demo] Report: {report_path} | by_ticker keys: {list(by_ticker.keys())} | "
        f"per-ticker Q: outputs/rl_qtable_<TICKER>.json",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.train_days <= args.lookback:
        print("train_days must be > lookback", file=sys.stderr)
        sys.exit(1)
    if args.train_days > args.total_days:
        print("train_days cannot exceed total-days", file=sys.stderr)
        sys.exit(1)

    if args.multi_demo:
        if args.warm_start:
            print("--warm-start is ignored in --multi-demo (each symbol trains from scratch)", file=sys.stderr)
        run_multi_demo(args)
        return

    q_out = Path(args.q_out)
    q, report = train_one_symbol(args, args.ticker, args.company, q_out, seed_offset=0)
    save_q_table(q)
    write_json(Path(args.report_out), report)
    print(json.dumps(report, indent=2), flush=True)
    print(f"[train] done. Report: {args.report_out}", flush=True)


if __name__ == "__main__":
    main()
