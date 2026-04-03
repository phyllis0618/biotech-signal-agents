#!/usr/bin/env python3
"""Apply trader decision to a trade_id and update RL Q-table + feedback log."""

from __future__ import annotations

import argparse

from src.trading.feedback_store import append_feedback


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record trader review for a proposed trade")
    p.add_argument("--trade-id", required=True, help="trade_id from FinalReport JSON")
    p.add_argument("--ticker", required=True)
    p.add_argument(
        "--coordinator-signal",
        required=True,
        choices=["long", "short", "no_trade"],
        help="Coordinator output (same as report.coordinator_signal)",
    )
    p.add_argument(
        "--coordinator-confidence",
        type=int,
        required=True,
        help="Coordinator confidence before RL",
    )
    p.add_argument(
        "--rl-action",
        required=True,
        choices=["long", "short", "no_trade"],
        help="RL-selected action (report.rl_action)",
    )
    p.add_argument(
        "--decision",
        required=True,
        choices=["approved", "rejected", "deferred"],
    )
    p.add_argument("--guidance", default="", help="Trader notes / instructions for the desk")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    row = append_feedback(
        trade_id=args.trade_id,
        ticker=args.ticker,
        coordinator_signal=args.coordinator_signal,
        coordinator_confidence=args.coordinator_confidence,
        rl_action_taken=args.rl_action,
        decision=args.decision,
        trader_guidance=args.guidance or None,
        update_rl=True,
    )
    print("Recorded:", row)


if __name__ == "__main__":
    main()
