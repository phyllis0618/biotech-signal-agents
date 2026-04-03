#!/usr/bin/env python3
"""
Joint backtest: Phase-1 Agentic Research -> Phase-2 BiotechTradingEnv rollout.

Example:
  python -m scripts.run_joint_backtest --ticker AMGN --company Amgen --policy buy_bias
"""

from __future__ import annotations

import argparse
import json

from src.rl.research_bridge import run_joint_rollout


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Research → RL env joint demo")
    p.add_argument("--ticker", required=True)
    p.add_argument("--company", required=True)
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--policy",
        choices=["hold", "random", "buy_bias"],
        default="buy_bias",
        help="Stub policy (connect trained PPO.act() here later)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = run_joint_rollout(
        args.ticker,
        args.company,
        max_steps=args.max_steps,
        seed=args.seed,
        policy=args.policy,
    )
    print(json.dumps(out, indent=2, ensure_ascii=True, default=str))


if __name__ == "__main__":
    main()
